"""
data server provides the class to serve data to training
 - from a collection of bin files in a filesystem
 - from messages in a stream in interactive session (to be implmeneted)
"""

import multiprocessing
import time
import os
import random
import mmap
import functools
from pathlib import Path

import numpy as np

from dataclasses import dataclass
from graph2tac.hash import get_split_label
from graph2tac.loader.helpers import get_all_fnames

import tqdm
from graph2tac.common import uuid

FILE_MULTIPLIER = np.array([2**32], dtype=np.uint64)

from graph2tac.loader.clib.loader import (
    get_data,
    get_proof_steps_size,
    build_subgraphs,
    get_step_state,
    get_step_state_text,
    get_step_label,
    get_step_label_text,
    get_step_hash_and_size,
    get_graph_constants,
    get_node_label_subgraph,
    build_def_clusters,
    get_def_cluster_subgraph,
    get_proof_step,
    get_node_label_to_hash,
    get_hash_to_name,
    capnp_unpack,
    files_get_scc_components,
    get_buf_def,
    get_buf_tactics,
    build_data_online_from_buf,
    data_online_extend,
    data_online_resize,
    get_def_deps_online,
    get_local_to_global_file_idx,
    get_scc_components,
    get_subgraph_online,
    get_proof_step_online,
    get_proof_step_online_text,
    get_graph_constants_online
)



@dataclass
class GraphConstants:
    tactic_num: int
    # tactic_max_arg_num: int
    edge_label_num: int
    base_node_label_num: int
    node_label_num: int
    cluster_subgraphs_num: int
    tactic_index_to_numargs: np.ndarray  # dtype = np.uint32
    tactic_index_to_string: list[str]    # tactic names
    tactic_index_to_hash: np.ndarray
    global_context: np.ndarray
    label_to_names: list[str]
    label_in_spine: list[bool]



@dataclass
class DataPoint:
    proof_step_idx: int
    state: tuple[np.ndarray]
    state_tf_gnn: tuple[np.ndarray]
    action: tuple[np.ndarray]
    state_text: bytes
    action_base_text: bytes
    action_interm_text: bytes
    action_text: bytes
    def_name: bytes
    step_in_proof: int
    file_name: Path


@dataclass
class DefIndexTable:
    idx_to_global_node: list
    global_node_to_idx: dict
    idx_to_name: list
    idx_to_hash: list
    idx_in_spine: list[bool]


def find_bin_files(data_dir: str):
    """
    returns the list of bin files (packed) in root dir
    """
    return [Path(data_dir).joinpath(f + '.bin') for f in get_all_fnames(data_dir, '**/*.bin')]

def find_binx_files(data_dir: str):
    """
    returns the list of binx files (unpacked) in root dir
    """
    return [Path(data_dir).joinpath(f + '.binx') for f in get_all_fnames(data_dir, '**/*.binx')]


def unpack_bin_files(data_dir):
    for fname in find_bin_files(data_dir):
        print(f"unpacking {fname}")
        capnp_unpack(os.fsencode(fname))


def top_sort_dataset_modules(data_dir: Path):
    fnames = find_bin_files(data_dir)
    root_dir = data_dir
    comp = files_get_scc_components(os.fsencode(root_dir), list(map(os.fsencode, fnames)))
    return list(map(lambda x: fnames[x], comp))

def get_global_def_table(buf_list, message_type: str, restrict_to_spine=False):
    """
    returns a list of indexed by buffer index
    each element in the above list is a list of tuples: (node_idx, def_hash, def_name)
    """
    res = []
    for buf_object in buf_list:
        temp = get_buf_def(buf_object, message_type, restrict_to_spine)
        res.append(temp)
    return res


def local_to_global_file_idx(data_dir: Path, fnames: list[Path]):
    return get_local_to_global_file_idx(os.fsencode(data_dir), list(map(os.fsencode,fnames)))

def get_global_nodes_in_spine(buf_list, message_type):
    global_nodes_in_spine = set()
    global_def_table_spine = get_global_def_table(buf_list, message_type, restrict_to_spine=True)
    for file_idx, file_table in enumerate(global_def_table_spine):
        for node_idx, def_hash, def_name in zip(*file_table):
            global_nodes_in_spine.add((file_idx * FILE_MULTIPLIER +  node_idx).item())
    return global_nodes_in_spine

def build_def_index(global_def_table, global_nodes_in_spine: set[int]) -> DefIndexTable:
    def_idx_to_global_node = []
    def_idx_to_name = []
    def_idx_to_hash = []
    global_node_to_def_idx = {}
    def_idx_in_spine = []

    for file_idx, (node_indexes, def_hashes, def_names) in enumerate(global_def_table):
        for node_idx, def_hash, def_name in zip(node_indexes, def_hashes, def_names):
            global_node_id = (file_idx * FILE_MULTIPLIER + node_idx).item()
            global_node_to_def_idx[global_node_id] = len(def_idx_to_global_node)
            def_idx_to_global_node.append(global_node_id)
            def_idx_to_name.append(def_name)
            def_idx_to_hash.append(def_hash)
            def_idx_in_spine.append(global_node_id in global_nodes_in_spine)
    return DefIndexTable(def_idx_to_global_node, global_node_to_def_idx, def_idx_to_name, def_idx_to_hash, def_idx_in_spine)




def def_hash_of_tactic_point(def_index_table, tactic_point):
    def_idx = def_index_table.global_node_to_idx[(tactic_point[2]*FILE_MULTIPLIER + tactic_point[3]).item()]
    return def_index_table.idx_to_hash[def_idx]

def def_name_of_tactic_point(def_index_table, tactic_point):
    def_idx = def_index_table.global_node_to_idx[(tactic_point[2]*FILE_MULTIPLIER + tactic_point[3]).item()]
    return def_index_table.idx_to_name[def_idx]

def def_hash_split(x, prob, seed):
    return get_split_label(x[7].item(), prob, seed)


def get_tactic_hash_num_args_collision(tactical_data, fnames):
    tactical_data = tactical_data[np.lexsort((tactical_data[:,1], tactical_data[:,0]))]
    data_error_report = []
    for tactic_group_by_hash in np.split(tactical_data, np.unique(tactical_data[:, 0], axis=0, return_index=True)[1][1:]):
        tactic_hash_arg_collisions = np.unique(tactic_group_by_hash[:,0:2], axis=0, return_index=True)[1]
        if len(tactic_hash_arg_collisions) != 1:
            collision_examples = tactic_group_by_hash[tactic_hash_arg_collisions].tolist()
            for collision_example in collision_examples:
                tactic_hash, num_args, file_idx, def_idx, proof_step_idx, outcome_idx = collision_example
                data_error_report.append(f"tactic_hash {tactic_hash}, num_args {num_args}, file {fnames[file_idx]}, def {def_idx}, proof_step {proof_step_idx}, outcome {outcome_idx}")
    return data_error_report


class Data2():
    """
    this is low-level api for refactored online loader from mmaped capnp bin files
    """
    def __init__(self, data_dir, restrict_to_spine, bfs_option, max_subgraph_size, split=(8,1,1), split_random_seed=0):
        """

        """
        self.__bfs_option = bfs_option
        self.__max_subgraph_size = max_subgraph_size
        self.__split = split
        self.__split_random_seed = split_random_seed
        t00 = time.time()
        print(f"LOADING | indexing and top sorting bin files in {data_dir}...", end='')
        self.__fnames = top_sort_dataset_modules(data_dir)
        print(f"done.")
        print(f"LOADING | preparing data from {len(self.__fnames)} files.")

        print(f"LOADING | constructing file reference table...", end='')
        self.__local_to_global_files = local_to_global_file_idx(data_dir, self.__fnames)
        print(f"done.")

        print(f"LOADING | indexing all definitions...", end='')
        t0 = time.time()

        buf_list = []
        for fname in self.__fnames:
            with open(fname,'rb') as fin:
                buf_object = memoryview(mmap.mmap(fin.fileno(), 0,
                                                  flags=mmap.MAP_SHARED,
                                                  prot=mmap.PROT_READ))
            buf_list.append(buf_object)

        self.__global_def_table = get_global_def_table(buf_list, "dataset", restrict_to_spine)


        global_nodes_in_spine = get_global_nodes_in_spine(buf_list, "dataset")
        print("LOADING | definitions in spine: {len(global_nodes_in_spine)}")

        self.__def_index_table = build_def_index(self.__global_def_table, global_nodes_in_spine)

        self.__def_idx_to_node = np.array(self.__def_index_table.idx_to_global_node, dtype=np.uint64)
        print(f"LOADING | definitions: {len(self.__def_idx_to_node)} ")


        t1 = time.time()
        print(f"Indexed {len(self.__def_index_table.idx_to_global_node)} definitions in {t1-t0:.6f} seconds.")
        print(f"LOADING | indexing all tactical action-outcomes in {len(self.__fnames)} files...", end='')
        t0 = time.time()
        self.__tactical_data = []
        for file_idx, (buf_object, fname, def_table) in enumerate(zip(buf_list, map(os.fsencode, self.__fnames), self.__global_def_table)):
            (tactic_hashes,
             number_arguments,
             def_node_indexes,
             proof_step_indexes,
             outcome_indexes,
             local_roots) = get_buf_tactics(buf_object, def_table[0], fname)

            file_indexes = np.full(def_node_indexes.shape, file_idx, dtype=np.uint64)
            self.__tactical_data.append(np.transpose(np.stack([tactic_hashes, number_arguments, file_indexes, def_node_indexes, proof_step_indexes, outcome_indexes, local_roots])))

        if len(self.__tactical_data) == 0:
            self.__tactical_data = np.zeros(shape=(0,7), dtype=np.uint64)
            self.__max_num_args = None
        else:
            self.__tactical_data = np.concatenate(self.__tactical_data)
            self.__max_num_args  = np.max(self.__tactical_data[:,1]).item()

        data_error_report = get_tactic_hash_num_args_collision(self.__tactical_data, self.__fnames)
        if data_error_report:
            raise Exception(f"the data contains tactic_hash num_args collisions, for example \n" + "\n".join(data_error_report))
        ############################################################################################################
        ####### if you want  tactic indices chronological, consider the following code                          ####
        ####### tactics_args = tactical_data[np.sort(np.unique(tactical_data[:,0], return_index=True)[1]),:2]   ####
        ############################################################################################################

        tactic_hashes = np.unique(self.__tactical_data[:,0])
        tactic_indexes = np.arange(len(tactic_hashes), dtype=np.uint32)
        def_hashes = np.array([def_hash_of_tactic_point(self.__def_index_table, tactical_point) for tactical_point in self.__tactical_data])
        def_hashes = np.reshape(def_hashes, (len(self.__tactical_data),1))
        self.__tactical_data = np.concatenate([self.__tactical_data, def_hashes], axis=1)

        t1 = time.time()
        print(f"LOADING | Indexed {len(self.__tactical_data)} tactical action-outcomes in {t1-t0:.6f} seconds.")


        print(f"LOADING | mmaping all capnp files and building data loader hash tables...", end='')
        t0 = time.time()


        self.__c_data_online = build_data_online_from_buf(
            self.__def_idx_to_node,
            tactic_hashes,
            tactic_indexes)
        print("done.")

        fname_list = list(map(os.fsencode, self.__fnames))
        n_files_in_data = data_online_extend(self.__c_data_online, buf_list, fname_list,  self.__local_to_global_files, "dataset")

        print(f"LOADING | data_online maps  {n_files_in_data}")

        t1 = time.time()
        print(f"{n_files_in_data} buffer objects processed in {t1-t0:.6f} seconds.")


        max_subgraph_size_for_definition_dependencies = max_subgraph_size
        print(f"LOADING | in def_dependencies: max_subgraph_size={max_subgraph_size} bfs_option={bfs_option}")

        print(f"LOADING | constructing shallow expansions of all definitions to build the graph of definition dependencies...", end="")
        t0 = time.time()
        self.__def_deps_ids = get_def_deps_online(self.__c_data_online, bfs_option, max_subgraph_size_for_definition_dependencies)
        t1 = time.time()
        print(f"done in {t1-t0:.6f} seconds.")
        print(f"LOADING | NOTICE: the graph of definition dependencies should be precomputed and recorded to capnp bin files at the time "
               "of generation of the dataset. It is inefficient to recompute this graph every time dataserver is initialized.")

        print(f"LOADING | building strongly connected components (def clusters) in the meta graph of definition dependencies...", end="")
        t0 = time.time()
        self.__def_scc_clusters = get_scc_components(self.__def_deps_ids)
        t1 = time.time()
        print(f"done in {t1-t0:.6f} seconds. Constructed {len(self.__def_scc_clusters)} def clusters.")
        t01 = time.time()
        print(f"LOADING | DataServer is fully initialized in {t01-t00:.6f} seconds and is ready to stream.")

    def tactical_data(self):
        return self.__tactical_data

    def graph_constants(self):
        # if we want to switch to chronological order of tactics, let's use this:
        # tactics_args = self.__tactical_data[np.sort(np.unique(self.__tactical_data[:,0], return_index=True)[1]),:2]

        tdataidx = np.unique(self.__tactical_data[:,0], return_index=True)[1].astype(np.uint64)
        thash_narg = self.__tactical_data[tdataidx, :2]
        uniquetac_idx = np.arange(thash_narg.shape[0], dtype=thash_narg.dtype).reshape((thash_narg.shape[0], 1))
        tdataidx_reshaped = tdataidx.reshape(thash_narg.shape[0],1)
        idx_thash_narg_tdataidx = np.concatenate([uniquetac_idx, thash_narg, tdataidx_reshaped], axis=1)

        tactic_num = len(tdataidx)
        tactic_index_to_hash = idx_thash_narg_tdataidx[:,1]
        tactic_index_to_numargs = idx_thash_narg_tdataidx[:,2]

        # DEPRECATE
        #if len(tactic_index_to_numargs) > 0:
        #    tactic_max_arg_num = np.max(tactic_index_to_numargs).item()
        #else:
        #    tactic_max_arg_num = 0
        # self.__tactic_max_arg_num = tactic_max_arg_num       # we should move out the mutable state of max_arg_num to the clients

        tactic_index_to_string = list(self.get_proof_step_text(tdataidx)[1] for tdataidx in idx_thash_narg_tdataidx[:,3])

        base_node_label_names, edge_label_num = get_graph_constants_online();
        base_node_label_num = len(base_node_label_names)

        global_context = np.arange(base_node_label_num, base_node_label_num + len(self.__def_index_table.idx_to_hash), dtype=np.uint32)

        label_to_names = [x.decode() for x in (base_node_label_names +  self.__def_index_table.idx_to_name)]
        label_in_spine = [True] * base_node_label_num + self.__def_index_table.idx_in_spine

        if len(global_context) > 0:
            node_label_num = np.max(global_context).item() + 1
        else:
            node_label_num = 0
        return GraphConstants(tactic_num=tactic_num,
                              # tactic_max_arg_num=tactic_max_arg_num,  #DEPRECATE
                              edge_label_num=edge_label_num,
                              base_node_label_num=base_node_label_num,
                              node_label_num=node_label_num,
                              cluster_subgraphs_num=len(self.__def_scc_clusters),
                              tactic_index_to_numargs=tactic_index_to_numargs,
                              tactic_index_to_string=tactic_index_to_string,
                              tactic_index_to_hash=tactic_index_to_hash,
                              global_context=global_context,
                              label_to_names=label_to_names,
                              label_in_spine=label_in_spine)


    def def_name_of_tactic_point(self, data_point_idx):
        return def_name_of_tactic_point(self.__def_index_table, self.__tactical_data[data_point_idx])


    def get_proof_step_text(self, data_point_idx):
        tactic_point = self.__tactical_data[data_point_idx]
        state_text, action_base_text, action_interm_text, action_text = get_proof_step_online_text(self.__c_data_online, tactic_point[2:6])
        return state_text, action_base_text, action_interm_text, action_text


    def get_proof_step(self, data_point_idx,
                       # tactic_max_arg_num=None,  #DEPRECATE
                       skip_text=False):
        # if tactic_max_arg_num is None:
        #    assert self.__tactic_max_arg_num is not None   #DEPRECATE
        # max_arg_size = self.__tactic_max_arg_num
        tactic_point = self.__tactical_data[data_point_idx]
        tactic_hash, num_args, file_idx, def_node_idx, proof_step_idx, outcome_idx, state_root_node, def_hash = tactic_point
        res = get_subgraph_online(
            self.__c_data_online, file_idx*FILE_MULTIPLIER + state_root_node, self.__bfs_option, self.__max_subgraph_size, False)
        nodes, edges, edge_labels, edges_offset, global_visited, _, _, _, _ = res
        edges_grouped_by_label = np.split(edges, edges_offset)

        context, root, tactic_index, args = get_proof_step_online(self.__c_data_online, tactic_point[2:6], global_visited)

        # REFACTOR and move out the following 3 lines to network files
        # tail_args = np.tile(np.array([[0,len(context)]]), (max_arg_size-len(args),1))
        # a = np.full((len(args),), True)
        # b = np.full((max_arg_size - len(args),), False)
        # mirek_mask = np.concatenate([a,b])
        # mirek_args = np.concatenate([args, tail_args], axis=0)

        def_name = def_name_of_tactic_point(self.__def_index_table, tactic_point)

        if not skip_text:
            state_text, action_base_text, action_interm_text, action_text = self.get_proof_step_text(data_point_idx)
        else:
            state_text, action_base_text, action_interm_text, action_text = "","","",""

        # TEXT ANNOTATION TO BE IMPLEMENTED
        return DataPoint(data_point_idx,
                         state=(nodes, edges_grouped_by_label, root, context),
                         state_tf_gnn=(nodes, edges[:,0], edges[:,1], edge_labels, root, context),
                         action=(tactic_index, args),
                         state_text=state_text,
                         action_base_text=action_base_text,
                         action_interm_text=action_interm_text,
                         action_text=action_text,
                         def_name=def_name,
                         step_in_proof=proof_step_idx.item(),
                         file_name=self.__fnames[file_idx])



    def select_data_points(self, split_fun, label):
        """
        returns a list of datapoint indices that have given label

        split is a function that you provide: data_point --> label

        since datapoint has all fields addressing the datapoint (definition, file, proof step etc)
        you can be flexible with the splits
        """
        return [i for i, tactical_point in enumerate(self.__tactical_data) if split_fun(tactical_point) == label]

    def data_train(self):
        return self.select_data_points(split_fun=lambda x: def_hash_split(x, self.__split, self.__split_random_seed), label=0)

    def data_valid(self):
        return self.select_data_points(split_fun=lambda x: def_hash_split(x, self.__split, self.__split_random_seed), label=1)

    def data_test(self):
        return self.select_data_points(split_fun=lambda x: def_hash_split(x, self.__split, self.__split_random_seed), label=2)



    def step_state(self, data_point_idx: int, tf_gnn=False):
        if tf_gnn:
            return self.get_proof_step(data_point_idx, skip_text=True).state_tf_gnn
        else:
            return self.get_proof_step(data_point_idx, skip_text=True).state

    def step_state_text(self, data_point_idx: int):
        return self.get_proof_step(data_point_idx).state_text

    def step_label_text(self, data_point_idx: int):
        res = self.get_proof_step(data_point_idx)
        return (res.action_base_text, res.action_interm_text, res.action_text)

    def step_label(self, data_point_idx):
        return self.get_proof_step(data_point_idx, skip_text=True).action

    def def_cluster_subgraph(self, cluster_idx, tf_gnn=False, max_arg_size=None):
        if max_arg_size is None:
            max_arg_size = self.__max_subgraph_size

        def_cluster = self.__def_scc_clusters[cluster_idx]

        res = get_subgraph_online(self.__c_data_online, self.__def_idx_to_node[def_cluster], self.__bfs_option, max_arg_size, False)

        nodes, edges, edge_labels, edges_offset, global_visited, _, _, _, _ = res
        edges_grouped_by_label = np.split(edges, edges_offset)

        if tf_gnn:
            return (nodes, edges[:, 0], edges[:, 1], edge_labels, len(def_cluster))
        else:
            return (nodes, edges_grouped_by_label, len(def_cluster))






class Data():
    """
    this is low-level api class which will provide the sources
    of state action pairs to the training network
    using as its own source various mechanisms:

    - importing from a set of bin files in a directory
    - from interactive session with coq
    """

    def __init__(self, project_dir: Path,   split_random_seed=0):
        self.__default_num_proc = max(1, multiprocessing.cpu_count() // 2)
        self.__project_dir = project_dir
        os.makedirs(self.__project_dir, exist_ok=True)
        if self.__project_dir.is_dir():
            print(f"Data: using {str(self.__project_dir)} for processed data storage")
        else:
            raise Exception(f"Failed to create or find {project_dir}")
        self.__fnames = None
        self.__c_data = None
        self.__split_labels = None
        self.__train_steps = None
        self.__valid_steps = None
        self.__test_steps = None
        self.__split_random_seed = split_random_seed
        print(f"LOADING | ")

    def file_name(self, file_idx) -> str:
        return self.__fnames[file_idx]

    def import_data(self, data_dir: Path,
                             bfs_option,
                             max_subgraph_size,
                             split=(8,1,1),
                             num_proc=None,
                             ignore_def_hash=False):
        """
        imports data from capnp .bin files using capnp v3 format
        """
        t0 = time.time()


        self.__fnames = top_sort_dataset_modules(data_dir)

        print(f"LOADING | processing {len(self.__fnames)} .binx files")

        if num_proc is None:
            num_proc = self.__default_num_proc

        self.__c_data = get_data(os.fsencode(data_dir),
                                           os.fsencode(self.__project_dir),
                                           list(map(os.fsencode, self.__fnames)),
                                           num_proc, bfs_option, max_subgraph_size, ignore_def_hash)
        t1 = time.time()
        print(f"LOADING | c_data call complete")
        print(f"LOADING | building dataset complete in {t1-t0} seconds")

        self.__split_dataset(split)
        t2 = time.time()
        print(f"LOADING | splitting data to train, valid, test complete in {t2-t1} seconds")

        build_def_clusters(self.__c_data, bfs_option, max_subgraph_size)
        t3 = time.time()
        print(f"LOADING | building def clusters complete {t3-t2} seconds")


    def __split_dataset(self, split):
        self.__split_labels = [get_split_label(self.step_hash(i), split,
                                               seed=self.__split_random_seed)
                               for i in range(self.proof_steps_size())]

        self.__train_steps = [step_idx for step_idx in range(self.proof_steps_size())
                              if self.__split_labels[step_idx] == 0]

        self.__valid_steps = [step_idx for step_idx in range(self.proof_steps_size())
                              if self.__split_labels[step_idx] == 1]

        self.__test_steps  = [step_idx for step_idx in range(self.proof_steps_size())
                              if self.__split_labels[step_idx] == 2]


    def proof_steps_size(self):
        """
        returns number of proof steps
        """
        return get_proof_steps_size(self.__c_data)

    def get_proof_step(self, proof_step_idx):
        """
        returns a raw proof_step tuple
        """
        return get_proof_step(self.__c_data, proof_step_idx)


    def step_state(self, proof_step: int, tf_gnn=False):
        """
        returns the step state by default in the format:
        (node colors, edges grouped by type, root, context)

        if tf_gnn then returns
        (node_colors, edge_sources, edge_targets, edge_labels, root, context)
        """
        node_labels, edges, edge_labels, edges_split_offset, root, context = get_step_state(self.__c_data, proof_step)

        if tf_gnn:
            return (node_labels, edges[:, 0], edges[:, 1], edge_labels, root, context)
        else:
            edges_grouped_by_label = np.split(edges, edges_split_offset)
            return (node_labels, edges_grouped_by_label, root, context)

    def step_state_text(self, proof_step) -> bytes:
        """
        returns step state text as python bytes
        """
        return get_step_state_text(self.__c_data, proof_step)



    def node_label_subgraph(self, node_label: int, tf_gnn=False):
        """
        returns the subgraph for the definition node passed by its assigned node label
        """
        assert (self.graph_constants().base_node_label_num <=
                node_label
                < self.graph_constants().node_label_num), "the node_label must be of used_definition"
        node_labels, edges, edges_labels, edges_split_offset = get_node_label_subgraph(self.__c_data, node_label)
        if tf_gnn:
            return (node_labels, edges[:, 0], edges[:, 1], edges_labels)
        else:
            edges_grouped_by_label = np.split(edges, edges_split_offset)
            return (node_labels, edges_grouped_by_label)


    def def_cluster_subgraph(self, cluster_idx: int, tf_gnn=False):
        """
        returns subgraph for the def cluster_idx in the same format as node_label_subgraph
        and also number of roots (entry points to the cluster strongly connected component)
        """
        # assert (0 <= cluster_idx < self.graph_constants().cluster_subgraphs_num), f"the cluster_idx {cluster_idx} must be bounded by cluster_subgraphs_num {self.graph_constants().cluster_subgraphs_num}"

        node_labels, edges, edges_labels, edges_split_offset, nroots = get_def_cluster_subgraph(self.__c_data, cluster_idx)

        if tf_gnn:
            return (node_labels, edges[:, 0], edges[:, 1], edges_labels, nroots)
        else:
            edges_grouped_by_label = np.split(edges, edges_split_offset)
            return (node_labels, edges_grouped_by_label, nroots)


    def step_label(self, proof_step: int):
        """
        return the step local label in the v3 format:
        (tac_idx, list[ctxt_idx], mask_args)
        """
        return get_step_label(self.__c_data, proof_step, True)

    def step_label_text(self, proof_step: int):
        """
        return the step local label in the text format:
        tactic_base_text, tactic_interm_text, tactic_text
        """
        return get_step_label_text(self.__c_data, proof_step)

    def step_hash(self, proof_step: int):
        """
        return the step hash as defined by the definition hash
        """
        return get_step_hash_and_size(self.__c_data, proof_step)[0]

    def step_nodes_size(self, proof_step: int):
        """
        returns the step node size
        """
        return get_step_hash_and_size(self.__c_data, proof_step)[1]

    def data_train(self):
        """
        returns the list of step_idx with the train label
        """
        return self.__train_steps

    def data_valid(self):
        """
        returns the list of step_idx with the valid label
        """
        return self.__valid_steps

    def data_test(self):
        """
        returns the list of step_idx with the valid label
        """
        return self.__test_steps

    def graph_constants(self):
        """
        returns
        tactic_num: int
#        tactic_max_arg_num: int   DEPRECATE
        edge_label_num: int
        base_node_label_num: int
        node_label_num: int
        cluster_subgraphs_num: int
        tactic_index_to_numargs: np.ndarray  # dtype = np.uint32
        tactic_index_to_string: list[str]    # tactic names
        tactic_index_to_hash: np.ndarray
        global_context: np.ndarray
        node_label_to_hash: np.ndarray
        class_to_names: list[str]

        packed as GraphConstants dataclass object
        """
        assert False, "need to remove tactic_max_arg_num from the second position in legacy graph_constants "
        # OR JUST REMOVE tactic_max_arg_num when packaging to graph_constants
        return GraphConstants(* get_graph_constants(self.__c_data))

    def node_label_to_hash(self):
        """
        debug function: color to hash
        """
        return get_node_label_to_hash(self.__c_data)


    def hash_to_name(self, node_hash: int):
        """
        debug function: hash to name
        """
        return get_hash_to_name(self.__c_data, node_hash)





class Iterator:
    def __init__(self, source, *fs):
        self.__stream__ = iter(source)
        self.__fs = fs
        if hasattr(source, '__len__'):
            self.__size = len(source)
        elif hasattr(source, '__length_hint__'):
            self.__size = source.__length_hint__()
        else:
            self.__size = None

    def __iter__(self):
        return self

    def __next__(self):
        step = next(self.__stream__)
        return tuple(f(step) for f in self.__fs)

    def __len__(self):
        return self.__size





class DataServer:
    """
    implements higher level api compatible with current train.py
    wrapping lower level of Data()
    """
    def __init__(self,
                 data_dir: Path,
                 work_dir: Path = Path('.'),
                 split: tuple[int] = (8, 1, 1),
                 cross_valid_fold: int = 0,
                 bfs_option = False,
                 max_subgraph_size: int = 20000,
                 split_random_seed = 0,
                 num_proc = None,
                 ignore_def_hash: bool = True,
                 restrict_to_spine: bool = False,
                 legacy = False
    ):
        """
        - data_dir is the directory containing capnp schema file and *.bin files

        - split is a len 3 tuple of integers defining the relative
        proportion of split as (train, valid, test).

        - cross_fold_level is an integer in the range [0, max_cross_fold)
        where max_cross_fold = (split[0] + split[1]) // split[0]
        with default split = [8, 1, 1] we have max_cross_fold = (8 + 1) // 1 = 9
        and therefore cross_valid_fold can take values in range(9)
        (that is from 0 to 8 including) --
        NOT YET IMPLEMENTED

        - num_proc number of threads for the loader
            (if not passed or None then the optimal number of threads if computed by the loader)

        ignore_def_hash: used uint64 of (file_idx, node_idx) instead of def hash
        for internal def tables
        """
        assert cross_valid_fold == 0, "cross_valid_fold is not yet implemented"

        self.__online_data = Data2(data_dir, restrict_to_spine, bfs_option, max_subgraph_size, split, split_random_seed=split_random_seed)


        self.__cluster_subgraphs = None
        self.__cluster_subgraphs_tf_gnn = None

        if legacy:
            self.__project_dir = work_dir.joinpath(uuid(data_dir))
            self.__data = Data(self.__project_dir, split_random_seed=split_random_seed)
            self.__data.import_data(data_dir, bfs_option, max_subgraph_size,
                                             split, num_proc=num_proc, ignore_def_hash=ignore_def_hash)


    def data_point(self, proof_step_idx, legacy=False):
        if not legacy:
            return self.__online_data.get_proof_step(proof_step_idx)
        else:
            proof_step_raw = self.__data.get_proof_step(proof_step_idx)
            return DataPoint(proof_step_idx,
                             self.__data.step_state(proof_step_idx, tf_gnn=False),
                             self.__data.step_state(proof_step_idx, tf_gnn=True),
                             self.__data.step_label(proof_step_idx),
                             self.__data.step_state_text(proof_step_idx),
                             self.__data.step_label_text(proof_step_idx)[0],
                             self.__data.step_label_text(proof_step_idx)[1],
                             self.__data.step_label_text(proof_step_idx)[2],
                             proof_step_raw[2],
                             proof_step_raw[3],
                             self.__data.file_name(proof_step_raw[10])
                             )



    def data_train(self, shuffled=False, tf_gnn=False, as_text=False, legacy=False):
        """
        returns a stream of training data, optionally shuffled
        with the default format (state, action) where

        state = (node_labels, edges_grouped_by_label, root, context)
        action = (tactic_id, argument_list_of_context_indices, mask of argument list)

        with option tf_gnn
        instead of edges_grouped_by_label you get edges in the format
        (sources, targets, classes)

        with option as_text
        returns state as a state_text (python bytes)

        returns action as a tuple (tactic_base_text, tactic_interm_text, tactic_text)
        (depending on your goals choose what you use from this tuple)
        """
        data = self.__data if legacy else self.__online_data

        shuffled_copy = list(data.data_train())
        if shuffled:
            random.shuffle(shuffled_copy)

        if as_text:
            return Iterator(shuffled_copy,
                            data.step_state_text,
                            data.step_label_text,
                            lambda x: x)
        else:
            return Iterator(shuffled_copy,
                            functools.partial(data.step_state, tf_gnn=tf_gnn),
                            functools.partial(data.step_label),
                            lambda x: x)



    def data_valid(self, tf_gnn=False, as_text=False, legacy=False):
        """
        returns a stream of validation data in the same format as  data_train
        see doc_string for data_train for the format
        """
        data = self.__data if legacy else self.__online_data


        if as_text:
            return Iterator(data.data_valid(),
                            data.step_state_text,
                            data.step_label_text,
                            lambda x: x)
        else:
            return Iterator(data.data_valid(),
                            functools.partial(data.step_state, tf_gnn=tf_gnn),
                            functools.partial(data.step_label),
                            lambda x: x)




    def def_class_subgraph(self, node_label_idx, tf_gnn, legacy=False):
        assert legacy, "Not implemented in refactored loader, who use this function?"
        """
        get the subgraph associated to the definition index in compressed
        sense. The node_label_idx must be in the range
        [graph_constants().base_node_label_num: graph_constants().node_label_num]

        tf_gnn=False returns
        (nodes, edges_grouped_by_label)

        tf_gnn=True returns
        (nodes, edge_sources, edge_targets, edge_labels)
        """

        return self.__data.node_label_subgraph(node_label_idx, tf_gnn=tf_gnn)

    def def_cluster_subgraph(self, cluster_idx, tf_gnn, legacy=False):
        """
        returns a definition cluster with cluster_idx
        in the same format as def_class_subgraph (see docstring there)

        but also an integer K that denotes the number of definitions in this cluster
        the entry points to the definition nodes are the first K in the list of returned nodes
        """
        data = self.__data if legacy else self.__online_data
        return data.def_cluster_subgraph(cluster_idx, tf_gnn=tf_gnn)


    def def_cluster_subgraphs(self, tf_gnn):
        """
        returns a list of subgraphs for definition clusters
        in the same format as def_class_subgraph (see docstring there)

        but also an integer K that denotes the number of definitions in this cluster
        the entry points to the definition nodes are the first K in the list of returned nodes

        """
        print(f"LOADING | requested {self.graph_constants().cluster_subgraphs_num}")
        if tf_gnn:
            if self.__cluster_subgraphs_tf_gnn is None:
                self.__cluster_subgraphs_tf_gnn = []
                for idx in tqdm.tqdm(range(self.graph_constants().cluster_subgraphs_num)):
                    self.__cluster_subgraphs_tf_gnn.append(self.def_cluster_subgraph(idx, tf_gnn))
            return self.__cluster_subgraphs_tf_gnn
        else:
            if self.__cluster_subgraphs is None:
                self.__cluster_subgraphs = []
                for idx in tqdm.tqdm(range(self.graph_constants().cluster_subgraphs_num)):
                    self.__cluster_subgraphs.append(self.def_cluster_subgraph(idx, tf_gnn))
            return self.__cluster_subgraphs


    def graph_constants(self, legacy=False):
        if legacy:
            print("WARNING! Legacy graph constants")
            return self.__data.graph_constants()
        else:
            return self.__online_data.graph_constants()


    def _debug_data(self):
        """
        you shouldn't call this method
        for debug purposes only
        """
        return self.__data

    def _debug_online_data(self):
        return self.__online_data
