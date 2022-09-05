"""
data server provides the class to serve data to training
 - from a collection of bin files in a filesystem
 - from messages in a stream in interactive session (to be implemented)
"""
from typing import List, Tuple, Callable, NewType

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

# nodes, edges, edge_labels, edge_offsets
LoaderGraph = NewType('LoaderGraph', Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray])

# name, step, is_faithful
ProofstateMetadata = NewType('ProofstateMetadata', Tuple[bytes, int, bool])

# context_node_ids, available_global_context
ProofstateContext = NewType('ProofstateContext', Tuple[np.ndarray, np.ndarray])

# loader_graph, root, context, metadata
LoaderProofstate = NewType('LoaderProofstate', Tuple[LoaderGraph, int, ProofstateContext, ProofstateMetadata])

# loader_graph, num_definitions, definition_names
LoaderDefinition = NewType('LoaderDefinition', Tuple[LoaderGraph, int, np.ndarray])


from graph2tac.loader.clib.loader import (
    files_get_scc_components,
    get_buf_def,
    get_def_previouses,
    get_buf_tactics,
    build_data_online_from_buf,
    data_online_extend,
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
    edge_label_num: int
    base_node_label_num: int
    node_label_num: int
    cluster_subgraphs_num: int
    tactic_index_to_numargs: np.ndarray  # dtype = np.uint32
    tactic_index_to_string: List[bytes]    # tactic names
    tactic_index_to_hash: np.ndarray
    global_context: np.ndarray
    label_to_names: List[str]
    label_in_spine: List[bool]
    max_subgraph_size: int


@dataclass
class DataPoint:
    proof_step_idx: int
    graph: LoaderGraph
    local_context: np.ndarray
    available_global_context: np.ndarray
    root: int
    action: Tuple[np.ndarray, np.ndarray]
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
    idx_to_global_context: list[set] 


def graph_as(model: str, graph: Tuple) -> Tuple:
    """
    input: the graph as loader returns it
    output: the graph in a format that model consumes where the model is "tf2" or "tfgnn"

    """
    nodes, edges, edge_labels, edges_offset = graph

    if model == "tf_gnn":
        return nodes, edges[:, 0], edges[:, 1], edge_labels
    if model == "tf2":
        edges_grouped_by_label = np.split(edges, edges_offset)
        return nodes, edges_grouped_by_label

    raise Exception(f"currently supported graph format is for models tf2 or tfgnn, but received {model}")


def find_bin_files(data_dir: Path) -> List[Path]:
    """
    returns the list of bin files (packed) in root dir
    """
    return [Path(data_dir).joinpath(f + '.bin') for f in get_all_fnames(data_dir, '**/*.bin')]


def find_binx_files(data_dir: Path) -> List[Path]:
    """
    returns the list of binx files (unpacked) in root dir
    """
    return [Path(data_dir).joinpath(f + '.binx') for f in get_all_fnames(data_dir, '**/*.binx')]


def top_sort_dataset_modules(data_dir: Path) -> List[Path]:
    fnames = find_bin_files(data_dir)
    root_dir = data_dir
    comp = files_get_scc_components(os.fsencode(root_dir), list(map(os.fsencode, fnames)))
    return list(map(lambda x: fnames[x], comp))


def get_global_def_table(buf_list, message_type: str, restrict_to_spine=False):
    """
    returns a list of buffer_def_tables
    buffer_def_table is ((node_idx_list, def_hash_list, def_name_list), representative)
    """
    return [get_buf_def(buf_object, message_type, restrict_to_spine) for buf_object in buf_list]

def local_to_global_file_idx(data_dir: Path, fnames: list[Path]):
    return get_local_to_global_file_idx(os.fsencode(data_dir), list(map(os.fsencode,fnames)))


def get_global_nodes_in_spine(buf_list, message_type):
    global_nodes_in_spine = set()
    global_def_table_spine = get_global_def_table(buf_list, message_type, restrict_to_spine=True)
    for file_idx, (file_table, representative) in enumerate(global_def_table_spine):
        for node_idx, def_hash, def_name in zip(*file_table):
            global_nodes_in_spine.add((file_idx,node_idx))
    return global_nodes_in_spine


def process_def_previouses(buf_list, message_type, def_idx_to_global_context,
                           def_idx_to_global_node, global_node_to_def_idx,  def_idx, local_to_global, representatives):
    if def_idx_to_global_context[def_idx] is not None:
        return
            
    def_global_node = def_idx_to_global_node[def_idx]

    def_file_idx, def_node_idx = def_global_node

    def_loc_previouses, def_ext_previouses = get_def_previouses(buf_list[def_file_idx], message_type, def_node_idx)


    previous_nodes = []
    for node_idx in def_loc_previouses:
        previous_nodes.append((def_file_idx,  node_idx))
    for dep_idx in def_ext_previouses:
        p_file_idx = local_to_global[def_file_idx][dep_idx]
        previous_nodes.append((p_file_idx, representatives[p_file_idx]))

    def_global_context = set()

    for previous_global_node in previous_nodes:

        previous_idx = global_node_to_def_idx[previous_global_node]
        
        process_def_previouses(buf_list, message_type, def_idx_to_global_context,
                               def_idx_to_global_node, global_node_to_def_idx, previous_idx, local_to_global, representatives)

        def_global_context.add(previous_idx)
        def_global_context.update(def_idx_to_global_context[previous_idx])

    def_idx_to_global_context[def_idx] = def_global_context

    
    

def build_def_index(global_def_table,
                    global_nodes_in_spine: set[tuple[int,int]],
                    buf_list,
                    message_type,
                    local_to_global
                    ) -> DefIndexTable:
    def_idx_to_global_node = []
    def_idx_to_name = []
    def_idx_to_hash = []
    global_node_to_def_idx = {}
    def_idx_in_spine = []
    
    representatives = []
    for file_idx, (def_data, representative) in enumerate(global_def_table):
        representatives.append(representative)
        for node_idx, def_hash, def_name in zip(*def_data):
            global_node_id = (file_idx, node_idx)
            global_node_to_def_idx[global_node_id] = len(def_idx_to_global_node)
            
            def_idx_to_global_node.append(global_node_id)
            def_idx_to_name.append(def_name)
            def_idx_to_hash.append(def_hash)
            def_idx_in_spine.append(global_node_id in global_nodes_in_spine)

    # in a second pass build global context per definition (need another pass because

    def_idx_to_global_context = [None for _ in def_idx_to_global_node]
    for (def_idx, def_global_node) in enumerate(def_idx_to_global_node):
        process_def_previouses(buf_list, message_type, def_idx_to_global_context, def_idx_to_global_node, global_node_to_def_idx, def_idx, local_to_global, representatives)


    return DefIndexTable(def_idx_to_global_node, global_node_to_def_idx, def_idx_to_name, def_idx_to_hash, def_idx_in_spine, def_idx_to_global_context)





def def_hash_of_tactic_point(def_index_table, tactic_point):
    def_idx = def_index_table.global_node_to_idx[(tactic_point[2].item(), tactic_point[3].item())]

    return def_index_table.idx_to_hash[def_idx]


def def_name_of_tactic_point(def_index_table, tactic_point):
    def_idx = def_index_table.global_node_to_idx[(tactic_point[2].item(), tactic_point[3].item())]
    return def_index_table.idx_to_name[def_idx]


def def_hash_split(tactical_point, prob: List[float], seed: int) -> int:
    return get_split_label(tactical_point[7].item(), prob, seed)


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


class Data2:
    """
    this is low-level api for refactored online loader from mmaped capnp bin files
    """

    # TODO (after TF2 deprecation): Remove data-splitting functionality, training logic should take care of this
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

        self.__def_index_table = build_def_index(self.__global_def_table, global_nodes_in_spine, buf_list, "dataset", self.__local_to_global_files)

        self.__def_idx_to_node = np.array(self.__def_index_table.idx_to_global_node, dtype=np.uint32)
        print(f"LOADING | definitions: {len(self.__def_idx_to_node)} ")


        t1 = time.time()
        print(f"Indexed {len(self.__def_index_table.idx_to_global_node)} definitions in {t1-t0:.6f} seconds.")
        print(f"LOADING | indexing all tactical action-outcomes in {len(self.__fnames)} files...", end='')
        t0 = time.time()
        self.__tactical_data = []
        for file_idx, (buf_object, fname, (def_table, representative)) in enumerate(zip(buf_list, map(os.fsencode, self.__fnames), self.__global_def_table)):
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

        tactic_index_to_string = list(self.get_proof_step_text(tdataidx)[1] for tdataidx in idx_thash_narg_tdataidx[:,3])

        base_node_label_names, edge_label_num = get_graph_constants_online()
        base_node_label_num = len(base_node_label_names)

        global_context = np.arange(base_node_label_num, base_node_label_num + len(self.__def_index_table.idx_to_hash), dtype=np.uint32)

        label_to_names = [x.decode() for x in (base_node_label_names +  self.__def_index_table.idx_to_name)]
        label_in_spine = [True] * base_node_label_num + self.__def_index_table.idx_in_spine

        if len(global_context) > 0:
            node_label_num = np.max(global_context).item() + 1
        else:
            node_label_num = 0
        return GraphConstants(tactic_num=tactic_num,
                              edge_label_num=edge_label_num,
                              base_node_label_num=base_node_label_num,
                              node_label_num=node_label_num,
                              cluster_subgraphs_num=len(self.__def_scc_clusters),
                              tactic_index_to_numargs=tactic_index_to_numargs,
                              tactic_index_to_string=tactic_index_to_string,
                              tactic_index_to_hash=tactic_index_to_hash,
                              global_context=global_context,
                              label_to_names=label_to_names,
                              label_in_spine=label_in_spine,
                              max_subgraph_size=self.__max_subgraph_size)

    def def_index_table(self):
        return self.__def_index_table
    
    def def_name_of_tactic_point(self, data_point_idx):
        return def_name_of_tactic_point(self.__def_index_table, self.__tactical_data[data_point_idx])

    def get_proof_step_text(self, data_point_idx):
        tactic_point = self.__tactical_data[data_point_idx]
        state_text, action_base_text, action_interm_text, action_text = get_proof_step_online_text(self.__c_data_online, tactic_point[2:6])
        return state_text, action_base_text, action_interm_text, action_text

    def get_proof_step(self, data_point_idx,
                       skip_text=False):
        tactic_point = self.__tactical_data[data_point_idx]
        tactic_hash, num_args, file_idx, def_node_idx, proof_step_idx, outcome_idx, state_root_node, def_hash = tactic_point
        res = get_subgraph_online(
            self.__c_data_online,
            np.array([(file_idx.item(), state_root_node.item())], dtype=np.uint32),
            self.__bfs_option, self.__max_subgraph_size, False)
        nodes, edges, edge_labels, edges_offset, global_visited, _, _, _, _ = res

        context, root, tactic_index, args = get_proof_step_online(self.__c_data_online, tactic_point[2:6], global_visited)


        def_name = def_name_of_tactic_point(self.__def_index_table, tactic_point)

        if not skip_text:
            state_text, action_base_text, action_interm_text, action_text = self.get_proof_step_text(data_point_idx)
        else:
            state_text, action_base_text, action_interm_text, action_text = "","","",""

        # TEXT ANNOTATION TO BE IMPLEMENTED
        return DataPoint(data_point_idx,
                         graph=LoaderGraph( (nodes, edges, edge_labels, edges_offset) ),
                         local_context=context,
                         available_global_context=self.step_global_context(data_point_idx),
                         root = root,
                         action=(tactic_index, args),
                         state_text=state_text,
                         action_base_text=action_base_text,
                         action_interm_text=action_interm_text,
                         action_text=action_text,
                         def_name=def_name,
                         step_in_proof=proof_step_idx.item(),
                         file_name=self.__fnames[file_idx])

    # TODO (after TF2 deprecation): Remove data-splitting functionality, training logic should take care of this
    def select_data_points(self, split_fun: Callable[[...], int], label: int) -> List[int]:
        """
        returns a list of datapoint indices that have given label

        split is a function that you provide: data_point --> label

        since datapoint has all fields addressing the datapoint (definition, file, proof step etc)
        you can be flexible with the splits
        """
        return [i for i, tactical_point in enumerate(self.__tactical_data) if split_fun(tactical_point) == label]

    # TODO (after TF2 deprecation): Remove data-splitting functionality, training logic should take care of this
    def data_train(self):
        return self.select_data_points(split_fun=lambda x: def_hash_split(x, self.__split, self.__split_random_seed),
                                       label=0)

    # TODO (after TF2 deprecation): Remove data-splitting functionality, training logic should take care of this
    def data_valid(self):
        return self.select_data_points(split_fun=lambda x: def_hash_split(x, self.__split, self.__split_random_seed),
                                       label=1)

    # TODO (after TF2 deprecation): Remove data-splitting functionality, training logic should take care of this
    def data_test(self):
        return self.select_data_points(split_fun=lambda x: def_hash_split(x, self.__split, self.__split_random_seed),
                                       label=2)

    def step_state(self, data_point_idx: int) -> LoaderProofstate:
        proof_step = self.get_proof_step(data_point_idx, skip_text=False)

        loader_graph = proof_step.graph

        root = proof_step.root

        context_node_ids = proof_step.local_context
        available_global_context = proof_step.available_global_context
        context = ProofstateContext( (context_node_ids, available_global_context) )

        name = proof_step.def_name
        step = proof_step.step_in_proof
        is_faithful = proof_step.action_text.replace(b'@', b'')==proof_step.action_interm_text.replace(b'@', b'')
        metadata = ProofstateMetadata( (name, step, is_faithful) )

        return LoaderProofstate( (loader_graph, root, context, metadata) )

    def step_global_context(self, data_point_idx: int) -> np.ndarray:
        """
        return dynamic global context as a numpy array of indices into GraphConstants.global_context
        """
        _, _, file_idx, def_node_idx, _, _, _, _  = self.__tactical_data[data_point_idx]
        def_idx = self.__def_index_table.global_node_to_idx[(file_idx, def_node_idx)]
        # def_name = self.__def_index_table.idx_to_name[def_idx]
        def_global_context = self.__def_index_table.idx_to_global_context[def_idx]
        return np.array(list(def_global_context), dtype=np.uint32)


    def step_state_text(self, data_point_idx: int):
        return self.get_proof_step(data_point_idx).state_text

    def step_label_text(self, data_point_idx: int):
        res = self.get_proof_step(data_point_idx)
        return res.action_base_text, res.action_interm_text, res.action_text

    def step_label(self, data_point_idx):
        return self.get_proof_step(data_point_idx, skip_text=True).action

    def def_cluster_subgraph(self, cluster_idx,  max_arg_size=None) -> LoaderDefinition:
        if max_arg_size is None:
            max_arg_size = self.__max_subgraph_size

        def_cluster = self.__def_scc_clusters[cluster_idx]

        res = get_subgraph_online(self.__c_data_online, self.__def_idx_to_node[def_cluster], self.__bfs_option, max_arg_size, False)

        nodes, edges, edge_labels, edges_offset, _ , _, _, _, _ = res

        base_node_label_names, _ = get_graph_constants_online()
        label_to_names = np.array(base_node_label_names +  self.__def_index_table.idx_to_name)

        num_definitions = len(def_cluster)
        definition_labels = nodes[:num_definitions]
        definition_names = label_to_names[definition_labels]

        graph = LoaderGraph( (nodes, edges, edge_labels, edges_offset) )
        return LoaderDefinition( (graph, num_definitions, definition_names) )


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
    # TODO (after TF2 deprecation): Remove data-splitting functionality, training logic should take care of this
    def __init__(self,
                 data_dir: Path,
                 max_subgraph_size,
                 split: Tuple[int, int, int] = (8, 1, 1),
                 bfs_option = True,
                 split_random_seed = 0,
                 restrict_to_spine: bool = False,
    ):
        """
        - data_dir is the directory containing capnp schema file and *.bin files

        - split is a len 3 tuple of integers defining the relative
        proportion of split as (train, valid, test).

        - num_proc number of threads for the loader
            (if not passed or None then the optimal number of threads if computed by the loader)

        ignore_def_hash: used uint64 of (file_idx, node_idx) instead of def hash
        for internal def tables
        """
        self.__online_data = Data2(data_dir, restrict_to_spine, bfs_option, max_subgraph_size, split, split_random_seed=split_random_seed)
        self.__cluster_subgraphs = None

    # TODO (after TF2 deprecation): Remove this API, data should be accessible in bulk, use Data2 for debugging
    def data_point(self, proof_step_idx: int):
        return self.__online_data.get_proof_step(proof_step_idx)

    # TODO (after TF2 deprecation): Remove shuffling from here, training logic should take care of this
    def data_train(self, shuffled: bool = False,  as_text: bool = False):
        """
        returns a stream of training data, optionally shuffled
        with the default format (state, action) where

        state = (graph, root, context)
        action = (tactic_id, argument_list_of_context_indices, mask of argument list)

        with option as_text
        returns state as a state_text (python bytes)

        returns action as a tuple (tactic_base_text, tactic_interm_text, tactic_text)
        (depending on your goals choose what you use from this tuple)
        """
        data =  self.__online_data

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
                            functools.partial(data.step_state),
                            functools.partial(data.step_label),
                            lambda x: x)

    # TODO (after TF2 deprecation): Remove train / validation split, training logic should take care of this
    def data_valid(self, as_text: bool = False):
        """
        returns a stream of validation data in the same format as  data_train
        see doc_string for data_train for the format
        """
        data = self.__online_data


        if as_text:
            return Iterator(data.data_valid(),
                            data.step_state_text,
                            data.step_label_text,
                            lambda x: x)
        else:
            return Iterator(data.data_valid(),
                            functools.partial(data.step_state),
                            functools.partial(data.step_label),
                            lambda x: x)

    # TODO (after TF2 deprecation): Remove this API, data should be accessible in bulk, use Data2 for debugging
    def def_cluster_subgraph(self, cluster_idx: int):
        """
        returns a definition cluster with cluster_idx
        in the same format as def_class_subgraph (see docstring there)

        but also an integer K that denotes the number of definitions in this cluster
        the entry points to the definition nodes are the first K in the list of returned nodes
        """
        return self.__online_data.def_cluster_subgraph(cluster_idx)

    def def_cluster_subgraphs(self):
        """
        returns a generator of subgraphs for definition clusters
        in the same format as def_class_subgraph (see docstring there)

        NOTE: caching should take place somewhere else, if necessary
        """
        return map(self.def_cluster_subgraph, range(self.graph_constants().cluster_subgraphs_num))

    def graph_constants(self):
        return self.__online_data.graph_constants()

    def _debug_online_data(self):
        return self.__online_data
