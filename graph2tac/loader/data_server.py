from heapq import heappush, heappop
from pathlib import Path
import yaml
import pytact.common
from pytact.data_reader import data_reader, Outcome, Node, Definition
from numpy.typing import NDArray
import numpy as np
from collections import deque
import itertools
import random
from graph2tac.hash import get_split_label
from collections import defaultdict
from typing import Iterable, Optional, List, Tuple, Dict
import tensorflow as tf
import tensorflow_gnn as tfgnn

from graph2tac.loader.data_classes import *
import pytact.graph_api_capnp as graph_api_capnp

class IterableLen:
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length
    def __iter__(self):
        return iter(self.iterable)
    def __next__(self):
        return next(self.iterable)
    def __len__(self):
        return self.length

def filtermap(f, it):
    for x in it:
        y = f(x)
        if y is not None: yield y

class AbstractDataServer:
    def __init__(self,
                 data_config,
    ):
        if data_config.symmetrization not in (BIDIRECTIONAL, UNDIRECTED, None):
            raise ValueError(f'{data_config.symmetrization} is not a valid graph symmetrization scheme (use {BIDIRECTIONAL}, {UNDIRECTED} or None)')
        self.data_config = data_config

        self._initialize_edge_labels()
        self._node_i_to_name = list(graph_api_capnp.Graph.Node.Label.schema.union_fields)
        self._base_node_label_num = len(self._node_i_to_name)
        self._node_i_to_ident = [0]*self._base_node_label_num
        self._node_i_in_spine = [True]*self._base_node_label_num
        self._definition_label = graph_api_capnp.Graph.Node.Label.definition.value
        self._edges_to_ignore = (graph_api_capnp.EdgeClassification.constOpaqueDef,)
        self._node_to_node_i = dict()
        self._tactic_to_i = dict()
        self._tactic_i_count = []
        self._tactic_i_to_numargs = []
        self._tactic_i_to_string = []
        self._tactic_i_to_hash = []
        self._def_name_dtype = str

    def _initialize_edge_labels(self):
        total_labels = len(graph_api_capnp.EdgeClassification.schema.enumerants)
        conflatable = [[x.raw for x in c.conflatable] for c in graph_api_capnp.conflatableEdges]
        conflation = [None]*total_labels
        for i,c in enumerate(conflatable):
            for x in c:
                conflation[x] = i
        edge_labels = [None]*total_labels
        edge_label_num = 0
        for group in conflatable:
            for x in group:
                edge_labels[x] = edge_label_num
            edge_label_num += 1
        for i in range(total_labels):
            if edge_labels[i] is not None: continue
            if conflation[i] is not None:
                continue
            else:
                edge_labels[i] = edge_label_num
            edge_label_num += 1
        self._edge_labels = edge_labels
        self._base_edge_label_num = edge_label_num
        if self.data_config.symmetrization == BIDIRECTIONAL:
            edge_label_num *= 2
        if self.data_config.add_self_edges:
            edge_label_num += 1
        self._edge_label_num = edge_label_num

    # Definition / Tactic registration

    def _register_definition(self, d):
        node = d.node
        if node in self._node_to_node_i:
            return self._node_to_node_i[node]
        new_i = len(self._node_i_to_name)
        self._node_to_node_i[node] = new_i
        self._node_i_to_name.append(d.name)
        self._node_i_to_ident.append(d.node.identity)
        self._node_i_in_spine.append(False)
        return new_i
        
    def _get_node_label_index(self, node):
        if int(node.label.which) == self._definition_label:
            return self._node_to_node_i[node]
        else:
            return int(node.label.which)

    def _register_tactic(self, tactic_usage):
        ident = tactic_usage.tactic.ident
        if ident in self._tactic_to_i:
            self._tactic_i_count[self._tactic_to_i[ident]] += 1
            return
        if len(tactic_usage.outcomes) == 0:
            return
        index = len(self._tactic_to_i)
        self._tactic_to_i[ident] = index
        self._tactic_i_count.append(1)
        self._tactic_i_to_numargs.append(len(tactic_usage.outcomes[0].tactic_arguments))
        self._tactic_i_to_string.append(tactic_usage.tactic.base_text)
        self._tactic_i_to_hash.append(ident)

    # Obtaining data

    def _downward_closure(self, roots):
        bfs_option = self.data_config.bfs_option
        max_graph_size = self.data_config.max_subgraph_size
        if self.data_config.stop_at_definitions:
            stop_at = self._definition_label
        else:
            stop_at = None
        nodes = list(roots)
        node_to_i = { node : i for i,node in enumerate(roots) }
        q = deque(enumerate(roots))
        edges = []
        while q:
            if bfs_option: xi,x = q.popleft()
            else: xi,x = q.pop()
            for e,y in x.children:
                if e in self._edges_to_ignore: continue
                yi = node_to_i.get(y, len(nodes))
                if yi == len(nodes):
                    if len(nodes) == max_graph_size: continue
                    node_to_i[y] = yi
                    nodes.append(y)
                    if int(y.label.which) != stop_at:
                        q.append((yi,y))
                edges.append((xi, yi, self._edge_labels[e]))

        node_labels = [
            self._get_node_label_index(node)
            for node in nodes
        ]

        if edges or (nodes and self.data_config.add_self_edges):
            edges_by_labels = [[] for _ in range(self._base_edge_label_num)]
            for edge in edges:
                edges_by_labels[edge[2]].append(edge)
            if self.data_config.symmetrization == UNDIRECTED:
                for e in edges_by_labels:
                    e.extend(
                        (b,a,l)
                        for a,b,l in list(e)
                    )
            elif self.data_config.symmetrization == BIDIRECTIONAL:
                edges_by_labels.extend(
                    [
                        (b,a, l+self._base_edge_label_num)
                        for a,b,l in e
                    ]
                    for e in list(edges_by_labels)
                )
            if self.data_config.add_self_edges:
                l = len(edges_by_labels)
                edges_by_labels.append([
                    [a,a,l] for a in range(len(nodes))
                ])
            edge_offsets = np.cumsum([
                len(x) for x in edges_by_labels
            ])[:-1].astype(np.uint32)
            edges = list(itertools.chain.from_iterable(edges_by_labels))
            edges = np.array(edges, dtype = np.uint32)
        else:
            edges = np.zeros([0,3], dtype = np.uint32)
            edge_offsets = np.zeros(self._edge_label_num, dtype = np.uint32)
        graph = LoaderGraph(
            nodes = np.array(node_labels, dtype=np.uint32),
            edges = edges[:,:2],
            edge_labels = edges[:,2],
            edge_offsets = edge_offsets,
        )

        return graph, node_to_i

    def cluster_to_graph(self, cluster):
        roots = [x.node for x in cluster]
        graph, _ = self._downward_closure(roots)

        return LoaderDefinition(
            graph = graph,
            num_definitions = len(roots),
            definition_names = np.array([
                n.definition.name
                for n in roots
            ], dtype = self._def_name_dtype)
        )

class DataToTFGNN:
    """
    Class for exporting Loader structures to TFGNN
    """
    MAX_LABEL_TOKENS = 128
    def __init__(self):
        vocabulary = [
            chr(i) for i in range(ord('a'), ord('z')+1)
        ] + [
            chr(i) for i in range(ord('A'), ord('Z')+1)
        ] + [
            chr(i) for i in range(ord('0'), ord('9')+1)
        ] + ["_", "'", "."]
        self._label_tokenizer = tf.keras.layers.TextVectorization(standardize=None,
                                                                  split='character',
                                                                  ngrams=None,
                                                                  output_mode='int',
                                                                  max_tokens=self.MAX_LABEL_TOKENS,
                                                                  vocabulary = vocabulary,
                                                                  ragged=True)

    @staticmethod
    def graph_to_graph_tensor(graph: LoaderGraph, context) -> tfgnn.GraphTensor:
        node_labels = tf.convert_to_tensor(graph.nodes, dtype = tf.int64)
        edge_labels = tf.convert_to_tensor(graph.edge_labels, dtype = tf.int64)
        sources = tf.convert_to_tensor(graph.edges[:,0], dtype = tf.int32)
        targets = tf.convert_to_tensor(graph.edges[:,1], dtype = tf.int32)

        node_set = tfgnn.NodeSet.from_fields(features={'node_label': node_labels},
                                             sizes=tf.shape(node_labels))

        adjacency = tfgnn.Adjacency.from_indices(source=('node', sources),
                                                 target=('node', targets))

        edge_set = tfgnn.EdgeSet.from_fields(features={'edge_label': edge_labels},
                                             sizes=tf.shape(edge_labels),
                                             adjacency=adjacency)

        return tfgnn.GraphTensor.from_pieces(node_sets={'node': node_set}, edge_sets={'edge': edge_set}, context = context)


    @staticmethod
    def proofstate_to_graph_tensor(state: LoaderProofstate, action: LoaderAction, id: int) -> tfgnn.GraphTensor:

        available_global_context = tf.convert_to_tensor(state.context.global_context, dtype = tf.int64)
        context_node_ids = tf.convert_to_tensor(state.context.local_context, dtype = tf.int64)

        local_context_length = tf.shape(context_node_ids, out_type=tf.int64)[0]

        local_arguments = tf.convert_to_tensor(action.local_args, dtype = tf.int64)
        global_arguments = tf.convert_to_tensor(action.global_args, dtype = tf.int64)

        context = tfgnn.Context.from_fields(features={
            'tactic': tf.convert_to_tensor([action.tactic_id], tf.int64),
            'local_context_ids': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(context_node_ids, axis=0),
                                                             row_splits_dtype=tf.int32),
            'global_context_ids': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(available_global_context, axis=0),
                                                              row_splits_dtype=tf.int32),
            'local_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(local_arguments, axis=0),
                                                           row_splits_dtype=tf.int32),
            'global_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(global_arguments, axis=0),
                                                            row_splits_dtype=tf.int32),
            'graph_id': tf.expand_dims(tf.convert_to_tensor(id, dtype = tf.int64), axis=0),
            'name': tf.expand_dims(state.metadata.name, axis=0),
            'step': tf.expand_dims(tf.convert_to_tensor(state.metadata.step, dtype = tf.int64), axis=0),
            'faithful': tf.expand_dims(tf.convert_to_tensor(state.metadata.is_faithful, dtype = tf.int64), axis=0)
        })
        return DataToTFGNN.graph_to_graph_tensor(state.graph, context)

    def definition_to_graph_tensor(self, defn: LoaderDefinition) -> tfgnn.GraphTensor:
        """Convert loader definition format to corresponding format for definition_data_spec"""

        num_definitions = tf.convert_to_tensor(defn.num_definitions, dtype = tf.int64)
        vectorized_definition_names = self._label_tokenizer(defn.definition_names)
        
        context = tfgnn.Context.from_fields(features={
            'num_definitions': tf.expand_dims(num_definitions, axis=0),
            'definition_name_vectors': tf.expand_dims(vectorized_definition_names.with_row_splits_dtype(tf.int32), axis=0)
        })
        return self.graph_to_graph_tensor(defn.graph, context)

TRAIN = 0
VALID = 1
HOLDOUT = 2 # maybe unnecessary

class Splitter:
    def lemma(self, d : Definition) -> int:
        raise Exception("Not implemented")
    def definition_cluster(self, d : List[Definition]) -> int:
        raise Exception("Not implemented")
    def assign_data_server(self, data_server):
        pass

class SplitDisabled(Splitter):
    def lemma(self, d : Definition) -> int:
        return TRAIN
    def definition_cluster(self, d : List[Definition]) -> int:
        return TRAIN

class SplitByHash(Splitter):
    def __init__(self, proportions : List[int], random_seed : int):
        self.proportions = proportions
        self.random_seed = random_seed
    def lemma(self, d : Definition) -> int:
        # to make it identical to vasily's loader
        ident_64 = np.array(int(d.node.identity)).astype("uint64").item()  # casts to uint64 (w/ overflow) w/o deprication warning
        return get_split_label(ident_64, self.proportions, self.random_seed)
    def definition_cluster(self, d : List[Definition]) -> int:
        return TRAIN

class SplitByFilePrefix(Splitter):
    def __init__(self, prefixes_per_label : List[List[str]]):
        self.prefixes_per_label = [
            (label+1, prefixes)
            for label, prefixes in enumerate(prefixes_per_label)
        ]
    def lemma(self, d : Definition):
        return self.graphid_to_label[d.node.graph]
    def definition_cluster(self, d : List[Definition]):
        return self.graphid_to_label[d[0].node.graph]
    def assign_data_server(self, data_server):

        # conversion from graph indices to filenames
        graphid_to_fname = {
            file_data.graph : str(fname)
            for fname, file_data in data_server._data.items()
        }
        assert set(graphid_to_fname.keys()) == set(range(len(data_server._data)))
        graphid_to_fname = [
            graphid_to_fname[i]
            for i in range(len(data_server._data))
        ]

        # conversion from graph indices to labels + check that all prefixes are used
        unused_prefixes = set().union(*[prefixes for label,prefixes in self.prefixes_per_label])
        graphid_to_label = []
        for fname in graphid_to_fname:
            label = TRAIN
            for l,prefixes in self.prefixes_per_label:
                for prefix in prefixes:
                    if fname.startswith(prefix):
                        label = max(l, label)
                        unused_prefixes.discard(prefix)
            graphid_to_label.append(label)

        if unused_prefixes:
            raise Exception(f"Some split prefixes not found:, {sorted(unused_prefixes)}")

        # check that the labels have correct dependencies
        for fname,d in data_server._data.items():
            label = graphid_to_label[d.graph]
            for dep_fname in data_server._data[fname].dependencies:
                dep_label = graphid_to_label[data_server._data[dep_fname].graph]
                if dep_label > label:
                    raise Exception(f"Dependency-inconsistent prefix split: {fname} (label {label}) depends on {dep_fname} (label {dep_label})")

        # save labels for efficient split
        self.graphid_to_label = graphid_to_label

def get_splitter(split_method, split):
    if split_method == "hash": return SplitByHash(**split)
    elif split_method == "file_prefix": return SplitByFilePrefix(split)
    elif split_method == "disabled": return SplitDisabled()
    else: raise Exception(f"Unexpected split method: '{split_method}'")

class DataServer(AbstractDataServer):
    def __init__(self, data_dir : Path, dataset_config : DatasetConfig):
        super().__init__(dataset_config.data_config)
        self.data_dir = data_dir
        self.dataset_config = dataset_config

        self._data_to_tfgnn = None
        self._proof_steps : List[Tuple[Outcome, Definition, int]] = []
        self._def_clusters : List[List[Definition]] = []
        self._repr_to_spine : Dict[Definition, NDArray[np.uint32]] = dict()
        self._repr_to_filedeps : Dict[Definition, List[Definition]] = dict()
        self._repr_to_recdeps : Dict[Definition, List[Definition]] = dict()
        self._def_to_file_ctx : Dict[Definition, Tuple[List[Definition], NDArray[np.uint32]]] = dict()

        self._reader = data_reader(Path(self.data_dir))
        self._data = self._reader.__enter__()

        # build the arrays of proofsteps, definition clusters, tactics, file dependencies
        fnames = self.topo_file_order()
        for name in fnames:
            file_data = self._data[name]
            self._load_file(file_data)

        self._exclude_unique_tactics()

        # precalculate def_file_ctx in a forward order to prevent stack overflow,
        # calculate the maximum definition length
        max_def_name_len = 0
        for cluster in self._def_clusters:
            for d in cluster:
                self.get_def_file_ctx(d)
                max_def_name_len = max(len(d.name), max_def_name_len)
        self._def_name_dtype = "U"+str(max_def_name_len)

        self.split = get_splitter(
            self.dataset_config.split_method,
            self.dataset_config.split
        )
        self.split.assign_data_server(self)

        # split the shuffle seed into two random number generators, one for proof states and one for definitions
        rng = random.Random(self.dataset_config.shuffle_random_seed)
        self.rng_proofstates = random.Random(rng.random())
        self.rng_definitions = random.Random(rng.random())

    def graph_constants(self):
        total_node_label_num = len(self._node_i_to_name)
        return GraphConstants(
            data_config = self.data_config,
            tactic_num = len(self._tactic_to_i),
            edge_label_num = self._edge_label_num,
            base_node_label_num = self._base_node_label_num,
            node_label_num = total_node_label_num,
            cluster_subgraphs_num = len(self._def_clusters),
            tactic_index_to_numargs = self._tactic_i_to_numargs,
            tactic_index_to_string = self._tactic_i_to_string,
            tactic_index_to_hash = self._tactic_i_to_hash,
            label_to_names = self._node_i_to_name,
            label_to_ident = self._node_i_to_ident,
            label_in_spine = self._node_i_in_spine,
        )

    def topo_file_order(self):

        # prepare secondary sorting key
        fname_to_key = {
            fname : (self.data_dir / fname).stat().st_size
            for fname in self._data.keys()
        }
        fname_to_revdeps = defaultdict(list)
        fname_to_deps = dict()

        # prepare graph and the heap
        heap = []
        for fname, d in self._data.items():
            fname_to_deps[fname] = set(d.dependencies)
            if len(d.dependencies) == 0:
                heappush(heap, (fname_to_key[fname], fname))
            else:
                for dep in d.dependencies:
                    fname_to_revdeps[dep].append(fname)

        # process the heap
        file_order = []
        while heap:
            _, fname = heappop(heap)
            file_order.append(fname)
            for rdep in fname_to_revdeps[fname]:
                deps = fname_to_deps[rdep]
                deps.remove(fname)
                if not deps:
                    heappush(heap, (fname_to_key[rdep], rdep))

        assert len(file_order) == len(self._data)
        return file_order
    
    def _load_file(self, file_data):
        # load proof steps
        for d in file_data.definitions(spine_only = self.dataset_config.restrict_to_spine):
            self._register_definition(d)
            proof = d.proof
            if proof is None: continue
            for index,tactic_usage in enumerate(proof):
                if tactic_usage.tactic is None: continue
                self._register_tactic(tactic_usage)
                for outcome in tactic_usage.outcomes:
                    self._proof_steps.append((outcome,d,index))

        # load clusters
        self._def_clusters.extend(file_data.clustered_definitions())

        if file_data.representative is not None:
            # set node_i_in_spine
            spine = []
            filedeps = []
            for d in file_data.definitions(spine_only = True, across_files = False):
                node_i = self._node_to_node_i[d.node]
                spine.append(node_i)
                self._node_i_in_spine[node_i] = True
                filedeps.extend(d.external_previous)

            r = file_data.representative
            self._repr_to_spine[r] = np.array(spine, dtype = np.uint32)
            self._repr_to_filedeps[r] = filedeps

    def _exclude_unique_tactics(self):
        if self.dataset_config.required_tactic_occurrence <= 1: return

        new_to_ori = [
            i for i,cnt in enumerate(self._tactic_i_count)
            if cnt >= self.dataset_config.required_tactic_occurrence
        ]
        ori_to_new = {
            ori : new
            for new, ori in enumerate(new_to_ori)
        }

        # update proofsteps
        self._proof_steps = [
            proof_step
            for proof_step in self._proof_steps
            if self._tactic_to_i[proof_step[0].tactic.ident] in ori_to_new
        ]

        # reindex tactics
        self._tactic_to_i = {
            ident : ori_to_new[ori]
            for ident, ori in self._tactic_to_i.items()
            if ori in ori_to_new
        }
        tactic_lists = (
            self._tactic_i_count,
            self._tactic_i_to_numargs,
            self._tactic_i_to_string,
            self._tactic_i_to_hash,
        )
        for tactic_list in tactic_lists:
            tactic_list[:] = [tactic_list[ori] for ori in new_to_ori]

        # sanity check
        assert len(self._tactic_i_to_hash) == len(self._tactic_to_i)
        for ident,i in self._tactic_to_i.items():
            assert self._tactic_i_to_hash[i] == ident

    def get_recdeps(self, representative):
        res = self._repr_to_recdeps.get(representative, None)
        if res is not None: return res

        deps_list = [representative]
        deps_set = set([representative])
        for dep in self._repr_to_filedeps[representative]:
            if dep in deps_set: continue
            subdeps_list = self.get_recdeps(dep)
            deps_list.extend(filter(lambda x: x not in deps_set, subdeps_list))
            deps_set.update(subdeps_list)
        self._repr_to_recdeps[representative] = deps_list
        return deps_list

    def get_def_file_ctx(self, definition):
        res = self._def_to_file_ctx.get(definition, None)
        if res is not None: return res

        if definition.previous is not None:
            prev_filedeps, prev_filectx = self.get_def_file_ctx(definition.previous)
        else:
            prev_filedeps, prev_filectx = [], np.zeros([0], dtype = np.uint32)

        if not definition.external_previous:
            filedeps, filectx = prev_filedeps, prev_filectx
        else:
            filedeps = list(prev_filedeps)
            filedeps_s = set(prev_filedeps)
            for dep in definition.external_previous:
                if dep in filedeps_s: continue
                subdeps_list = self.get_recdeps(dep)
                filedeps.extend(filter(lambda x: x not in filedeps_s, subdeps_list))
                filedeps_s.update(subdeps_list)

            if not filedeps: filectx = prev_filectx
            filectx = np.concatenate([
                self._repr_to_spine[dep]
                for dep in filedeps
            ])

        res = filedeps, filectx
        self._def_to_file_ctx[definition] = res        
        return res

    def datapoint_graph(self, i):
        proof_step, definition, index = self._proof_steps[i]
        graph, node_to_i = self._downward_closure([proof_step.before.root])
        root_i = 0
        local_context = proof_step.before.context
        local_context_i = [node_to_i[n] for n in local_context]

        available_global_context = np.array([
            self._node_to_node_i[ctx_def.node]
            for ctx_def in definition.global_context(across_files = False)
        ], dtype = np.uint32)
        filedeps, filectx = self.get_def_file_ctx(definition)
        available_global_context = np.concatenate([
            filectx,
            available_global_context,
        ])

        context = ProofstateContext(
            local_context=np.array(local_context_i, dtype = np.uint32),
            global_context=np.array(available_global_context, dtype = np.uint32),
        )

        is_faithful = proof_step.tactic.text.replace('@', '') == proof_step.tactic.interm_text.replace('@', '')
        if self.dataset_config.exclude_not_faithful and not is_faithful: return None

        metadata = ProofstateMetadata(
            name=definition.name.encode('utf-8'),
            step=index,
            is_faithful=is_faithful
        )

        proofstate = LoaderProofstate(
            graph=graph,
            root=root_i,
            context=context,
            metadata=metadata
        )

        tactic_i = self._tactic_to_i[proof_step.tactic.ident]

        has_none_argument = False

        local_args = []

        global_args = []
        local_nodes_to_ctx_i = { node : i for i,node in enumerate(local_context) }
        for arg in proof_step.tactic_arguments:
            arg_local = local_nodes_to_ctx_i.get(arg, -1)
            if arg_local < 0: arg_global = self._node_to_node_i.get(arg, -1)
            else: arg_global = -1
            # node reindexing: "node label" -> "index to global_context"
            if arg_global >= 0:
                [arg_global] = np.flatnonzero(available_global_context == arg_global)
            if arg_global < 0 and arg_local < 0: has_none_argument = True
            local_args.append(arg_local)
            global_args.append(arg_global)

        local_args = np.array(local_args, dtype = np.int64)
        global_args = np.array(global_args, dtype = np.int64)

        if has_none_argument and self.dataset_config.exclude_none_arguments: return None

        action = LoaderAction(
            tactic_id=tactic_i,
            local_args=local_args,
            global_args=global_args,
        )
        return proofstate, action, i

    def datapoint_text(self, i):
        proof_step, _, _ = self._proof_steps[i]
        state_text = proof_step.before.text
        label_text = proof_step.tactic.text
        return state_text, label_text

    def datapoint_indices(self, *labels):
        if not labels: return list(range(len(self._proof_steps)))
        else: return [
            i for i,(_,d,_) in enumerate(self._proof_steps)
            if self.split.lemma(d) in labels
        ]

    def get_datapoints(self, label : int, shuffled: bool = False) -> Iterable[Tuple[LoaderProofstate, LoaderAction, int]]:
        ids = self.datapoint_indices(label)
        if shuffled:
            self.rng_proofstates.shuffle(ids)
        return IterableLen(filtermap(self.datapoint_graph, ids), len(ids))

    def data_train(self, shuffled: bool = False) -> Iterable[Tuple[LoaderProofstate, LoaderAction, int]]:
        return self.get_datapoints(TRAIN, shuffled = shuffled)

    def data_valid(self) -> Iterable[Tuple[LoaderProofstate, LoaderAction, int]]:
        return self.get_datapoints(VALID)

    def def_cluster_indices(self, *labels):
        if not labels: return list(range(len(self._def_clusters)))
        else: return [
            i for i,ds in enumerate(self._def_clusters)
            if self.split.definition_cluster(ds) in labels
        ]

    def def_cluster_subgraph(self, i : int) -> LoaderDefinition:
        return self.cluster_to_graph(self._def_clusters[i])

    def def_cluster_subgraphs(self, label : int = TRAIN, shuffled: bool = False) -> Iterable[LoaderDefinition]:
        ids = self.def_cluster_indices(label)
        if shuffled:
            self.rng_definitions.shuffle(ids)
        return IterableLen(map(self.def_cluster_subgraph, ids), len(ids))

    # YAML config stuff

    def get_config(self) -> dict:
        config_d = dict(self.dataset_config.__dict__)
        config_d['data_config'] = config_d['data_config'].__dict__
        return config_d

    @classmethod
    def from_yaml_config(cls, data_dir: Path, yaml_filepath: Path) -> "DataServer":
        """
        Create a DataLoaderDataset from a YAML configuration file

        @param data_dir: the directory containing the data
        @param yaml_filepath: the filepath to the YAML file containing the
        @return: a DataLoaderDataset object
        """
        with yaml_filepath.open() as yaml_file:
            config_d = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        config_d['data_config'] = DataConfig(**config_d['data_config'])
        return cls(data_dir=data_dir, dataset_config = DatasetConfig(**config_d))

    # export for TFGNN

    def _get_data_to_tfgnn(self):
        if self._data_to_tfgnn is None:
            self._data_to_tfgnn = DataToTFGNN()
        return self._data_to_tfgnn

    def proofstates_tfgnn(self, label, shuffle) -> tf.data.Dataset:
        """
        Returns a pair of proof-state datasets for train and validation.

        @param shuffle: whether to shuffle the resulting datasets
        @return: a dataset of (GraphTensor, label) pairs
        """

        exporter = self._get_data_to_tfgnn()
        graph_id_spec = tf.TensorSpec([], tf.int64)
        # get proof-states
        proofstate_dataset = tf.data.Dataset.from_generator(
            lambda: self.get_datapoints(label, shuffle),
            output_signature=(LoaderProofstateSpec, LoaderActionSpec, graph_id_spec),
        )
        proofstate_dataset = proofstate_dataset.map(exporter.proofstate_to_graph_tensor)

        return proofstate_dataset

    def definitions_tfgnn(self, label, shuffle) -> tf.data.Dataset:
        """
        Returns the definition dataset.

        @param shuffle: whether to shuffle the resulting dataset
        @return: a dataset with all the definition clusters
        """
        exporter = self._get_data_to_tfgnn()
        res = tf.data.Dataset.from_generator(
            lambda: self.def_cluster_subgraphs(label, shuffle),
            output_signature=LoaderDefinitionSpec,
        )
        res = res.map(exporter.definition_to_graph_tensor)
        return res
