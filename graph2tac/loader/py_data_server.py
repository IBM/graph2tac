from pathlib import Path
import pytact.common
import pytact.data_reader
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from collections import deque
import itertools
import random
from graph2tac.hash import get_split_label

from graph2tac.loader.data_classes import *
import pytact.graph_api_capnp as graph_api_capnp

from graph2tac.loader.data_server import top_sort_dataset_modules

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

class DataServer:
    def __init__(self,
                 data_dir: Path,
                 max_subgraph_size,
                 split: tuple[int, int, int] = (8, 1, 1),
                 bfs_option = True,
                 split_random_seed = 0,
                 restrict_to_spine: bool = False,
                 stop_at_definitions: bool = True,
                 fname_order_key = None,
    ):
        self.data_dir = data_dir
        self.max_subgraph_size = max_subgraph_size
        self.split = split
        self.bfs_option = bfs_option
        self.split_random_seed = split_random_seed
        self.restrict_to_spine = restrict_to_spine
        self.stop_at_definitions = stop_at_definitions

        self._initialize()

        self._reader = pytact.data_reader.data_reader(Path(data_dir))
        self._data = self._reader.__enter__()

        #fnames = list(self._data.keys())
        fnames = [
            fname.relative_to(data_dir)
            for fname in top_sort_dataset_modules(data_dir)
        ]
        assert set(fnames) == set(self._data.keys())
        if fname_order_key is not None:
            fnames.sort(key = fname_order_key)
        for name in fnames:
            file_data = self._data[name]
            self._load_file(file_data)

        # precalculate def_file_ctx in a forward order to prevent stack overflow
        for cluster in self._def_clusters:
            for d in cluster:
                self.get_def_file_ctx(d)

    def graph_constants(self):
        total_node_label_num = len(self._node_i_to_name)
        return GraphConstants(
            tactic_num = len(self._tactic_to_i),
            edge_label_num = self._edge_label_num,
            base_node_label_num = self._base_node_label_num,
            node_label_num = total_node_label_num,
            cluster_subgraphs_num = len(self._def_clusters),
            tactic_index_to_numargs = np.array(self._tactic_i_to_numargs, dtype = np.uint64),
            tactic_index_to_string = list(self._tactic_i_to_bytes),
            tactic_index_to_hash = np.array(self._tactic_i_to_hash, dtype = np.uint64),
            global_context = np.arange(self._base_node_label_num, total_node_label_num, dtype = np.uint32),
            label_to_names = self._node_i_to_name,
            label_in_spine = self._node_i_in_spine,
            max_subgraph_size = self.max_subgraph_size,
        )

    def _initialize(self):
        self._initialize_edge_labels()
        self._node_i_to_name = list(graph_api_capnp.Graph.Node.Label.schema.union_fields)
        self._base_node_label_num = len(self._node_i_to_name)
        self._node_i_in_spine = [True]*self._base_node_label_num
        self._definition_label = graph_api_capnp.Graph.Node.Label.definition.value
        self._edges_to_ignore = (graph_api_capnp.EdgeClassification.constOpaqueDef,)
        self._def_node_to_i = dict()
        self._tactic_to_i = dict()
        self._tactic_i_to_numargs = []
        self._tactic_i_to_bytes = []
        self._tactic_i_to_hash = []
        self._proof_steps : list[tuple[Outcome, Definition, int]] = []
        self._def_clusters : list[list[Definition]] = []
        self._repr_to_spine : dict[Definition, NDArray[np.uint32]] = dict()
        self._repr_to_filedeps : dict[Definition, list[Definition]] = dict()
        self._repr_to_recdeps : dict[Definition, list[Definition]] = dict()
        self._def_to_file_ctx : dict[Definition, tuple[list[Definition], NDArray[np.uint32]]] = dict()

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
        self._edge_label_num = edge_label_num
        
    def _load_file(self, file_data):
        # load proof steps
        for d in file_data.definitions(spine_only = self.restrict_to_spine):
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
                node_i = self._def_node_to_i[d.node]
                spine.append(node_i)
                self._node_i_in_spine[node_i] = True
                filedeps.extend(d.external_previous)

            r = file_data.representative
            self._repr_to_spine[r] = np.array(spine, dtype = np.uint32) - self._base_node_label_num
            self._repr_to_filedeps[r] = filedeps

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

    def _register_definition(self, d):
        node = d.node
        if node in self._def_node_to_i: return
        new_i = len(self._node_i_to_name)
        self._def_node_to_i[node] = new_i
        self._node_i_to_name.append(d.name)
        self._node_i_in_spine.append(False)
        
    def _get_node_label_index(self, node):
        if int(node.label.which) == self._definition_label:
            return self._def_node_to_i[node]
        else:
            return int(node.label.which)

    def _register_tactic(self, tactic_usage):
        if tactic_usage.tactic.ident in self._tactic_to_i:
            return
        if len(tactic_usage.outcomes) == 0:
            return
        index = len(self._tactic_to_i)
        self._tactic_to_i[tactic_usage.tactic.ident] = index
        self._tactic_i_to_numargs.append(len(tactic_usage.outcomes[0].tactic_arguments))
        self._tactic_i_to_bytes.append(bytes(tactic_usage.tactic.base_text, 'utf-8'))
        self._tactic_i_to_hash.append(tactic_usage.tactic.ident)

    def _downward_closure(self, roots):
        bfs_option = self.bfs_option
        max_graph_size = self.max_subgraph_size
        if self.stop_at_definitions:
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

        if edges:
            edges_by_labels = [[] for _ in range(self._edge_label_num)]
            for edge in edges:
                edges_by_labels[edge[2]].append(edge)
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

    def _datapoint_graph(self, i):
        proof_step, definition, index = self._proof_steps[i]
        graph, node_to_i = self._downward_closure(
            [proof_step.before.root]
        )
        root_i = 0
        local_context = proof_step.before.context
        local_context_i = [node_to_i[n] for n in local_context]

        available_global_context = np.array([
            self._def_node_to_i[ctx_def.node]
            for ctx_def in definition.global_context(across_files = False)
        ], dtype = np.uint32) - self._base_node_label_num
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

        if proof_step.tactic_arguments:
            arguments = []
            local_nodes_to_ctx_i = { node : i for i,node in enumerate(local_context) }
            for arg in proof_step.tactic_arguments:
                arg_local = local_nodes_to_ctx_i.get(arg, None)
                if arg_local is not None:
                    arguments.append([0, arg_local])
                    continue
                arg_global = self._def_node_to_i.get(arg, None)
                if arg_global is not None and arg_global >= self._base_node_label_num:
                    arg_global -= self._base_node_label_num
                    arguments.append([1, arg_global])
                    continue
                else:
                    arguments.append([0, len(local_context_i)])
            arguments = np.array(arguments, dtype = np.uint32)
        else:
            arguments = np.zeros([0,2], dtype = np.uint32)

        action = LoaderAction(
            tactic_id=tactic_i,
            args=arguments,
        )
        return proofstate, action, i

    def _datapoint_text(self, i):
        proof_step, _, _ = self._proof_steps[i]
        state_text = proof_step.before.text
        label_text = proof_step.tactic.text
        return state_text, label_text

    def _select_data_points(self, label):
        return [
            i for i,(_,d,_) in enumerate(self._proof_steps)
            if get_split_label(
                    np.array(d.node.identity, dtype = np.uint64).item(),
                    self.split, self.split_random_seed
            ) == label
        ]

    def data_train(self, shuffled: bool = False,  as_text: bool = False):
        train_ids = self._select_data_points(0)
        if shuffled:
            train_ids = list(train_ids)
            random.shuffle(train_ids)

        if as_text:
            return IterableLen(map(self._datapoint_text, train_ids), len(train_ids))
        else:
            return IterableLen(map(self._datapoint_graph, train_ids), len(train_ids))

    def data_valid(self, as_text: bool = False):
        valid_ids = self._select_data_points(1)
        if as_text:
            return IterableLen(map(self._datapoint_text, valid_ids), len(valid_ids))
        else:
            return IterableLen(map(self._datapoint_graph, valid_ids), len(valid_ids))

    def def_cluster_subgraph(self, i):
        cluster = self._def_clusters[i]
        roots = [x.node for x in cluster]
        graph, _ = self._downward_closure(roots)

        return LoaderDefinition(
            graph = graph,
            num_definitions = len(roots),
            definition_names = np.array([
                n.definition.name
                for n in roots
            ], dtype = 'S61')
        )

    def def_cluster_subgraphs(self):
        return IterableLen(map(self.def_cluster_subgraph, range(len(self._def_clusters))), len(self._def_clusters))
