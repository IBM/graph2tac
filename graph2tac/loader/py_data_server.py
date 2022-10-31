from pathlib import Path
import capnp
import pytact.common
import pytact.data_reader
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from collections import deque
import itertools
from graph2tac.hash import get_split_label

capnp.remove_import_hook()
graph_api_capnp = pytact.common.graph_api_capnp()
graph_api_capnp = capnp.load(graph_api_capnp)

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


def custom_dataclass_repr(obj) -> str:
    """String repr for a dataclass, but removing the array data."""
    kws = {k: f"array(shape={v.shape}, dytype={v.dtype})" if isinstance(v, np.ndarray) else repr(v) for k,v in obj.__dict__.items()}
    return obj.__class__.__qualname__ + "(" + ", ".join(k + "=" + v for k,v in kws.items()) + ")"


@dataclass(repr=False)
class LoaderGraph:
    """A single graph as it comes from the loader"""
    nodes: NDArray[np.uint32]  # [nodes]
    """Node class labels.  Shape: [number of nodes]"""
    edges: NDArray[np.uint32]  # [edges,2]
    """Edges as source-target pairs indexed into `nodes`.  Shape: [number of edges, 2]"""
    edge_labels: NDArray[np.uint32]  # [edges]
    """Edge labels.  Shape: [number of edges]"""
    edge_offsets: NDArray[np.uint32]  # [edge_labels]
    """Start position in `edges` of each edge label.  Shape: [number of edge labels]"""

    def __repr__(self):
        return custom_dataclass_repr(self)


@dataclass(repr=False)
class ProofstateMetadata:
    """Metadata for a single proof state"""
    name: bytes
    """Theorem name"""
    step: int
    """Step of the proof"""
    is_faithful: bool
    """Represents if the action is faithful, meaning that the action fully encodes the original tactic"""

    def __repr__(self):
        return custom_dataclass_repr(self)

@dataclass(repr=False)
class ProofstateContext:
    """Local and global context for a single proofstate"""
    local_context: NDArray[np.uint32]  # [loc_cxt]
    """Local context as indices into the `node` field of the corresponding graph.  Shape: [size of local context]"""
    global_context: NDArray[np.uint32]  # [global_cxt]
    """
    Global context as indices into list of all global identifiers.
    Shape: [size of global context]
    
    The index is into the list of global definitions maintained by the dataserver and kept in the model
    (not the list of node labels).  Hence the smallest `global_context` value is usually 0.
    """

    def __repr__(self):
        return custom_dataclass_repr(self)

@dataclass(repr=False)
class LoaderProofstate:
    """A single proofstate as it comes from the loader"""
    graph: LoaderGraph
    """Proofstate graph as it comes from the loader"""
    root: int
    """Index of the root node in `graph.nodes`.  Currently the root is always 0."""
    context: ProofstateContext
    """Local and global context"""
    metadata: ProofstateMetadata
    """Proofstate metadata"""

    def __repr__(self):
        return custom_dataclass_repr(self)

@dataclass(repr=False)
class LoaderAction:
    """A single tactic as it comes from the loader"""
    tactic_id: int
    """Base tactic id"""
    args: NDArray[np.uint32]
    """
    Tactic arguments (local or global or none).  Using the indices in the `context` field of the proofstate.
    
    If `args[i] == [0, l] for l < local_cxt_size`, then the `i`th arg is a local context variable with index `l`.
    If `args[i] == [0, local_cxt_size]`, then the `i`th arg is a none argument (can't be represented).
    If `args[i] == [1, g]` for `g > 0`, then the `i`th arg is a global definition with index `g`.
    
    The local argument is an index into `context.local_context` for the corresponding proofstate,
    but the global argument is the index into the list of global definitions returned by the dataserver and kept
    in the model.  In particular, the global argument is neither an index into `context.global_context`
    for the proofstate nor the node label of a definition (which would instead be offset by the number of base node labels).

    Hence the smallest possible global argument index is 0 and it can be larger than
    `len(context.global_context)` for the corresponding proofstate.

    Shape: [number of args for this tactic, 2]
    """

    def __repr__(self):
        return custom_dataclass_repr(self)

@dataclass(repr=False)
class LoaderDefinition:
    """A single definition declaration as it comes from the loader"""
    graph: LoaderGraph
    """Definition graph as it comes from the loader"""
    num_definitions: int
    """
    Number of definitions in the graph.
    
    The roots for these definitions are the nodes of the `graph.nodes`.
    (E.g. inductive types are packaged as a single graph with
    mulitple definitions, one for the type and one for each constructor
    """
    definition_names: NDArray[np.object_]
    """Names of the definitions.  Stored as either bytestrings or strings.  Shape: [number of definitions in the graph]"""

    def __repr__(self):
        return custom_dataclass_repr(self)

@dataclass
class GraphConstants:
    tactic_num: int
    edge_label_num: int
    base_node_label_num: int
    node_label_num: int
    cluster_subgraphs_num: int
    tactic_index_to_numargs: np.ndarray  # dtype = np.uint32
    tactic_index_to_string: list[bytes]    # tactic names
    tactic_index_to_hash: np.ndarray
    global_context: np.ndarray
    label_to_names: list[str]
    label_in_spine: list[bool]
    max_subgraph_size: int

@dataclass
class DataPoint:
    """
    A proofstate and action datapoint from the loader
    
    This is used mostly for investigation into the dataset.
    """
    proof_step_idx: int
    """Datapoint unique index from the loader."""
    graph: LoaderGraph
    """Single proofstate graph"""
    local_context: NDArray[np.uint32]
    """Local context as indices into `graph.node` of the corresponding graph.  Shape: [size of local context]"""
    available_global_context: NDArray[np.uint32]
    """
    Global context as indices into list of all global identifiers.
    Shape: [size of global context]
    
    The index is into the list of global definitions maintained by the dataserver and kept in the model
    (not the list of node labels).  Hence the smallest `global_context` value is usually 0.
    """
    root: int
    """Index of the root node in `graph.nodes`.  Currently the root is always 0."""
    action: LoaderAction
    """The tactic applied to the proof state"""
    state_text: bytes
    """Plain text representation of the proof state"""
    action_base_text: bytes
    """The base tactic represented at plain text, e.g. 'apply _'"""
    action_interm_text: bytes
    """The tactic represented at plain text most similar to how it is represented coming out of the loader"""
    action_text: bytes
    """The tactic represented as plain text, representing the full"""
    def_name: bytes
    """The theorem this proof step is in.  Can be used to align to the orginal data."""
    step_in_proof: int
    """The step ix of the proof.  Can be used to align to the orginal data."""
    file_name: Path
    """The coq file this proof step is in.  Can be used to align to the orginal data."""


class DataServer:
    def __init__(self,
                 data_dir: Path,
                 max_subgraph_size,
                 split: tuple[int, int, int] = (8, 1, 1),
                 bfs_option = True,
                 split_random_seed = 0,
                 restrict_to_spine: bool = False,
                 stop_at_definitions: bool = True,
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
        for name, file_data in self._data.items():
            self._load_file(file_data)

    def graph_constants(self):
        total_node_label_num = len(self._node_i_to_name)
        return GraphConstants(
            tactic_num = len(self._tactic_to_i),
            edge_label_num = self._edge_label_num,
            base_node_label_num = self._base_node_label_num,
            node_label_num = total_node_label_num,
            cluster_subgraphs_num = len(self._def_clusters),
            tactic_index_to_numargs = np.array(self._tactic_i_to_numargs, dtype = np.uint32),
            tactic_index_to_string = list(self._tactic_i_to_bytes),
            tactic_index_to_hash = np.array(self._tactic_i_to_hash, dtype = np.uint32),
            global_context = np.arange(self._base_node_label_num, total_node_label_num),
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

    def _initialize_edge_labels(self):
        total_labels = len(graph_api_capnp.EdgeClassification.schema.enumerants)
        conflatable = [[x.raw for x in c.conflatable] for c in graph_api_capnp.conflatableEdges]
        conflation = [None]*total_labels
        for i,c in enumerate(conflatable):
            for x in c:
                conflation[x] = i
        edge_labels = [None]*total_labels
        edge_label_num = 0
        for i in range(total_labels):
            if edge_labels[i] is not None: continue
            if conflation[i] is not None:
                for x in conflatable[conflation[i]]:
                    edge_labels[x] = edge_label_num
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
            index = 0
            for tactic_usage in proof:
                if tactic_usage.tactic is None: continue
                self._register_tactic(tactic_usage)
                for outcome in tactic_usage.outcomes:
                    self._proof_steps.append((outcome,d,index))
                    index += 1

        # load clusters
        self._def_clusters.extend(file_data.clustered_definitions())

        # set node_i_in_spine
        for d in file_data.definitions(spine_only = True, across_files = False):
            node_i = self._def_node_to_i[d.node]
            self._node_i_in_spine[node_i] = True

    def _register_definition(self, d):
        node = d.node
        if node in self._def_node_to_i: return
        new_i = len(self._node_i_to_name)
        self._def_node_to_i[node] = new_i
        self._node_i_to_name.append(d.name)
        self._node_i_in_spine.append(False)
        
    def _get_node_label_index(self, node):
        if node.label.which.raw == self._definition_label:
            return self._def_node_to_i[node]
        else:
            return node.label.which.raw

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
        while q and len(nodes) < max_graph_size:
            if bfs_option: xi,x = q.popleft()
            else: xi,x = q.pop()
            if x.label.which.raw != stop_at:
                for e,y in x.children:
                    yi = node_to_i.get(y, len(nodes))
                    if yi == len(nodes):
                        node_to_i[y] = yi
                        nodes.append(y)
                        q.append((yi,y))
                    el = self._edge_labels[e.raw]
                    if el in self._edges_to_ignore: continue
                    edges.append((xi, yi, el))

        node_labels = [
            self._get_node_label_index(node)
            for node in nodes
        ]

        if edges:
            edges_by_labels = [[] for _ in range(self._edge_label_num)]
            for edge in edges:
                edges_by_labels[edge[2]].append(edge)
            edge_offsets = np.cumsum([0]+[
                len(x) for x in edges_by_labels
            ])[:-1]
            edges = list(itertools.chain.from_iterable(edges_by_labels))
            edges = np.array(edges, dtype = int)
        else:
            edges = np.zeros([0,3], dtype = int)
            edge_offsets = np.zeros(self._edge_label_num, dtype = int)
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
        available_global_context = [
            self._def_node_to_i[ctx_def.node] - self._base_node_label_num
            for ctx_def in definition.global_context()
        ]

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
            if get_split_label(d.node.identity, self.split, self.split_random_seed) == label
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
            definition_names = np.array([n.definition.name for n in roots])
        )

    def def_cluster_subgraphs(self):
        return IterableLen(map(self.get_cluster_subgraph, range(len(self._def_clusters))), len(self._def_clusters))
