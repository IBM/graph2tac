from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

def custom_dataclass_repr(obj) -> str:
    """String repr for a dataclass, but removing the array data."""
    kws = {k: f"array(shape={v.shape}, dtype={v.dtype})" if isinstance(v, np.ndarray) else repr(v) for k,v in obj.__dict__.items()}
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
    tactic_index_to_numargs: list[int]
    tactic_index_to_string: list[str]    # tactic names
    tactic_index_to_hash: list[int]
    global_context: list[int]
    label_to_names: list[str]
    label_in_spine: list[bool]
    max_subgraph_size: int
