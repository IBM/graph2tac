from dataclasses import dataclass
from typing import Optional, Any, List
import tensorflow as tf
from collections import namedtuple

int64_spec = tf.TensorSpec(shape=(), dtype=tf.int64)
str_spec = tf.TensorSpec(shape=(), dtype=tf.string)

def namedtuple_with_spec(namedtuple_name, **item_spec_d):
    res = namedtuple(namedtuple_name, list(item_spec_d.keys()))
    spec = res(**item_spec_d)
    return res, spec

LoaderGraph, LoaderGraphSpec = namedtuple_with_spec(
    "LoaderGraph",
    nodes = tf.TensorSpec([None], tf.int64), # Node class labels
    edges = tf.TensorSpec([None,2], tf.int32), # Edges as source-target pairs indexed into `nodes`
    edge_labels = tf.TensorSpec([None], tf.int64),
    edge_offsets = tf.TensorSpec([None], tf.int32), # Start position in `edges` of each edge label.
)

ProofstateMetadata, ProofstateMetadataSpec = namedtuple_with_spec(
    "ProofstateMetadata",
    name = tf.TensorSpec([], tf.string),
    step = tf.TensorSpec([], tf.int64),
    is_faithful = tf.TensorSpec([], tf.int64),
)

ProofstateContext, ProofstateContextSpec = namedtuple_with_spec(
    "ProofstateMetadata",
    local_context = tf.TensorSpec([None], dtype=tf.int64),
    # Local context as indices into the `node` field of the corresponding graph.
    global_context = tf.TensorSpec([None], dtype=tf.int64),
    # Global context as indices into list of all global identifiers.
    # Shape: [size of global context]
    # The index is into the list of global definitions maintained by the dataserver and kept in the model
    # (not the list of node labels).  Hence the smallest `global_context` value is usually 0.
)

LoaderProofstate, LoaderProofstateSpec = namedtuple_with_spec(
    "LoaderProofstate",
    graph = LoaderGraphSpec,
    root = tf.TensorSpec([], dtype=tf.int64), # Index of the root node in `graph.nodes`. Currently always 0.
    context = ProofstateContextSpec,
    metadata = ProofstateMetadataSpec,
)

LoaderAction, LoaderActionSpec = namedtuple_with_spec(
    "LoaderAction",
    tactic_id = tf.TensorSpec([], tf.int64), # Base tactic id
    local_args = tf.TensorSpec([None], dtype=tf.int64),
    global_args = tf.TensorSpec([None], dtype=tf.int64),
    # Tactic arguments (local or global or none).  Using the indices in the `context` field of the proofstate.
    
    # If `local_args[i] == l' for 'l >= 0', then the `i`th arg is a local context variable with index `l`.
    # If `global_args[i] == g` for `g >= 0`, then the `i`th arg is a global definition with index `g`.
    # Otherwise, the values are set to '-1'
    
    # The local and global arguments are indices into `context.local_context` and `context.global_context` respectively.
)

LoaderDefinition, LoaderDefinitionSpec = namedtuple_with_spec(
    "LoaderDefinition",
    graph = LoaderGraphSpec,
    num_definitions = tf.TensorSpec([], tf.int64),
    # Number of definitions in the graph.
    # The roots for these definitions are the nodes of the `graph.nodes`.
    # (E.g. inductive types are packaged as a single graph with
    # mulitple definitions, one for the type and one for each constructor
    definition_names = tf.TensorSpec([None], tf.string),
)

# possible symmetrizations
BIDIRECTIONAL = 'bidirectional'
UNDIRECTED = 'undirected'

@dataclass
class DataConfig:
    max_subgraph_size: int
    bfs_option: bool
    stop_at_definitions: bool
    symmetrization: Optional[str] # "bidirectional" or "undirected" or None
    add_self_edges: bool

@dataclass
class DatasetConfig:
    data_config: DataConfig
    split_method : str # "hash" or "file_prefix"
    split : Any # further argument for get_splitter
    restrict_to_spine : bool
    exclude_none_arguments : bool
    exclude_not_faithful : bool
    required_tactic_occurrence : int
    shuffle_random_seed : int

@dataclass
class GraphConstants:
    data_config : DataConfig
    tactic_num: int
    edge_label_num: int
    base_node_label_num: int
    node_label_num: int
    cluster_subgraphs_num: int
    tactic_index_to_numargs: List[int]
    tactic_index_to_string: List[str]    # tactic names
    tactic_index_to_hash: List[int]
    label_to_names: List[str]
    label_to_ident: List[int]
    label_in_spine: List[bool]
