"""
Parameters for the model in model.py.  These classes are:
- created from yaml files (training, prediction, etc)
- used to build the model
- stored inside the model wrapper class
- saved in yaml files along with the model weights
"""

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import Optional
import yaml

# TODO(jrute): This is really just a copy of GraphConstants and should be just replaced with it
# The only reason this was made was because of the non-serializable data in graph constants,
# but that is not as much of a problem with the way Fidel is serializing and deserializing the yaml.

@dataclass_json
@dataclass
class ModelDatasetConstants:
    """
    Dataset constants for building the model.

    These will be inferred automatically during training, but should
    be explicitly set for inference.
    """
    tactic_num: int  # number of base tactics to predict
    tactic_max_arg_num: int  # max number of args for a tactic
    edge_label_num: int  # number of edge labels
    base_node_label_num: int  # number of base node labels
    node_label_num: int  # number of extended node labels, where each def has its own node label
    tactic_index_to_numargs: list[int]  # num of args for each base tactic (in order of tactic id)
    tactic_index_to_hash: list[int]
    node_label_to_name: list[str]
    node_label_in_spine: list[bool]
    global_context: list[int] # list of constants that can be predicted
    max_subgraph_size: int

@dataclass_json
@dataclass
class ModelParams:
    # dataset constants
    dataset_consts: Optional[ModelDatasetConstants] = None # Should be None for training

    # preprocessing
    ignore_definitions: bool = False  # initial node embedding depends only on base node class
    normalize_def_embeddings: bool = True  # force node and def embeddings to be unit vectors
    single_edge_label: bool = False  # ignore edge labels (except for edge direction and self-edges)
    symmetric_edges: bool = False  # assign both directions of an edge the same edge label
    #TODO(jrute): Should this be expressed negatively, e.g. no_self_edges?
    self_edges: bool = True  # add reflexive self edges to the graph

    # main body
    total_hops: int = 10  # number of message passing hops
    node_dim: int = 64  # node embedding dimension for each layer
    message_passing_layer: str = "conv2"  # the message passing layer type
    norm_msgs: bool = False  # a setting for conv2 message passing
    nonlin_position: Optional[str] = None  # None, "after_mp", "after_dropout", "after_add", "after_norm"
    nonlin_type: str = "relu"
    residuals: bool = False
    residual_dropout: bool = False
    residual_norm: Optional[str] = None  # None, "layer_norm", "batch_norm"

    # final layers
    final_collapse: bool = False
    aggreg_max: bool = False

    # definition training
    use_same_graph_nn_weights_for_def_training: bool = True

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml_file(yaml_file: Path) -> "ModelParams":
        with yaml_file.open() as f:
            param_dict=yaml.load(f, Loader=yaml.FullLoader)
            if param_dict is None:  # empty file
                param_dict = {}
            return ModelParams.from_dict(param_dict)
