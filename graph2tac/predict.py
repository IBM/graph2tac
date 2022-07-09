from typing import List, Tuple, Optional

import numpy as np

from graph2tac.loader.data_server import GraphConstants


NUMPY_NDIM_LIMIT = 32


def cartesian_product(*arrays):
    """
    using the code from  https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """
    la = len(arrays)

    if la > 32:
        print(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


class Predict:
    """
    Common prediction API to load training checkpoints and make predictions in order to interact with an evaluator.
    This class exposes the following methods:
        - `initialize`: set the global context during evaluation
        - `compute_new_definitions`: update node label embeddings using definition cluster graphs
        - `ranked_predictions`: make predictions for a single proof-state
        - `get_tactic_index_to_numargs`: access the value of `tactic_index_to_numargs` seen during training
        - `get_tactic_index_to_hash`: access the value of `tactic_index_to_hash` seen during training
        - `get_label_to_name`: access the value of `label_to_name` seen during training
        - `get_label_in_spine`: access the value of `label_in_spine` seen during training
        - `get_max_subgraph_size`: access the value of `max_subgraph_size` seen during training
    """
    _graph_constants: GraphConstants

    def __init__(self, graph_constants: GraphConstants):
        """

        @param graph_constants: the graph constants seen during training
        """
        self._graph_constants = graph_constants
        assert self._graph_constants is not None

    def get_tactic_index_to_numargs(self) -> np.ndarray:
        """
        [ Public API ] Returns the tactic_index_to_numargs seen during training
        """
        return self._graph_constants.tactic_index_to_numargs

    def get_tactic_index_to_hash(self) -> np.ndarray:
        """
        [ Public API ] Returns the tactic_index_to_hash seen during training
        """
        return self._graph_constants.tactic_index_to_hash

    def get_label_to_name(self) -> List[str]:
        """
        [ Public API ] Returns the label_to_names seen during training
        """
        return self._graph_constants.label_to_names

    def get_label_in_spine(self) -> List[bool]:
        """
        [ Public API ] Returns the label_in_spine seen during training
        """
        return self._graph_constants.label_in_spine

    def get_max_subgraph_size(self) -> int:
        """
        [ Public API ] Returns the max_subgraph_size seen during training
        """
        return self._graph_constants.max_subgraph_size

    def initialize(self, global_context: Optional[List[int]] = None) -> None:
        """
        [ Public API ] Initializes the model to use a different global context than the one seen during training.

        @param global_context: a replacement for the original global_context in the GraphConstants seen during training
        """
        raise NotImplementedError('initialize should be implemented by sub-classes')

    def ranked_predictions(self,
                           state: Tuple,
                           allowed_model_tactics: List,
                           available_global: Optional[np.ndarray],
                           tactic_expand_bound: int,
                           total_expand_bound: int
                           ) -> Tuple[np.ndarray, List]:
        """
        [ Public API ] Returns actions ordered by their corresponding probabilities, as computed by the model.
            - The sum of probabilities is constrained to be a positive real number less or equal to 1.0
            - The set of actions returned by this function is a subset of all potential actions.
            - The probabilities for all potential actions sum up to 1.0, as normalized by the way of tf.nn.softmax.

        @param state: (graph, root, context)
        @param allowed_model_tactics:
        @param available_global: np.array of indices into global_context
        @param tactic_expand_bound:
        @param total_expand_bound:
        @return: a pair (ranked_actions, ranked_values)
        """
        raise NotImplementedError('ranked_predictions should be implemented by sub-classes')

    def compute_new_definitions(self, new_cluster_subgraphs : List) -> None:
        """
        [ Public API ] Updates definition embeddings using the model to process definition cluster subgraphs.

        @param new_cluster_subgraphs: a list of definition clusters
        """
        raise NotImplementedError('compute_new_definitions should be implemented by sub-classes')
