from typing import List, Tuple, Optional, Callable, Dict

import pickle
import numpy as np
from pathlib import Path

from graph2tac.common import logger
from graph2tac.loader.data_server import GraphConstants, LoaderProofstate, LoaderDefinition


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


def predict_api_debugging(api_call: Callable):
    api_name = api_call.__name__
    def api_call_with_debug(self: "Predict", *args: List, **kwargs: Dict):
        if self._debug_dir is not None:
            debug_message_file = self._run_dir / f'{self._debug_message_number}.pickle'
            logger.debug(f'logging {api_name} call to {debug_message_file}')
            with debug_message_file.open('wb') as pickle_jar:
                pickle.dump((api_name, args, kwargs), pickle_jar)
            self._debug_message_number += 1
        return api_call(self, *args, **kwargs)
    return api_call_with_debug


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

    def __init__(self, graph_constants: GraphConstants, debug_dir: Optional[Path] = None):
        """

        @param graph_constants: the graph constants seen during training
        @param debug_dir: a directory where all api calls will be logged for debugging purposes
        """
        self._graph_constants = graph_constants
        self._debug_dir = debug_dir
        assert self._graph_constants is not None

        # if debug mode is on, initialize evaluation logging directory
        if self._debug_dir is not None:
            self._debug_dir.mkdir(exist_ok=True)

            run_dirs = [int(run_dir.name) for run_dir in self._debug_dir.glob('*') if run_dir.is_dir()]
            run_number = max(run_dirs) + 1 if run_dirs else 1

            self._run_dir = debug_dir / str(run_number)
            self._run_dir.mkdir()
            logger.info(f'running Predict in debug mode, messages will be stored at {self._run_dir}')

            self._debug_message_number = 0

    @predict_api_debugging
    def get_tactic_index_to_numargs(self) -> np.ndarray:
        """
        [ Public API ] Returns the tactic_index_to_numargs seen during training
        """
        return self._graph_constants.tactic_index_to_numargs

    @predict_api_debugging
    def get_tactic_index_to_hash(self) -> np.ndarray:
        """
        [ Public API ] Returns the tactic_index_to_hash seen during training
        """
        return self._graph_constants.tactic_index_to_hash

    @predict_api_debugging
    def get_label_to_name(self) -> List[str]:
        """
        [ Public API ] Returns the label_to_names seen during training
        """
        return self._graph_constants.label_to_names

    @predict_api_debugging
    def get_label_in_spine(self) -> List[bool]:
        """
        [ Public API ] Returns the label_in_spine seen during training
        """
        return self._graph_constants.label_in_spine

    @predict_api_debugging
    def get_max_subgraph_size(self) -> int:
        """
        [ Public API ] Returns the max_subgraph_size seen during training
        """
        return self._graph_constants.max_subgraph_size

    @predict_api_debugging
    def initialize(self, global_context: Optional[List[int]] = None) -> None:
        """
        [ Public API ] Initializes the model to use a different global context than the one seen during training.

        @param global_context: a replacement for the original global_context in the GraphConstants seen during training
        """
        raise NotImplementedError('initialize should be implemented by sub-classes')

    @predict_api_debugging
    def ranked_predictions(self,
                           state: LoaderProofstate,
                           allowed_model_tactics: List[int],
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

    @predict_api_debugging
    def compute_new_definitions(self, new_cluster_subgraphs : List[LoaderDefinition]) -> None:
        """
        [ Public API ] Updates definition embeddings using the model to process definition cluster subgraphs.

        @param new_cluster_subgraphs: a list of definition clusters
        """
        raise NotImplementedError('compute_new_definitions should be implemented by sub-classes')
