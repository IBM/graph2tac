from typing import List, Tuple, Optional, Callable, Dict, TypeVar

import time
import pickle
import numpy as np
from pathlib import Path

from graph2tac.common import logger
from graph2tac.loader.data_classes import GraphConstants, LoaderProofstate, LoaderDefinition

# TODO(jrute): This probably isn't needed now
# but I haven't measured the timings without it.
# It prevents tactics with too many arguments from being used.
NUMPY_NDIM_LIMIT = 32

RT = TypeVar('RT')  # return type

def predict_api_debugging(api_call: Callable[..., RT]) -> Callable[..., RT]:
    api_name = api_call.__name__
    def api_call_with_debug(self: "Predict", *args: List, **kwargs: Dict) -> RT:
        if self._debug_dir is not None:
            debug_message_file = self._run_dir / f'{self._debug_message_number}.pickle'
            logger.debug(f'logging {api_name} call to {debug_message_file}')
            with debug_message_file.open('wb') as pickle_jar:
                pickle.dump((api_name, args, kwargs), pickle_jar)
            self._debug_message_number += 1
        start = time.time()
        return_value = api_call(self, *args, **kwargs)
        end = time.time()

        total_time, num_calls = self._timings.get(api_name, (0.0, 0))
        self._timings[api_name] = (total_time + end - start, num_calls+1)
        return return_value
    return api_call_with_debug


class Predict:
    """
    Common prediction API to load training checkpoints and make predictions in order to interact with an evaluator.
    This class exposes the following methods:
        - `allocate_definitions`: set the global context during evaluation
        - `compute_new_definitions`: update node label embeddings using definition cluster graphs
        - `ranked_predictions`: make predictions for a single proof-state
    and the following attribute:
        - `graph_constants`: Dataset constants of type `GraphConstants` seen during training
    """
    graph_constants: GraphConstants
    _debug_dir: Optional[Path]
    _timings: Dict[str, Tuple[float, int]]

    def __init__(self,
                 graph_constants: GraphConstants,
                 tactic_expand_bound: int,
                 search_expand_bound: int,
                 debug_dir: Optional[Path] = None,
    ):
        """

        @param graph_constants: the graph constants seen during training
        @param tactic_expand_bound: how many base tactics to select (may be ignored by the subclass)
        @param search_expand_bound: how many total tactics to return
        @param debug_dir: a directory where all api calls will be logged for debugging purposes
        """
        self.graph_constants = graph_constants 
        self._debug_dir = debug_dir
        self._tactic_expand_bound = tactic_expand_bound
        self._search_expand_bound = search_expand_bound
        self._timings = {}
        assert self.graph_constants is not None

        # if debug mode is on, initialize evaluation logging directory
        if self._debug_dir is not None:
            self._debug_dir.mkdir(exist_ok=True)

            run_dirs = [int(run_dir.name) for run_dir in self._debug_dir.glob('*') if run_dir.is_dir()]
            run_number = max(run_dirs) + 1 if run_dirs else 1

            self._run_dir = self._debug_dir / str(run_number)
            self._run_dir.mkdir()
            logger.info(f'running Predict in debug mode, messages will be stored at {self._run_dir}')

            self._debug_message_number = 0

    @predict_api_debugging
    def allocate_definitions(self, new_node_label_num) -> None:
        """
        [ Public API ] Prepares sufficient size for (new) model's definition

        @param new_node_label_num: required size of the array of nodes stored in the model
        """
        raise NotImplementedError('allocate_definitions should be implemented by sub-classes')

    @predict_api_debugging
    def ranked_predictions(self,
                           state: LoaderProofstate,
                           allowed_model_tactics: List[int],
                           available_global: Optional[np.ndarray],
                           ) -> Tuple[np.ndarray, List]:
        """
        [ Public API ] Returns actions ordered by their corresponding probabilities, as computed by the model.
            - The sum of probabilities is constrained to be a positive real number less or equal to 1.0
            - The set of actions returned by this function is a subset of all potential actions.
            - The probabilities for all potential actions sum up to 1.0, as normalized by the way of tf.nn.softmax.

        @param state: (graph, root, context)
        @param allowed_model_tactics:
        @param available_global: np.array of indices into global_context
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
