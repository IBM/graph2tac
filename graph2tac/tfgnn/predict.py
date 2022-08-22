from typing import Tuple, List, Union, Iterable, Callable, Optional

import re
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from dataclasses import dataclass
from pathlib import Path

from graph2tac.loader.data_server import GraphConstants, LoaderProofstate, LoaderDefinition
from graph2tac.tfgnn.graph_schema import strip_graph
from graph2tac.tfgnn.dataset import Dataset, DataServerDataset
from graph2tac.tfgnn.tasks import PredictionTask, GlobalArgumentPrediction, DefinitionTask, BASE_TACTIC_PREDICTION, LOCAL_ARGUMENT_PREDICTION, GLOBAL_ARGUMENT_PREDICTION, _local_arguments_logits
from graph2tac.tfgnn.models import GraphEmbedding, LogitsFromEmbeddings
from graph2tac.tfgnn.train import Trainer
from graph2tac.common import logger
from graph2tac.predict import Predict, predict_api_debugging, cartesian_product, NUMPY_NDIM_LIMIT


class Inference:
    """
    Container class for a single inference for a given proof-state.
    Subclasses should implement the following methods:
        - `numpy`: converts the inference into numpy format for Vasily's evaluation framework
        - `evaluate`: returns whether the inference is correct or not
    """
    value: float
    numpy: Callable[[], np.ndarray]
    evaluate: Callable[[int, tf.Tensor, tf.Tensor], bool]


@dataclass
class TacticInference(Inference):
    """
    Container class for a single base tactic inference for a given proof-state.
    """
    value: float
    tactic_id: int

    def numpy(self) -> np.ndarray:
        return np.array([[self.tactic_id, self.tactic_id]], dtype=np.uint32)

    def evaluate(self, tactic_id: int, local_arguments: tf.Tensor, global_arguments: tf.Tensor) -> bool:
        return tactic_id == self.tactic_id


@dataclass
class LocalArgumentInference(Inference):
    """
    Container class for a single base tactic and local arguments inference for a given proof-state.
    """
    value: float
    tactic_id: int
    local_arguments: tf.Tensor

    def numpy(self) -> np.ndarray:
        top_row = np.insert(np.zeros_like(self.local_arguments), 0, self.tactic_id)
        bottom_row = np.insert(self.local_arguments, 0, self.tactic_id)
        return np.stack([top_row, bottom_row], axis=-1).astype(np.uint32)

    def evaluate(self, tactic_id: int, local_arguments: tf.Tensor, global_arguments: tf.Tensor) -> bool:
        return tactic_id == self.tactic_id and tf.reduce_all(local_arguments == self.local_arguments)


@dataclass
class GlobalArgumentInference(Inference):
    """
    Container class for a single base tactic and local+global arguments inference for a given proof-state.
    """
    value: float
    tactic_id: int
    local_arguments: tf.Tensor
    global_arguments: tf.Tensor

    def numpy(self) -> np.ndarray:
        top_row = np.insert(np.where(self.global_arguments==-1, 0, 1), 0, self.tactic_id)
        bottom_row = np.insert(np.where(self.global_arguments==-1, self.local_arguments, self.global_arguments), 0, self.tactic_id)
        return np.stack([top_row, bottom_row], axis=-1).astype(np.uint32)

    def evaluate(self, tactic_id: int, local_arguments: tf.Tensor, global_arguments: tf.Tensor) -> bool:
        return tactic_id == self.tactic_id and tf.reduce_all(local_arguments == self.local_arguments) and tf.reduce_all(global_arguments == self.global_arguments)


@dataclass
class PredictOutput:
    """
    Container class for a list of predictions for a given proof-state.
    """
    state: Optional[LoaderProofstate]
    predictions: List[Inference]

    def p_total(self) -> float:
        """
        Computes the total probability captured by all the predictions for this proof-state.
        """
        return sum(np.exp(pred.value) for pred in self.predictions)

    def sort(self) -> None:
        """
        Sorts all the predictions in descending order according to their value (log of probability).
        """
        self.predictions.sort(key=lambda prediction: -prediction.value)

    def numpy(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Converts all predictions to numpy format for interaction with Vasily's evaluation framework.
        """
        self.sort()
        return [pred.numpy() for pred in self.predictions], np.array([np.exp(pred.value) for pred in self.predictions])

    def _evaluate(self,
                  tactic_id: int,
                  local_arguments: tf.Tensor,
                  global_arguments: tf.Tensor,
                  search_expand_bound: Optional[int] = None):
        self.sort()
        predictions = self.predictions[:search_expand_bound] if search_expand_bound is not None else self.predictions
        return any(inference.evaluate(tactic_id, local_arguments, global_arguments) for inference in predictions)

    def evaluate(self, action: Tuple, search_expand_bound: Optional[int] = None) -> bool:
        """
        Evaluate an action in tuple format.
        """
        loader_graph, root, context, proofstate_info = self.state
        local_context_ids, global_context_ids = context
        local_context_length = tf.shape(local_context_ids, out_type=tf.int64)[0]

        tactic_id, arguments_array = action
        tactic_id = tf.cast(tactic_id, dtype=tf.int64)
        arguments_array = tf.cast(arguments_array, dtype=tf.int64)
        action = DataServerDataset._action_to_arguments((tactic_id, arguments_array), local_context_length)
        return self._evaluate(*action, search_expand_bound=search_expand_bound)


class TFGNNPredict(Predict):
    def __init__(self,
                 log_dir: Path,
                 debug_dir: Optional[Path] = None,
                 checkpoint_number: Optional[int] = None,
                 exclude_tactics: Optional[List[str]] = None,
                 numpy_output: bool = True
                 ):
        """
        @param log_dir: the directory for the checkpoint that is to be loaded (as passed to the Trainer class)
        @param debug_dir: set to a directory to dump pickle files for every API call that is made
        @param checkpoint_number: the checkpoint number we want to load (use `None` for the latest checkpoint)
        @param exclude_tactics: a list of tactic names to exclude from all predictions
        @param numpy_output: set to True to return the predictions as a tuple of numpy arrays (for evaluation purposes)
        """
        self.numpy_output = numpy_output

        # create dummy dataset for pre-processing purposes
        graph_constants_filepath = log_dir / 'config' / 'graph_constants.yaml'
        with graph_constants_filepath.open('r') as yml_file:
            graph_constants = GraphConstants(**yaml.load(yml_file, Loader=yaml.UnsafeLoader))

        dataset_yaml_filepath = log_dir / 'config' / 'dataset.yaml'
        with dataset_yaml_filepath.open('r') as yml_file:
            dataset = Dataset(graph_constants=graph_constants, **yaml.load(yml_file, Loader=yaml.SafeLoader))
        self._preprocess = dataset._preprocess

        # call to parent constructor to defines self._graph_constants
        super().__init__(graph_constants=graph_constants, debug_dir=debug_dir)

        # to build dummy proofstates we will need to use a tactic taking no arguments
        self._dummy_tactic_id = tf.argmin(graph_constants.tactic_index_to_numargs)  # num_arguments == 0

        # the decoding mechanism currently does not support tactics with more than NUMPY_NDIM_LIMIT
        self._tactic_mask = tf.constant(graph_constants.tactic_index_to_numargs<NUMPY_NDIM_LIMIT)

        # mask tactics explicitly excluded from predictions
        if exclude_tactics is not None:
            exclude_tactics = set(exclude_tactics)
            self._tactic_mask &= np.array([(tactic_name.decode() not in exclude_tactics) for tactic_name in graph_constants.tactic_index_to_string])

        # when doing local argument predictions, if the local context is empty we mask all tactics which take arguments
        self._tactic_mask_no_arguments = tf.constant(graph_constants.tactic_index_to_numargs == 0)

        # create prediction task
        prediction_yaml_filepath = log_dir / 'config' / 'prediction.yaml'
        self.prediction_task = PredictionTask.from_yaml_config(graph_constants=dataset.graph_constants(),
                                                               yaml_filepath=prediction_yaml_filepath)

        # create definition task
        definition_yaml_filepath = log_dir / 'config' / 'definition.yaml'
        if definition_yaml_filepath.is_file():
            self.definition_task = DefinitionTask.from_yaml_config(graph_embedding=self.prediction_task.graph_embedding,
                                                                   gnn=self.prediction_task.gnn,
                                                                   yaml_filepath=definition_yaml_filepath)
        else:
            self.definition_task = None

        # load training checkpoint
        checkpoint = tf.train.Checkpoint(prediction_task=self.prediction_task.checkpoint)
        if self.definition_task is not None:
            checkpoint.definition_task = self.definition_task

        checkpoints_path = log_dir / 'ckpt'
        available_checkpoints = {int(re.search('ckpt-(\d+).index', str(ckpt)).group(1)): ckpt.with_suffix('')
                                 for ckpt in checkpoints_path.glob('*.index')}
        if checkpoint_number is None:
            checkpoint_number = max(available_checkpoints.keys())
            logger.info(f'no checkpoint number specified, using latest available checkpoint #{checkpoint_number}')
        elif checkpoint_number not in available_checkpoints.keys():
            logger.error(f'checkpoint #{checkpoint_number} is not available')
            raise ValueError(f'checkpoint number {checkpoint_number} not found')

        try:
            load_status = checkpoint.restore(save_path=str(available_checkpoints[checkpoint_number]))
        except tf.errors.OpError as error:
            logger.error(f'unable to restore checkpoint #{checkpoint_number}!')
            raise error
        else:
            load_status.expect_partial().assert_nontrivial_match().assert_existing_objects_matched().run_restore_ops()
            logger.info(f'restored checkpoint #{checkpoint_number}!')

        # choose the prediction method according to the task type
        prediction_task_type = self.prediction_task.get_config().get('prediction_task_type')
        if prediction_task_type == BASE_TACTIC_PREDICTION:
            self._batch_ranked_predictions = self._base_tactic_prediction
        elif prediction_task_type == LOCAL_ARGUMENT_PREDICTION:
            self._batch_ranked_predictions = self._local_argument_prediction
        elif prediction_task_type == GLOBAL_ARGUMENT_PREDICTION:
            self._batch_ranked_predictions = self._global_argument_prediction
        else:
            raise ValueError(f'{prediction_task_type} is not a supported prediction task type')

    @staticmethod
    def _extend_graph_embedding(graph_embedding: GraphEmbedding, new_node_label_num: int) -> GraphEmbedding:
        new_graph_embedding = GraphEmbedding(node_label_num=new_node_label_num,
                                             edge_label_num=graph_embedding._edge_label_num,
                                             hidden_size=graph_embedding._hidden_size)

        new_graph_embedding._edge_embedding = graph_embedding._edge_embedding

        new_labels = graph_embedding._node_label_num + tf.range(new_node_label_num - graph_embedding._node_label_num)
        new_embeddings = tf.concat([graph_embedding._node_embedding.embeddings, new_graph_embedding._node_embedding(new_labels)], axis=0)

        new_graph_embedding._node_embedding.set_weights([new_embeddings])
        return new_graph_embedding

    @predict_api_debugging
    def initialize(self, global_context: Optional[List[int]] = None) -> None:
        if global_context is not None:
            # update the global context
            self._graph_constants.global_context = tf.constant(global_context, dtype=tf.int32)

            # extend the embedding table if necessary
            new_node_label_num = max(global_context)+1
            if new_node_label_num > self._graph_constants.node_label_num:
                logger.info(f'extending global context from {self._graph_constants.node_label_num} to {new_node_label_num} elements')
                new_graph_embedding = self._extend_graph_embedding(graph_embedding=self.prediction_task.graph_embedding,
                                                                   new_node_label_num=new_node_label_num)
                self.prediction_task.graph_embedding = new_graph_embedding
                if self.definition_task is not None:
                    self.definition_task._graph_embedding = new_graph_embedding
                self._graph_constants.node_label_num = new_node_label_num

            # update the global arguments logits head (always necessary, because the local context may shrink!)
            self.prediction_task.global_arguments_logits = LogitsFromEmbeddings(
                embedding_matrix=self.prediction_task.graph_embedding._node_embedding.embeddings,
                valid_indices=tf.constant(self._graph_constants.global_context, dtype=tf.int32),
                name=GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS
            )

    @predict_api_debugging
    def compute_new_definitions(self, new_cluster_subgraphs: List[LoaderDefinition]) -> None:
        if self.definition_task is None:
            raise RuntimeError('cannot update definitions when a definition task is not present')
        definition_graph = self._make_definition_batch(new_cluster_subgraphs)

        scalar_definition_graph = definition_graph.merge_batch_to_components()
        definition_embeddings = self.definition_task(scalar_definition_graph).flat_values
        defined_labels = Trainer._get_defined_labels(definition_graph).flat_values

        node_label_num = self.prediction_task.graph_embedding._node_label_num
        hidden_size = self.prediction_task.graph_embedding._hidden_size
        update_mask = tf.reduce_sum(tf.one_hot(defined_labels, node_label_num, axis=0), axis=-1, keepdims=True)
        new_embeddings = update_mask * tf.scatter_nd(indices=tf.expand_dims(defined_labels, axis=-1),
                                                     updates=definition_embeddings,
                                                     shape=(node_label_num, hidden_size))
        old_embeddings = (1 - update_mask) * self.prediction_task.graph_embedding._node_embedding.embeddings
        self.prediction_task.graph_embedding._node_embedding.set_weights([new_embeddings + old_embeddings])

    def _dummy_proofstate_data_generator(self, states: List[Tuple]):
        for state in states:
            action = (self._dummy_tactic_id, tf.zeros(shape=(0, 2), dtype=tf.int64))
            graph_id = tf.constant(-1, dtype=tf.int64)
            yield state, action, graph_id

    def _make_dummy_proofstate_dataset(self, states: List[Tuple]) -> tf.data.Dataset:
        """
        Create a dataset of (dummy) proofstate graphs.

        @param states: list of proof-states in tuple form (as returned by the data loader)
        @return: a `tf.data.Dataset` producing `GraphTensor` objects following the `proofstate_graph_spec` schema
        """
        dataset = tf.data.Dataset.from_generator(lambda: self._dummy_proofstate_data_generator(states),
                                                 output_signature=DataServerDataset.proofstate_data_spec)
        dataset = dataset.map(DataServerDataset._make_proofstate_graph_tensor)
        dataset = dataset.apply(self._preprocess)
        return dataset

    def _make_definition_batch(self, new_cluster_subgraphs: Iterable[LoaderDefinition]) -> tfgnn.GraphTensor:
        """
        Create a dataset of definition graphs.

        @param new_cluster_subgraphs: list of definition clusters in tuple form (as returned by the data loader)
        @return: a `GraphTensor` containing the definition clusters, following the `definition_graph_spec` schema
        """
        dataset = tf.data.Dataset.from_generator(lambda: new_cluster_subgraphs,
                                                 output_signature=DataServerDataset.definition_data_spec)
        dataset = dataset.map(DataServerDataset._make_definition_graph_tensor)
        dataset = self._preprocess(dataset)
        return dataset.batch(Dataset.MAX_DEFINITIONS).get_single_element()

    @staticmethod
    def _logits_decoder(logits: tf.Tensor, total_expand_bound: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements the same decoding mechanism as in `Predict` from `graph2tac.tf2.predict`.
        """
        num_arguments = tf.shape(logits)[0]
        if num_arguments == 0:
            return np.array([[]], dtype=np.uint32), np.array([0], dtype=np.float32)

        logits = tf.math.log_softmax(logits).numpy()

        expand_bound = int(total_expand_bound**(1/num_arguments))

        sorted_indices = np.argsort(-logits).astype(dtype=np.uint32)
        restricted_indices = sorted_indices[:, :expand_bound]
        arg_combinations = cartesian_product(*restricted_indices)
        first_index = np.tile(np.arange(num_arguments), (arg_combinations.shape[0], 1))
        combination_values = np.sum(logits[first_index, arg_combinations], axis=1)
        return arg_combinations, combination_values

    @staticmethod
    def _expand_local_arguments_combinations(tactic_id: int,
                                             arg_combinations: np.ndarray,
                                             combination_values: np.ndarray
                                             ) -> List[Inference]:
        """
        Produces a list of `Prediction` objects with various local argument combinations for a given tactic.
        """
        return [LocalArgumentInference(tactic_id=int(tactic_id),
                                       local_arguments=tf.cast(arguments, dtype=tf.int64),
                                       value=float(value))
                for arguments, value in zip(arg_combinations, combination_values)]

    @staticmethod
    def _expand_global_arguments_combinations(tactic_id: int,
                                              arg_combinations: np.ndarray,
                                              combination_values: np.ndarray,
                                              global_context_size: int
                                              ):
        """
        Produces a list of `Prediction` objects with various local and global argument combinations for a given tactic.
        """
        return [GlobalArgumentInference(tactic_id=int(tactic_id),
                                        local_arguments=tf.where(arguments < global_context_size, -1, arguments - global_context_size),
                                        global_arguments=tf.where(arguments < global_context_size, arguments, -1),
                                        value=float(value))
                for arguments, value in zip(tf.cast(arg_combinations, dtype=tf.int64), combination_values)]

    def _top_k_tactics(self,
                       scalar_proofstate_graph: tfgnn.GraphTensor,
                       tactic_expand_bound: int,
                       mask_tactics_with_arguments: tf.Tensor,
                       allowed_tactics
                       ):
        """
        Returns the top `tactic_expand_bound` tactics for each of the proof-states in the input.
        This is the first stage of any  prediction process.
        """
        # get the graph tensor with hidden states
        bare_graph = strip_graph(scalar_proofstate_graph)
        embedded_graph = self.prediction_task.graph_embedding(bare_graph, training=False)
        hidden_graph = self.prediction_task.gnn(embedded_graph, training=False)

        # predict the tactic embedding and logits
        tactic_embedding = self.prediction_task.tactic_head(hidden_graph, training=False)
        tactic_logits = self.prediction_task.tactic_logits_from_embeddings(tactic_embedding, training=False)

        # mask and normalize the tactic logits
        tactic_mask = tf.where(tf.expand_dims(mask_tactics_with_arguments, axis=1),
                               tf.expand_dims(self._tactic_mask_no_arguments, axis=0),
                               tf.expand_dims(self._tactic_mask, axis=0))
        if allowed_tactics is not None:
            tactic_num = self._graph_constants.tactic_num
            allowed_tactics_mask = tf.reduce_any(tf.cast(tf.one_hot(allowed_tactics, tactic_num), tf.bool), axis=0)
            tactic_mask &= tf.expand_dims(allowed_tactics_mask, axis=0)

        tactic_expand_bound = min(tactic_expand_bound, tf.reduce_sum(tf.cast(tactic_mask, tf.int32)))
        tactic_logits = tf.math.log_softmax(tactic_logits + tf.math.log(tf.cast(tactic_mask, tf.float32)), axis=-1)

        # get the top tactic_expand_bound tactics, their logits and their number of arguments
        top_k = tf.math.top_k(tactic_logits, k=tactic_expand_bound)
        return top_k, hidden_graph

    def _base_tactic_prediction(self,
                                scalar_proofstate_graph: tfgnn.GraphTensor,
                                tactic_expand_bound: int,
                                total_expand_bound: int,
                                allowed_model_tactics: Optional[Iterable[int]] = None
                                ) -> List[PredictOutput]:
        """
        Produces base tactic predictions with no arguments (for the base tactic prediction task).
        """

        # in principle, we should never predict tactics which require arguments
        # mask_tactics_with_arguments = tf.ones(shape=(scalar_proofstate_graph.num_components,), dtype=tf.bool)
        mask_tactics_with_arguments = tf.zeros(shape=(scalar_proofstate_graph.num_components,), dtype=tf.bool)

        top_k, _ = self._top_k_tactics(scalar_proofstate_graph=scalar_proofstate_graph,
                                       tactic_expand_bound=tactic_expand_bound,
                                       mask_tactics_with_arguments=mask_tactics_with_arguments,
                                       allowed_tactics=allowed_model_tactics)

        batch_predictions = []
        for top_k_tactics, top_k_logits in zip(top_k.indices, top_k.values):
            predictions = [TacticInference(tactic_id=int(tactic_id), value=float(value))
                           for tactic_id, value in zip(top_k_tactics, top_k_logits)]
            batch_predictions.append(PredictOutput(state=None, predictions=predictions))
        return batch_predictions

    def _local_argument_prediction(self,
                                   scalar_proofstate_graph: tfgnn.GraphTensor,
                                   tactic_expand_bound: int,
                                   total_expand_bound: int,
                                   allowed_model_tactics: Optional[Iterable[int]] = None
                                   ) -> List[PredictOutput]:
        """
        Produces base tactic and local argument predictions (for the local argument prediction task).
        """
        # get the batch size
        batch_size = scalar_proofstate_graph.num_components.numpy()

        # get the local context node ids from the input graph
        batch_local_context_ids = scalar_proofstate_graph.context['local_context_ids']

        # we should only use tactics with arguments when the local context is non-empty
        mask_tactics_with_arguments = batch_local_context_ids.row_lengths() == 0

        # get the top tactic_expand_bound tactics and input/output graphs
        top_k, hidden_graph = self._top_k_tactics(scalar_proofstate_graph=scalar_proofstate_graph,
                                                  tactic_expand_bound=tactic_expand_bound,
                                                  mask_tactics_with_arguments=mask_tactics_with_arguments,
                                                  allowed_tactics=allowed_model_tactics)

        # get the number of arguments for each tactic (as with top_k elements, the shape is [batch_size, k])
        top_k_num_arguments = tf.gather(tf.constant(self.prediction_task._graph_constants.tactic_index_to_numargs, dtype=tf.int32), top_k.indices)

        batch_predictions = [PredictOutput(state=None, predictions=[]) for _ in range(batch_size)]
        for batch_tactic, batch_tactic_logits, batch_num_arguments in zip(tf.unstack(top_k.indices, axis=1),
                                                                          tf.unstack(top_k.values, axis=1),
                                                                          tf.unstack(top_k_num_arguments, axis=1)):
            if tf.reduce_sum(batch_num_arguments) > 0:
                # obtain hidden states for all the arguments
                tactic_embedding = self.prediction_task.tactic_embedding(batch_tactic)
                hidden_state_sequences = self.prediction_task.arguments_head((hidden_graph, tactic_embedding, batch_num_arguments), training=False)

                # compute logits for all arguments while masking out non-local-context nodes
                batch_arguments_logits = _local_arguments_logits(scalar_proofstate_graph, hidden_graph, hidden_state_sequences)
                for state_prediction, tactic_id, tactic_value, num_arguments, arguments_logits, local_context_ids in zip(batch_predictions, batch_tactic, batch_tactic_logits, batch_num_arguments, batch_arguments_logits, batch_local_context_ids):
                    local_context_length = tf.shape(local_context_ids)[0]
                    local_context_logits = arguments_logits[:num_arguments,:local_context_length]
                    arg_combinations, combination_values = self._logits_decoder(local_context_logits,
                                                                                total_expand_bound=total_expand_bound)
                    combination_values += tactic_value.numpy()

                    state_prediction.predictions.extend(
                        self._expand_local_arguments_combinations(tactic_id=int(tactic_id),
                                                                  arg_combinations=arg_combinations,
                                                                  combination_values=combination_values)
                    )
            else:
                for state_prediction, tactic_id, tactic_value in zip(batch_predictions, batch_tactic, batch_tactic_logits):
                    prediction = LocalArgumentInference(tactic_id=int(tactic_id),
                                                         local_arguments=tf.constant([], dtype=tf.int64),
                                                         value=float(tactic_value))
                    state_prediction.predictions.append(prediction)
        return batch_predictions

    def _global_argument_prediction(self,
                                    scalar_proofstate_graph: tfgnn.GraphTensor,
                                    tactic_expand_bound: int,
                                    total_expand_bound: int,
                                    allowed_model_tactics: Optional[Iterable[int]] = None
                                    ) -> List[PredictOutput]:
        # get the batch size
        batch_size = scalar_proofstate_graph.num_components.numpy()

        # TODO: Here we should check whether the local and global context are both empty (unlikely)
        mask_tactics_with_arguments = tf.zeros(shape=(scalar_proofstate_graph.num_components,), dtype=bool)

        # get the top tactic_expand_bound tactics and input/output graphs
        top_k, hidden_graph = self._top_k_tactics(scalar_proofstate_graph=scalar_proofstate_graph,
                                                  tactic_expand_bound=tactic_expand_bound,
                                                  mask_tactics_with_arguments=mask_tactics_with_arguments,
                                                  allowed_tactics=allowed_model_tactics)

        # get the local context node ids from the input graph
        batch_local_context_ids = scalar_proofstate_graph.context['local_context_ids']

        # get the number of arguments for each tactic (as with top_k elements, the shape is [batch_size, k])
        top_k_num_arguments = tf.gather(tf.constant(self.prediction_task._graph_constants.tactic_index_to_numargs, dtype=tf.int32), top_k.indices)

        batch_predictions = [PredictOutput(state=None, predictions=[]) for _ in range(batch_size)]
        for batch_tactic, batch_tactic_logits, batch_num_arguments in zip(tf.unstack(top_k.indices, axis=1),
                                                                          tf.unstack(top_k.values, axis=1),
                                                                          tf.unstack(top_k_num_arguments, axis=1)):
            if tf.reduce_sum(batch_num_arguments) > 0:
                tactic_embedding = self.prediction_task.tactic_embedding(batch_tactic)
                hidden_state_sequences = self.prediction_task.arguments_head((hidden_graph, tactic_embedding, batch_num_arguments), training=False)

                batch_local_arguments_logits = _local_arguments_logits(scalar_proofstate_graph, hidden_graph, hidden_state_sequences)

                batch_global_arguments_logits = self.prediction_task.global_arguments_logits(hidden_state_sequences.to_tensor())
                global_context_size = int(tf.shape(batch_global_arguments_logits)[-1])
                for state_prediction, tactic_id, tactic_value, num_arguments, local_arguments_logits, global_arguments_logits, local_context_ids in zip(batch_predictions, batch_tactic, batch_tactic_logits, batch_num_arguments, batch_local_arguments_logits, batch_global_arguments_logits, batch_local_context_ids):
                    local_context_length = tf.shape(local_context_ids)[0]
                    local_context_logits = local_arguments_logits[:num_arguments, :local_context_length]
                    logits = tf.concat([global_arguments_logits[:num_arguments,:], local_context_logits], axis=-1)
                    arg_combinations, combination_values = self._logits_decoder(logits,
                                                                                total_expand_bound=total_expand_bound)
                    combination_values += tactic_value.numpy()
                    state_prediction.predictions.extend(
                        self._expand_global_arguments_combinations(tactic_id=int(tactic_id),
                                                                   arg_combinations=arg_combinations,
                                                                   combination_values=combination_values,
                                                                   global_context_size=global_context_size)
                    )
            else:
                for state_prediction, tactic_id, tactic_value in zip(batch_predictions, batch_tactic, batch_tactic_logits):
                    prediction = GlobalArgumentInference(tactic_id=int(tactic_id),
                                                         local_arguments=tf.constant([], dtype=tf.int64),
                                                         global_arguments=tf.constant([], dtype=tf.int64),
                                                         value=float(tactic_value))
                    state_prediction.predictions.append(prediction)
        return batch_predictions

    def batch_ranked_predictions(self,
                                 states: List[LoaderProofstate],
                                 tactic_expand_bound: int,
                                 total_expand_bound: int,
                                 allowed_model_tactics: Optional[Iterable[int]] = None
                                 ) -> List[Union[PredictOutput, Tuple[List[np.ndarray], np.ndarray]]]:
        # convert the input to a batch of graph tensor (rank 1)
        proofstate_graph = self._make_dummy_proofstate_dataset(states).batch(len(states)).get_single_element()

        # convert into a scalar graph (rank 0)
        scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()

        batch_predict_output = self._batch_ranked_predictions(scalar_proofstate_graph=scalar_proofstate_graph,
                                                              tactic_expand_bound=tactic_expand_bound,
                                                              total_expand_bound=total_expand_bound,
                                                              allowed_model_tactics=allowed_model_tactics)
        for state, predict_output in zip(states, batch_predict_output):
            predict_output.state = state

        if not self.numpy_output:
            return batch_predict_output
        else:
            return [predict_output.numpy() for predict_output in batch_predict_output]

    @predict_api_debugging
    def ranked_predictions(self,
                           state: LoaderProofstate,
                           tactic_expand_bound: int,
                           total_expand_bound: int,
                           available_global: Optional[np.ndarray] = None,
                           allowed_model_tactics: Optional[Iterable[int]] = None
                           ) -> Union[PredictOutput, Tuple[np.ndarray, np.ndarray]]:
        """
        Produces predictions for a single proof-state.
        """
        if available_global is not None:
            raise NotImplementedError('available_global is not supported yet')

        return self.batch_ranked_predictions(states=[state],
                                             allowed_model_tactics=allowed_model_tactics,
                                             tactic_expand_bound=tactic_expand_bound,
                                             total_expand_bound=total_expand_bound)[0]

    def _evaluate(self,
                  proofstate_graph_dataset: tf.data.Dataset,
                  batch_size: int,
                  tactic_expand_bound: int,
                  total_expand_bound: int,
                  search_expand_bound: Optional[int] = None,
                  allowed_model_tactics: Optional[Iterable[int]] = None
                  ) -> Tuple[float, float]:
        predictions = []
        tactic = []
        local_arguments = []
        global_arguments = []
        names = []
        for proofstate_graph in iter(proofstate_graph_dataset.batch(batch_size)):
            scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()
            batch_predict_output = self._batch_ranked_predictions(scalar_proofstate_graph=scalar_proofstate_graph,
                                                                  tactic_expand_bound=tactic_expand_bound,
                                                                  total_expand_bound=total_expand_bound,
                                                                  allowed_model_tactics=allowed_model_tactics)
            predictions.extend(batch_predict_output)
            tactic.append(scalar_proofstate_graph.context['tactic'])
            local_arguments.append(scalar_proofstate_graph.context['local_arguments'])
            global_arguments.append(scalar_proofstate_graph.context['global_arguments'])
            names.append(scalar_proofstate_graph.context['name'])

        tactic = tf.concat(tactic, axis=0)
        local_arguments = tf.concat(local_arguments, axis=0)
        global_arguments = tf.concat(global_arguments, axis=0)
        names = tf.concat(names, axis=0).numpy()

        per_proofstate = []
        per_lemma = {}
        for action, name, predict_output in zip(zip(tactic, local_arguments, global_arguments), names, predictions):
            result = predict_output._evaluate(*action, search_expand_bound=search_expand_bound)
            per_proofstate.append(result)

            per_lemma[name] = (per_lemma.get(name, True) and result)
        per_proofstate_result = np.array(per_proofstate).mean()
        per_lemma_result = np.array(list(per_lemma.values())).mean()
        return per_proofstate_result, per_lemma_result

    def evaluate(self,
                 state_action_pairs: Iterable[Tuple[LoaderProofstate, Tuple]],
                 batch_size: int,
                 tactic_expand_bound: int,
                 total_expand_bound: int,
                 search_expand_bound: Optional[int] = None,
                 allowed_model_tactics: Optional[Iterable[int]] = None
                 ) -> Tuple[float, float]:
        states, actions = zip(*state_action_pairs)
        proofstate_graph_dataset = self._make_dummy_proofstate_dataset(states).batch(batch_size)

        predictions = []
        for proofstate_graph in iter(proofstate_graph_dataset):
            scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()
            batch_predict_output = self._batch_ranked_predictions(scalar_proofstate_graph=scalar_proofstate_graph,
                                                                  tactic_expand_bound=tactic_expand_bound,
                                                                  total_expand_bound=total_expand_bound,
                                                                  allowed_model_tactics=allowed_model_tactics)
            predictions.extend(batch_predict_output)

        per_proofstate = []
        per_lemma = {}
        for (state, action), predict_output in zip(state_action_pairs, predictions):
            predict_output.state = state
            result = predict_output.evaluate(action, search_expand_bound=search_expand_bound)
            per_proofstate.append(result)

            _, _, _, (name, _, _) = state
            per_lemma[name] = (per_lemma.get(name, True) and result)
        per_proofstate_result = np.array(per_proofstate).mean()
        per_lemma_result = np.array(list(per_lemma.values())).mean()
        return per_proofstate_result, per_lemma_result
