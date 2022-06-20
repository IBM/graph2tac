from typing import Tuple, List, NamedTuple, Union, Iterable, Callable, Optional

import yaml
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from dataclasses import dataclass
from pathlib import Path

from graph2tac.loader.data_server import GraphConstants
from graph2tac.tf2.predict import cartesian_product
from graph2tac.tfgnn.graph_schema import proofstate_graph_spec, definition_graph_spec, strip_graph
from graph2tac.tfgnn.dataset import Dataset, DataLoaderDataset
from graph2tac.tfgnn.tasks import PredictionTask, GlobalArgumentPrediction, DefinitionTask, BASE_TACTIC_PREDICTION, LOCAL_ARGUMENT_PREDICTION, GLOBAL_ARGUMENT_PREDICTION, _local_arguments_logits
from graph2tac.tfgnn.models import GraphEmbedding, LogitsFromEmbeddings
from graph2tac.tfgnn.train import Trainer


NUMPY_NDIM_LIMIT = 32


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
    state: Optional[Tuple]
    predictions: List[Inference]

    def p_total(self) -> float:
        """
        Computes the total probability captured by all the predictions for this proof-state.
        """
        return sum(np.exp(prediction.value) for prediction in self.predictions)

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
        return [pred.numpy() for pred in self.predictions], np.array([pred.value for pred in self.predictions])

    def _evaluate(self, tactic_id: int, local_arguments: tf.Tensor, global_arguments: tf.Tensor):
        return any(pred.evaluate(tactic_id, local_arguments, global_arguments) for pred in self.predictions)

    def evaluate(self, action: Tuple) -> bool:
        """
        Evaluate an action in tuple format.
        """
        (_, _, _, _, _, context_node_ids) = self.state
        local_context_length = tf.shape(context_node_ids, out_type=tf.int64)[0]

        tactic_id, arguments_array = action
        tactic_id = tf.cast(tactic_id, dtype=tf.int64)
        arguments_array = tf.cast(arguments_array, dtype=tf.int64)
        action = DataLoaderDataset._action_to_arguments((tactic_id, arguments_array), local_context_length)
        return self._evaluate(*action)


class Predict:
    """
    Class to load training checkpoints and make predictions in order to interact with an evaluator.
    This class exposes the following methods:
        - `initialize`: set the global context during evaluation
        - `compute_new_definitions`: update node label embeddings using definition cluster graphs
        - `ranked_predictions`: make predictions for a single proof-state
        - `batch_ranked_predictions`: make predictions for a batch of proof-states
        - `get_tactic_index_to_numargs`: access the value of `tactic_index_to_numargs` seen during training
        - `get_tactic_index_to_hash`: access the value of `tactic_index_to_hash` seen during training
        - `get_node_label_to_name`: access the value of `node_label_to_name` seen during training
        - `get_node_label_in_spine`: access the value of `node_label_in_spine` seen during training
    """
    def __init__(self, log_dir: Path, numpy_output: bool = True):
        """
        @param log_dir: the directory for the checkpoint that is to be loaded (as passed to the Trainer class)
        @param numpy_output: set to True to return the predictions as a tuple of numpy arrays (for evaluation purposes)
        """
        self.numpy_output = numpy_output

        # create dummy dataset for pre-processing purposes
        graph_constants_filepath = log_dir / 'config' / 'graph_constants.yaml'
        with graph_constants_filepath.open('r') as yml_file:
            self._graph_constants = GraphConstants(**yaml.load(yml_file, Loader=yaml.UnsafeLoader))

        dataset_yaml_filepath = log_dir / 'config' / 'dataset.yaml'
        with dataset_yaml_filepath.open('r') as yml_file:
            dataset = Dataset(**yaml.load(yml_file, Loader=yaml.SafeLoader))
        dataset._graph_constants = self._graph_constants

        self._preprocess = dataset._preprocess
        self._dummy_tactic_id = tf.argmin(dataset._graph_constants.tactic_index_to_numargs)  # num_arguments == 0
        self._tactic_logits_mask = tf.constant(dataset._graph_constants.tactic_index_to_numargs<NUMPY_NDIM_LIMIT)

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
        checkpoint_path = log_dir / 'ckpt'
        checkpoint = tf.train.Checkpoint(prediction_task=self.prediction_task.checkpoint)
        if self.definition_task is not None:
            checkpoint.definition_task = self.definition_task

        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=None)
        try:
            checkpoint_restored = checkpoint_manager.restore_or_initialize()
        except tf.errors.NotFoundError as error:
            print(f'unable to restore checkpoint from {checkpoint_path}!')
            raise error
        else:
            if checkpoint_restored is not None:
                print(f'restored checkpoint {checkpoint_restored}!')

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

    def get_tactic_index_to_numargs(self):
        """
        Public API
        """
        return self._graph_constants.tactic_index_to_numargs

    def get_tactic_index_to_hash(self):
        """
        Public API
        """
        return self._graph_constants.tactic_index_to_hash

    def get_node_label_to_name(self):
        """
        Public API
        """
        return self._graph_constants.label_to_names

    def get_node_label_in_spine(self) -> Iterable:
        """
        Public API
        """
        return self._graph_constants.label_in_spine

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

    def initialize(self, global_context: Optional[List[int]] = None):
        if global_context is not None:
            self._graph_constants.global_context = tf.constant(global_context, dtype=tf.int32)
            new_node_label_num = max(global_context)+1
            if new_node_label_num > self._graph_constants.node_label_num:
                self.prediction_task.graph_embedding = self._extend_graph_embedding(self.prediction_task.graph_embedding, new_node_label_num)

                self.prediction_task.global_arguments_logits = LogitsFromEmbeddings(
                    embedding_matrix=self.prediction_task.graph_embedding._node_embedding.embeddings,
                    valid_indices=tf.constant(self._graph_constants.global_context, dtype=tf.int32),
                    name=GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS
                )

    def compute_new_definitions(self, new_cluster_subgraphs: List[Tuple]) -> None:
        definition_graph = self._make_definition_batch(new_cluster_subgraphs)

        masked_definition_graph = Trainer._mask_defined_labels(definition_graph)
        scalar_definition_graph = masked_definition_graph.merge_batch_to_components()
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
                                                 output_signature=DataLoaderDataset.proofstate_data_spec)
        dataset = dataset.map(DataLoaderDataset._make_proofstate_graph_tensor)
        dataset = self._preprocess(dataset, shuffle=False)
        return dataset

    def _make_definition_batch(self, new_cluster_subgraphs: List[Tuple]) -> tfgnn.GraphTensor:
        """
        Create a dataset of definition graphs.

        @param new_cluster_subgraphs: list of definition clusters in tuple form (as returned by the data loader)
        @return: a `GraphTensor` containing the definition clusters, following the `definition_graph_spec` schema
        """
        dataset = tf.data.Dataset.from_generator(lambda: new_cluster_subgraphs,
                                                 output_signature=DataLoaderDataset.definition_data_spec)
        dataset = dataset.map(DataLoaderDataset._make_definition_graph_tensor)
        dataset = self._preprocess(dataset, shuffle=False)
        return dataset.batch(len(new_cluster_subgraphs)).get_single_element()

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
                       allowed_model_tactics: Optional[List[int]]
                       ) -> Tuple[NamedTuple, tfgnn.GraphTensor]:
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
        tactic_logits_mask = self._tactic_logits_mask
        if allowed_model_tactics is not None:
            tactic_num = tf.shape(tactic_logits)[-1]
            tactic_logits_mask &= tf.reduce_any(tf.cast(tf.one_hot(allowed_model_tactics, tactic_num), tf.bool), axis=0)
        tactic_expand_bound = min(tactic_expand_bound, tf.reduce_sum(tf.cast(tactic_logits_mask, tf.int32)))
        tactic_logits = tf.math.log_softmax(tactic_logits + tf.math.log(tf.cast(tactic_logits_mask, tf.float32)), axis=-1)

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
        top_k, _ = self._top_k_tactics(scalar_proofstate_graph=scalar_proofstate_graph,
                                       tactic_expand_bound=tactic_expand_bound,
                                       allowed_model_tactics=allowed_model_tactics)

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

        # get the top tactic_expand_bound tactics and input/output graphs
        top_k, hidden_graph = self._top_k_tactics(scalar_proofstate_graph=scalar_proofstate_graph,
                                                  tactic_expand_bound=tactic_expand_bound,
                                                  allowed_model_tactics=allowed_model_tactics)

        # get the local context node ids from the input graph
        context_node_ids = scalar_proofstate_graph.context['context_node_ids']

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
                for state_prediction, tactic_id, tactic_value, num_arguments, arguments_logits, local_context_ids in zip(batch_predictions, batch_tactic, batch_tactic_logits, batch_num_arguments, batch_arguments_logits, context_node_ids):
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

        # get the top tactic_expand_bound tactics and input/output graphs
        top_k, hidden_graph = self._top_k_tactics(scalar_proofstate_graph=scalar_proofstate_graph,
                                                  tactic_expand_bound=tactic_expand_bound,
                                                  allowed_model_tactics=allowed_model_tactics)

        # get the local context node ids from the input graph
        context_node_ids = scalar_proofstate_graph.context['context_node_ids']

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
                for state_prediction, tactic_id, tactic_value, num_arguments, local_arguments_logits, global_arguments_logits, local_context_ids in zip(batch_predictions, batch_tactic, batch_tactic_logits, batch_num_arguments, batch_local_arguments_logits, batch_global_arguments_logits, context_node_ids):
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
                                 states: List[Tuple],
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

    def ranked_predictions(self,
                           state: Tuple,
                           tactic_expand_bound: int,
                           total_expand_bound: int,
                           allowed_model_tactics: Optional[Iterable[int]] = None
                           ) -> Union[PredictOutput, Tuple[np.ndarray, np.ndarray]]:
        """
        Produces predictions for a single proof-state.
        """
        return self.batch_ranked_predictions(states=[state],
                                             allowed_model_tactics=allowed_model_tactics,
                                             tactic_expand_bound=tactic_expand_bound,
                                             total_expand_bound=total_expand_bound)[0]

    def _evaluate(self,
                  proofstate_graph_dataset: tf.data.Dataset,
                  batch_size: int,
                  tactic_expand_bound: int,
                  total_expand_bound: int,
                  allowed_model_tactics: Optional[Iterable[int]] = None
                  ) -> float:
        predictions = []
        tactic = []
        local_arguments = []
        global_arguments = []
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

        tactic = tf.concat(tactic, axis=0)
        local_arguments = tf.concat(local_arguments, axis=0)
        global_arguments = tf.concat(global_arguments, axis=0)

        results = []
        for tactic_id, local_arguments, global_arguments, predict_output in zip(tactic, local_arguments, global_arguments, predictions):
            results.append(predict_output._evaluate(tactic_id, local_arguments, global_arguments))
        return np.array(results).mean()

    def evaluate(self,
                 state_action_pairs: Iterable[Tuple[Tuple, Tuple]],
                 batch_size: int,
                 tactic_expand_bound: int,
                 total_expand_bound: int,
                 allowed_model_tactics: Optional[Iterable[int]] = None
                 ) -> float:
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

        results = []
        for (state, action), predict_output in zip(state_action_pairs, predictions):
            predict_output.state = state
            results.append(predict_output.evaluate(action))
        return np.array(results).mean()
