from typing import Tuple, List, Union, Iterable, Callable, Optional

import re
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from dataclasses import dataclass
from pathlib import Path

from graph2tac.loader.data_classes import DataConfig, GraphConstants, LoaderAction, LoaderActionSpec, LoaderProofstate, LoaderProofstateSpec, LoaderDefinition, LoaderDefinitionSpec
from graph2tac.loader.data_server import DataToTFGNN
from graph2tac.tfgnn.tasks import PredictionTask, TacticPrediction, DefinitionTask, GLOBAL_ARGUMENT_PREDICTION
from graph2tac.tfgnn.models import GraphEmbedding, LogitsFromEmbeddings
from graph2tac.tfgnn.train import Trainer
from graph2tac.common import logger
from graph2tac.predict import Predict, predict_api_debugging, cartesian_product, NUMPY_NDIM_LIMIT
from graph2tac.tfgnn.graph_schema import vectorized_definition_graph_spec, proofstate_graph_spec, batch_graph_spec
from graph2tac.tfgnn.stack_graph_tensors import stack_graph_tensors

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
        top_row = np.insert(np.where(self.global_arguments == -1, 0, 1), 0, self.tactic_id)
        bottom_row = np.insert(np.where(self.global_arguments == -1, self.local_arguments, self.global_arguments), 0, self.tactic_id)
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

    def evaluate(self, action: LoaderAction, search_expand_bound: Optional[int] = None) -> bool:
        """
        Evaluate an action in the loader format
        """
        local_context_ids = self.state.context.local_context
        local_context_length = tf.shape(local_context_ids, out_type=tf.int64)[0]

        tactic_id = tf.cast(action.tactic_id, dtype=tf.int64)
        arguments_array = tf.cast(action.args, dtype=tf.int64)
        local_args = tf.cast(action.local_args, dtype=tf.int64)
        global_args = tf.cast(action.global_args, dtype=tf.int64)

        return self._evaluate(tactic_id, local_args, global_args, search_expand_bound=search_expand_bound)


class TFGNNPredict(Predict):
    def __init__(self,
                 log_dir: Path,
                 tactic_expand_bound: int,
                 debug_dir: Optional[Path] = None,
                 checkpoint_number: Optional[int] = None,
                 exclude_tactics: Optional[List[str]] = None,
                 allocation_reserve: float = 0.5,
                 numpy_output: bool = True,
                 ):
        """
        @param log_dir: the directory for the checkpoint that is to be loaded (as passed to the Trainer class)
        @param debug_dir: set to a directory to dump pickle files for every API call that is made
        @param checkpoint_number: the checkpoint number we want to load (use `None` for the latest checkpoint)
        @param exclude_tactics: a list of tactic names to exclude from all predictions
        @param allocation_reserve: proportional size of extra allocated space when resizing the nodes embedding array
        @param numpy_output: set to True to return the predictions as a tuple of numpy arrays (for evaluation purposes)
        """

        self._exporter = DataToTFGNN()
        self._allocation_reserve = allocation_reserve

        # create dummy dataset for pre-processing purposes
        graph_constants_filepath = log_dir / 'config' / 'graph_constants.yaml'
        with graph_constants_filepath.open('r') as yml_file:
            graph_constants_d = yaml.load(yml_file, Loader=yaml.UnsafeLoader)
            graph_constants_d['data_config'] = DataConfig(**graph_constants_d['data_config'])
            graph_constants = GraphConstants(**graph_constants_d)

        # call to parent constructor to defines self.graph_constants
        super().__init__(
            graph_constants=graph_constants,
            tactic_expand_bound=tactic_expand_bound,
            debug_dir=debug_dir,
        )

        # to build dummy proofstates we will need to use a tactic taking no arguments
        self._dummy_tactic_id = tf.argmin(graph_constants.tactic_index_to_numargs)  # num_arguments == 0

        # the decoding mechanism currently does not support tactics with more than NUMPY_NDIM_LIMIT
        self.fixed_tactic_mask = tf.constant(np.array(graph_constants.tactic_index_to_numargs) < NUMPY_NDIM_LIMIT)

        # mask tactics explicitly excluded from predictions
        if exclude_tactics is not None:
            exclude_tactics = set(exclude_tactics)
            self.fixed_tactic_mask &= tf.constant([(tactic_name not in exclude_tactics) for tactic_name in graph_constants.tactic_index_to_string])

        # create prediction task
        prediction_yaml_filepath = log_dir / 'config' / 'prediction.yaml'
        self.prediction_task = PredictionTask.from_yaml_config(graph_constants=graph_constants,
                                                               yaml_filepath=prediction_yaml_filepath)
        self.prediction_task_type = self.prediction_task.get_config()['prediction_task_type']

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
            checkpoint.definition_task = self.definition_task.get_checkpoint()

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

        node_label_num = self.graph_constants.node_label_num
        extra_label_num = round(self._allocation_reserve*node_label_num)
        if extra_label_num > node_label_num: self._allocate_definitions(node_label_num + extra_label_num)
        self._compile_network()

    def _allocate_definitions(self, new_node_label_num) -> None: # explicit change of the network array

        logger.info(f'extending global context from {self.graph_constants.node_label_num} to {new_node_label_num} elements')
        new_graph_emb_layer = self.prediction_task.graph_embedding.extend_embeddings(new_node_label_num)
        self.prediction_task.graph_embedding = new_graph_emb_layer

        if self.definition_task is not None:
            self.definition_task._graph_embedding = new_graph_emb_layer
        self.graph_constants.node_label_num = new_node_label_num
        self.prediction_task.global_arguments_logits.update_embedding_matrix(
            embedding_matrix=self.prediction_task.graph_embedding.get_node_embeddings()
        )

    @predict_api_debugging
    def allocate_definitions(self, new_node_label_num : int) -> None:
        if self.prediction_task_type != GLOBAL_ARGUMENT_PREDICTION:
            # no need to update anything if we are not going to use the global context
            return

        if new_node_label_num <= self.graph_constants.node_label_num:
            # already have sufficient array
            return

        new_node_label_num += round(self._allocation_reserve*new_node_label_num)

        self._allocate_definitions(new_node_label_num)
        self._compile_network()

    @predict_api_debugging
    def compute_new_definitions(self, new_cluster_subgraphs: List[LoaderDefinition]) -> None:
        if self.definition_task is None:
            raise RuntimeError('cannot update definitions when a definition task is not present')

        assert len(new_cluster_subgraphs) == 1
        self._compute_and_replace_definition_embs(new_cluster_subgraphs[0])

    @tf.function(input_signature = (LoaderProofstateSpec,))
    def _make_proofstate_graph_tensor(self, state : LoaderProofstate):
        action = LoaderAction(
            self._dummy_tactic_id,
            tf.zeros(shape=(0), dtype=tf.int64),
            tf.zeros(shape=(0), dtype=tf.int64),
        )
        graph_id = tf.constant(-1, dtype=tf.int64)
        x = DataToTFGNN.proofstate_to_graph_tensor(state, action, graph_id)
        return x

    def _compile_network(self):
        @tf.function(input_signature = (LoaderDefinitionSpec,))
        def compute_and_replace_definition_embs(loader_definition):
            graph_tensor = self._exporter.definition_to_graph_tensor(loader_definition)
            definition_graph = stack_graph_tensors([graph_tensor])

            scalar_definition_graph = definition_graph.merge_batch_to_components()
            definition_embeddings = self.definition_task(scalar_definition_graph).flat_values
            defined_labels = Trainer._get_defined_labels(definition_graph).flat_values

            self.prediction_task.graph_embedding.update_node_embeddings(
                embeddings=definition_embeddings,
                indices=defined_labels
            )
        self._compute_and_replace_definition_embs = compute_and_replace_definition_embs

        inference_model_bare = self.prediction_task.create_inference_model(
            tactic_expand_bound=self._tactic_expand_bound,
            graph_constants=self.graph_constants
        )
        allowed_model_tactics_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32)
        @tf.function(input_signature = (LoaderProofstateSpec, allowed_model_tactics_spec))
        def inference_model(state, allowed_model_tactics):
            tactic_mask = tf.scatter_nd(
                indices = tf.expand_dims(allowed_model_tactics, axis = 1),
                updates = tf.ones_like(allowed_model_tactics, dtype=bool),
                shape = [self.graph_constants.tactic_num]
            )
            graph_tensor_single = self._make_proofstate_graph_tensor(state)
            graph_tensor_stacked = stack_graph_tensors([graph_tensor_single])
            inference_output = inference_model_bare({
                self.prediction_task.PROOFSTATE_GRAPH: graph_tensor_stacked,
                self.prediction_task.TACTIC_MASK: tf.expand_dims(tactic_mask, axis=0),
            })
            return inference_output
        self._inference_model = inference_model

    # Currently not used
    def _make_proofstate_batch(self, datapoints : Iterable[LoaderProofstate]):
        return stack_graph_tensors([
            self._make_proofstate_graph_tensor(x)
            for x in datapoints
        ])

    @staticmethod
    def _logits_decoder(logits: tf.Tensor, total_expand_bound: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decoding mechanism
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

    @classmethod
    def _expand_arguments_logits(cls,
                                 total_expand_bound: int,
                                 num_arguments: int,
                                 local_context_size: int,
                                 global_context_size: int,
                                 tactic: tf.Tensor,
                                 tactic_logits: tf.Tensor,
                                 local_arguments_logits: Optional[tf.Tensor] = None,
                                 global_arguments_logits: Optional[tf.Tensor] = None
                                 ) -> List[Inference]:
        if local_arguments_logits is None:
            # this is a base tactic prediction
            return [TacticInference(value=float(tactic_logits.numpy()), tactic_id=int(tactic.numpy()))]
        elif global_arguments_logits is None:
            # this is a base tactic plus local arguments prediction
            logits = local_arguments_logits[:num_arguments, :local_context_size]
            arg_combinations, combination_values = cls._logits_decoder(logits=logits,
                                                                       total_expand_bound=total_expand_bound)
            combination_values += tactic_logits.numpy()
            return [LocalArgumentInference(value=float(value),
                                           tactic_id=int(tactic.numpy()),
                                           local_arguments=local_arguments)
                    for local_arguments, value in zip(tf.cast(arg_combinations, dtype=tf.int64), combination_values)]
        else:
            # this is a base tactic plus local and global arguments prediction
            combined_arguments_logits = tf.concat([global_arguments_logits[:num_arguments, :], local_arguments_logits[:num_arguments, :local_context_size]], axis=-1)
            arg_combinations, combination_values = cls._logits_decoder(logits=combined_arguments_logits,
                                                                       total_expand_bound=total_expand_bound)
            combination_values += tactic_logits.numpy()
            return [GlobalArgumentInference(value=float(value),
                                            tactic_id=int(tactic.numpy()),
                                            local_arguments=tf.where(arguments < global_context_size, -1, arguments - global_context_size),
                                            global_arguments=tf.where(arguments < global_context_size, arguments, -1))
                    for arguments, value in zip(tf.cast(arg_combinations, dtype=tf.int64), combination_values)]

    @predict_api_debugging
    def ranked_predictions(self,
                           state: LoaderProofstate,
                           total_expand_bound: int,
                           available_global: Optional[np.ndarray] = None,
                           allowed_model_tactics: Optional[Iterable[int]] = None
                           ) -> Union[PredictOutput, Tuple[np.ndarray, np.ndarray]]:
        """
        Produces predictions for a single proof-state.
        """
        if available_global is not None:
            raise NotImplementedError('available_global is not supported yet')

        inference_output = self._inference_model(state, allowed_model_tactics)
        predict_output = PredictOutput(state=None, predictions=[])

        # go over the tactic_expand_bound batches
        for proofstate_batch_output in zip(*inference_output.values()):
            # go over the individual proofstates in a batch
            inference_data = {
                output_name: output_value[0]
                for output_name, output_value in zip(inference_output.keys(), proofstate_batch_output)
            }

            num_arguments = self.graph_constants.tactic_index_to_numargs[inference_data[TacticPrediction.TACTIC]]
            predictions = self._expand_arguments_logits(total_expand_bound=total_expand_bound,
                                                        num_arguments=num_arguments,
                                                        local_context_size=len(state.context.local_context),
                                                        global_context_size=len(state.context.global_context),
                                                        **inference_data)
            predict_output.predictions.extend(filter(lambda inference: inference.value > -float('inf'), predictions))

        # fill in the states in loader format
        predict_output.state = state

        # return predictions in the appropriate format
        return predict_output.numpy()

    # (!) NOT MAINTAINED
    def _batch_ranked_predictions(self,
                                  proofstate_graph: tfgnn.GraphTensor,
                                  tactic_expand_bound: int,
                                  total_expand_bound: int,
                                  tactic_mask: tf.Tensor
                                  ) -> List[PredictOutput]:

        raise Exception("Running unmaintained code. Delete this line at your own risk.")
        
        inference_model = self._inference_model(tactic_expand_bound)

        inference_output = inference_model({self.prediction_task.PROOFSTATE_GRAPH: proofstate_graph,
                                            self.prediction_task.TACTIC_MASK: tactic_mask})

        _, local_context_sizes = proofstate_graph.context['local_context_ids'].nested_row_lengths()

        batch_size = int(proofstate_graph.total_num_components.numpy())

        predict_outputs = [PredictOutput(state=None, predictions=[]) for _ in range(batch_size)]

        # go over the tactic_expand_bound batches
        for proofstate_batch_output in zip(*inference_output.values()):
            # go over the individual proofstates in a batch
            for predict_output, proofstate_output, local_context_size in zip(predict_outputs,
                                                                             zip(*proofstate_batch_output),
                                                                             local_context_sizes):
                inference_data = {output_name: output_value for output_name, output_value in zip(inference_output.keys(), proofstate_output)}
                num_arguments = self.graph_constants.tactic_index_to_numargs[inference_data[TacticPrediction.TACTIC]]
                predictions = self._expand_arguments_logits(total_expand_bound=total_expand_bound,
                                                            num_arguments=num_arguments,
                                                            local_context_size=local_context_size,
                                                            global_context_size=global_context_size,
                                                            **inference_data)
                predict_output.predictions.extend(filter(lambda inference: inference.value > -float('inf'), predictions))
        return predict_outputs

    
    # (!) NOT MAINTAINED
    def _evaluate(self,
                  proofstate_graph_dataset: tf.data.Dataset,
                  batch_size: int,
                  tactic_expand_bound: int,
                  total_expand_bound: int,
                  search_expand_bound: Optional[int] = None,
                  allowed_model_tactics: Optional[Iterable[int]] = None
                  ) -> Tuple[float, float]:

        raise Exception("Running unmaintained code. Delete this line at your own risk.")

        tactic_mask = self._tactic_mask_from_allowed_model_tactics(allowed_model_tactics)

        predictions = []
        tactic = []
        local_arguments = []
        global_arguments = []
        names = []
        for proofstate_graph in iter(proofstate_graph_dataset.batch(batch_size)):
            scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()
            batch_tactic_mask = tf.repeat(tf.expand_dims(tactic_mask, axis=0), repeats=proofstate_graph.total_num_components, axis=0)
            batch_predict_output = self._batch_ranked_predictions(proofstate_graph=proofstate_graph,
                                                                  tactic_expand_bound=tactic_expand_bound,
                                                                  total_expand_bound=total_expand_bound,
                                                                  tactic_mask=batch_tactic_mask)
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
