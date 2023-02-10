from typing import Tuple, List, Union, Iterable, Callable, Optional

import re
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from dataclasses import dataclass
from pathlib import Path

from graph2tac.loader.data_classes import GraphConstants, LoaderAction, LoaderProofstate, LoaderDefinition
from graph2tac.tfgnn.dataset import Dataset, DataServerDataset
from graph2tac.tfgnn.tasks import PredictionTask, TacticPrediction, DefinitionTask, GLOBAL_ARGUMENT_PREDICTION
from graph2tac.tfgnn.models import GraphEmbedding, LogitsFromEmbeddings
from graph2tac.tfgnn.train import Trainer
from graph2tac.common import logger
from graph2tac.predict import Predict, predict_api_debugging, cartesian_product, NUMPY_NDIM_LIMIT


def stack_dicts_with(f, ds):
    keys = ds[0].keys()
    assert all(d.keys() == keys for d in ds)
    return {
        key : f([d[key] for d in ds])
        for key in keys
    }

def stack_maybe_ragged(xs):
    if isinstance(xs[0], tf.RaggedTensor):
        return tf.ragged.stack(xs)
    else:
        return tf.stack(xs)

def stack_contexts(cs):
    return tfgnn.Context.from_fields(
        sizes = tf.stack([c.sizes for c in cs]),
        features = stack_dicts_with(stack_maybe_ragged, [c.features for c in cs]),
    )

def stack_node_sets(nss):
    sizes = tf.stack([ns.sizes for ns in nss])
    features = stack_dicts_with(tf.ragged.stack, [ns.features for ns in nss])
    return tfgnn.NodeSet.from_fields(
        sizes = sizes,
        features = features,
    )

def stack_edge_sets(ess):
    sizes = tf.stack([es.sizes for es in ess])
    features = stack_dicts_with(tf.ragged.stack, [es.features for es in ess])
    source_name = ess[0].adjacency.source_name
    target_name = ess[0].adjacency.target_name
    assert all(es.adjacency.source_name == source_name for es in ess)
    assert all(es.adjacency.target_name == target_name for es in ess)
    source = tf.ragged.stack([es.adjacency.source for es in ess])
    target = tf.ragged.stack([es.adjacency.target for es in ess])
    return tfgnn.EdgeSet.from_fields(
        sizes = sizes,
        features = features,
        adjacency = tfgnn.Adjacency.from_indices(
            source = (source_name, source),
            target = (target_name, target),
        ),
    )

def stack_graph_tensors(gts):
    context = stack_contexts([gt.context for gt in gts])
    node_sets = stack_dicts_with(stack_node_sets, [gt.node_sets for gt in gts])
    edge_sets = stack_dicts_with(stack_edge_sets, [gt.edge_sets for gt in gts])
    return tfgnn.GraphTensor.from_pieces(context = context, node_sets = node_sets, edge_sets = edge_sets)


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
        local_args, global_args = DataServerDataset._split_action_arguments(arguments_array, local_context_length)
        return self._evaluate(tactic_id, local_args, global_args, search_expand_bound=search_expand_bound)


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

        # initialize inference model cache (most of the time we always use one model for a fixed tactic_expand_bound)
        self._inference_model_cache = {}
        self.cached_definition_computation = None

        # create dummy dataset for pre-processing purposes
        graph_constants_filepath = log_dir / 'config' / 'graph_constants.yaml'
        with graph_constants_filepath.open('r') as yml_file:
            graph_constants = GraphConstants(**yaml.load(yml_file, Loader=yaml.UnsafeLoader))

        dataset_yaml_filepath = log_dir / 'config' / 'dataset.yaml'
        with dataset_yaml_filepath.open('r') as yml_file:
            dataset = Dataset(graph_constants=graph_constants, **yaml.load(yml_file, Loader=yaml.SafeLoader))
        self._dataset = dataset

        # call to parent constructor to defines self._graph_constants
        super().__init__(graph_constants=graph_constants, debug_dir=debug_dir)

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
        self.prediction_task = PredictionTask.from_yaml_config(graph_constants=dataset.graph_constants(),
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

    @predict_api_debugging
    def initialize(self, global_context: Optional[List[int]] = None) -> None:
        if self.prediction_task_type != GLOBAL_ARGUMENT_PREDICTION:
            # no need to update anything if we are not going to use the global context
            return

        if global_context is not None:
            # update the global context
            self._graph_constants.global_context = global_context

            # extend the embedding table if necessary
            new_node_label_num = max(global_context)+1
            if new_node_label_num > self._graph_constants.node_label_num:
                logger.info(f'extending global context from {self._graph_constants.node_label_num} to {new_node_label_num} elements')
                new_graph_emb_layer = self.prediction_task.graph_embedding.extend_embeddings(new_node_label_num)
                self.prediction_task.graph_embedding = new_graph_emb_layer
                
                if self.definition_task is not None:
                    self.definition_task._graph_embedding = new_graph_emb_layer
                self._graph_constants.node_label_num = new_node_label_num

            # update the global arguments logits head (always necessary, because the global context may shrink!)
            self.prediction_task.global_arguments_logits.update_embedding_matrix(
                embedding_matrix=self.prediction_task.graph_embedding.get_node_embeddings(),
                valid_indices=tf.constant(self._graph_constants.global_context, dtype=tf.int32)
            )

            # clear the inference model cache to force re-creation of the inference models using the new layers
            self._inference_model_cache = {}
            self.cached_definition_computation = None

    def _fetch_definition_computation(self):
        """
        The definition computation needs to be rebuilt when the embeddings table is swapped out.

        When the embeddings table is changed, this cached function is deleted and
        we rebuild it here when we need it.
        """

        if self.cached_definition_computation is not None:
            return self.cached_definition_computation
        
        @tf.function(input_signature = DataServerDataset.definition_data_spec)
        def _compute_and_replace_definition_embs(loader_graph, num_definitions, definition_names):
            definition_graph = stack_graph_tensors([
                self._make_definition_graph_tensor_from_data(loader_graph, num_definitions, definition_names)
            ])

            scalar_definition_graph = definition_graph.merge_batch_to_components()
            definition_embeddings = self.definition_task(scalar_definition_graph).flat_values
            defined_labels = Trainer._get_defined_labels(definition_graph).flat_values

            self.prediction_task.graph_embedding.update_node_embeddings(
                embeddings=definition_embeddings,
                indices=defined_labels
            )
        
        self.cached_definition_computation = _compute_and_replace_definition_embs
        return _compute_and_replace_definition_embs
    
    @predict_api_debugging
    def compute_new_definitions(self, new_cluster_subgraphs: List[LoaderDefinition]) -> None:
        if self.definition_task is None:
            raise RuntimeError('cannot update definitions when a definition task is not present')

        assert len(new_cluster_subgraphs) == 1
        new_cluster_subgraph = DataServerDataset._loader_to_definition_data(new_cluster_subgraphs[0])

        compute_and_replace_definition_embs = self._fetch_definition_computation()
        compute_and_replace_definition_embs(*new_cluster_subgraph)
        

    @tf.function(input_signature = DataServerDataset.proofstate_data_spec)
    def _make_proofstate_graph_tensor_from_data(self, state, action, graph_id):
        x = DataServerDataset._make_proofstate_graph_tensor(state, action, graph_id)
        x = self._dataset._preprocess_single(x)
        return x

    def _make_proofstate_graph_tensor(self, state : LoaderProofstate):
        action = LoaderAction(self._dummy_tactic_id, tf.zeros(shape=(0, 2), dtype=tf.int64))
        graph_id = tf.constant(-1, dtype=tf.int64)
        x = DataServerDataset._loader_to_proofstate_data((state, action, graph_id))
        x = self._make_proofstate_graph_tensor_from_data(*x)
        return x

    def _make_proofstate_batch(self, datapoints : Iterable[LoaderProofstate]):
        return stack_graph_tensors([
            self._make_proofstate_graph_tensor(x)
            for x in datapoints
        ])

    def _dummy_proofstate_data_generator(self, states: List[LoaderProofstate]):
        for state in states:
            action = LoaderAction(self._dummy_tactic_id, tf.zeros(shape=(0, 2), dtype=tf.int64))
            graph_id = tf.constant(-1, dtype=tf.int64)
            yield DataServerDataset._loader_to_proofstate_data((state, action, graph_id))

    def _make_dummy_proofstate_dataset(self, states: List[LoaderProofstate]) -> tf.data.Dataset:
        """
        Create a dataset of (dummy) proofstate graphs.

        @param states: list of proof-states in tuple form (as returned by the data loader)
        @return: a `tf.data.Dataset` producing `GraphTensor` objects following the `proofstate_graph_spec` schema
        """
        dataset = tf.data.Dataset.from_generator(lambda: self._dummy_proofstate_data_generator(states),
                                                 output_signature=DataServerDataset.proofstate_data_spec)
        dataset = dataset.map(DataServerDataset._make_proofstate_graph_tensor)
        dataset = dataset.apply(self._dataset._preprocess)
        return dataset

    @tf.function(input_signature = DataServerDataset.definition_data_spec)
    def _make_definition_graph_tensor_from_data(self, loader_graph, num_definitions, definition_names):
        x = DataServerDataset._make_definition_graph_tensor(loader_graph, num_definitions, definition_names)
        x = self._dataset._preprocess_single(x)
        x = self._dataset.tokenize_definition_graph(x)
        return x

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

    def _inference_model(self, tactic_expand_bound: int) -> tf.keras.Model:
        if tactic_expand_bound not in self._inference_model_cache.keys():
            inference_model = self.prediction_task.create_inference_model(tactic_expand_bound=tactic_expand_bound,
                                                                          graph_constants=self._graph_constants)
            self._inference_model_cache[tactic_expand_bound] = inference_model
        return self._inference_model_cache[tactic_expand_bound]

    def _batch_ranked_predictions(self,
                                  proofstate_graph: tfgnn.GraphTensor,
                                  tactic_expand_bound: int,
                                  total_expand_bound: int,
                                  tactic_mask: tf.Tensor
                                  ) -> List[PredictOutput]:

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
                num_arguments = self._graph_constants.tactic_index_to_numargs[inference_data[TacticPrediction.TACTIC]]
                predictions = self._expand_arguments_logits(total_expand_bound=total_expand_bound,
                                                            num_arguments=num_arguments,
                                                            local_context_size=local_context_size,
                                                            global_context_size=len(self._graph_constants.global_context),
                                                            **inference_data)
                predict_output.predictions.extend(filter(lambda inference: inference.value > -float('inf'), predictions))
        return predict_outputs

    def _tactic_mask_from_allowed_model_tactics(self, allowed_model_tactics: Optional[Iterable[int]]) -> tf.Tensor:
        tactic_num = self._graph_constants.tactic_num
        if allowed_model_tactics is not None:
            tactic_mask = tf.reduce_any(tf.cast(tf.one_hot(allowed_model_tactics, tactic_num), tf.bool), axis=0)
        else:
            tactic_mask = tf.ones(shape=(tactic_num,), dtype=tf.bool)
        return tactic_mask & self.fixed_tactic_mask

    def batch_ranked_predictions(self,
                                 states: List[LoaderProofstate],
                                 tactic_expand_bound: int,
                                 total_expand_bound: int,
                                 allowed_model_tactics: Optional[Iterable[int]] = None
                                 ) -> List[Union[PredictOutput, Tuple[List[np.ndarray], np.ndarray]]]:
        # convert the input to a batch of graph tensors (rank 1)
        proofstate_graph = self._make_proofstate_batch(states)

        # create the tactic mask input
        tactic_mask = self._tactic_mask_from_allowed_model_tactics(allowed_model_tactics)
        tactic_mask = tf.repeat(tf.expand_dims(tactic_mask, axis=0), repeats=len(states), axis=0)

        # make predictions
        batch_predict_output = self._batch_ranked_predictions(proofstate_graph=proofstate_graph,
                                                              tactic_expand_bound=tactic_expand_bound,
                                                              total_expand_bound=total_expand_bound,
                                                              tactic_mask=tactic_mask)

        # fill in the states in loader format
        for state, predict_output in zip(states, batch_predict_output):
            predict_output.state = state

        # return predictions in the appropriate format
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

    def evaluate(self,
                 state_action_pairs: Iterable[Tuple[LoaderProofstate, LoaderAction]],
                 batch_size: int,
                 tactic_expand_bound: int,
                 total_expand_bound: int,
                 search_expand_bound: Optional[int] = None,
                 allowed_model_tactics: Optional[Iterable[int]] = None
                 ) -> Tuple[float, float]:
        states = [state for state, _ in state_action_pairs]
        proofstate_graph_dataset = self._make_dummy_proofstate_dataset(states).batch(batch_size)

        tactic_mask = self._tactic_mask_from_allowed_model_tactics(allowed_model_tactics)

        predictions: list[PredictOutput] = []
        for proofstate_graph in iter(proofstate_graph_dataset):
            batch_tactic_mask = tf.repeat(tf.expand_dims(tactic_mask, axis=0), repeats=proofstate_graph.total_num_components, axis=0)
            batch_predict_output = self._batch_ranked_predictions(proofstate_graph=proofstate_graph,
                                                                  tactic_expand_bound=tactic_expand_bound,
                                                                  total_expand_bound=total_expand_bound,
                                                                  tactic_mask=batch_tactic_mask)
            predictions.extend(batch_predict_output)

        per_proofstate = []
        per_lemma = {}
        for (state, action), predict_output in zip(state_action_pairs, predictions):
            predict_output.state = state
            result = predict_output.evaluate(action, search_expand_bound=search_expand_bound)
            per_proofstate.append(result)

            name = state.metadata.name
            per_lemma[name] = (per_lemma.get(name, True) and result)
        per_proofstate_result = np.array(per_proofstate).mean()
        per_lemma_result = np.array(list(per_lemma.values())).mean()
        return per_proofstate_result, per_lemma_result
