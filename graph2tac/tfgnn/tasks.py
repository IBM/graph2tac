from typing import List, Dict, Iterable, Tuple, Any, Callable, Optional

import yaml
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path

from graph2tac.loader.data_server import GraphConstants
from graph2tac.tfgnn.graph_schema import proofstate_graph_spec, batch_graph_spec, strip_graph
from graph2tac.tfgnn.models import GraphEmbedding, LogitsFromEmbeddings, get_gnn_constructor, get_arguments_head_constructor, get_tactic_head_constructor, get_definition_head_constructor


BASE_TACTIC_PREDICTION = 'base_tactic_prediction'
LOCAL_ARGUMENT_PREDICTION = 'local_argument_prediction'
GLOBAL_ARGUMENT_PREDICTION = 'global_argument_prediction'


def arguments_filter(y_true: tf.RaggedTensor, y_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Extracts the local arguments which are not None from the ground truth and predictions

    @param y_true: the labels for the arguments, with shape [batch_size, 1, None(num_arguments)]
    @param y_pred: the logits for the arguments, with shape [batch_size, max(num_arguments), context_size]
    @return: a tuple whose first element contains the not-None arguments, the second element being the logits corresponding to each not-None argument
    """
    # convert y_true to a dense tensor padding with -1 values (also used for None arguments);
    # remove spurious dimension (y_true was created from a non-scalar graph)
    # [ batch_size, max(num_arguments) ]
    arguments_tensor = tf.squeeze(y_true.to_tensor(default_value=-1), axis=1)

    # we want to compute only go over the positions that are not None
    positions = tf.where(arguments_tensor != -1)

    # keep only these positions in the both y_true and y_pred
    arguments_true = tf.gather_nd(arguments_tensor, positions)
    arguments_pred = tf.gather_nd(y_pred, positions)
    return arguments_true, arguments_pred


def _local_arguments_logits(scalar_proofstate_graph: tfgnn.GraphTensor,
                            hidden_graph: tfgnn.GraphTensor,
                            hidden_state_sequences: tf.RaggedTensor
                            ) -> tf.Tensor:
    """
    Computes logits for local arguments from the hidden states and the local context node ids.
    """
    # the sizes of the components of this graph
    # [ batch_size, ]
    sizes = tf.cast(scalar_proofstate_graph.node_sets['node'].sizes, dtype=tf.int64)

    # the offsets of the node ids for each graph component
    # [ batch_size, 1 ]
    cumulative_sizes = tf.expand_dims(tf.cumsum(sizes, exclusive=True), axis=-1)

    # the node ids for the local context nodes, shifted per components
    # [ batch_size, None(num_context_nodes) ]
    local_context_ids = cumulative_sizes + scalar_proofstate_graph.context['local_context_ids']

    # the hidden states for the nodes in the local context
    # [ batch_size, None(num_context_nodes), hidden_size ]
    context_node_hidden_states = tf.gather(hidden_graph.node_sets['node']['hidden_state'],
                                           local_context_ids).with_row_splits_dtype(tf.int64)

    # the logits for each local context node to be each local argument
    # [ batch_size, max(num_arguments), max(num_context_nodes) ]
    arguments_logits = tf.matmul(hidden_state_sequences.to_tensor(),
                                 context_node_hidden_states.to_tensor(),
                                 transpose_b=True)

    # a mask for the local context nodes that actually exist
    # [ batch_size,  max(num_context_nodes) ]
    context_mask = tf.cast(tf.zeros_like(scalar_proofstate_graph.context['local_context_ids']),
                           dtype=tf.float32).to_tensor(-float('inf'))

    # the masked logits
    # [ batch_size, max(num_arguments), max(num_context_nodes) ]
    return arguments_logits + tf.expand_dims(context_mask, axis=1)


@tf.function
def _local_arguments_pred(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.argmax(y_pred, axis=-1) if tf.shape(y_pred)[-1] > 0 else tf.zeros_like(y_true)


class LocalArgumentSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Used to compute the sparse categorical crossentropy loss for the local argument prediction task.

    NOTES:
        - `-1` arguments correspond to `None`
        - logits **are not** assumed to be normalized
    """
    def call(self, y_true, y_pred):
        """
        @param y_true: ids for the arguments, with shape [ batch_size, 1, None(num_arguments) ]
        @param y_pred: logits for each argument position, with shape [ batch_size, None(num_arguments), num_categories ]
        @return: a vector of length equal to the total number of not-None arguments within this batch
        """
        arguments_true, arguments_pred = arguments_filter(y_true, y_pred)

        if tf.size(arguments_pred) == 0:
            # deal with the edge case where there is no local context or no local arguments to predict in the batch
            return tf.zeros_like(arguments_true, dtype=tf.float32)
        else:
            # the local context is non-empty and we have at least one argument, so the following doesn't fail
            return tf.nn.sparse_softmax_cross_entropy_with_logits(arguments_true, arguments_pred)


class GlobalArgumentSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Used to compute the sparse categorical crossentropy loss for the global argument prediction task.

    NOTES:
        - `-1` arguments correspond to `None` or a different type of argument
        - logits **are** assumed to be normalized
    """

    def call(self, y_true, y_pred):
        """
        @param y_true: ids for the arguments, with shape [ batch_size, 1, None(num_arguments) ]
        @param y_pred: logits for each argument position, with shape [ batch_size, None(num_arguments), num_categories ]
        @return: a vector of length equal to the total number of not-None arguments within this batch
        """
        arguments_true, arguments_pred = arguments_filter(y_true, y_pred)

        if tf.size(arguments_pred) == 0:
            # deal with the edge case where there is no global context or no global arguments to predict in the batch
            return tf.zeros_like(arguments_true, dtype=tf.float32)
        else:
            # the context is non-empty and we have at least one argument, so the following doesn't fail
            return -tf.gather(arguments_pred, arguments_true, batch_dims=1)


class ArgumentSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    """
    Per-argument sparse categorical accuracy, excluding None arguments.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        arguments_true, arguments_pred = arguments_filter(y_true, y_pred)
        if tf.shape(arguments_pred)[-1] > 0:
            super().update_state(arguments_true, arguments_pred, sample_weight)


class DefinitionMeanSquaredError(tf.keras.losses.MeanSquaredError):
    """
    Mean-squared-error loss for the definition task, summing over the multiple definitions in a given definition graph.
    """
    def call(self, y_true, y_pred):
        return tf.reduce_sum(super().call(y_true, y_pred), axis=-1)


class MixedMetricsCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        for metric in self.model.mixed_metrics:
            metric.reset_state()

    def on_test_begin(self, logs=None):
        for metric in self.model.mixed_metrics:
            metric.reset_state()

    def on_predict_begin(self, logs=None):
        for metric in self.model.mixed_metrics:
            metric.reset_state()

    def on_epoch_begin(self, batch, logs=None):
        for metric in self.model.mixed_metrics:
            metric.reset_state()


class LocalArgumentModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arguments_seq_accuracy = tf.keras.metrics.Mean(name=LocalArgumentPrediction.ARGUMENTS_SEQ_ACCURACY)
        self.strict_accuracy = tf.keras.metrics.Mean(name=LocalArgumentPrediction.STRICT_ACCURACY)
        self.mixed_metrics = [self.arguments_seq_accuracy, self.strict_accuracy]

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metric_results = super().compute_metrics(x, y, y_pred, sample_weight)

        tactic = y[LocalArgumentPrediction.TACTIC_LOGITS]
        tactic_logits = y_pred[LocalArgumentPrediction.TACTIC_LOGITS]
        tactic_accuracy = tf.keras.metrics.sparse_categorical_accuracy(tactic, tactic_logits)

        arguments = y[LocalArgumentPrediction.ARGUMENTS_LOGITS]
        arguments_logits = y_pred[LocalArgumentPrediction.ARGUMENTS_LOGITS]
        arguments_true = tf.squeeze(arguments.to_tensor(default_value=0), axis=1)
        arguments_pred = _local_arguments_pred(arguments_true, arguments_logits)
        arguments_seq_accuracy = tf.cast(tf.reduce_all(arguments_true == arguments_pred, axis=-1), dtype=tf.float32)
        arguments_seq_mask = tf.cast(tf.reduce_min(arguments_true, axis=-1) > -1, dtype=tf.float32)

        self.arguments_seq_accuracy.update_state(arguments_seq_mask * arguments_seq_accuracy,
                                                 sample_weight=sample_weight)
        self.strict_accuracy.update_state(arguments_seq_mask * arguments_seq_accuracy * tactic_accuracy,
                                          sample_weight=sample_weight)

        metric_results[LocalArgumentPrediction.ARGUMENTS_SEQ_ACCURACY] = self.arguments_seq_accuracy.result()
        metric_results[LocalArgumentPrediction.STRICT_ACCURACY] = self.strict_accuracy.result()
        return metric_results


class GlobalArgumentModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arguments_seq_accuracy = tf.keras.metrics.Mean(name=GlobalArgumentPrediction.ARGUMENTS_SEQ_ACCURACY)
        self.strict_accuracy = tf.keras.metrics.Mean(name=GlobalArgumentPrediction.STRICT_ACCURACY)
        self.mixed_metrics = [self.arguments_seq_accuracy, self.strict_accuracy]

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metric_results = super().compute_metrics(x, y, y_pred, sample_weight)

        tactic = y[LocalArgumentPrediction.TACTIC_LOGITS]
        tactic_logits = y_pred[LocalArgumentPrediction.TACTIC_LOGITS]
        tactic_accuracy = tf.keras.metrics.sparse_categorical_accuracy(tactic, tactic_logits)

        local_arguments = y[GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS]
        local_arguments_logits = y_pred[GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS]
        local_arguments_best_logit = tf.reduce_max(local_arguments_logits, axis=-1)
        local_arguments_pred = _local_arguments_pred(y_true=tf.squeeze(local_arguments.to_tensor(-1), axis=1),
                                                     y_pred=local_arguments_logits)

        global_arguments = y[GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS]
        global_arguments_logits = y_pred[GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS]
        global_arguments_best_logit = tf.reduce_max(global_arguments_logits, axis=-1)
        global_arguments_pred = tf.argmax(global_arguments_logits, axis=-1)

        arguments_logits = tf.stack([local_arguments_best_logit, global_arguments_best_logit])
        arguments_pred = tf.where(tf.argmax(arguments_logits) == 0, local_arguments_pred, global_arguments_pred)

        arguments_true = tf.reduce_max(tf.stack([local_arguments, global_arguments], axis=-1), axis=-1)
        arguments_true = tf.squeeze(arguments_true.to_tensor(default_value=0), axis=1)

        arguments_seq_accuracy = tf.cast(tf.reduce_all(arguments_pred == arguments_true, axis=-1), tf.float32)
        arguments_seq_mask = tf.cast(tf.reduce_min(arguments_true, axis=-1) > -1, dtype=tf.float32)

        self.arguments_seq_accuracy.update_state(arguments_seq_mask * arguments_seq_accuracy,
                                                 sample_weight=sample_weight)
        self.strict_accuracy.update_state(arguments_seq_mask * arguments_seq_accuracy * tactic_accuracy,
                                          sample_weight=sample_weight)

        metric_results[GlobalArgumentPrediction.ARGUMENTS_SEQ_ACCURACY] = self.arguments_seq_accuracy.result()
        metric_results[GlobalArgumentPrediction.STRICT_ACCURACY] = self.strict_accuracy.result()
        return metric_results


class PredictionTask:
    """
    Base class for the various prediction tasks that we will define.

    Subclasses should implement the following methods:
        - create_input_output
        - _create_prediction_model
        - _loss
        - _metrics

    Additionally, they may override the following methods
        - loss_weights
        - callbacks

    They can also implement any other methods that may be necessary (for prediction, see graph2tac.tfgnn. predict).
    """
    TACTIC_LOGITS = 'tactic_logits'

    create_input_output: Callable[[tfgnn.GraphTensor], Tuple[tfgnn.GraphTensor, tf.Tensor]]
    _create_prediction_model: Callable[..., tf.keras.Model]
    _loss: Callable[[], Dict[str, tf.keras.losses.Loss]]
    _metrics: Callable[[], Dict[str, List[tf.keras.metrics.Metric]]]

    def __init__(self,
                 graph_constants: GraphConstants,
                 hidden_size: int,
                 tactic_embedding_size: int,
                 gnn_type: str,
                 gnn_config: dict,
                 tactic_head_type: str,
                 tactic_head_config: dict,
                 **kwargs
                 ):
        """
        @param graph_constants: a GraphConstants object for the graphs that will be consumed by the model
        @param hidden_size: the (globally shared) hidden size
        @param tactic_embedding_size: the (globally shared) size of tactic embeddings
        @param gnn_type: the type of GNN component to use
        @param gnn_config: the hyperparameters to be passed to GNN constructor
        @param tactic_head_type: the type of tactic head to use
        @param tactic_head_config: the hyperparameters to be passed to tactic_head_function
        @param kwargs: other arguments are passed to the _create_prediction_model implemented by subclasses
        """
        self._graph_constants = graph_constants
        self._hidden_size = hidden_size
        self._tactic_embedding_size = tactic_embedding_size
        self._gnn_type = gnn_type
        self._tactic_head_type = tactic_head_type

        # we have to clear the Keras session to make sure layer names are consistently chosen
        # NOTE: this would break multi-gpu training using MirroredStrategy, so be careful with layer names in that case
#        if not isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy) and not isinstance(tf.distribute.get_strategy, tf.distribute.OneDeviceStrategy):
#            tf.keras.backend.clear_session()

        self.graph_embedding = GraphEmbedding(node_label_num=graph_constants.node_label_num,
                                              edge_label_num=graph_constants.edge_label_num,
                                              hidden_size=hidden_size)

        self.tactic_embedding = tf.keras.layers.Embedding(input_dim=graph_constants.tactic_num,
                                                          output_dim=tactic_embedding_size)

        # we initialize embeddings here so that we can use them immediately when constructing models
        self.graph_embedding._node_embedding(tf.range(graph_constants.node_label_num))
        self.tactic_embedding(tf.range(graph_constants.tactic_num))

        self.tactic_logits_from_embeddings = LogitsFromEmbeddings(embedding_matrix=self.tactic_embedding.embeddings,
                                                                  valid_indices=tf.range(graph_constants.tactic_num),
                                                                  name=self.TACTIC_LOGITS)

        gnn_constructor = get_gnn_constructor(gnn_type)
        self.gnn = gnn_constructor(hidden_size=hidden_size, **gnn_config)

        tactic_head_constructor = get_tactic_head_constructor(tactic_head_type)
        self.tactic_head = tactic_head_constructor(tactic_embedding_size=tactic_embedding_size, **tactic_head_config)

        self.prediction_model = self._create_prediction_model(**kwargs)

        self.checkpoint = tf.train.Checkpoint(prediction_model=self.prediction_model)

    def get_config(self):
        gnn_config = self.gnn.get_config()
        gnn_config.pop('hidden_size')

        tactic_head_config = self.tactic_head.get_config()
        tactic_head_config.pop('tactic_embedding_size')

        return {
            'hidden_size': self._hidden_size,
            'tactic_embedding_size': self._tactic_embedding_size,
            'gnn_type': self._gnn_type,
            'gnn_config': gnn_config,
            'tactic_head_type': self._tactic_head_type,
            'tactic_head_config': tactic_head_config
        }

    @staticmethod
    def from_yaml_config(graph_constants: GraphConstants, yaml_filepath: Path) -> "PredictionTask":
        """
        Create an instance of this class from a YAML configuration file.

        @param graph_constants: a GraphConstants object for the graphs that will be consumed by the model
        @param yaml_filepath: the filepath to a YAML file containing all other arguments to the constructor
        @return: a PredictionTask object
        """
        with yaml_filepath.open() as yaml_file:
            task_config = yaml.load(yaml_file, Loader=yaml.SafeLoader)

        prediction_task_type = task_config.pop('prediction_task_type')
        prediction_task_constructor = get_prediction_task_constructor(prediction_task_type)
        return prediction_task_constructor(graph_constants=graph_constants, **task_config)

    def loss(self) -> Dict[str, tf.keras.losses.Loss]:
        """
        Provides the losses for this task, to be used with keras model.compile()

        :return: a dictionary mapping the model's outputs to their corresponding losses
        """
        with self.prediction_model.distribute_strategy.scope():
            return self._loss()

    def metrics(self) -> Dict[str, List[tf.keras.metrics.Metric]]:
        """
        Provides the metrics for this task, to be used with `tf.keras.Model.compile()`

        :return: a dictionary mapping the model's outputs to a list of metrics, for use with `model.compile`
        """
        with self.prediction_model.distribute_strategy.scope():
            return self._metrics()

    def loss_weights(self) -> Dict[str, float]:
        """
        Provides the loss weights for this task, to be used with keras model.compile()

        :return: a dictionary mapping the model's outputs to their corresponding loss weights
        """
        return {loss_name: 1.0 for loss_name in self._loss().keys()}

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Provides basic callbacks for this task, to be used with keras model.compile()

        :return: a list of keras callbacks
        """
        return []

    def from_trainer_checkpoint(self, save_path: str) -> None:
        """
        Loads a checkpoint created by the training script.

        @param save_path: the full path to the checkpoint we want to load
        """
        load_status = tf.train.Checkpoint(prediction_task=self.checkpoint).restore(save_path)
        load_status.expect_partial().assert_nontrivial_match().run_restore_ops()


class TacticPrediction(PredictionTask):
    """
    Wrapper for the base tactic prediction task
    """

    def get_config(self):
        config = super().get_config()

        config.update({
            'prediction_task_type': BASE_TACTIC_PREDICTION,
        })
        return config

    def _create_prediction_model(self) -> tf.keras.Model:
        """
        Combines a GNN component with a tactic head to produce an end-to-end model for the base tactic prediction task.
        The resulting model is for training purposes, and produces tactic logits.

        :return: a keras model consuming proof-state graphs and producing tactic logits
        """

        proofstate_graph = tf.keras.layers.Input(type_spec=batch_graph_spec(proofstate_graph_spec),
                                                 name='proofstate_graph')
        scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()

        bare_graph = strip_graph(scalar_proofstate_graph)
        embedded_graph = self.graph_embedding(bare_graph)  # noqa [ PyCallingNonCallable ]

        hidden_graph = self.gnn(embedded_graph)
        tactic_embedding = self.tactic_head(hidden_graph)
        tactic_logits = self.tactic_logits_from_embeddings(tactic_embedding)  # noqa [ PyCallingNonCallable ]

        return tf.keras.Model(inputs=proofstate_graph, outputs={TacticPrediction.TACTIC_LOGITS: tactic_logits})

    @staticmethod
    def create_input_output(graph_tensor: tfgnn.GraphTensor) -> Tuple:
        return graph_tensor, {TacticPrediction.TACTIC_LOGITS: graph_tensor.context['tactic']}

    @staticmethod
    def _loss() -> Dict[str, tf.keras.losses.Loss]:
        return {TacticPrediction.TACTIC_LOGITS: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}

    @staticmethod
    def _metrics() -> Dict[str, List[tf.keras.metrics.Metric]]:
        return {TacticPrediction.TACTIC_LOGITS: [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]}


class LocalArgumentPrediction(PredictionTask):
    """
    Wrapper for the local argument prediction task
    """
    ARGUMENTS_LOGITS = 'arguments_logits'
    ARGUMENTS_SEQ_ACCURACY = 'arguments_seq_accuracy'
    STRICT_ACCURACY = 'strict_accuracy'

    def __init__(self,
                 arguments_loss_coefficient: float = 1.0,
                 **kwargs
                 ):
        """
        @param arguments_loss_coefficient: the weight of the loss term for the arguments (base tactic loss has weight 1)
        @param kwargs: arguments to be passed to the PredictionTask constructor
        """
        super().__init__(**kwargs)
        self._arguments_loss_coefficient = arguments_loss_coefficient

    def get_config(self):
        config = super().get_config()

        arguments_head_config = self.arguments_head.get_config()
        arguments_head_config.pop('hidden_size')
        arguments_head_config.pop('tactic_embedding_size')

        config.update({
            'prediction_task_type': LOCAL_ARGUMENT_PREDICTION,
            'arguments_loss_coefficient': self._arguments_loss_coefficient,
            'arguments_head_type': self._arguments_head_type,
            'arguments_head_config': arguments_head_config
        })

        return config

    def _create_prediction_model(self,
                                 arguments_head_type: str,
                                 arguments_head_config: dict
                                 ) -> tf.keras.Model:
        """
        Combines a GNN component with a tactic head and an arguments head to produce an end-to-end model for the
        local argument prediction task. The resulting model is weakly autoregressive and for training purposes only,
        producing both tactic logits and argument logits for each local context node and argument position.

        @param arguments_head_type: the type of arguments head to use
        @param arguments_head_config: the hyperparameters to be used for the arguments head
        @return: a keras model consuming graphs and producing tactic logits and local arguments logits
        """
        self._arguments_head_type = arguments_head_type

        arguments_head_constructor = get_arguments_head_constructor(arguments_head_type)
        self.arguments_head = arguments_head_constructor(hidden_size=self._hidden_size,
                                                         tactic_embedding_size=self._tactic_embedding_size,
                                                         **arguments_head_config)

        self.arguments_logits = tf.keras.layers.Lambda(lambda inputs: _local_arguments_logits(*inputs),
                                                       name=self.ARGUMENTS_LOGITS)

        proofstate_graph = tf.keras.layers.Input(type_spec=batch_graph_spec(proofstate_graph_spec),
                                                 name='proofstate_graph')
        scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()
        # in_context = _context_mask(scalar_proofstate_graph)

        bare_graph = strip_graph(scalar_proofstate_graph)
        embedded_graph = self.graph_embedding(bare_graph)  # noqa [ PyCallingNonCallable ]
        hidden_graph = self.gnn(embedded_graph)

        tactic_embedding = self.tactic_head(hidden_graph)
        tactic_logits = self.tactic_logits_from_embeddings(tactic_embedding)  # noqa [ PyCallingNonCallable ]

        tactic = scalar_proofstate_graph.context['tactic']
        num_arguments = tf.gather(tf.constant(self._graph_constants.tactic_index_to_numargs, dtype=tf.int64), tactic)
        hidden_state_sequences = self.arguments_head((hidden_graph, self.tactic_embedding(tactic), num_arguments))
        arguments_logits = self.arguments_logits((scalar_proofstate_graph, hidden_graph, hidden_state_sequences))

        return LocalArgumentModel(inputs=proofstate_graph,
                                  outputs={LocalArgumentPrediction.TACTIC_LOGITS: tactic_logits,
                                           LocalArgumentPrediction.ARGUMENTS_LOGITS: arguments_logits})

    @staticmethod
    def create_input_output(graph_tensor: tfgnn.GraphTensor) -> Tuple[Any, Dict[str, Any]]:
        outputs = {LocalArgumentPrediction.TACTIC_LOGITS: graph_tensor.context['tactic'],
                   LocalArgumentPrediction.ARGUMENTS_LOGITS: graph_tensor.context['local_arguments']}
        return graph_tensor, outputs

    @staticmethod
    def _loss() -> Dict[str, tf.keras.losses.Loss]:
        return {LocalArgumentPrediction.TACTIC_LOGITS: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                LocalArgumentPrediction.ARGUMENTS_LOGITS: LocalArgumentSparseCategoricalCrossentropy()}

    @staticmethod
    def _metrics() -> Dict[str, Iterable[tf.keras.metrics.Metric]]:
        return {LocalArgumentPrediction.TACTIC_LOGITS: [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                LocalArgumentPrediction.ARGUMENTS_LOGITS: [ArgumentSparseCategoricalAccuracy(name='accuracy')]}

    def loss_weights(self) -> Dict[str, float]:
        return {LocalArgumentPrediction.TACTIC_LOGITS: 1.0,
                LocalArgumentPrediction.ARGUMENTS_LOGITS: self._arguments_loss_coefficient}

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [MixedMetricsCallback()]


class GlobalArgumentPrediction(PredictionTask):
    LOCAL_ARGUMENTS_LOGITS = 'local_arguments_logits'
    GLOBAL_ARGUMENTS_LOGITS = 'global_arguments_logits'
    ARGUMENTS_SEQ_ACCURACY = 'arguments_seq_accuracy'
    STRICT_ACCURACY = 'strict_accuracy'

    def __init__(self, arguments_loss_coefficient: float = 1.0, **kwargs):
        """
        @param arguments_loss_coefficient: the weight of the loss term for the arguments (base tactic loss has weight 1)
        @param kwargs: arguments to be passed to the PredictionTask constructor
        """
        super().__init__(**kwargs)
        self._arguments_loss_coefficient = arguments_loss_coefficient

    def get_config(self):
        config = super().get_config()

        arguments_head_config = self.arguments_head.get_config()
        arguments_head_config.pop('hidden_size')
        arguments_head_config.pop('tactic_embedding_size')

        config.update({
            'prediction_task_type': GLOBAL_ARGUMENT_PREDICTION,
            'arguments_loss_coefficient': self._arguments_loss_coefficient,
            'arguments_head_type': self._arguments_head_type,
            'arguments_head_config': arguments_head_config
        })

        return config

    def _create_prediction_model(self,
                                 arguments_head_type: str,
                                 arguments_head_config: dict
                                 ) -> tf.keras.Model:
        """
        Combines a GNN component with a tactic head and an arguments head to produce an end-to-end model for the
        global argument prediction task. The resulting model is for training purposes, and produces tactic logits
        and argument logits (for each local context node / global context id per argument position).

        @param arguments_head_type: the type of arguments head to use
        @param arguments_head_config: the hyperparameters to be passed to the arguments head
        @return: a keras model consuming graphs and producing tactic and local/global arguments logits
        """
        self._arguments_head_type = arguments_head_type

        arguments_head_constructor = get_arguments_head_constructor(arguments_head_type)
        self.arguments_head = arguments_head_constructor(hidden_size=self._hidden_size,
                                                         tactic_embedding_size=self._tactic_embedding_size,
                                                         **arguments_head_config)

        self.global_arguments_logits = LogitsFromEmbeddings(
            embedding_matrix=self.graph_embedding._node_embedding.embeddings,
            valid_indices=tf.constant(self._graph_constants.global_context, dtype=tf.int32))

        self.local_arguments_logits_output = tf.keras.layers.Lambda(lambda x: x, name=self.LOCAL_ARGUMENTS_LOGITS)
        self.global_arguments_logits_output = tf.keras.layers.Lambda(lambda x: x, name=self.GLOBAL_ARGUMENTS_LOGITS)

        proofstate_graph = tf.keras.layers.Input(type_spec=batch_graph_spec(proofstate_graph_spec),
                                                 name='proofstate_graph')
        scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()

        bare_graph = strip_graph(scalar_proofstate_graph)
        embedded_graph = self.graph_embedding(bare_graph)  # noqa [ PyCallingNonCallable ]
        hidden_graph = self.gnn(embedded_graph)

        tactic_embedding = self.tactic_head(hidden_graph)
        tactic_logits = self.tactic_logits_from_embeddings(tactic_embedding)  # noqa [ PyCallingNonCallable ]

        tactic = scalar_proofstate_graph.context['tactic']
        num_arguments = tf.gather(tf.constant(self._graph_constants.tactic_index_to_numargs, dtype=tf.int64), tactic)
        hidden_state_sequences = self.arguments_head((hidden_graph, self.tactic_embedding(tactic), num_arguments))
        local_arguments_logits = _local_arguments_logits(scalar_proofstate_graph, hidden_graph, hidden_state_sequences)

        global_arguments_logits = self.global_arguments_logits(hidden_state_sequences.to_tensor())   # noqa

        # normalize logits making sure the log_softmax is numerically stable
        local_arguments_max_logit = tf.reduce_max(local_arguments_logits, axis=-1)
        global_arguments_max_logit = tf.reduce_max(global_arguments_logits, axis=-1)
        arguments_max_logit = tf.reduce_max(tf.stack([local_arguments_max_logit, global_arguments_max_logit], axis=-1), axis=-1, keepdims=True)
        local_arguments_logits -= arguments_max_logit
        global_arguments_logits -= arguments_max_logit

        local_arguments_logits_norm = tf.reduce_sum(tf.exp(local_arguments_logits), axis=-1, keepdims=True)
        global_arguments_logits_norm = tf.reduce_sum(tf.exp(global_arguments_logits), axis=-1, keepdims=True)
        norm = -tf.math.log(local_arguments_logits_norm + global_arguments_logits_norm)

        normalized_local_arguments_logits = self.local_arguments_logits_output(local_arguments_logits + norm)
        normalized_global_arguments_logits = self.global_arguments_logits_output(global_arguments_logits + norm)

        return GlobalArgumentModel(inputs=proofstate_graph,
                                   outputs={self.TACTIC_LOGITS: tactic_logits,
                                            self.LOCAL_ARGUMENTS_LOGITS: normalized_local_arguments_logits,
                                            self.GLOBAL_ARGUMENTS_LOGITS: normalized_global_arguments_logits})

    @staticmethod
    def create_input_output(graph_tensor: tfgnn.GraphTensor) -> Tuple[Any, Dict[str, Any]]:
        outputs = {GlobalArgumentPrediction.TACTIC_LOGITS: graph_tensor.context['tactic'],
                   GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: graph_tensor.context['local_arguments'],
                   GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: graph_tensor.context['global_arguments']}
        return graph_tensor, outputs

    @staticmethod
    def _loss() -> Dict[str, tf.keras.losses.Loss]:
        return {GlobalArgumentPrediction.TACTIC_LOGITS: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: GlobalArgumentSparseCategoricalCrossentropy(),
                GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: GlobalArgumentSparseCategoricalCrossentropy()}

    def loss_weights(self) -> Dict[str, float]:
        return {GlobalArgumentPrediction.TACTIC_LOGITS: 1.0,
                GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: self._arguments_loss_coefficient,
                GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: self._arguments_loss_coefficient}

    def _metrics(self) -> Dict[str, List[tf.keras.metrics.Metric]]:
        return {GlobalArgumentPrediction.TACTIC_LOGITS: [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: [ArgumentSparseCategoricalAccuracy(name='accuracy')],
                GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: [ArgumentSparseCategoricalAccuracy(name='accuracy')]}

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [MixedMetricsCallback()]


class DefinitionTask(tf.keras.layers.Layer):
    """
    A layer to compute definition embeddings from definition cluster graphs. The input graphs should:
        - be scalar graphs
        - follow the schema for `definition_graph_spec` in `graph2tac.tfgnn.graph_schema`
        - have the node labels for the definitions being defined masked to -1
    """
    def __init__(self,
                 graph_embedding: tf.keras.layers.Layer,
                 gnn: tf.keras.layers.Layer,
                 definition_head_type: str,
                 definition_head_config: dict,
                 name: str = 'definition_layer',
                 **kwargs):
        """
        @param graph_embedding: the GraphEmbedding layer from the prediction task
        @param gnn: the GNN layer from the prediction task
        @param definition_head_type: the type of definition head to use
        @param definition_head_config: the hyperparameters for the definition head
        @param name: the name of this layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._definition_head_type = definition_head_type

        self._graph_embedding = graph_embedding
        self._gnn = gnn

        definition_head_constructor = get_definition_head_constructor(definition_head_type)
        self._definition_head = definition_head_constructor(hidden_size=graph_embedding._hidden_size,
                                                            **definition_head_config)

    def get_config(self):
        config = super().get_config()

        definition_head_config = self._definition_head.get_config()
        definition_head_config.pop('hidden_size')

        config.update({
            'definition_head_type': self._definition_head_type,
            'definition_head_config': definition_head_config
        })
        return config

    @classmethod
    def from_yaml_config(cls,
                         graph_embedding: tf.keras.layers.Layer,
                         gnn: tf.keras.layers.Layer,
                         yaml_filepath: Path
                         ) -> Optional["DefinitionTask"]:
        with yaml_filepath.open() as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        if config.get('definition_head_type') is None:
            return None
        else:
            return cls(graph_embedding=graph_embedding, gnn=gnn, **config)

    @staticmethod
    def _mask_defined_embeddings(scalar_definition_graph: tfgnn.GraphTensor, embedded_graph: tfgnn.GraphTensor):
        num_definitions = tf.cast(scalar_definition_graph.context['num_definitions'], dtype=tf.int32)
        is_defined = tf.ragged.range(scalar_definition_graph.node_sets['node'].sizes) < tf.expand_dims(num_definitions, axis=-1)
        mask = tf.expand_dims(1 - tf.cast(is_defined.flat_values, dtype=tf.float32), axis=-1)
        masked_hidden_state = embedded_graph.node_sets['node']['hidden_state'] * mask
        return embedded_graph.replace_features(node_sets={'node': {'hidden_state': masked_hidden_state}})

    def call(self, scalar_definition_graph: tfgnn.GraphTensor, training: bool = False):
        bare_graph = strip_graph(scalar_definition_graph)
        embedded_graph = self._graph_embedding(bare_graph, training=training)
        masked_embedded_graph = self._mask_defined_embeddings(scalar_definition_graph, embedded_graph)
        hidden_graph = self._gnn(masked_embedded_graph, training=training)

        num_definitions = scalar_definition_graph.context['num_definitions']
        definition_body_embeddings = self._definition_head((hidden_graph, num_definitions), training=training)
        return definition_body_embeddings


def get_prediction_task_constructor(prediction_task_type: str) -> Callable[..., PredictionTask]:
    if prediction_task_type == BASE_TACTIC_PREDICTION:
        return TacticPrediction
    elif prediction_task_type == LOCAL_ARGUMENT_PREDICTION:
        return LocalArgumentPrediction
    elif prediction_task_type == GLOBAL_ARGUMENT_PREDICTION:
        return GlobalArgumentPrediction
    else:
        raise ValueError(f'{prediction_task_type} is not a valid prediction task type')
