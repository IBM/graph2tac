from typing import List, Dict, Iterable, Tuple, Callable, Optional, Union

import yaml
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path
import numpy as np

from graph2tac.loader.data_classes import GraphConstants
from graph2tac.tfgnn.graph_schema import proofstate_graph_spec, batch_graph_spec, strip_graph
from graph2tac.tfgnn.models import (GraphEmbedding,
                                    LogitsFromEmbeddings,
                                    get_gnn_constructor,
                                    get_arguments_head_constructor,
                                    get_tactic_head_constructor,
                                    get_definition_head_constructor)


BASE_TACTIC_PREDICTION = 'base_tactic_prediction'
LOCAL_ARGUMENT_PREDICTION = 'local_argument_prediction'
GLOBAL_ARGUMENT_PREDICTION = 'global_argument_prediction'


@tf.function
def _local_arguments_pred(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.argmax(y_pred, axis=-1) if tf.shape(y_pred)[-1] > 0 else tf.zeros_like(y_true)


class ArgumentSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Sparse categorical crossentropy loss for local and global args in the global argument prediction task.
    
    NOTES:
        - `-1` arguments correspond to `None` or a different type of argument (e.g. local vs global)
        - logits **are** assumed to be normalized
        - `sum_loss_over_tactic` parameter:
            - If `True`, the losses are summed across all arguments within a batch.
              When the local and global losses are added together, the combined loss is the negative
              log probability of the ground truth sequence of (non-None) arguments.
              Further, if argument loss weight is 1.0 then when the tactic, local, and global
              losses are added together, the combined loss is equal to the negative log probability of
              the full ground truth tactic, including base tactic and all (non-None) arguments.
            - If `False`, the "batch_size" of the output will not necessarily be the same as the number
              of elements in the batch.  This has the effect that the loss will be averaged by the number
              of global (or local) arguments in the batch before combining with other losses.
    """

    def __init__(self, sum_loss_over_tactic: bool, **kwargs):
        """
        @param average_per_tactic: whether to sum the argument losses over tactic count argument
        """
        super().__init__(**kwargs)
        self.sum_loss_over_tactic = sum_loss_over_tactic

    @staticmethod
    def arguments_filter(y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        """
        Extracts the local arguments which are not None from the ground truth and predictions.
        
        Returns a pair of ragged tensors with shapes:
        - [batch_size, None(num_nonempty_args)]
        - [batch_size, None(num_nonempty_args), None(context)]
        @param y_true: the labels for the arguments, with shape [batch_size, 1, None(num_arguments)]
        @param y_pred: the logits for the arguments, with shape [batch_size, None(num_arguments),  None(context)]
        @return: a tuple whose first element contains the non-None arguments, the second element being the logits corresponding to each non-None argument
        """
        # convert y_true to a dense tensor padding with -1 values (also used for None arguments);
        # remove spurious dimension (y_true was created from a non-scalar graph)
        # [ batch_size, max(num_arguments) ]
        arguments_tensor = tf.squeeze(y_true.to_tensor(default_value=-1), axis=1)

        # we want to compute only over the positions that are not None
        nrows = tf.shape(y_true, out_type=tf.int64)[0]
        positions = tf.where(arguments_tensor != -1)
        row_ids = positions[:, 0]

        # keep only these positions in the both y_true and y_pred
        arguments_true = tf.RaggedTensor.from_value_rowids(
            values=tf.gather_nd(arguments_tensor, positions),
            value_rowids=row_ids,
            nrows=nrows
        )
        arguments_pred = tf.RaggedTensor.from_value_rowids(
            values=tf.gather_nd(y_pred, positions),
            value_rowids=row_ids,
            nrows=nrows
        )
        # output shape: [batch, None(args)], [batch, None(args), globals]
        return arguments_true, arguments_pred

    @staticmethod
    def convert_to_ragged(y_true, y_pred):

        # y_true: # [batch_size, 1, None(args)]
        # y_pred: # [batch_size, max(num_arguments), context_size]

        # remove spurious dimension (y_true was created from a non-scalar graph)
        # find lengths
        # shape: [batch]
        y_true_lengths = tf.squeeze(y_true, 1).row_lengths()

        # use y_true_lenghts to set outer ragged shape
        # shape: [batch, None(args), context_size]
        y_pred = tf.RaggedTensor.from_tensor(y_pred, lengths=y_true_lengths)

        def tensor_to_ragged_filter(x, value_to_filter):
            # x shape: # [nrows, ncols]
            nrows = tf.shape(y_true, out_type=tf.int64)[0]
            positions = tf.where(x != value_to_filter)  # [nrows, 2]
            row_ids = positions[:, 0]  # [nrows]

            # return shape: [nrows, None(cols)]
            return tf.RaggedTensor.from_value_rowids(
                values=tf.gather_nd(x, positions),
                value_rowids=row_ids,
                nrows=nrows
            )

        # filter out -inf
        # shape: [batch, None(args), None(context)]
        y_pred = tf.ragged.map_flat_values(lambda x: tensor_to_ragged_filter(x, -np.inf), y_pred)

        return y_pred


    def call(self, y_true, y_pred):
        """
        @param y_true: ids for the arguments, with shape [ batch_size, 1, None(num_arguments) ]
        @param y_pred: logits for each argument position, with shape [ batch_size, None(num_arguments), num_categories ]
        @return: a vector of length equal to either the size of the batch or the number of arguments of the given type
        """
        # y_true shape: [batch, None(args)]
        # y_pred shape: [batch, None(args), None(context)]

        # filter out any arguments which have index -1, i.e. are None or of a different kind (global vs local)
        # shape: [batch, None(args)], [batch, None(args), None(context)]
        arguments_true, arguments_pred = self.arguments_filter(y_true, y_pred)
        
        # compute the cross entropy loss by using gather to find the corresponding logit
        # (and flip the sign to get cross entropy)
        arg_losses = -tf.gather(arguments_pred, arguments_true, batch_dims=2)

        # return the losses as a list of losses (it will be reduced automatically by keras to a single number)
        if self.sum_loss_over_tactic:
            # sum over all arguments in a batch element
            # shape: [batch]
            return tf.reduce_sum(arg_losses, axis=-1)
        else:
            # return one loss for each argument in the in the batch
            # shape: [num of args in ground truth for batch]
            return arg_losses.flat_values


class ArgumentSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    """
    Per-argument sparse categorical accuracy, excluding None arguments.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true shape: [batch, None(args)]
        # y_pred shape: [batch, None(args), None(context)]

        # filter out any arguments which have index -1, i.e. are None or of a different kind (global vs local)
        # shape: [batch, None(args)], [batch, None(args), None(context)]
        arguments_true, arguments_pred = ArgumentSparseCategoricalCrossentropy.arguments_filter(y_true, y_pred)
        
        # TODO(jrute): Is this the best way?  I'm increasing the context size again to be non-ragged.
        # shape: [batch, None(args), context_size]
        arguments_pred = arguments_pred.with_values(arguments_pred.values.to_tensor(default_value=-np.inf))

        if tf.shape(arguments_pred)[-1] > 0:
            super().update_state(arguments_true, arguments_pred, sample_weight)


class DefinitionNormSquaredLoss(tf.keras.losses.Loss):
    """
    Norm squared loss
    """
    def call(self, y_true, y_pred):
        # ignore y_true as it is zero
        return tf.reduce_sum(y_pred * y_pred, axis=-1)


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

        local_arguments = y[LocalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS]
        local_arguments_logits = y_pred[LocalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS]
        local_arguments_true = tf.squeeze(local_arguments.to_tensor(default_value=0), axis=1)
        local_arguments_pred = _local_arguments_pred(local_arguments_true, local_arguments_logits)
        arguments_seq_accuracy = tf.cast(tf.reduce_all(local_arguments_true == local_arguments_pred, axis=-1), dtype=tf.float32)
        arguments_seq_mask = tf.cast(tf.reduce_min(local_arguments_true, axis=-1) > -1, dtype=tf.float32)

        self.arguments_seq_accuracy.update_state(arguments_seq_mask * arguments_seq_accuracy,
                                                 sample_weight=sample_weight)
        self.strict_accuracy.update_state(arguments_seq_mask * arguments_seq_accuracy * tactic_accuracy,
                                          sample_weight=sample_weight)

        metric_results[LocalArgumentPrediction.ARGUMENTS_SEQ_ACCURACY] = self.arguments_seq_accuracy.result()
        metric_results[LocalArgumentPrediction.STRICT_ACCURACY] = self.strict_accuracy.result()
        return metric_results

@tf.function
def arg_best_logit_and_pred(
    logits: tf.RaggedTensor,  # [batch, None(args), None(context)]
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:  # ([batch, None(args)], [batch, None(args)])
    """Find the value and index of the best arguments logit

    If the context is empty, return -inf for the value and 0 for the index

    :param logits: Logits.  Shape: [batch, None(args), None(context)]
    :type logits: tf.RaggedTensor
    :return: value and index of the best logits.  Shape: [batch, None(arg)]
    :rtype: Tuple[tf.RaggedTensor, tf.RaggedTensor]
    """
    # pad context with -inf to make rows uniform
    logits = logits.with_values(logits.values.to_tensor(default_value=-np.inf))  # [batch, None(args), max(context)]
    if tf.shape(logits.values)[-1] == 0:
        # best_logit is -inf and pred is 0 if there are no logits
        template = logits.with_values(logits.value_rowids())  # [batch, None(args)]
        best_logit = tf.math.log(tf.zeros_like(template, dtype=tf.float32))  # [batch, None(args)]  
        pred = tf.zeros_like(template, dtype=tf.int64)  # [batch, None(args)]
    else:
        # since we pad context with -inf, the best_logit is -inf and pred is 0 if context is empty
        best_logit = tf.ragged.map_flat_values(tf.reduce_max, logits, axis=-1)  # [batch, None(args)]
        pred = tf.ragged.map_flat_values(tf.argmax, logits, axis=-1, output_type=tf.int64)  # [batch, None(args)]
    return (best_logit, pred)


class GlobalArgumentModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arguments_seq_accuracy = tf.keras.metrics.Mean(name=GlobalArgumentPrediction.ARGUMENTS_SEQ_ACCURACY)
        self.strict_accuracy = tf.keras.metrics.Mean(name=GlobalArgumentPrediction.STRICT_ACCURACY)
        self.mixed_metrics = [self.arguments_seq_accuracy, self.strict_accuracy]

    def compute_metrics(self, x, y, y_pred, sample_weight):
        metric_results = super().compute_metrics(x, y, y_pred, sample_weight)

        # tactic accuracy
        tactic = y[LocalArgumentPrediction.TACTIC_LOGITS]
        tactic_logits = y_pred[LocalArgumentPrediction.TACTIC_LOGITS]
        tactic_accuracy = tf.keras.metrics.sparse_categorical_accuracy(tactic, tactic_logits)

        # local arguments
        local_arguments = y[GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS]  # [batch, 1, None(args)]
        local_arguments_logits = y_pred[GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS]  # [batch, None(args), None(context]
        local_true = tf.squeeze(local_arguments, 1)  # [batch, None(args)]
        local_best_logit, local_pred = arg_best_logit_and_pred(local_arguments_logits)
        
        # global arguments
        global_arguments = y[GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS]  # [batch, 1, None(args)]
        global_arguments_logits = y_pred[GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS]  # [batch, None(args), None(context]
        global_true = tf.squeeze(global_arguments, 1)  # [batch, None(args)]
        global_best_logit, global_pred = arg_best_logit_and_pred(global_arguments_logits)

        # all arguments
        # for ground truth, we can take which ever argument is higher since the other is -1
        # for the ground true, local and global can both be -1 which means the true argument is None
        arg_true_is_local = (local_true >= global_true)  # [batch, None(args)]
        arg_true_ix = tf.where(local_true >= global_true, local_true, global_true)  # [batch, None(args)]

        arg_pred_is_local = (local_best_logit >= global_best_logit)  # [batch, None(args)]
        arg_pred_ix = tf.where(local_best_logit >= global_best_logit, local_pred, global_pred)  # [batch, None(args)]
        
        # strict accuracies
        # check that every argument in the sequence is correct
        # if any are None (i.e. ground truth local and global is -1) then it is marked as incorrect (since model can't produce None by design)
        seq_arg_accuracy_is_local =  tf.reduce_all(tf.equal(arg_true_is_local.with_row_splits_dtype(tf.int64), arg_pred_is_local), axis=-1)  # [batch]
        seq_arg_accuracy_ix =  tf.reduce_all(tf.equal(arg_true_ix.with_row_splits_dtype(tf.int64), arg_pred_ix), axis=-1)  # [batch]
        seq_arg_accuracy = tf.cast(seq_arg_accuracy_is_local & seq_arg_accuracy_ix, dtype=tf.float32)  # [batch]
        strict_accuracy = tactic_accuracy * seq_arg_accuracy  # [batch]

        self.arguments_seq_accuracy.update_state(seq_arg_accuracy, sample_weight=sample_weight)
        self.strict_accuracy.update_state(strict_accuracy, sample_weight=sample_weight)

        metric_results[GlobalArgumentPrediction.ARGUMENTS_SEQ_ACCURACY] = self.arguments_seq_accuracy.result()
        metric_results[GlobalArgumentPrediction.STRICT_ACCURACY] = self.strict_accuracy.result()
        return metric_results


class PredictionTask:
    """
    Base class for the various prediction tasks that we will define.

    Subclasses should implement the following methods:
        - create_input_output
        - create_train_model
        - create_inference_model
        - loss
        - metrics

    Additionally, they may override the following methods
        - loss_weights
        - callbacks

    They can also implement any other methods that may be necessary (for prediction, see graph2tac.tfgnn. predict).
    """
    create_input_output: Callable[[tfgnn.GraphTensor], Tuple[tfgnn.GraphTensor, Union[tf.Tensor, tf.RaggedTensor]]]
    create_train_model: Callable[[], tf.keras.Model]
    create_inference_model: Callable[..., tf.keras.Model]
    loss: Callable[[], Dict[str, tf.keras.losses.Loss]]
    metrics: Callable[[], Dict[str, List[tf.keras.metrics.Metric]]]

    PROOFSTATE_GRAPH = 'proofstate_graph'

    def __init__(self,
                 graph_constants: GraphConstants,
                 hidden_size: int,
                 unit_norm_embs: bool,
                 gnn_type: str,
                 gnn_config: dict
                 ):
        """
        @param graph_constants: a GraphConstants object for the graphs that will be consumed by the model
        @param hidden_size: the (globally shared) hidden size
        @param unit_norm_embs: whether to restrict embeddings to the unit norm
        @param gnn_type: the type of GNN component to use
        @param gnn_config: the hyperparameters to be passed to GNN constructor
        """
        self._graph_constants = graph_constants
        self._hidden_size = hidden_size
        self._unit_norm_embs = unit_norm_embs
        self._gnn_type = gnn_type

        # we have to clear the Keras session to make sure layer names are consistently chosen
        # NOTE: this would break multi-gpu training using MirroredStrategy, so be careful with layer names in that case
        # if not isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy) and not isinstance(tf.distribute.get_strategy, tf.distribute.OneDeviceStrategy):
        #     tf.keras.backend.clear_session()

        # create and initialize node and edge embeddings
        self.graph_embedding = GraphEmbedding(
            node_label_num=graph_constants.node_label_num,
            edge_label_num=graph_constants.edge_label_num,
            hidden_size=hidden_size,
            unit_normalize=unit_norm_embs
        )
        self.graph_embedding.lookup_node_embedding(tf.range(graph_constants.node_label_num))

        # create the GNN component
        gnn_constructor = get_gnn_constructor(gnn_type)
        self.gnn = gnn_constructor(hidden_size=hidden_size, **gnn_config)

        # create checkpoint with both layers created above
        self.checkpoint = tf.train.Checkpoint(graph_embedding=self.graph_embedding, gnn=self.gnn)

    def get_config(self):
        gnn_config = self.gnn.get_config()
        gnn_config.pop('hidden_size')

        return {
            'hidden_size': self._hidden_size,
            'gnn_type': self._gnn_type,
            'unit_norm_embs': self._unit_norm_embs,
            'gnn_config': gnn_config,
        }

    @staticmethod
    def from_yaml_config(graph_constants: GraphConstants,
                         yaml_filepath: Path
                         ) -> Union["TacticPrediction", "LocalArgumentPrediction", "GlobalArgumentPrediction"]:
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

    def loss_weights(self) -> Dict[str, float]:
        """
        Provides the loss weights for this task, to be used with keras model.compile()

        @return: a dictionary mapping the model's outputs to their corresponding loss weights
        """
        return {loss_name: 1.0 for loss_name in self.loss().keys()}

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Provides basic callbacks for this task, to be used with keras model.compile()

        @return: a list of keras callbacks
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
    Wrapper for the base tactic prediction task.
    """
    TACTIC: str = 'tactic'
    TACTIC_LOGITS = 'tactic_logits'
    TACTIC_MASK = 'tactic_mask'

    def __init__(self,
                 tactic_embedding_size: int,
                 tactic_head_type: str,
                 tactic_head_config: dict,
                 **kwargs):
        """
        @param tactic_embedding_size: the (globally shared) size of tactic embeddings
        @param tactic_head_type: the type of tactic head to use
        @param tactic_head_config: the hyperparameters to be passed to tactic_head_function
        @param kwargs: other arguments are passed on to the PredictionTask class constructor
        """
        super().__init__(**kwargs)
        self._tactic_embedding_size = tactic_embedding_size
        self._tactic_head_type = tactic_head_type

        # create and initialize tactic embeddings
        self.tactic_embedding = tf.keras.layers.Embedding(input_dim=self._graph_constants.tactic_num,
                                                          output_dim=tactic_embedding_size)
        self.tactic_embedding(tf.range(self._graph_constants.tactic_num))

        # create the tactic head
        tactic_head_constructor = get_tactic_head_constructor(tactic_head_type)
        self.tactic_head = tactic_head_constructor(tactic_embedding_size=tactic_embedding_size, **tactic_head_config)

        # a layer to compute tactic logits from tactic embeddings
        self.tactic_logits_from_embeddings = LogitsFromEmbeddings(
            embedding_matrix=self.tactic_embedding.embeddings,
            cosine_similarity=False,
            name=self.TACTIC_LOGITS
        )

        # update checkpoint with new layers
        self.checkpoint.tactic_embedding = self.tactic_embedding
        self.checkpoint.tactic_head = self.tactic_head
        self.checkpoint.tactic_logits_from_embeddings = self.tactic_logits_from_embeddings

    @staticmethod
    def _top_k_tactics(tactic_logits: tf.Tensor,  # [batch, tactics]
                       tactic_mask: tf.Tensor,  # [batch, tactics]
                       tactic_expand_bound: int
                       ) -> Tuple[tf.Tensor, tf.Tensor]:  #([batch, top_k_tactics], [batch, top_k_tactics])
        tactic_logits = tf.math.log_softmax(tactic_logits + tf.math.log(tf.cast(tactic_mask, tf.float32)), axis=-1)

        # Sometimes the number of tactics is less than the tactic_expand_bound
        # In that case return only the number of tactics.  It is ok if some have probability 0.
        top_k = tf.math.top_k(tactic_logits, k=tf.minimum(tactic_expand_bound, tf.shape(tactic_logits)[1]))
        # ([batch, top_k_tactics], [batch, top_k_tactics])
        return top_k.indices, top_k.values

    def _tactic_logits_and_hidden_graph(self,
                                        scalar_proofstate_graph: tfgnn.GraphTensor
                                        ) -> Tuple[tf.Tensor, tfgnn.GraphTensor]:
        bare_graph = strip_graph(scalar_proofstate_graph)
        embedded_graph = self.graph_embedding(bare_graph)  # noqa [ PyCallingNonCallable ]
        hidden_graph = self.gnn(embedded_graph)
        
        tactic_embedding = self.tactic_head(hidden_graph)
        tactic_logits = self.tactic_logits_from_embeddings(tactic_embedding)  # noqa [ PyCallingNonCallable ]
        return tactic_logits, hidden_graph

    def get_config(self):
        config = super().get_config()

        tactic_head_config = self.tactic_head.get_config()
        tactic_head_config.pop('tactic_embedding_size')

        config.update({
            'prediction_task_type': BASE_TACTIC_PREDICTION,
            'tactic_embedding_size': self._tactic_embedding_size,
            'tactic_head_type': self._tactic_head_type,
            'tactic_head_config': tactic_head_config
        })
        return config

    def create_train_model(self) -> tf.keras.Model:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    def create_inference_model(self, tactic_expand_bound: int, graph_constants: GraphConstants) -> tf.keras.Model:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    @staticmethod
    def create_input_output(graph_tensor: tfgnn.GraphTensor) -> Tuple[tfgnn.GraphTensor, Dict[str, tf.Tensor]]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    @staticmethod
    def loss() -> Dict[str, tf.keras.losses.Loss]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    @staticmethod
    def metrics() -> Dict[str, List[tf.keras.metrics.Metric]]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")


class GlobalEmbeddings(tf.keras.layers.Layer):
    def __init__(
        self,
        global_embeddings_layer: LogitsFromEmbeddings,
        dynamic_global_context: bool,
        name="global_embeddings",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.global_embeddings_layer = global_embeddings_layer
        self.dynamic_global_context = dynamic_global_context

    def call(
        self, 
        global_context: tf.RaggedTensor,  # [batch, None(context)]
        inference: bool = False,
        training=False
    ) -> tf.Tensor: # [batch, None(context), hdim]
        # [valid_context, hdim]
        all_embeddings = self.global_embeddings_layer.get_keys_embeddings()

        # note: inference model always uses dynamic global context
        if self.dynamic_global_context or inference:
            # select the global context for each batch
            # this is equivalent to `tf.gather(all_embeddings, global_context)`
            # but *much faster* when inference is run on CPU
            # [batch, None(context), hdim]
            return global_context.with_values(tf.gather(all_embeddings, global_context.values))
        else:
            # currently this would mess up later indexing
            raise NotImplementedError("dynamical_global_context must be set to true")
            # repeat all the embeddings for each batch element
            # [batch, valid_context, hdim]
            batch_size = tf.shape(global_context)[0]
            global_embeddings = tf.tile(tf.expand_dims(all_embeddings, axis=0), multiples=[batch_size, 1, 1])
            # [batch, None(context), hdim]
            return tf.RaggedTensor.from_tensor(global_embeddings, ragged_rank=1)


class QueryKeyMul(tf.keras.layers.Layer):
    """Ragged Query-Key multiplication

    Calculate inner product of ragged key and query tensors.
    It is one of the most computationally intensive in the network and hence has been heavily
    optimized.

    :param method: One of multiple methods to computing the query and key.
    :type tf: str
    :param name: Layer name
    :type tf: str
    """

    def __init__(
        self,
        method : str = "broadcast_ragged",
        name : str = "query_key_mul",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.method = method

    @staticmethod
    def _mul_map_fn(
        queries : tf.RaggedTensor,  # shape: # [batch, None(args), hdim]
        keys : tf.RaggedTensor,  # shape: # [batch, None(context), hdim]
    ) -> tf.RaggedTensor:  # shape: # [batch, None(args), None(context)]
        """
        Compute the inner product via tf.map_fun.  Won't run on a GPU.
        """
        @tf.function
        def linear_op(qk):
            x = tf.einsum("ij, kj -> ik", qk[0], qk[1])
            x = tf.RaggedTensor.from_tensor(x)
            return x
        return tf.map_fn(
            linear_op,
            elems=[queries, keys],
            fn_output_signature=tf.RaggedTensorSpec(shape=[None, None])
        )
    
    @staticmethod
    def _mul_broadcast_ragged(
        queries : tf.RaggedTensor,  # shape: # [batch, None(args), hdim]
        keys : tf.RaggedTensor,  # shape: # [batch, None(context), hdim]
    ) -> tf.RaggedTensor:  # shape: # [batch, None(args), None(context)]
        """
        Compute the inner product by broadcasting the query and key tensors.
        
        Both are broadcast to shape [batch, None(args), None(context), hdim]
        before multiplying.  Special care is taken to avoid operations which
        take place too long on the CPU.
        """
        # broadcast ragged keys to be shape [batch, None(args), None(context), hdim]
        # these lines are equivalent to tf.gather(keys, queries.value_rowids())
        # but faster since the ragged gather is only done on the indices and
        # not the key vectors.  Ragged gathers take place on the CPU.
        key_ix = keys.with_values(tf.range(tf.shape(keys.values)[0]))  # [batch, None(context)]
        keys_values_ixs = tf.gather(key_ix, queries.value_rowids())  # [batch-args, None(context)]
        keys_values = tf.gather(keys.values, keys_values_ixs)  # [batch-args, None(context), hdim]
        
        # broadcast ragged queries to shape [batch, None(args), None(context), hdim]
        queries_values_values = tf.gather(queries.values, keys_values.value_rowids())  # [batch-args-context, hdim]

        # multiply key and query
        # can just multiply the ragged values arrays since they have same shape
        logits_values_values = tf.einsum("ij,ij->i", keys_values.values, queries_values_values)  # [batch-args-context]
        logits_values = keys_values.with_values(logits_values_values)  # [batch-args, None(context)]
        logits = queries.with_values(logits_values)  # [batch, None(args), None(context)]
        return logits
    
    @staticmethod
    def _mul_ragged_to_dense_to_ragged(
        queries : tf.RaggedTensor,  # shape: # [batch, None(args), hdim]
        keys : tf.RaggedTensor,  # shape: # [batch, None(context), hdim]
    ) -> tf.RaggedTensor:  # shape: # [batch, None(args), None(context)]
        """
        Compute the inner product by converting ragged arrays to tensors first.
        
        This can be computationally intensive when some batch elements
        have large numbers of arguments
        """
        # convert arrays to non-ragged
        # these two line are the most computationally intensive in this method
        # RaggedTensor.to_tensor seems to take place on the CPU.
        queries_dense = queries.to_tensor()    # [batch, max(args), hdim]
        keys_dense = keys.to_tensor()         # [batch, max(context), hdim]
        
        # multiply
        logits_dense = tf.einsum("ijl,ikl->ijk", queries_dense, keys_dense)  # [batch, max(args), max(context)]

        # convert back to ragged
        logits_part_ragged = tf.RaggedTensor.from_tensor(logits_dense, lengths=queries.row_lengths())  # [batch, None(args), max(context)]
        lengths = queries.with_values(tf.gather(keys.row_lengths(), queries.value_rowids()))  # [batch, None(args)]
        logits_values = tf.RaggedTensor.from_tensor(logits_part_ragged.values, lengths=lengths.values)
        logits = logits_part_ragged.with_values(logits_values)
        return logits

    @staticmethod
    def _mul_singleton_batch(
        queries : tf.RaggedTensor,  # shape: # [1, None(args), hdim]
        keys : tf.RaggedTensor,  # shape: # [1, None(context), hdim]
    ) -> tf.RaggedTensor:  # shape: # [1, None(args), None(context)]
        """
        If outer batch dim is 1 (which is often the case during inference),
        then compute inner product on value tensors directly.
        """
        tf.assert_equal(tf.shape(keys)[0], 1)

        query_values = queries.values  # [args, hdim]
        key_values = keys.values  # [context, hdim]
#
        # multiply inner values
        logits_tensor = tf.einsum("ik,jk->ij", query_values, key_values)  # [args, context]
        logits_tensor = tf.expand_dims(logits_tensor, axis=0)  # [1, args, context]
        
        # reshape back into a ragged tensor
        return tf.RaggedTensor.from_tensor(logits_tensor, ragged_rank=2)  # [1, None(args), None(context)]
    
    def call(
        self, 
        queries: tf.RaggedTensor, # [batch, None(args), hdim]
        keys: tf.RaggedTensor, # [batch, None(context), hdim]
        training=False
    ) -> tf.RaggedTensor: # [batch, None(args), None(context)]
        """Compute the ragged inner product of ragged queries and keys.

        :param queries: Queries associated with argument positions. Shape [batch, None(args), hdim]
        :type queries: tf.RaggedTensor
        :param keys: Keys associated with local or global context. Shape [batch, None(context), hdim]
        :type keys: tf.RaggedTensor
        :return: Logits.  Shape [batch, None(args), None(context)]
        :rtype: tf.RaggedTensor
        """
        # TODO(jrute): Clean up row split types to be consistent across model code
        queries = queries.with_row_splits_dtype(tf.int64)
        keys = keys.with_row_splits_dtype(tf.int64)
        if tf.shape(queries)[0] == 1:
            return self._mul_singleton_batch(queries, keys)
        elif self.method == "map_fn":
            return self._mul_map_fn(queries, keys)
        elif self.method == "broadcast_ragged":
            return self._mul_broadcast_ragged(queries, keys)
        elif self.method == "ragged_to_dense_to_ragged":
            return self._mul_ragged_to_dense_to_ragged(queries, keys)
        else:
            raise Exception(f"Unsupported multiplication method: {self.method}")

class QueryKeyMulGlobal(tf.keras.layers.Layer):
    """Layer for computing global logits.

    Take inner product of tensors.  If using cosine similarity, 
    then unit normalize before the inner product and divide by a
    learned temperature tensor.  (The temperature parameter is 
    because cosine similarity otherwise only produces logits between
    -1 and 1.)

    :param name: layer name
    :type name: str
    :param cosine_similarity: Whether to use cosine similarlity with learned temperature parameter.
    :type name: str
    """
    def __init__(
        self,
        name="query_key_mul_global",
        cosine_similarity: bool = True,
        temp: Optional[tf.Variable] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._cosine_similarity = cosine_similarity
        if self._cosine_similarity:
            # since cosine similarity is between -1.0 and 1.0
            # we add a learned temperature parameter
            # so logits can be in a wider or narrower range -1/temp to 1/temp

            assert temp is not None
            self._temp = temp
        self.query_key_mul = QueryKeyMul()

    def unit_normalize_tensor(self, x: tf.Tensor) ->  tf.Tensor:
        x_norm = tf.norm(x, axis=-1, keepdims=True)
        return x / x_norm

    def unit_normalize_ragged(self, rt: tf.RaggedTensor) -> tf.RaggedTensor:
        return tf.ragged.map_flat_values(self.unit_normalize_tensor, rt)

    def call(
        self, 
        queries: tf.RaggedTensor, # [batch, None(args), hdim]
        keys: tf.RaggedTensor, # [batch, context, hdim]
        training=False
    ) -> tf.Tensor: # [batch, max(args), context]        
        if self._cosine_similarity:
            # normalize embeddings before taking inner product
            if training:
                # the key embeddings are unit normalized already,
                # but this ensures they stay normalized during training
                keys = self.unit_normalize_ragged(keys) 
            queries = self.unit_normalize_ragged(queries)
            
        logits = self.query_key_mul(queries, keys)

        if self._cosine_similarity:
            logits = logits / self._temp
        
        return logits


class LocalArgumentPrediction(TacticPrediction):
    """
    Wrapper for the base tactic plus local argument prediction tasks.
    """
    LOCAL_ARGUMENTS_LOGITS = 'local_arguments_logits'

    ARGUMENTS_SEQ_ACCURACY = 'arguments_seq_accuracy'
    STRICT_ACCURACY = 'strict_accuracy'

    def __init__(self,
                 arguments_head_type: str,
                 arguments_head_config: dict,
                 arguments_loss_coefficient: float = 1.0,
                 **kwargs
                 ):
        """
        @param arguments_head_type: the type of arguments head to use
        @param arguments_head_config: the hyperparameters to be used for the arguments head
        @param arguments_loss_coefficient: the weight of the loss term for the arguments (base tactic loss has weight 1)
        @param kwargs: other arguments are  passed to the TacticPrediction constructor
        """
        super().__init__(**kwargs)
        self._arguments_head_type = arguments_head_type
        self._arguments_loss_coefficient = arguments_loss_coefficient

        # create arguments head
        arguments_head_constructor = get_arguments_head_constructor(arguments_head_type)
        self.arguments_head = arguments_head_constructor(hidden_size=self._hidden_size,
                                                         tactic_embedding_size=self._tactic_embedding_size,
                                                         **arguments_head_config)

        # we use trivial lambda layers to appropriately rename outputs
        self.local_arguments_logits_output = tf.keras.layers.Lambda(lambda x: x, name=self.LOCAL_ARGUMENTS_LOGITS)

        # update checkpoint with new layers
        self.checkpoint.arguments_head = self.arguments_head

    def get_config(self):
        config = super().get_config()

        arguments_head_config = self.arguments_head.get_config()
        arguments_head_config.pop('hidden_size')
        arguments_head_config.pop('tactic_embedding_size')

        config.update({
            'prediction_task_type': LOCAL_ARGUMENT_PREDICTION,
            'arguments_head_type': self._arguments_head_type,
            'arguments_head_config': arguments_head_config,
            'arguments_loss_coefficient': self._arguments_loss_coefficient
        })
        return config

    @staticmethod
    def _local_context_hidden(
        scalar_proofstate_graph: tfgnn.GraphTensor,
        hidden_graph: tfgnn.GraphTensor,
    ) -> tf.RaggedTensor:  # [batch_size, None(local_context), hidden_size]
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
        return tf.gather(hidden_graph.node_sets['node']['hidden_state'],
                                               local_context_ids).with_row_splits_dtype(tf.int64)

    def _hidden_state_sequences(
            self,
            hidden_graph: tfgnn.GraphTensor,  # [batch]
            tactic: tf.Tensor,  # [batch]
        ) -> tf.RaggedTensor:  # [batch, None(args), hdim]
        # [batch]
        num_arguments = tf.gather(tf.constant(self._graph_constants.tactic_index_to_numargs, dtype=tf.int64), tactic)
        # [batch, hdim]
        hidden_state = hidden_graph.context["hidden_state"]
        # [batch, tactic_hdim]
        tactic_embedding = self.tactic_embedding(tactic)
        # [batch, None(args), hdim]
        return self.arguments_head((hidden_state, tactic_embedding, num_arguments))

    def _hidden_state_sequences_inference(
        self, 
        hidden_graph: tfgnn.GraphTensor,  # [batch] 
        tactic: tf.Tensor,  # [batch, tactic_expand_bound]
    ) -> tf.RaggedTensor:  # [batch*tactic_expand_bound, None(args), hdim]
        batch_size = tf.shape(tactic)[0]
        tactic_expand_bound = tf.shape(tactic)[1]

        tactic = tf.reshape(tactic, shape=[batch_size*tactic_expand_bound])  # [batch*tactic_expand_bound]
        
        # [batch*tactic_expand_bound]
        num_arguments = tf.gather(tf.constant(self._graph_constants.tactic_index_to_numargs, dtype=tf.int64), tactic)
        
        hidden_state = hidden_graph.context["hidden_state"]  # [batch, hdim]
        hidden_state = tf.expand_dims(hidden_state, axis=1)  # [batch, 1, hdim] 
        hidden_state = tf.tile(hidden_state, multiples=[1, tactic_expand_bound, 1])  # [batch, tactic_expand_bound, hdim]
        hidden_state = tf.reshape(hidden_state, shape=[batch_size*tactic_expand_bound, tf.shape(hidden_state)[2]])  # [batch*tactic_expand_bound, hdim]

        # [batch*tactic_expand_bound, tactic_hdim]
        tactic_embedding = self.tactic_embedding(tactic)
        # [batch*tactic_expand_bound, None(args), hdim]
        return self.arguments_head((hidden_state, tactic_embedding, num_arguments))

    def create_train_model(self) -> tf.keras.Model:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    def create_inference_model(self, tactic_expand_bound: int, graph_constants: GraphConstants) -> tf.keras.Model:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    @staticmethod
    def create_input_output(graph_tensor: tfgnn.GraphTensor) -> Tuple[tfgnn.GraphTensor, Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    @staticmethod
    def loss() -> Dict[str, tf.keras.losses.Loss]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    @staticmethod
    def metrics() -> Dict[str, Iterable[tf.keras.metrics.Metric]]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    def loss_weights(self) -> Dict[str, float]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        raise NotImplemented("Use GlobalArgumentPrediction instead")


class GlobalArgumentPrediction(LocalArgumentPrediction):
    """
    Wrapper for the base tactic plus local and global argument prediction tasks.
    """
    GLOBAL_ARGUMENTS_LOGITS = 'global_arguments_logits'

    def __init__(self,
                 dynamic_global_context: bool = False,
                 global_cosine_similarity: bool = False,
                 sum_loss_over_tactic: bool = False,
                 **kwargs):
        """
        @param dynamic_global_context: whether to restrict the global context to available definitions only
        @param global_cosine_similarity: whether to use cosine similarity to calculate global arg logits
        @param sum_loss_over_tactic: whether to sum the argument losses over the tactic
        @param kwargs: arguments to be passed to the LocalArgumentPrediction constructor
        """
        super().__init__(**kwargs)
        self._dynamic_global_context = dynamic_global_context
        self._sum_loss_over_tactic = sum_loss_over_tactic
        self._global_cosine_similarity = global_cosine_similarity
        
        self.global_arguments_head = tf.keras.layers.Dense(self._hidden_size)
        self.local_arguments_head = tf.keras.layers.Dense(self._hidden_size)

        # create a layer to extract logits from the node label embeddings
        self.global_arguments_logits = LogitsFromEmbeddings(
            embedding_matrix=self.graph_embedding.get_node_embeddings(),
            cosine_similarity=self._global_cosine_similarity
        )
        self.global_embeddings = GlobalEmbeddings(
            global_embeddings_layer=self.global_arguments_logits,
            dynamic_global_context=self._dynamic_global_context
        )
        self.global_logits = QueryKeyMulGlobal(
            cosine_similarity=self._global_cosine_similarity,
            temp=self.global_arguments_logits._temp if self._global_cosine_similarity else None
        )

        # we use trivial lambda layers to appropriately rename outputs
        self.local_arguments_logits_output = tf.keras.layers.Lambda(lambda x: x, name=self.LOCAL_ARGUMENTS_LOGITS)
        self.global_arguments_logits_output = tf.keras.layers.Lambda(lambda x: x, name=self.GLOBAL_ARGUMENTS_LOGITS)

        # update checkpoint with new layers
        self.checkpoint.local_arguments_head = self.local_arguments_head
        self.checkpoint.global_arguments_head = self.global_arguments_head
        self.checkpoint.global_arguments_logits = self.global_arguments_logits

    def get_config(self):
        config = super().get_config()

        config.update({
            'prediction_task_type': GLOBAL_ARGUMENT_PREDICTION,
            'dynamic_global_context': self._dynamic_global_context,
            'global_cosine_similarity': self._global_cosine_similarity,
            'sum_loss_over_tactic': self._sum_loss_over_tactic
        })
        return config

    @staticmethod
    def _global_arguments_logits_mask(scalar_proofstate_graph: tfgnn.GraphTensor,
                                      global_context_size: int) -> tf.Tensor:
        """
        @param scalar_proofstate_graph: the proofstate graph containing ids for the available global context definitions
        @param global_context_size: the size of the full global context
        @return: a mask for logits of the global context, taking into account the definitions that are actually available
        """
        global_context_ids = scalar_proofstate_graph.context['global_context_ids']

        indices = tf.stack([
            tf.cast(global_context_ids.value_rowids(), tf.int64),
            global_context_ids.values
        ], axis = -1)
        updates = tf.ones_like(global_context_ids.values, dtype=tf.float32)
        shape = [global_context_ids.nrows(), global_context_size]

        return tf.math.log(tf.scatter_nd(indices, updates, shape))  # [batch_size, global_cxt]

    @staticmethod
    def _normalize_logits(local_arguments_logits: tf.Tensor, global_arguments_logits: tf.Tensor) -> Tuple[
        tf.Tensor, tf.Tensor]:
        """
        Normalize local and global arguments logits making sure the log_softmax is numerically stable.
        """
        local_arguments_max_logit = tf.reduce_max(local_arguments_logits, axis=-1)
        global_arguments_max_logit = tf.reduce_max(global_arguments_logits, axis=-1)
        arguments_max_logit = tf.reduce_max(tf.stack([local_arguments_max_logit, global_arguments_max_logit], axis=-1),
                                            axis=-1, keepdims=True)
        local_arguments_logits -= arguments_max_logit
        global_arguments_logits -= arguments_max_logit
        
        local_arguments_logits_norm = tf.reduce_sum(tf.exp(local_arguments_logits), axis=-1, keepdims=True)
        global_arguments_logits_norm = tf.reduce_sum(tf.exp(global_arguments_logits), axis=-1, keepdims=True)
        norm = -tf.math.log(local_arguments_logits_norm + global_arguments_logits_norm)
        return local_arguments_logits + norm, global_arguments_logits + norm

    def _ragged_log_softmax_double(
        self,
        logits0: tf.RaggedTensor,  # [batch_dim, (ragged_dim)]
        logits1: tf.RaggedTensor,  # [batch_dim, (ragged_dim)]
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:  # [batch_dim, (ragged_dim)]

        # subtract off max value to make stable 
        max_logits0 = tf.expand_dims(tf.reduce_max(logits0, axis=-1), -1)  # [total(args), 1]
        max_logits1 = tf.expand_dims(tf.reduce_max(logits1, axis=-1), -1)  # [total(args), 1]
        max_logits = tf.maximum(max_logits0, max_logits1)
        logits0 = logits0 - max_logits  # [batch_dim, (ragged_dim)]
        logits1 = logits1 - max_logits  # [batch_dim, (ragged_dim)]

        # elementwise exp
        exp_logits0 = tf.math.exp(logits0)  # [batch_dim, (ragged_dim)]
        exp_logits1 = tf.math.exp(logits1)  # [batch_dim, (ragged_dim)]
        sum_exp0 = tf.expand_dims(tf.reduce_sum(exp_logits0, axis=-1), -1)  # [batch_dim, 1]
        sum_exp1 = tf.expand_dims(tf.reduce_sum(exp_logits1, axis=-1), -1)  # [batch_dim, 1]
        sum_exp = sum_exp0 + sum_exp1  # [batch_dim, 1]

        # divide in log space to get log probability
        log_sum = tf.math.log(sum_exp)  # [batch_dim, 1]
        return (logits0 - log_sum, logits1 - log_sum) # [batch_dim, (ragged_dim)]

    def _log_softmax_logits(
        self,
        local_arguments_logits: tf.RaggedTensor,  # [batch, None(args), None(context)]
        global_arguments_logits: tf.RaggedTensor,  # [batch, None(args), None(context)]
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:  # ([batch, None(args), None(context)], [batch, None(args), None(context)])
        """
        Normalize local and global arguments logits making sure the log_softmax is numerically stable.
        """
        # perform softmax on inner ragged logits
        # [total(args), None(local_context)], # [total(args), None(global_context)]
        local_logits_values, global_logits_values = \
            self._ragged_log_softmax_double(local_arguments_logits.values, global_arguments_logits.values)
        
        # shape back into double ragged tensors
        # [total(args), None(local_context)]
        local_logits = local_arguments_logits.with_values(local_logits_values)
        # [total(args), None(global_context)]
        global_logits = global_arguments_logits.with_values(global_logits_values)

        return local_logits, global_logits

    def create_train_model(self) -> tf.keras.Model:
        """
        Combines a GNN component with a tactic head and an arguments head to produce an end-to-end model for the
        global argument prediction task. The resulting model is for training purposes, and produces tactic logits
        and argument logits (for each local context node / global context id per argument position).

        @return: a keras model consuming graphs and producing tactic and local/global arguments logits
        """
        proofstate_graph = tf.keras.layers.Input(type_spec=batch_graph_spec(proofstate_graph_spec),
                                                 name=self.PROOFSTATE_GRAPH)
        scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()

        tactic_logits, hidden_graph = self._tactic_logits_and_hidden_graph(scalar_proofstate_graph)

        hidden_state_sequences = self._hidden_state_sequences(hidden_graph=hidden_graph,
                                                              tactic=scalar_proofstate_graph.context['tactic'])
        # [batch, None(args), hdim]
        local_hidden_state_sequences = self.local_arguments_head(hidden_state_sequences)
        # [batch, None(context), hdim]
        local_context_hidden = self._local_context_hidden(
            scalar_proofstate_graph,
            hidden_graph,
        )
        # [batch_size, None(args), None(context)]
        local_arguments_logits = QueryKeyMul()(local_hidden_state_sequences, local_context_hidden)
        
        # [batch, None(args), hdim]
        global_hidden_state_sequences = self.global_arguments_head(hidden_state_sequences)
        # [batch, None(context), hdim]
        global_embeddings = self.global_embeddings(scalar_proofstate_graph.context['global_context_ids'])
        # [batch, None(args), None(context)]
        global_arguments_logits = self.global_logits(queries=global_hidden_state_sequences, keys=global_embeddings)
        
        normalized_local_arguments_logits, normalized_global_arguments_logits = self._log_softmax_logits(local_arguments_logits=local_arguments_logits, global_arguments_logits=global_arguments_logits)

        local_arguments_logits_output = self.local_arguments_logits_output(normalized_local_arguments_logits)
        global_arguments_logits_output = self.global_arguments_logits_output(normalized_global_arguments_logits)
        
        return GlobalArgumentModel(inputs=proofstate_graph,
                                   outputs={self.TACTIC_LOGITS: tactic_logits,
                                            self.LOCAL_ARGUMENTS_LOGITS: local_arguments_logits_output,
                                            self.GLOBAL_ARGUMENTS_LOGITS: global_arguments_logits_output})

    def create_inference_model(self, tactic_expand_bound: int, graph_constants: GraphConstants) -> tf.keras.Model:
        """
        Combines a GNN component with a tactic head and an arguments head to produce an end-to-end model for the
        global argument prediction task. The resulting model is for inference purposes, and produces tactics, their logits
        and argument logits (for each local context node / global context id per argument position).

        @warning: we do not use the GraphConstants saved in this task since during inference these may not be up-to-date
        @param tactic_expand_bound: the number of base tactic predictions to produce for each proofstate
        @param graph_constants: the graph constants to use during inference (with a possibly updated global context)
        @return: a keras model consuming graphs and producing tactic and local/global arguments logits
        """
        proofstate_graph = tf.keras.layers.Input(type_spec=batch_graph_spec(proofstate_graph_spec),
                                                 name=self.PROOFSTATE_GRAPH)
        scalar_proofstate_graph = proofstate_graph.merge_batch_to_components()

        # [tactic_num]
        tactic_mask = tf.keras.Input(shape=(graph_constants.tactic_num,), dtype=tf.bool, name=self.TACTIC_MASK)

        # [batch_size, tactic_num]
        tactic_logits, hidden_graph = self._tactic_logits_and_hidden_graph(scalar_proofstate_graph)

        # [tactic_num, ]
        no_argument_tactics_mask = tf.constant(graph_constants.tactic_index_to_numargs, dtype = tf.int64) == 0
        all_tactics_mask = tf.ones(graph_constants.tactic_num, dtype=tf.bool)

        # [batch_size, ]
        no_local_context_proofstates = scalar_proofstate_graph.context['local_context_ids'].row_lengths() == 0
        no_global_context_proofstates = scalar_proofstate_graph.context['global_context_ids'].row_lengths() == 0
        no_context_proofstates = no_local_context_proofstates & no_global_context_proofstates

        # [batch_size, tactic_num]
        proofstate_tactic_mask = tf.where(tf.expand_dims(no_context_proofstates, axis=-1),
                                          tf.expand_dims(no_argument_tactics_mask, axis=0),
                                          tf.expand_dims(all_tactics_mask, axis=0))
        # [batch_size, top_k_tactics], [batch_size, top_k_tactics] 
        tactic, top_k_values = self._top_k_tactics(tactic_logits=tactic_logits,
                                                          tactic_mask=proofstate_tactic_mask & tactic_mask,
                                                          tactic_expand_bound=tactic_expand_bound)

        # [batch_size, None(context), hdim]
        local_context_hidden = self._local_context_hidden(
            scalar_proofstate_graph,
            hidden_graph,
        )
        # [batch_size, None(context), hdim]
        global_embeddings = self.global_embeddings(
            scalar_proofstate_graph.context['global_context_ids'],
            inference=True
        )
        
        batch_size = tf.shape(tactic)[0]
        top_k_tactics_cnt = tf.shape(tactic)[1]  # this may be less than tactic_expand_bound

        # [batch*top_k_tactics, None(args), hdim]
        hidden_state_sequences = self._hidden_state_sequences_inference(hidden_graph=hidden_graph, tactic=tactic)
        # [batch*top_k_tactics]
        tactic_arg_cnt = hidden_state_sequences.row_lengths()
        # [batch]
        batch_arg_cnt = tf.reduce_sum(tf.reshape(tactic_arg_cnt, shape=[batch_size, top_k_tactics_cnt]), axis=-1)
        
        # [batch*top_k_tactics, None(args), hdim]
        local_hidden_state_sequences = self.local_arguments_head(hidden_state_sequences)
        # [batch, None(top_k_tactics*args), hdim]
        local_hidden_state_sequences = tf.RaggedTensor.from_row_lengths(
            values=local_hidden_state_sequences.values,  # [batch-tactic-arg, hdim]
            row_lengths=batch_arg_cnt
        )
        # [batch, None(top_k_tactics*args), None(context)]
        local_arguments_logits = QueryKeyMul()(local_hidden_state_sequences, local_context_hidden)
        # [batch*top_k_tactics, None(args), None(context)]
        local_arguments_logits = tf.RaggedTensor.from_row_lengths(
            values=local_arguments_logits.values,  # [batch-tactic-arg, None(context)]
            row_lengths=tactic_arg_cnt
        )
        
        # [batch*top_k_tactics, None(args), hdim]
        global_hidden_state_sequences = self.global_arguments_head(hidden_state_sequences)
        # [batch, None(top_k_tactics*args), hdim]
        global_hidden_state_sequences = tf.RaggedTensor.from_row_lengths(
            values=global_hidden_state_sequences.values,  # [batch-tactic-arg, None(context)]
            row_lengths=batch_arg_cnt
        )
        # [batch, None(top_k_tactics*args), None(context)]
        global_arguments_logits = self.global_logits(queries=global_hidden_state_sequences, keys=global_embeddings)
        # [batch*top_k_tactics, None(args), None(context)]
        global_arguments_logits = tf.RaggedTensor.from_row_lengths(
            values=global_arguments_logits.values,  # [batch-tactic-arg, None(context)]
            row_lengths=tactic_arg_cnt
        )

        return tf.keras.Model(inputs={self.PROOFSTATE_GRAPH: proofstate_graph, self.TACTIC_MASK: tactic_mask},
                              outputs={self.TACTIC: tactic,  # [batch, top_k_tactics]
                                       self.TACTIC_LOGITS: top_k_values,  # [batch, top_k_tactics]
                                       self.LOCAL_ARGUMENTS_LOGITS: local_arguments_logits,  # [batch*top_k_tactics, None(args), None(context)]
                                       self.GLOBAL_ARGUMENTS_LOGITS: global_arguments_logits})  # [batch*top_k_tactics, None(args), None(context)]

    @staticmethod
    def create_input_output(graph_tensor: tfgnn.GraphTensor) -> Tuple[
        tfgnn.GraphTensor, Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]:
        outputs = {GlobalArgumentPrediction.TACTIC_LOGITS: graph_tensor.context['tactic'],
                   GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: graph_tensor.context['local_arguments'],
                   GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: graph_tensor.context['global_arguments']}
        return graph_tensor, outputs

    def loss(self) -> Dict[str, tf.keras.losses.Loss]:
        return {GlobalArgumentPrediction.TACTIC_LOGITS: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: ArgumentSparseCategoricalCrossentropy(sum_loss_over_tactic=self._sum_loss_over_tactic),
                GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: ArgumentSparseCategoricalCrossentropy(sum_loss_over_tactic=self._sum_loss_over_tactic)}

    def loss_weights(self) -> Dict[str, float]:
        return {GlobalArgumentPrediction.TACTIC_LOGITS: 1.0,
                GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: self._arguments_loss_coefficient,
                GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: self._arguments_loss_coefficient}

    def metrics(self) -> Dict[str, List[tf.keras.metrics.Metric]]:
        return {GlobalArgumentPrediction.TACTIC_LOGITS: [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
                GlobalArgumentPrediction.LOCAL_ARGUMENTS_LOGITS: [ArgumentSparseCategoricalAccuracy(name='accuracy')],
                GlobalArgumentPrediction.GLOBAL_ARGUMENTS_LOGITS: [ArgumentSparseCategoricalAccuracy(name='accuracy')]}

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [MixedMetricsCallback()]


class DefinitionTask(tf.keras.layers.Layer):
    """
    A layer to compute definition embeddings from definition cluster graphs. The input graphs should:
        - be scalar graphs
        - follow the schema for `vectorized_definition_graph_spec` in `graph2tac.tfgnn.graph_schema`
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

        self.definition_head = definition_head_constructor(
            hidden_size=graph_embedding._hidden_size,
            unit_normalize=graph_embedding._unit_normalize,
            **definition_head_config
        )

    def get_checkpoint(self) -> tf.train.Checkpoint:
        """
        @return: a checkpoint tracking any **new** variables created by this layer
        """
        return tf.train.Checkpoint(definition_head=self.definition_head)

    def get_config(self):
        config = super().get_config()

        definition_head_config = self.definition_head.get_config()
        definition_head_config.pop('hidden_size')  # use the setting from graph embedding
        definition_head_config.pop('unit_normalize')  # use the setting from graph embedding

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
        definition_name_vectors = scalar_definition_graph.context['definition_name_vectors']
        definition_body_embeddings = self.definition_head((hidden_graph, num_definitions, definition_name_vectors), training=training)
        return definition_body_embeddings


def get_prediction_task_constructor(prediction_task_type: str
                                    ) -> Callable[..., Union[TacticPrediction, LocalArgumentPrediction, GlobalArgumentPrediction]]:
    if prediction_task_type == BASE_TACTIC_PREDICTION:
        return TacticPrediction
    elif prediction_task_type == LOCAL_ARGUMENT_PREDICTION:
        return LocalArgumentPrediction
    elif prediction_task_type == GLOBAL_ARGUMENT_PREDICTION:
        return GlobalArgumentPrediction
    else:
        raise ValueError(f'{prediction_task_type} is not a valid prediction task type')
