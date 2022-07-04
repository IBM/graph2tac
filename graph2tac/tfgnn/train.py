from typing import Optional, Dict, Tuple

import yaml
import argparse
import atexit
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path
from math import ceil

from graph2tac.tfgnn.dataset import Dataset, DataServerDataset, TFRecordDataset
from graph2tac.tfgnn.tasks import PredictionTask, DefinitionTask, DefinitionMeanSquaredError
from graph2tac.tfgnn.graph_schema import definition_graph_spec, batch_graph_spec
from graph2tac.tfgnn.q_checkpoint_manager import QCheckpointManager


class Trainer:
    """
    This class encapsulates the training logic for the various tasks we can define.
    """

    DEFINITION_EMBEDDING = 'definition_embedding'

    def __init__(self,
                 dataset: Dataset,
                 prediction_task: PredictionTask,
                 optimizer_type: str,
                 optimizer_config: Optional[dict] = None,
                 definition_task: Optional[DefinitionTask] = None,
                 definition_loss_coefficient: Optional[float] = None,
                 l2_regularization_coefficient: Optional[float] = None,
                 log_dir: Optional[Path] = None,
                 max_to_keep: int = 1,
                 keep_checkpoint_every_n_hours: Optional[int] = None,
                 qsaving: Optional[float] = None):
        """
        @param dataset:
        @param prediction_task:
        @param optimizer_type:
        @param optimizer_config:
        @param definition_task:
        @param definition_loss_coefficient:
        @param l2_regularization_coefficient:
        @param log_dir:
        @param max_to_keep:
        @param keep_checkpoint_every_n_hours:
        @param qsaving:
        """
        # dataset
        self.dataset = dataset

        # prediction task
        self.prediction_task = prediction_task

        # optimizer
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config if optimizer_config is not None else {}
        self.optimizer = tf.keras.optimizers.get(optimizer_type).from_config(self.optimizer_config)

        # definition task
        self.definition_task = definition_task
        self.definition_loss_coefficient = definition_loss_coefficient

        # regularization
        self.l2_regularization_coefficient = l2_regularization_coefficient

        # logging
        self.log_dir = log_dir
        self.max_to_keep = max_to_keep
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self.qsaving = qsaving

        self.trained_epochs = tf.Variable(initial_value=0, trainable=False)
        self.checkpoint = tf.train.Checkpoint(prediction_task=prediction_task.checkpoint,
                                              optimizer=self.optimizer,
                                              trained_epochs=self.trained_epochs)
        if definition_task is not None:
            self.checkpoint.definition_task = definition_task

        with prediction_task.prediction_model.distribute_strategy.scope():
            if self.definition_task is not None:
                self.train_model = self._create_train_model()
            else:
                self.train_model = self.prediction_task.prediction_model

            if self.l2_regularization_coefficient is not None:
                self.train_model.add_loss(self._l2_regularization)

    def get_config(self):
        config = {
            'dataset': self.dataset.get_config(),
            'prediction_task': self.prediction_task.get_config(),
            'optimizer_type': self.optimizer_type,
            'optimizer_config': self.optimizer.get_config(),
            'definition_task': self.definition_task.get_config() if self.definition_task is not None else None,
            'definition_loss_coefficient': self.definition_loss_coefficient,
            'l2_regularization_coefficient': self.l2_regularization_coefficient,
        }
        return config

    @classmethod
    def from_yaml_config(cls,
                         dataset: Dataset,
                         trainer_config: Path,
                         prediction_task_config: Path,
                         definition_task_config: Optional[Path] = None,
                         log_dir: Optional[Path] = None
                         ) -> "Trainer":
        with trainer_config.open() as yaml_file:
            trainer_config = yaml.load(yaml_file, Loader=yaml.SafeLoader)

        prediction_task = PredictionTask.from_yaml_config(graph_constants=dataset.graph_constants(),
                                                          yaml_filepath=prediction_task_config)

        if definition_task_config is not None:
            definition_task = DefinitionTask.from_yaml_config(graph_embedding=prediction_task.graph_embedding,
                                                              gnn=prediction_task.gnn,
                                                              yaml_filepath=definition_task_config)
        else:
            definition_task = None

        return cls(dataset=dataset,
                   prediction_task=prediction_task,
                   definition_task=definition_task,
                   log_dir=log_dir,
                   **trainer_config)

    def _callbacks(self):
        callbacks = self.prediction_task.callbacks()

        trained_epochs_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.trained_epochs.assign(epoch + 1))
        callbacks.append(trained_epochs_callback)

        if self.log_dir is not None:
            # create directory for logs
            self.log_dir.mkdir(exist_ok=True)

            # save configuration
            config_dir = self.log_dir / 'config'
            config_dir.mkdir(exist_ok=True)

            graph_constants_filepath = config_dir / 'graph_constants.yaml'
            graph_constants_filepath.write_text(yaml.dump(self.dataset._graph_constants.__dict__))

            dataset_yaml_filepath = config_dir / 'dataset.yaml'
            dataset_yaml_filepath.write_text(yaml.dump(self.dataset.get_config()))

            prediction_yaml_filepath = config_dir / 'prediction.yaml'
            prediction_yaml_filepath.write_text(yaml.dump(self.prediction_task.get_config()))

            if self.definition_task is not None:
                definition_yaml_filepath = config_dir / 'definition.yaml'
                definition_yaml_filepath.write_text(yaml.dump(self.definition_task.get_config()))

            # checkpointing callback
            checkpoint_path = self.log_dir / 'ckpt'
            checkpoint_manager = QCheckpointManager(self.checkpoint,
                                                    checkpoint_path,
                                                    max_to_keep=self.max_to_keep,
                                                    keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours,
                                                    qsaving=self.qsaving)
            try:
                checkpoint_restored = checkpoint_manager.restore_or_initialize()
            except tf.errors.NotFoundError as error:
                print(f'unable to restore checkpoint from {checkpoint_path}!')
                raise error
            else:
                if checkpoint_restored is not None:
                    print(f'Restored checkpoint {checkpoint_restored}!')

            save_checkpoint_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: checkpoint_manager.save())
            callbacks.append(save_checkpoint_callback)

            callbacks.append(ExtendedTensorBoard(trainer=self,
                                                 log_dir=self.log_dir,
                                                 write_graph=False,
                                                 write_steps_per_second=True,
                                                 update_freq='epoch'))
        return callbacks

    def _input_output_mixing(self, prediction_task_io, definition_graph):
        proofstate_graph, outputs = prediction_task_io

        outputs.update({self.DEFINITION_EMBEDDING: tf.zeros(shape=(1, self.prediction_task._hidden_size))})

        return (proofstate_graph, definition_graph), outputs

    def _prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        # create input-output pairs
        dataset = dataset.map(self.prediction_task.create_input_output)

        # add definitions if necessary
        if self.definition_task is not None:
            count = ceil(self.dataset.total_proofstates()/self.dataset.total_definitions())
            definitions = self.dataset.definitions(shuffle=False)
            definitions = definitions.repeat(count).shuffle(self.dataset.SHUFFLE_BUFFER_SIZE)
            dataset = tf.data.Dataset.zip(datasets=(dataset, definitions))
            dataset = dataset.map(self._input_output_mixing)

        return dataset

    def _l2_regularization(self):
        return tf.reduce_sum([tf.keras.regularizers.l2(l2=self.l2_regularization_coefficient)(weight) for weight in self.train_model.trainable_weights])

    @staticmethod
    def _mask_defined_labels(definition_graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
        num_definitions = tf.cast(definition_graph.context['num_definitions'], dtype=tf.int32)
        is_defined = tf.ragged.range(tf.squeeze(definition_graph.node_sets['node'].sizes, axis=-1)) < num_definitions
        masked_node_labels = tf.where(is_defined.with_row_splits_dtype(tf.int32),
                                      tf.constant(-1, dtype=tf.int64),  # TODO: This fails on CPU
                                      definition_graph.node_sets['node']['node_label'])
        return definition_graph.replace_features(node_sets={'node': {'node_label': masked_node_labels}})

    @staticmethod
    def _get_defined_labels(definition_graph: tfgnn.GraphTensor) -> tf.Tensor:
        cumulative_sizes = tf.cumsum(definition_graph.node_sets['node'].sizes, exclusive=True)
        definition_nodes = tf.ragged.range(tf.squeeze(definition_graph.context['num_definitions'], axis=-1)) + tf.cast(cumulative_sizes, dtype=tf.int64)
        defined_labels = tf.gather(definition_graph.node_sets['node']['node_label'].flat_values, definition_nodes)
        return defined_labels

    def _create_train_model(self):
        (proofstate_graph,) = self.prediction_task.prediction_model.inputs
        prediction_outputs = self.prediction_task.prediction_model(proofstate_graph)

        # compute definition body embeddings
        definition_graph = tf.keras.layers.Input(type_spec=batch_graph_spec(definition_graph_spec))
        masked_definition_graph = self._mask_defined_labels(definition_graph)
        scalar_definition_graph = masked_definition_graph.merge_batch_to_components()
        definition_body_embeddings = self.definition_task(scalar_definition_graph)  # noqa [ PyCallingNonCallable ]

        # get learned definition embeddings
        defined_labels = self._get_defined_labels(definition_graph)
        definition_id_embeddings = self.prediction_task.graph_embedding._node_embedding(defined_labels)

        normalization = tf.sqrt(tf.cast(definition_graph.context['num_definitions'], dtype=tf.float32))
        embedding_difference = (definition_body_embeddings - definition_id_embeddings) / tf.expand_dims(normalization, axis=-1)

        # this is an ugly hack to preserve the output names
        outputs = {name: tf.keras.layers.Lambda(lambda x: x, name=name)(output) for name, output in prediction_outputs.items()}
        outputs.update({self.DEFINITION_EMBEDDING: tf.keras.layers.Lambda(lambda x: x, name=self.DEFINITION_EMBEDDING)(embedding_difference)})

        # make sure we construct the correct type of model
        model_constructor = type(self.prediction_task.prediction_model)
        model = model_constructor(inputs=(proofstate_graph, definition_graph), outputs=outputs)
        return model

    def _loss(self) -> Dict[str, tf.keras.losses.Loss]:
        loss = self.prediction_task.loss()
        if self.definition_task is not None:
            loss.update({self.DEFINITION_EMBEDDING: DefinitionMeanSquaredError()})
        return loss

    def _loss_weights(self) -> Dict[str, float]:
        loss_weights = self.prediction_task.loss_weights()
        if self.definition_task is not None:
            loss_weights.update({self.DEFINITION_EMBEDDING: self.definition_loss_coefficient})
        return loss_weights

    def run(self,
            total_epochs: int,
            batch_size: int,
            split: Tuple[int, int],
            split_random_seed: int
            ) -> tf.keras.callbacks.History:
        """
        @param total_epochs: the total number of epochs to train for (will automatically resume from last trained epoch)
        @param batch_size: the global batch size to use
        @param split: a pair of integers specifying the training/validation split, as passed to Dataset.proofstates()
        @param split_random_seed: a seed for the training/validation split, as passed to Dataset.proofstates()
        @return: the training history
        """
        self.train_model.compile(loss=self._loss(),
                                 loss_weights=self._loss_weights(),
                                 optimizer=self.optimizer,
                                 metrics=self.prediction_task.metrics())

        train_proofstates, valid_proofstates = self.dataset.proofstates(split=split,
                                                                        split_random_seed=split_random_seed,
                                                                        shuffle=True)

        train_proofstates = train_proofstates.apply(self._prepare_dataset).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        valid_proofstates = valid_proofstates.apply(self._prepare_dataset).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        history = self.train_model.fit(train_proofstates,
                                       validation_data=valid_proofstates,
                                       initial_epoch=self.trained_epochs.numpy(),
                                       epochs=total_epochs,
                                       callbacks=self._callbacks())
        return history


class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    A callback extending the standard TensorBoard callback to additionally write the model summary, training config, etc
    """
    def __init__(self, trainer: Trainer, line_length: int = 200, **kwargs):
        self.line_length = line_length
        self.trainer_config = trainer.get_config()
        self.dataset_stats = trainer.dataset.stats()
        super().__init__(**kwargs)

    @property
    def _text_writer(self):
        if 'text' not in self._writers:
            self._writers['text'] = tf.summary.create_file_writer(self.log_dir)
        return self._writers['text']

    def on_train_begin(self, logs=None):
        # TODO: here we should check that we are not resuming training
        model_summary = []
        self.model.summary(print_fn=lambda line: model_summary.append(line))

        with self._text_writer.as_default():
            tf.summary.text(name='model summary', data='<pre>\n' + '\n'.join(model_summary) + '\n</pre>', step=0)
            tf.summary.text(name='trainer config', data='<pre>\n' + yaml.dump(self.trainer_config) + '\n</pre>', step=0)
            tf.summary.text(name='dataset stats', data='<pre>\n' + yaml.dump(self.dataset_stats) + '\n</pre>', step=0)
        super().on_train_begin(logs)


def main():
    parser = argparse.ArgumentParser(description="Train")

    # dataset specification
    dataset_source = parser.add_mutually_exclusive_group(required=True)
    dataset_source.add_argument("--data-dir", metavar="DIRECTORY", type=Path,
                                help="Location of the capnp dataset")
    dataset_source.add_argument("--tfrecord-prefix", metavar="TFRECORD_PREFIX", type=Path,
                                help="Prefix for the .tfrecord and .yml files in the TFRecord dataset")
    parser.add_argument("--dataset-config", metavar="DATASET_YAML", type=Path, required=True,
                        help=f"YAML file with the configuration for the dataset")

    # task specification
    parser.add_argument("--prediction-task-config", metavar="PREDICTION_YAML", type=Path, required=True,
                                    help="YAML file with the configuration for the prediction task")
    parser.add_argument("--definition-task-config", metavar="DEFINITION_YAML", type=Path,
                        help="YAML file with the configuration for the definition task")

    # training specification
    parser.add_argument("--trainer-config", metavar="TRAINING_YAML", type=Path, required=True,
                        help="YAML file with the configuration for training")
    parser.add_argument("--run-config", metavar="RUN_YAML", type=Path, required=True,
                        help="YAML file with the configuration for training")
    parser.add_argument("--log", type=Path, metavar="DIRECTORY",
                        help="Directory where checkpoints and logs are kept")

    # device specification
    parser.add_argument("--gpu", type=str,
                        help="GPUs to use for training ('/gpu:0', '/gpu:1', ... or use 'all' for multi-GPU training)")
    args = parser.parse_args()

    # read the run parameters and set the global seed
    if not args.run_config.is_file():
        parser.error(f'--run-config {args.run_config} must be a YAML file')
    with args.run_config.open() as yaml_file:
        run_config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    tf.random.set_seed(run_config['tf_seed'])

    # dataset creation
    if args.data_dir is not None and not args.data_dir.is_dir():
        parser.error(f'--data-dir {args.data_dir} must be an existing directory')
    if not args.dataset_config.is_file():
        parser.error(f'--dataset-config {args.dataset_config} must be a YAML file')

    if args.data_dir is not None:
        dataset = DataServerDataset.from_yaml_config(data_dir=args.data_dir,
                                                     yaml_filepath=args.dataset_config)
    else:
        dataset = TFRecordDataset.from_yaml_config(tfrecord_prefix=args.tfrecord_prefix,
                                                   yaml_filepath=args.dataset_config)

    # choice of distribution strategy
    if args.gpu == 'all':
        strategy = tf.distribute.MirroredStrategy()

        # fix a tf.distribute bug upon exiting training with MirroredStrategy
        atexit.register(strategy._extended._collective_ops._pool.close)

        # set sharding policy for the dataset
        dataset.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    elif args.gpu is not None:
        strategy = tf.distribute.OneDeviceStrategy(args.gpu)
    else:
        strategy = tf.distribute.get_strategy()

    # trainer creation
    if not args.prediction_task_config.is_file():
        parser.error(f'--prediction-task-config {args.prediction_task_config} must be a YAML file')

    if args.definition_task_config is not None and not args.definition_task_config.is_file():
        parser.error(f'--definition-task-config {args.definition_task_config} should be a YAML file')

    if not args.trainer_config.is_file():
        parser.error(f'--trainer-config {args.trainer_config} should be a YAML file')

    with strategy.scope():
        trainer = Trainer.from_yaml_config(dataset=dataset,
                                           trainer_config=args.trainer_config,
                                           prediction_task_config=args.prediction_task_config,
                                           definition_task_config=args.definition_task_config,
                                           log_dir=args.log)

    # training
    trainer.run(total_epochs=run_config['total_epochs'],
                batch_size=run_config['batch_size'],
                split=run_config['split'],
                split_random_seed=run_config['split_random_seed'])


if __name__ == "__main__":
    main()
