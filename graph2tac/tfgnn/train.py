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
from graph2tac.tfgnn.graph_schema import vectorized_definition_graph_spec, batch_graph_spec
from graph2tac.tfgnn.train_utils import QCheckpointManager, ExtendedTensorBoard, DefinitionLossScheduler
from graph2tac.common import logger


class Trainer:
    """
    This class encapsulates the training logic for the various tasks we can define.
    """

    DEFINITION_EMBEDDING = 'definition_embedding'

    def __init__(self,
                 dataset: Dataset,
                 prediction_task: PredictionTask,
                 serialized_optimizer: Dict,
                 definition_task: Optional[DefinitionTask] = None,
                 definition_loss_coefficient: Optional[float] = None,
                 definition_loss_schedule: Optional[Dict] = None,
                 l2_regularization_coefficient: Optional[float] = None,
                 log_dir: Optional[Path] = None,
                 max_to_keep: Optional[int] = 1,
                 keep_checkpoint_every_n_hours: Optional[int] = None,
                 qsaving: Optional[float] = None):
        """
        @param dataset: a `graph2tac.tfgnn.dataset.Dataset` object providing proof-states and definitions
        @param prediction_task: the `graph2tac.tfgnn.tasks.PredictionTask` to use for proofstates
        @param serialized_optimizer: the optimizer to use, as serialized by tf.keras.optimizers.serialize
        @param definition_task: the `graph2tac.tfgnn.tasks.DefinitionTask` to use for definitions (or `None` to skip)
        @param definition_loss_coefficient: the coefficient in front of the definition embeddings loss term
        @param definition_loss_schedule: the parameters for the `DefinitionLossScheduler`, if using
        @param l2_regularization_coefficient: the coefficient to use for L2 regularization
        @param log_dir: the directory where TensorBoard logs and checkpoints should be saved
        @param max_to_keep: the maximum number of checkpoints to keep (or `None` to keep all)
        @param keep_checkpoint_every_n_hours: optionally keep additional checkpoints every given number of hours
        @param qsaving: additionally keep checkpoints at epochs [qsaving^n] for n = 0, 1, 2, ...
        """
        # dataset
        self.dataset = dataset
        self.dataset_options = tf.data.Options()
        if isinstance(prediction_task.prediction_model.distribute_strategy, tf.distribute.MirroredStrategy):
            self.dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        # prediction task
        self.prediction_task = prediction_task

        # optimizer
        self.optimizer = tf.keras.optimizers.deserialize(serialized_optimizer)

        # definition task
        self.definition_task = definition_task
        if definition_task is not None:
            if definition_loss_coefficient is None:
                raise ValueError('the definition_loss_coefficient should be set whenever using a definition_task')
            else:
                self.definition_loss_coefficient = tf.Variable(initial_value=definition_loss_coefficient,
                                                               dtype=tf.float32)

            self.definition_loss_schedule = definition_loss_schedule
        else:
            self.definition_loss_coefficient = None
            self.definition_loss_schedule = None

        # regularization
        self.l2_regularization_coefficient = l2_regularization_coefficient

        # checkpoint
        self.trained_epochs = tf.Variable(initial_value=0, trainable=False)
        self.run_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        self.checkpoint = tf.train.Checkpoint(prediction_task=prediction_task.checkpoint,
                                              optimizer=self.optimizer,
                                              trained_epochs=self.trained_epochs,
                                              num_runs=self.run_counter)
        if definition_task is not None:
            self.checkpoint.definition_task = definition_task
            self.checkpoint.definition_loss_coefficient = self.definition_loss_coefficient

        # train model
        with prediction_task.prediction_model.distribute_strategy.scope():
            if self.definition_task is not None:
                self.train_model = self._create_train_model()
            else:
                self.train_model = self.prediction_task.prediction_model

            if self.l2_regularization_coefficient is not None:
                self.train_model.add_loss(self._l2_regularization)

        # logging
        self.log_dir = log_dir
        self.max_to_keep = max_to_keep
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self.qsaving = qsaving

        if log_dir is not None:
            # create directory for logs
            log_dir.mkdir(exist_ok=True)

            # restore latest checkpoint
            self.checkpoint_path = log_dir / 'ckpt'
            self.checkpoint_manager = QCheckpointManager(self.checkpoint,
                                                         self.checkpoint_path,
                                                         max_to_keep=max_to_keep,
                                                         keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
                                                         qsaving=qsaving)
            try:
                checkpoint_restored = self.checkpoint_manager.restore_or_initialize()
            except tf.errors.NotFoundError as error:
                logger.error(f'unable to restore checkpoint from {self.checkpoint_path}!')
                raise error
            else:
                if checkpoint_restored is not None:
                    logger.info(f'Restored checkpoint {checkpoint_restored}!')
                else:
                    self.checkpoint_manager.save(self.trained_epochs)

            # save configuration
            config_dir = log_dir / 'config'
            config_dir.mkdir(exist_ok=True)

            self._to_yaml_config(directory=config_dir,
                                 filename='graph_constants',
                                 config=dataset._graph_constants.__dict__)

            self._to_yaml_config(directory=config_dir,
                                 filename='dataset',
                                 config=dataset.get_config())

            self._to_yaml_config(directory=config_dir,
                                 filename='prediction',
                                 config=prediction_task.get_config())

            if self.definition_task is not None:
                self._to_yaml_config(directory=config_dir,
                                     filename='definition',
                                     config=definition_task.get_config())

    def _to_yaml_config(self, directory: Path, filename: str, config: Dict) -> None:
        """
        Exports a YAML file for a configuration dict, renaming according to the current run number if necessary.

        @param directory: the directory where the YAML file should be created
        @param filename: the target filename (without extension)
        @param config: the dict to export
        """
        filepath = directory / f'{filename}.yaml'
        if filepath.is_file():
            new_filepath = directory / f'{filename}-{self.run_counter.value()}.yaml'
            logger.info(f'{filepath} already exists, renaming to {new_filepath}')
            filepath.rename(new_filepath)
        filepath.write_text(yaml.dump(config))
    def get_config(self):
        config = {
            'dataset': self.dataset.get_config(),
            'prediction_task': self.prediction_task.get_config(),
            'serialized_optimizer': tf.keras.optimizers.serialize(self.optimizer),
            'definition_task': self.definition_task.get_config() if self.definition_task is not None else None,
            'definition_loss_coefficient': self.definition_loss_coefficient.value() if self.definition_loss_coefficient is not None else None,
            'definition_loss_schedule': self.definition_loss_schedule,
            'l2_regularization_coefficient': self.l2_regularization_coefficient,
            'max_to_keep': self.max_to_keep,
            'keep_checkpoint_every_n_hours': self.keep_checkpoint_every_n_hours,
            'qsaving': self.qsaving
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
        num_runs_callback = tf.keras.callbacks.LambdaCallback(on_train_begin=lambda logs: self.run_counter.assign_add(1))
        if self.definition_loss_schedule is not None:
            definition_loss_scheduler = DefinitionLossScheduler(definition_loss_coefficient=self.definition_loss_coefficient,
                                                                **self.definition_loss_schedule)
            callbacks.append(definition_loss_scheduler)

        callbacks.extend([trained_epochs_callback, num_runs_callback])

        if self.log_dir is not None:
            save_checkpoint_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.checkpoint_manager.save(self.trained_epochs))
            callbacks.append(save_checkpoint_callback)
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
            definitions = self.dataset.definitions(shuffle=False).map(self.dataset.tokenize_definition_graph)
            definitions = definitions.repeat().shuffle(self.dataset.SHUFFLE_BUFFER_SIZE)
            dataset = tf.data.Dataset.zip(datasets=(dataset, definitions))
            dataset = dataset.map(self._input_output_mixing)

        return dataset.with_options(self.dataset_options)

    def _l2_regularization(self):
        return tf.reduce_sum([tf.keras.regularizers.l2(l2=self.l2_regularization_coefficient)(weight) for weight in self.train_model.trainable_weights])

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
        definition_graph = tf.keras.layers.Input(type_spec=batch_graph_spec(vectorized_definition_graph_spec))
        scalar_definition_graph = definition_graph.merge_batch_to_components()
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
        # compile the training model
        self.train_model.compile(loss=self._loss(),
                                 loss_weights=self._loss_weights(),
                                 optimizer=self.optimizer,
                                 metrics=self.prediction_task.metrics())

        # get training data
        train_proofstates, valid_proofstates = self.dataset.proofstates(split=split,
                                                                        split_random_seed=split_random_seed,
                                                                        shuffle=True)
        train_proofstates = train_proofstates.apply(self._prepare_dataset).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        valid_proofstates = valid_proofstates.apply(self._prepare_dataset).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # prepare callbacks
        callbacks = self._callbacks()
        if self.log_dir is not None:
            tensorboard_callback = ExtendedTensorBoard(log_dir=self.log_dir,
                                                       write_graph=False,
                                                       write_steps_per_second=True,
                                                       update_freq='epoch',
                                                       run_counter=self.run_counter)
            callbacks.append(tensorboard_callback)

            # logs for this run
            tensorboard_callback.log_run(trainer_config=self.get_config(),
                                         dataset_stats=self.dataset.stats(split=split,
                                                                          split_random_seed=split_random_seed),
                                         run_config={
                                             'total_epochs': total_epochs,
                                             'batch-size': batch_size,
                                             'split': split,
                                             'split_random_seed': split_random_seed
                                         }
                                         )

        # run fit
        history = self.train_model.fit(train_proofstates,
                                       validation_data=valid_proofstates,
                                       initial_epoch=self.trained_epochs.numpy(),
                                       epochs=total_epochs,
                                       callbacks=callbacks)
        return history


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

    # logging level
    parser.add_argument("--log-level", type=int, metavar="LOG_LEVEL", default=20,
                        help="Logging level (defaults to 20, a.k.a. logging.INFO)")
    args = parser.parse_args()

    # set logging level
    logger.setLevel(args.log_level)

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
