from typing import Optional, Dict
from math import floor
from collections import OrderedDict
from datetime import datetime, timedelta

import re
import yaml
import tensorflow as tf


class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    A callback extending the standard TensorBoard callback.

    Additional log entries:
        - dataset statistics
        - model summary
        - trainer configuration
        - run parameters
        - available GPU devices
        - device peak memory usage per epoch

    """
    def __init__(self, run_counter: tf.Variable, *args, **kwargs):
        """
        @param run_counter: a tf.Variable counting the number of runs in the history (not updated here)
        @param args: additional arguments are passed on to the parent constructor
        @param kwargs: additional keyword arguments are passed on to the parent constructor
        """
        self.run_counter = run_counter
        self.devices = tf.config.list_physical_devices('GPU')
        self.device_names = [re.search(r'.*(GPU:\d)', device.name).group(1) for device in self.devices]
        self._epoch_begin_times = {}
        self._epoch_end_times = {}
        super().__init__(*args, **kwargs)

    @property
    def _text_writer(self) -> tf.summary.SummaryWriter:
        if 'text' not in self._writers:
            self._writers['text'] = tf.summary.create_file_writer(self.log_dir)
        return self._writers['text']

    def log_run(self,
                 trainer_config: Dict,
                 dataset_stats: Dict,
                 run_config: Dict
                 ):
        device_details = {device.name: tf.config.experimental.get_device_details(device)
                          for device in self.devices}

        with self._text_writer.as_default():
            tf.summary.text(name='devices',
                            data='<pre>\n' + yaml.dump(device_details) + '\n</pre>',
                            step=self.run_counter)
            tf.summary.text(name='trainer config',
                            data='<pre>\n' + yaml.dump(trainer_config) + '\n</pre>',
                            step=self.run_counter)
            tf.summary.text(name='dataset stats',
                            data='<pre>\n' + yaml.dump(dataset_stats) + '\n</pre>',
                            step=self.run_counter)
            tf.summary.text(name='run config',
                            data='<pre>\n' + yaml.dump(run_config) + '\n</pre>',
                            step=self.run_counter)

    def on_train_begin(self, logs: Optional[Dict] = None):
        # log model summary
        model_summary = []
        self.model.summary(print_fn=lambda line: model_summary.append(line))

        with self._text_writer.as_default():
            tf.summary.text(name='model summary',
                            data='<pre>\n' + '\n'.join(model_summary) + '\n</pre>',
                            step=self.run_counter)
        super().on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        timings = [self._epoch_end_times[epoch] - self._epoch_begin_times[epoch]
                   for epoch in self._epoch_end_times.keys()]
        num_epochs = len(timings)
        seconds_per_epoch = sum(timings, timedelta())/num_epochs

        with self._text_writer.as_default():
            tf.summary.text(name='run details',
                            data=f'- total epochs: {num_epochs}\n- avg. time per epoch: {seconds_per_epoch}',
                            step=self.run_counter)
        super().on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        # keep track of when the epoch starts
        self._epoch_begin_times[epoch] = datetime.now()

        # reset the stats for memory usage
        for device_name in self.device_names:
            tf.config.experimental.reset_memory_stats(device=device_name)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        # keep track of when the epoch ends
        self._epoch_end_times[epoch] = datetime.now()

        # log the epoch timing
        logs['epoch_duration'] = (self._epoch_end_times[epoch] - self._epoch_begin_times[epoch]).total_seconds()

        # log the learning rate if there is a learning rate schedule
        if isinstance(self.model.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            logs['learning_rate'] = self.model.optimizer.learning_rate(self.model.optimizer.iterations)

        # log memory usage during this epoch
        for device_name in self.device_names:
            memory_info = tf.config.experimental.get_memory_info(device=device_name)
            logs[f'{device_name}_peak_memory'] = memory_info['peak'] / 1024 / 1024 / 1024
        super().on_epoch_end(epoch, logs)


class QCheckpointManager(tf.train.CheckpointManager):
    """
    A CheckpointManager supporting Q-saving.
    """
    def __init__(self, *args, qsaving: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if qsaving is not None and qsaving <= 1:
            raise ValueError(f'qsaving parameter should be greater than 1, but got {qsaving}')
        self._qsaving = qsaving

    def _sweep(self):
        """
        Deletes or preserves managed checkpoints.
        Saves checkpoints at epoch = [qsaving^n] for n = 0, 1, ..., otherwise lets the parent CheckpointManager decide.
        """
        if not self._max_to_keep:
            # Does not update self._last_preserved_timestamp, since everything is kept
            # in the active set.
            return

        maybe_delete = OrderedDict()
        while len(self._maybe_delete) > self._max_to_keep:
            filename, timestamp = self._maybe_delete.popitem(last=False)
            filename_regex = re.search(r'.*-(\d+)', filename)
            if filename_regex is not None:
                epoch = int(filename_regex.group(1))

                # First checkpoint is always saved
                if epoch == 0:
                    continue

                # Checkpoints to be saved due to Q-saving
                if self._qsaving is not None:
                    q = 1
                    while q < epoch:
                        q *= self._qsaving
                    if epoch == floor(q):
                        continue
            # It is not for us to decide whether this checkpoint is kept or not
            maybe_delete[filename] = timestamp

        # The rest of the checkpoints will be processed at some later stage
        while len(self._maybe_delete):
            filename, timestamp = self._maybe_delete.popitem(last=False)
            maybe_delete[filename] = timestamp

        # Let the parent class do its thing
        self._maybe_delete = maybe_delete
        super()._sweep()


class DefinitionLossScheduler(tf.keras.callbacks.Callback):
    """
    A scheduler for the definition loss coefficient.
    """
    def __init__(self,
                 definition_loss_coefficient: tf.Variable,
                 blowup_rate: float,
                 steps: Optional[int] = None):
        """
        @param definition_loss_coefficient: the tf.Variable that is used as a coefficient for the definition loss term
        @param blowup_rate: the factor by which coefficient is multiplied
        @param steps: if `None`, the blowup_rate is applied every epoch; otherwise, it is applied (continuously) every `steps` batches
        """
        self.definition_loss_coefficient = definition_loss_coefficient
        self.blowup_rate = blowup_rate
        self.steps = steps

        if steps is not None:
            self.eff_blowup_rate = tf.pow(blowup_rate, 1/steps)

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.steps is not None:
            self.definition_loss_coefficient.assign(self.definition_loss_coefficient.value() * self.eff_blowup_rate)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        # log the latest definition loss coefficient used for this epoch
        logs['definition_loss_coefficient'] = self.definition_loss_coefficient.value()

        # update the definition loss coefficient, if necessary
        if self.steps is None:
            self.definition_loss_coefficient.assign(self.definition_loss_coefficient.value() * self.blowup_rate)
