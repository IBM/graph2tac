from typing import Optional, Dict
from math import floor
from collections import OrderedDict
from datetime import datetime

import re
import yaml
import tensorflow as tf


class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    """
    A callback extending the standard TensorBoard callback.

    Additional log entries:
        - model summary
        - total time per run
        - run parameters
        - available GPU devices
        - trainer configuration
        - dataset statistics
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
        self.device_names = [re.search('.*(GPU:\d)', device.name).group(1) for device in self.devices]
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
        self._train_begin_time = datetime.now()

        model_summary = []
        self.model.summary(print_fn=lambda line: model_summary.append(line))

        with self._text_writer.as_default():
            tf.summary.text(name='model summary',
                            data='<pre>\n' + '\n'.join(model_summary) + '\n</pre>',
                            step=self.run_counter)
        super().on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        self._train_end_time = datetime.now()

        with self._text_writer.as_default():
            tf.summary.text(name='run timings',
                            data=f'total time: {self._train_end_time-self._train_begin_time}',
                            step=self.run_counter)
        super().on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for device_name in self.device_names:
            tf.config.experimental.reset_memory_stats(device=device_name)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
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
        if self._qsaving is not None:
            if not self._max_to_keep:
                # Does not update self._last_preserved_timestamp, since everything is kept
                # in the active set.
                return

            maybe_delete = OrderedDict()
            while len(self._maybe_delete) > self._max_to_keep:
                filename, timestamp = self._maybe_delete.popitem(last=False)
                filename_regex = re.search('.*-(\d+)', filename)
                if filename_regex is not None:
                    epoch = int(filename_regex.group(1))
                    q = 1
                    while q < epoch:
                        q *= self._qsaving
                    if epoch == floor(q):
                        # This checkpoint will be saved due to Q-saving
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
