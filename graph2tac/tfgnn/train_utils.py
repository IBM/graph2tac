from typing import Optional
from math import floor
from collections import OrderedDict

import re
import tensorflow as tf


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
        Saves checkpoints at epoch = [q^n] for n = 0, 1, ..., otherwise lets the parent CheckpointManager decide.
        """
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
