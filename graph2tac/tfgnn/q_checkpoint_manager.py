from typing import Optional
from math import floor
from collections import OrderedDict

import re
import tensorflow as tf

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
