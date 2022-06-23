from tensorflow.python.lib.io import file_io
import numpy as np

def _delete_file_if_exists(filespec):
    """Deletes files matching `filespec`."""
    for pathname in file_io.get_matching_files(filespec):
        try:
            file_io.delete_file(pathname)
        except errors.NotFoundError:
            logging.warning(
                "Hit NotFoundError when deleting '%s', possibly because another "
                "process/thread is also deleting/moving the same file", pathname)

class QCheckpointManager(tf.train.CheckpointManager):
    def __init__(self, *args, qsaving = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert qsaving > 1
        self._qsaving = qsaving
    def _sweep(self):
        """Deletes or preserves managed checkpoints."""
        if not self._max_to_keep:
            # Does not update self._last_preserved_timestamp, since everything is kept
            # in the active set.
            return
        while len(self._maybe_delete) > self._max_to_keep:
            filename, timestamp = self._maybe_delete.popitem(last=False)
            # Even if we're keeping this checkpoint due to
            # keep_checkpoint_every_n_hours, we won't reference it to avoid
            # infinitely-growing CheckpointState protos.
            if (self._keep_checkpoint_every_n_hours
                and (timestamp - self._keep_checkpoint_every_n_hours * 3600.
                     >= self._last_preserved_timestamp)):
                self._last_preserved_timestamp = timestamp
                continue
            if self._qsaving:
                try:
                    #print(filename)
                    i = int(filename.split('-')[-1])
                    #print(i)
                except:
                    i = None
                if i is not None:
                    q = 1
                    while q < i: q *= self._qsaving
                    #print(i, np.floor(q))
                    if i == np.floor(q):
                        continue
            _delete_file_if_exists(filename + ".index")
            _delete_file_if_exists(filename + ".data-?????-of-?????")
