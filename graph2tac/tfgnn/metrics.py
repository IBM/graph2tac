import tensorflow as tf


class SparseCategoricalConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_categories, name='confusion_matrix', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_categories = num_categories
        self.confusion_matrix = self.add_weight(name='confusion_matrix',
                                                shape=(num_categories, num_categories),
                                                initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = 1.0 if sample_weight is None else sample_weight

        confusion_matrix = tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=-1), num_classes=self.num_categories)

        self.confusion_matrix.assign_add(sample_weight * tf.cast(confusion_matrix, dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.confusion_matrix, tf.reduce_sum(self.confusion_matrix, axis=-1, keepdims=True))

    def reset_state(self):
        self.value.assign(tf.zeros(shape=(self.num_categories, self.num_categories)))


class FullLocalArgumentSparseCategoricalAccuracy(tf.keras.metrics.Mean):
    def __init__(self, weighting_scheme=None, **kwargs):
        super().__init__(**kwargs)
        if weighting_scheme == 'length':
            self.clip_value_min = 0.0
            self.clip_value_max = float('inf')
        elif weighting_scheme == 'non-empty':
            self.clip_value_min = 0.0
            self.clip_value_max = 1.0
        else:
            self.clip_value_min = 1.0
            self.clip_value_max = 1.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        if tf.shape(y_pred)[-1] > 0:
            arguments_pred = tf.argmax(y_pred, axis=-1)

            positions = tf.where(tf.reduce_min(y_true, axis=-1) >= 0)
            valid_arguments_true = tf.gather_nd(y_true.to_tensor(), positions)
            valid_arguments_pred = tf.gather_nd(tf.expand_dims(arguments_pred, axis=1), positions)

            weights = tf.cast(tf.gather_nd(y_true.row_lengths(axis=-1), positions), dtype=tf.float32)
            clipped_weights = tf.clip_by_value(weights, clip_value_min=self.clip_value_min, clip_value_max=self.clip_value_max)

            super().update_state(tf.reduce_all(valid_arguments_true == valid_arguments_pred, axis=-1), clipped_weights)
