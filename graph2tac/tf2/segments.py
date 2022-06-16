import tensorflow as tf

def segment_safemax(data, segment_ids, num_segments):
    res = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
    return tf.where(res > -1e10, res, 0)

def segment_avgmax(data, segment_ids, num_segments):
    a = tf.math.unsorted_segment_mean(data, segment_ids, num_segments)
    m = segment_safemax(data, segment_ids, num_segments)
    return tf.concat([a,m], axis = 1)
def segment_summax(data, segment_ids, num_segments):
    a = tf.math.unsorted_segment_sum(data, segment_ids, num_segments)
    m = segment_safemax(data, segment_ids, num_segments)
    return tf.concat([a,m], axis = 1)

def segment_any(data, segment_ids, num_segments):
    data = tf.cast(data, tf.int32)
    nums = tf.math.unsorted_segment_sum(data, segment_ids, num_segments)
    res = tf.math.greater(nums, 0)
    return res
def segment_all(data, segment_ids, num_segments):
    return ~segment_any(~data, segment_ids, num_segments)

def segment_lens(ids, num_segments):
    return tf.math.unsorted_segment_sum(tf.ones_like(ids), ids, num_segments)

def gather_dim(data, indices):
    ind_dim = len(shape)
    if ind_dim == 1: return tf.gather(data, indices)
    ind_perm = list(reversed(range(ind_dim)))
    data_perm = ind_perm + list(range(ind_dim, len(data.shape)))
    data = tf.transpose(data, perm = data_perm)
    indices = tf.transpose(indices, perm = ind_perm)
    res = tf.gather(data, indices, batch_dims=ind_dim-1)
    res = tf.transpose(res, perm = data_perm)
    return res

def segment_softmax(data, segment_ids, num_segments):
    m0 = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
    data = data - tf.gather(m0, segment_ids) # avoid numerical errors
    e = tf.exp(data)
    es = tf.math.unsorted_segment_sum(e, segment_ids, num_segments)
    return e / tf.gather(es, segment_ids)

def segment_logsoftmax(data, segment_ids, num_segments):
    m0 = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
    data = data - tf.gather(m0, segment_ids) # avoid numerical errors
    es = tf.math.unsorted_segment_sum(tf.exp(data), segment_ids, num_segments)
    return data - tf.gather(tf.math.log(es), segment_ids)

def segment_argmax(data, segment_ids, num_segments):
    candidates = (
        data == tf.gather(
            tf.math.unsorted_segment_max(data, segment_ids, num_segments),
            segment_ids,
        )
    )
    cand_len = tf.shape(candidates)[0]
    nums = tf.range(cand_len)
    if len(candidates.shape) > 1:
        shape = [cand_len]+(len(candidates.shape)-1)*[1]
        nums = tf.reshape(nums, shape)
    return tf.maximum(-1,
        tf.math.unsorted_segment_max(
            tf.cast(candidates, tf.int32) * nums,
            segment_ids, num_segments
        ))

def norm_weights(x, V):
    x = tf.math.unsorted_segment_sum(x, x, V)
    x = tf.maximum(x, 1)
    x = tf.cast(x, float)
    x = tf.sqrt(x)
    x = tf.expand_dims(x, -1)
    return x
