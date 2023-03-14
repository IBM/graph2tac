import tensorflow as tf
import tensorflow_gnn as tfgnn

def stack_dicts_with(f, ds):
    keys = ds[0].keys()
    assert all(d.keys() == keys for d in ds)
    return {
        key : f([d[key] for d in ds])
        for key in keys
    }

def stack_maybe_ragged(xs):
    if isinstance(xs[0], tf.RaggedTensor):
        return tf.ragged.stack(xs)
    else:
        return tf.stack(xs)

def stack_contexts(cs):
    return tfgnn.Context.from_fields(
        sizes = tf.stack([c.sizes for c in cs]),
        features = stack_dicts_with(stack_maybe_ragged, [c.features for c in cs]),
    )

def stack_node_sets(nss):
    sizes = tf.stack([ns.sizes for ns in nss])
    features = stack_dicts_with(tf.ragged.stack, [ns.features for ns in nss])
    return tfgnn.NodeSet.from_fields(
        sizes = sizes,
        features = features,
    )

def stack_edge_sets(ess):
    sizes = tf.stack([es.sizes for es in ess])
    features = stack_dicts_with(tf.ragged.stack, [es.features for es in ess])
    source_name = ess[0].adjacency.source_name
    target_name = ess[0].adjacency.target_name
    assert all(es.adjacency.source_name == source_name for es in ess)
    assert all(es.adjacency.target_name == target_name for es in ess)
    source = tf.ragged.stack([es.adjacency.source for es in ess])
    target = tf.ragged.stack([es.adjacency.target for es in ess])
    return tfgnn.EdgeSet.from_fields(
        sizes = sizes,
        features = features,
        adjacency = tfgnn.Adjacency.from_indices(
            source = (source_name, source),
            target = (target_name, target),
        ),
    )

def stack_graph_tensors(gts):
    context = stack_contexts([gt.context for gt in gts])
    node_sets = stack_dicts_with(stack_node_sets, [gt.node_sets for gt in gts])
    edge_sets = stack_dicts_with(stack_edge_sets, [gt.edge_sets for gt in gts])
    return tfgnn.GraphTensor.from_pieces(context = context, node_sets = node_sets, edge_sets = edge_sets)
