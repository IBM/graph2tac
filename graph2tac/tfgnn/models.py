from typing import Iterable, Dict, Any, Callable, Optional, Tuple
from dataset import Dataset

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models.gat_v2 import GATv2GraphUpdate, GATv2Conv

SIMPLE_CONVOLUTION_GNN = 'simple_convolution_gnn'
GCN_CONVOLUTION_GNN = 'gcn_convolution_gnn'
ATTENTION_GNN = 'attention_gnn'
DENSE_TACTIC = 'dense_tactic'
DENSE_DEFINITION = 'dense_definition'
SIMPLE_RNN = 'simple_rnn'

class LogitsFromEmbeddings(tf.keras.layers.Layer):
    """
    Compute logits from an embedding matrix by taking the scalar product with a hidden state.
    NOTE: We don't use a lambda layer to make sure the embedding matrix is properly tracked.
    """
    def __init__(self,
                 embedding_matrix: tf.Variable,
                 valid_indices: tf.Tensor,
                 name: str = 'logits_from_embeddings',
                 **kwargs
                 ):
        self._embedding_matrix = embedding_matrix
        self._valid_indices = valid_indices
        super().__init__(name=name, **kwargs)

    def call(self, hidden_state, training=False):
        return tf.matmul(a=hidden_state, b=tf.gather(self._embedding_matrix, self._valid_indices), transpose_b=True)


class NodeSetDropout(tfgnn.keras.layers.MapFeatures):
    """
    Dropout layer for node features.
    """

    def __init__(self,
                 rate: float,
                 feature_name: tfgnn.FieldName = tfgnn.DEFAULT_STATE_NAME,
                 name: str = 'node_set_dropout',
                 **kwargs):
        """
        @param rate: the dropout rate
        @param feature_name: the feature to apply the dropout to
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        self._feature_name = feature_name
        self._dropout = tf.keras.layers.Dropout(rate=rate)
        super().__init__(name=name, node_sets_fn=self._node_sets_fn, **kwargs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'rate': self._dropout.rate,
            'feature_name': self._feature_name
        })
        return config

    def _node_sets_fn(self,
                      node_set: tfgnn.NodeSet,
                      node_set_name: tfgnn.NodeSetName
                      ) -> Dict[tfgnn.FieldName, Any]:
        features = dict(node_set.features)
        features[self._feature_name] = self._dropout(node_set[self._feature_name])
        return features


class NodeSetLayerNormalization(tfgnn.keras.layers.MapFeatures):
    """
    LayerNormalization for node features.
    """

    def __init__(self,
                 feature_name: tfgnn.FieldName = tfgnn.DEFAULT_STATE_NAME,
                 epsilon=0.001,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 name: str = 'node_set_layer_normalization',
                 **kwargs):
        """
        @param feature_name: the feature to normalize
        @param epsilon: see tf.keras.layers.LayerNormalization
        @param center: see tf.keras.layers.LayerNormalization
        @param scale: see tf.keras.layers.LayerNormalization
        @param beta_initializer: see tf.keras.layers.LayerNormalization
        @param gamma_initializer: see tf.keras.layers.LayerNormalization
        @param beta_regularizer: see tf.keras.layers.LayerNormalization
        @param gamma_regularizer: see tf.keras.layers.LayerNormalization
        @param beta_constraint: see tf.keras.layers.LayerNormalization
        @param gamma_constraint: see tf.keras.layers.LayerNormalization
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        self._feature_name = feature_name
        self._layer_normalization = tf.keras.layers.LayerNormalization(axis=-1,
                                                                       epsilon=epsilon,
                                                                       center=center,
                                                                       scale=scale,
                                                                       beta_initializer=beta_initializer,
                                                                       gamma_initializer=gamma_initializer,
                                                                       beta_regularizer=beta_regularizer,
                                                                       gamma_regularizer=gamma_regularizer,
                                                                       beta_constraint=beta_constraint,
                                                                       gamma_constraint=gamma_constraint)
        super().__init__(name=name, node_sets_fn=self._node_sets_fn, **kwargs)

    def get_config(self) -> dict:
        config = super().get_config()

        layer_normalization_config = self._layer_normalization.get_config()

        config.update({
            'feature_name': self._feature_name,
            'epsilon': layer_normalization_config['epsilon'],
            'center': layer_normalization_config['center'],
            'scale': layer_normalization_config['scale'],
            'beta_initializer': layer_normalization_config['beta_initializer'],
            'gamma_initializer': layer_normalization_config['gamma_initializer'],
            'beta_regularizer': layer_normalization_config['beta_regularizer'],
            'gamma_regularizer': layer_normalization_config['gamma_regularizer'],
            'beta_constraint': layer_normalization_config['beta_constraint'],
            'gamma_constraint': layer_normalization_config['gamma_constraint']
        })
        return config

    def _node_sets_fn(self,
                      node_set: tfgnn.NodeSet,
                      node_set_name: tfgnn.NodeSetName
                      ) -> Dict[tfgnn.FieldName, Any]:
        features = dict(node_set.features)
        features[self._feature_name] = self._layer_normalization(node_set[self._feature_name])
        return features


class HiddenStatePooling(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int,
                 reduce_type: str,
                 name='hidden_state_pooling',
                 **kwargs):
        """

        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param reduce_type: the type of reduction to perform to obtain the graph hidden state from the nodes
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._hidden_size = hidden_size
        self._reduce_type = reduce_type

        if reduce_type in tfgnn.get_registered_reduce_operation_names():
            self._pooler = tfgnn.keras.layers.Pool(tag=tfgnn.CONTEXT, reduce_type=reduce_type)
        elif reduce_type == 'attention':
            self._pooler = GATv2Conv(num_heads=1, per_head_channels=hidden_size, receiver_tag=tfgnn.CONTEXT)
        else:
            raise ValueError(f'{reduce_type} is not a valid pooling operation')

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'hidden_size': self._hidden_size,
            'reduce_type': self._reduce_type
        })
        return config

    def call(self, embedded_graph, training=False):
        pooled_hidden_state = self._pooler(embedded_graph, node_set_name='node', training=training)  # noqa [ PyCallingNonCallable ]
        return embedded_graph.replace_features(context={'hidden_state': pooled_hidden_state})


class GraphEmbedding(tfgnn.keras.layers.MapFeatures):
    """
    Embedding layer for graphs, maps node labels to hidden states and edge labels to edge embeddings.
    """

    def __init__(self,
                 node_label_num: int,
                 edge_label_num: int,
                 hidden_size: int,
                 name: str = 'graph_embedding',
                 **kwargs):
        """

        @param node_label_num: the number of node labels to embed
        @param edge_label_num: the number of edge labels to embed
        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        self._node_label_num = node_label_num
        self._edge_label_num = edge_label_num
        self._hidden_size = hidden_size

        self._node_embedding = tf.keras.layers.Embedding(input_dim=node_label_num,
                                                         output_dim=hidden_size,
                                                         name=f'{name}_node_embedding')

        self._edge_embedding = tf.keras.layers.Embedding(input_dim=edge_label_num,
                                                         output_dim=hidden_size,
                                                         name=f'{name}_edge_embedding')

        self._total_size = tfgnn.keras.layers.TotalSize()
        super().__init__(node_sets_fn=self._node_sets_fn,
                         edge_sets_fn=self._edge_sets_fn,
                         context_fn=self._context_fn,
                         name=name,
                         **kwargs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'node_label_num': self._node_label_num,
            'edge_label_num': self._edge_label_num,
            'hidden_size': self._hidden_size
        })
        return config

    def _node_sets_fn(self,
                      node_set: tfgnn.NodeSet,
                      node_set_name: tfgnn.NodeSetName
                      ) -> Dict[tfgnn.FieldName, Any]:
        return {'hidden_state': self._node_embedding(node_set['node_label'])}

    def _edge_sets_fn(self,
                      edge_set: tfgnn.EdgeSet,
                      edge_set_name: tfgnn.EdgeSetName
                      ) -> Dict[tfgnn.FieldName, Any]:
        return {'edge_embedding': self._edge_embedding(edge_set['edge_label'])}

    def _context_fn(self,
                    context: tfgnn.Context
                    ) -> Dict[tfgnn.FieldName, Any]:
        return {'hidden_state': tf.zeros(shape=(self._total_size(context), self._hidden_size))}


class NodeDegree(tf.keras.layers.Layer):
    """
    Layer to compute node in/out degrees (aggregated for all edge sets).
    """

    def __init__(self,
                 name: str = 'node_degree',
                 **kwargs):
        """
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)

    def call(self, graph_tensor: tfgnn.GraphTensor, training=False) -> tfgnn.GraphTensor:
        all_features = {}
        for node_set_name, node_set in graph_tensor.node_sets.items():
            all_features[node_set_name] = {'in_degree': tf.zeros(shape=tf.reduce_sum(node_set.sizes), dtype=tf.int64),
                                           'out_degree': tf.zeros(shape=tf.reduce_sum(node_set.sizes), dtype=tf.int64)}
            all_features[node_set_name].update(node_set.features)

        for edge_set_name, edge_set in graph_tensor.edge_sets.items():
            source_name = edge_set.adjacency.source_name
            target_name = edge_set.adjacency.target_name

            edge_ones = tf.ones(shape=tf.reduce_sum(edge_set.sizes), dtype=tf.int64)

            in_degree = tfgnn.pool_edges_to_node(graph_tensor=graph_tensor,
                                                 edge_set_name=edge_set_name,
                                                 node_tag=tfgnn.TARGET,
                                                 reduce_type='sum',
                                                 feature_value=edge_ones)
            all_features[target_name]['in_degree'] += in_degree

            out_degree = tfgnn.pool_edges_to_node(graph_tensor=graph_tensor,
                                                  edge_set_name=edge_set_name,
                                                  node_tag=tfgnn.SOURCE,
                                                  reduce_type='sum',
                                                  feature_value=edge_ones)
            all_features[source_name]['out_degree'] += out_degree

        return graph_tensor.replace_features(node_sets=all_features)


class GCNNormalization(tf.keras.layers.Layer):
    """
    Layer to compute the normalization coefficient used in the GCNConvolution layer.
    """
    def __init__(self, name: str = 'gcn_normalization', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, graph_tensor: tfgnn.GraphTensor, training=False) -> tfgnn.GraphTensor:
        all_features = {}
        for edge_set_name, edge_set in graph_tensor.edge_sets.items():
            out_degrees = tfgnn.broadcast_node_to_edges(graph_tensor=graph_tensor,
                                                        edge_set_name=edge_set_name,
                                                        node_tag=tfgnn.SOURCE,
                                                        feature_name='out_degree')

            in_degrees = tfgnn.broadcast_node_to_edges(graph_tensor=graph_tensor,
                                                       edge_set_name=edge_set_name,
                                                       node_tag=tfgnn.TARGET,
                                                       feature_name='in_degree')

            gcn_norm = tf.sqrt(tf.math.reciprocal_no_nan(tf.cast(in_degrees * out_degrees, dtype=tf.float32)))

            all_features[edge_set_name] = {'gcn_norm': gcn_norm}
            all_features[edge_set_name].update(edge_set.features)
        return graph_tensor.replace_features(edge_sets=all_features)


class GCNConvolution(tf.keras.layers.Layer):
    """
    Layer implementing the GCN layer as described in
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#implementing-the-gcn-layer
    """
    def __init__(self,
                 hidden_size: int,
                 dense_activation: Optional[str],
                 residual_activation: Optional[str],
                 dropout_rate: float,
                 source_feature_name: str = tfgnn.DEFAULT_STATE_NAME,
                 target_feature_name: str = tfgnn.DEFAULT_STATE_NAME,
                 edge_feature_name: Optional[str] = None,
                 name: str = 'gcn_convolution',
                 **kwargs):
        """
        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param dense_activation: the activation to use for the dense layers
        @param residual_activation: the activation to use after the residual connection following convolutions
        @param dropout_rate: the dropout rate on the node hidden states
        @param source_feature_name: the name of the feature to use as message in the source node set
        @param target_feature_name: the name of the feature to update in the target node set
        @param edge_feature_name: the name of the edge feature to concatenate to the message (or None to disable)
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        self._hidden_size = hidden_size
        self._dense_activation = dense_activation
        self._residual_activation = dense_activation
        self._dropout_rate = dropout_rate
        self._source_feature_name = source_feature_name
        self._target_feature_name = target_feature_name
        self._edge_feature_name = edge_feature_name

        self._edge_transformation = tf.keras.layers.Dense(units=hidden_size, activation=dense_activation)
        self._node_transformation = tf.keras.layers.Dense(units=hidden_size, activation=residual_activation)
        self._dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        super().__init__(name=name, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self._hidden_size,
            'dense_activation': self._dense_activation,
            'residual_activation': self._residual_activation,
            'dropout_rate': self._dropout_rate,
            'source_feature_name': self._source_feature_name,
            'target_feature_name': self._target_feature_name,
            'edge_feature_name': self._edge_feature_name
        })
        return config

    def call(self, graph_tensor: tfgnn.GraphTensor, edge_set_name: str, training=False) -> tfgnn.GraphTensor:
        edge_set = graph_tensor.edge_sets[edge_set_name]
        target_node_set_name = edge_set.adjacency.target_name

        edge_normalization = edge_set['gcn_norm']

        hidden_states = tfgnn.broadcast_node_to_edges(graph_tensor=graph_tensor,
                                                      edge_set_name=edge_set_name,
                                                      node_tag=tfgnn.SOURCE,
                                                      feature_name=self._source_feature_name)

        if self._edge_feature_name is not None:
            edge_embeddings = edge_set[self._edge_feature_name]
            hidden_states = tf.concat([hidden_states, edge_embeddings], axis=-1)

        messages =  tf.expand_dims(edge_normalization, axis=-1) * self._edge_transformation(hidden_states, training=training)
        messages = self._dropout(messages, training=training)

        pooled_messages = tfgnn.pool_edges_to_node(graph_tensor=graph_tensor,
                                                   edge_set_name=edge_set_name,
                                                   node_tag=tfgnn.TARGET,
                                                   reduce_type='sum',
                                                   feature_value=messages)

        hidden_states = self._node_transformation(pooled_messages, training=training)
        hidden_states = self._dropout(hidden_states, training=training)

        hidden_states += graph_tensor.node_sets[target_node_set_name][self._target_feature_name]

        all_features = {node_set_name: dict(node_set.features) for node_set_name, node_set in graph_tensor.node_sets.items()}
        all_features[target_node_set_name][self._target_feature_name] = hidden_states
        return graph_tensor.replace_features(node_sets=all_features)


class GATv2GNN(tf.keras.layers.Layer):
    """
    GNN layer built out of attention layers:
        - inputs should match the `hidden_graph_spec` in `graph_schema.py`
        - outputs match the `hidden_graph_spec` in `graph_schema.py`
    """

    def __init__(self,
                 hidden_size: int,
                 hops: int,
                 attention_config: dict,
                 final_reduce_type: str,
                 name: str = 'gatv2_gnn',
                 **kwargs):
        """
        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param hops: number of message-passing hops
        @param attention_config: additional arguments for all GATv2Convolution layers (e.g. use_bias, edge_dropout, ...)
        @param final_reduce_type: the reduction to perform on the final step
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._hidden_size = hidden_size
        self._attention_config = attention_config
        self._final_reduce_type = final_reduce_type

        self._attention_layers = [GATv2GraphUpdate(num_heads=1,
                                                   per_head_channels=hidden_size,
                                                   edge_set_name='edge',
                                                   sender_edge_feature='edge_embedding',
                                                   name=f'gat_v2_{i}',
                                                   **attention_config) for i in range(hops)]

        self._hidden_state_pooling = HiddenStatePooling(hidden_size=hidden_size, reduce_type=final_reduce_type)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'hidden_size': self._hidden_size,
            'hops': len(self._attention_layers),
            'attention_config': self._attention_config,
            'final_reduce_type': self._final_reduce_type
        })
        return config

    def call(self,
             embedded_graph: tfgnn.GraphTensor,
             training: bool = False
             ) -> tfgnn.GraphTensor:
        for attention_layer in self._attention_layers:
            embedded_graph = attention_layer(embedded_graph, training=training)

        output_graph = self._hidden_state_pooling(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]

        return output_graph


class SimpleConvolutionGNN(tf.keras.layers.Layer):
    """
    GNN layer built out of simple convolution layers:
        - inputs should match the `bare_graph_spec` in `graph_schema.py`
        - outputs match the `hidden_graph_spec` in `graph_schema.py`
    """

    def __init__(self,
                 hidden_size: int,
                 hops: int,
                 dense_activation: Optional[str],
                 residual_activation: Optional[str],
                 dropout_rate: float,
                 layer_norm: bool,
                 reduce_type: str,
                 final_reduce_type: str,
                 name: str = 'simple_convolution_gnn',
                 **kwargs):
        """
        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param hops: number of message-passing hops
        @param dense_activation: the activation to use for the dense layers
        @param residual_activation: the activation to use after the residual connection following convolutions
        @param dropout_rate: the dropout rate on the node hidden states
        @param layer_norm: whether to use layer normalization on node hidden states
        @param reduce_type: the reduction to perform after each convolution
        @param final_reduce_type: the reduction to perform on the final step
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._hidden_size = hidden_size
        self._dense_activation = dense_activation
        self._residual_activation = residual_activation
        self._dropout_rate = dropout_rate
        self._layer_norm = layer_norm
        self._reduce_type = reduce_type
        self._final_reduce_type = final_reduce_type

        conv_gnn_builder = tfgnn.keras.ConvGNNBuilder(convolutions_factory=self._convolutions_factory,
                                                      nodes_next_state_factory=self._nodes_next_state_factory,
                                                      receiver_tag=tfgnn.TARGET)
        self._convolutions = [conv_gnn_builder.Convolve() for _ in range(hops)]

        self._dropout = NodeSetDropout(dropout_rate)

        self._layer_normalization = NodeSetLayerNormalization()

        self._hidden_state_pooling = HiddenStatePooling(hidden_size=hidden_size, reduce_type=final_reduce_type)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'hidden_size': self._hidden_size,
            'hops': len(self._convolutions),
            'dense_activation': self._dense_activation,
            'residual_activation': self._residual_activation,
            'dropout_rate': self._dropout_rate,
            'layer_norm': self._layer_norm,
            'reduce_type': self._reduce_type,
            'final_reduce_type': self._final_reduce_type
        })
        return config

    def _convolutions_factory(self,
                              edge_set_name: tfgnn.EdgeSetName,
                              receiver_tag: tfgnn.IncidentNodeTag
                              ) -> tfgnn.keras.layers.SimpleConvolution:
        return tfgnn.keras.layers.SimpleConvolution(
            message_fn=tf.keras.layers.Dense(units=self._hidden_size, activation=self._dense_activation),
            reduce_type=self._reduce_type,
            combine_type='concat',
            receiver_tag=tfgnn.TARGET,
            sender_node_feature='hidden_state',
            sender_edge_feature='edge_embedding',
            name='simple_convolution'
        )

    def _nodes_next_state_factory(self, node_set_name: tfgnn.NodeSetName) -> tfgnn.keras.layers.ResidualNextState:
        return tfgnn.keras.layers.ResidualNextState(
            residual_block=tf.keras.layers.Dense(units=self._hidden_size, name=f'dense'),
            activation=self._residual_activation,
            name='residual'
        )

    def call(self,
             embedded_graph: tfgnn.GraphTensor,
             training: bool = False
             ) -> tfgnn.GraphTensor:
        for convolution in self._convolutions:
            embedded_graph = convolution(embedded_graph, training=training)
            embedded_graph = self._dropout(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]
            if self._layer_norm:
                embedded_graph = self._layer_normalization(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]

        output_graph = self._hidden_state_pooling(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]

        return output_graph


class GCNConvolutionGNN(tf.keras.layers.Layer):
    """
    GNN layer built out of GCN convolution layers:
        - inputs should match the `hidden_graph_spec` in `graph_schema.py`
        - outputs match the `hidden_graph_spec` in `graph_schema.py`
    """

    def __init__(self,
                 hidden_size: int,
                 hops: int,
                 dense_activation: str,
                 residual_activation: str,
                 dropout_rate: float,
                 layer_norm: bool,
                 final_reduce_type: str,
                 name: str = 'gcn_convolution_gnn',
                 **kwargs):
        """
        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param hops: number of message-passing hops
        @param dense_activation: the activation to use for the dense layers
        @param residual_activation: the activation to use after the residual connection following convolutions
        @param dropout_rate: the dropout rate on the node hidden states
        @param layer_norm: whether to use layer normalization on node hidden states
        @param final_reduce_type: the reduction to perform on the final step
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._hidden_size = hidden_size
        self._dense_activation = dense_activation
        self._residual_activation = residual_activation
        self._dropout_rate = dropout_rate
        self._layer_norm = layer_norm
        self._final_reduce_type = final_reduce_type

        self._node_degree = NodeDegree()
        self._gcn_normalization = GCNNormalization()
        self._convolutions = [GCNConvolution(hidden_size=hidden_size,
                                             dense_activation=dense_activation,
                                             residual_activation=residual_activation,
                                             dropout_rate=dropout_rate,
                                             edge_feature_name='edge_embedding') for _ in range(hops)]

        self._layer_normalization = NodeSetLayerNormalization()

        self._hidden_state_pooling = HiddenStatePooling(hidden_size=hidden_size, reduce_type=final_reduce_type)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'hidden_size': self._hidden_size,
            'hops': len(self._convolutions),
            'dense_activation': self._dense_activation,
            'residual_activation': self._residual_activation,
            'dropout_rate': self._dropout_rate,
            'layer_norm': self._layer_norm,
            'final_reduce_type': self._final_reduce_type
        })
        return config

    def call(self,
             embedded_graph: tfgnn.GraphTensor,
             training: bool = False
             ) -> tfgnn.GraphTensor:
        embedded_graph = self._node_degree(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]
        embedded_graph = self._gcn_normalization(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]

        for convolution in self._convolutions:
            embedded_graph = convolution(embedded_graph, edge_set_name='edge', training=training)  # noqa [ PyCallingNonCallable ]
            if self._layer_norm:
                embedded_graph = self._layer_normalization(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]

        output_graph = self._hidden_state_pooling(embedded_graph, training=training)  # noqa [ PyCallingNonCallable ]

        return output_graph


class DenseTacticHead(tf.keras.layers.Layer):
    """
    Tactic head composed of a series of dense hidden layers
        - inputs should match the `hidden_graph_spec` in `graph_schema.py`.
        - outputs a tensor with logits for each base tactic
    """

    def __init__(self,
                 tactic_embedding_size: int,
                 hidden_layers: Iterable[dict] = (),
                 name: str = 'dense_tactic_head',
                 **kwargs):
        """
        @param tactic_embedding_size: the size of the (hidden) tactic embedding dimension
        @param hidden_layers: the list of arguments to the dense hidden layers (units, activation, etc)
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._tactic_embedding_size = tactic_embedding_size

        self._hidden_layers = []
        for i, hidden_layer_config in enumerate(hidden_layers):
            hidden_layer_config['name'] = f'{name}_dense_{i}'
            self._hidden_layers.append(tf.keras.layers.Dense.from_config(hidden_layer_config))

        self._final_layer = tf.keras.layers.Dense(units=tactic_embedding_size, name=f'{name}_tactic_embedding')

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'tactic_embedding_size': self._tactic_embedding_size,
            'hidden_layers': [hidden_layer.get_config() for hidden_layer in self._hidden_layers]
        })
        return config

    def call(self,
             hidden_graph: tfgnn.GraphTensor,
             training: bool = False
             ) -> tf.Tensor:
        hidden_state = hidden_graph.context['hidden_state']

        for hidden_layer in self._hidden_layers:
            hidden_state = hidden_layer(hidden_state, training=training)

        tactic_embedding = self._final_layer(hidden_state, training=training)

        return tactic_embedding


class ArgumentsHead(tf.keras.layers.Layer):
    """
    Base layer for argument heads, taking care of the special case where there are no arguments to compute.
    Sub-classed layers should implement the _get_hidden_state_sequences method.
    """

    @staticmethod
    def _no_hidden_state_sequences(inputs):
        """
        Produces an empty arguments tensor with the right dimensions.

        @param inputs: a tuple with the same inputs as the arguments head receives
        @return: a RaggedTensor of shape [ None(batch_size), None(0), hidden_size ]
        """
        hidden_graph, tactic_embedding, num_arguments = inputs

        # TODO: This sometimes produces a warning about type inference failing; investigate
        return tf.RaggedTensor.from_row_lengths(
            values=tf.repeat(hidden_graph.context['hidden_state'], repeats=num_arguments, axis=0),
            row_lengths=num_arguments,
            validate=False
        )

    def call(self, inputs, training=False):
        hidden_graph, tactic_embedding, num_arguments = inputs

        if tf.reduce_sum(num_arguments) > 0:
            return self._get_hidden_state_sequences(inputs, training=training)
        else:
            return self._no_hidden_state_sequences(inputs)


class RNNArgumentsHead(ArgumentsHead):
    """
    Simple recurrent head to put on top of a GNN component to predict argument hidden states:
        - inputs should match the `hidden_graph_spec` in `graph_schema.py`.
        - outputs a ragged tensor with hidden states for each argument position
    """

    def __init__(self,
                 hidden_size: int,
                 tactic_embedding_size: int,
                 activation: str,
                 recurrent_depth: int,
                 recurrent_activation: str = 'tanh',
                 name: str = 'rnn_arguments_head',
                 **kwargs):
        """
        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param tactic_embedding_size: size of the tactic embeddings (used here as the dimension of the internal state)
        @param activation: the activation for the final output
        @param recurrent_depth: the number of stacked RNN layers
        @param recurrent_activation: the activation for the RNN layers
        @param name: the name of the output model
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._hidden_size = hidden_size
        self._tactic_embedding_size = tactic_embedding_size
        self._activation = activation
        self._recurrent_activation = recurrent_activation

        self._rnn_layers = [tf.keras.layers.SimpleRNN(units=tactic_embedding_size,
                                                      activation=recurrent_activation,
                                                      return_state=True,
                                                      return_sequences=True) for _ in range(recurrent_depth)]
        self._final_dense = tf.keras.layers.Dense(units=hidden_size, activation=activation)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'hidden_size': self._hidden_size,
            'tactic_embedding_size': self._tactic_embedding_size,
            'activation': self._activation,
            'recurrent_depth': len(self._rnn_layers),
            'recurrent_activation': self._recurrent_activation
        })
        return config

    def _get_hidden_state_sequences(self, inputs, training=False):
        hidden_graph, tactic_embedding, num_arguments = inputs

        hidden_state_sequences = tf.RaggedTensor.from_row_lengths(
            values=tf.repeat(hidden_graph.context['hidden_state'], repeats=num_arguments, axis=0),
            row_lengths=num_arguments,
            validate=False
        )
        internal_state = tactic_embedding
        for rnn_layer in self._rnn_layers:
            hidden_state_sequences, internal_state = rnn_layer(hidden_state_sequences, initial_state=internal_state, training=training)
        hidden_state_sequences = self._final_dense(hidden_state_sequences, training=training)

        return hidden_state_sequences


class DenseDefinitionHead(tf.keras.layers.Layer):
    """
    Definition head composed of a series of dense hidden layers
        - inputs should match the `hidden_graph_spec` in `graph_schema.py`.
        - outputs a tensor with an embedding for the definition graph
    """

    def __init__(self,
                 hidden_size: int,
                 hidden_layers: Iterable[dict] = (),
                 name_layer: Optional[dict],
                 name: str = 'dense_definition_head',
                 **kwargs):
        """

        @param hidden_size: size of the hidden states (needs to be compatible with other components!)
        @param hidden_layers: the list of arguments to the dense hidden layers (units, activation, etc)
        @param name: the name of the layer
        @param kwargs: passed on to parent constructor
        """
        super().__init__(name=name, **kwargs)
        self._hidden_size = hidden_size

        if name_layer is None:
            self._name_layer = None
        else:
            self._name_layer = tf.keras.Sequential([
                tf.keras.layers.Input(shape=[None], dtype=tf.float32, ragged=True),
                tf.keras.layers.Embedding(
                    input_dim=Dataset.MAX_LABEL_TOKENS,
                    output_dim=hidden_size,
                ),
                tf.keras.layers.Lambda(lambda x: x.to_tensor()), # align ragged tensor
                tf.keras.layers.deserialize(name_layer),
                #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
            ])

        self._hidden_layers = [tf.keras.layers.Dense.from_config(hidden_layer_config) for hidden_layer_config in hidden_layers]
        self._final_layer = tf.keras.layers.Dense(units=hidden_size)

    def get_config(self) -> dict:
        config = super().get_config()
        if self._name_layer is None:
            name_layer = None
        else:
            name_layer = self._name_layer.get_config()
        config.update({
            'hidden_size': self._hidden_size,
            'hidden_layers': [hidden_layer.get_config() for hidden_layer in self._hidden_layers],
            'name_layer': name_layer,
        })
        return config

    def call(self,
             inputs: Tuple[tfgnn.GraphTensor, tf.RaggedTensor, tf.RaggedTensor],
             training: bool = False
             ) -> tf.Tensor:
        hidden_graph, num_definitions, definition_name_vectors = inputs

        cumulative_sizes = tf.expand_dims(tf.cumsum(hidden_graph.node_sets['node'].sizes, exclusive=True), axis=-1)
        definition_nodes = tf.ragged.range(num_definitions) + tf.cast(cumulative_sizes, dtype=tf.int64)

        node_hidden_states = tf.gather(hidden_graph.node_sets['node']['hidden_state'], definition_nodes.flat_values)
        graph_hidden_states = tf.repeat(hidden_graph.context['hidden_state'], num_definitions, axis=0)

        hidden_state = [node_hidden_states, graph_hidden_states]
        if self._name_layer is not None:
            rnn_output = self._name_layer(definition_name_vectors.values)
            hidden_state.append(rnn_output)
        hidden_state = tf.concat(hidden_state, axis=-1)

        for hidden_layer in self._hidden_layers:
            hidden_state = hidden_layer(hidden_state, training=training)

        definition_embedding = self._final_layer(hidden_state, training=training)

        return tf.RaggedTensor.from_row_lengths(definition_embedding, num_definitions)


def get_gnn_constructor(gnn_type: str) -> Callable[..., tf.keras.layers.Layer]:
    if gnn_type == SIMPLE_CONVOLUTION_GNN:
        return SimpleConvolutionGNN
    elif gnn_type == GCN_CONVOLUTION_GNN:
        return GCNConvolutionGNN
    elif gnn_type == ATTENTION_GNN:
        return GATv2GNN
    else:
        raise ValueError(f'{gnn_type} is not a valid GNN type')


def get_tactic_head_constructor(gnn_type: str) -> Callable[..., tf.keras.layers.Layer]:
    if gnn_type == DENSE_TACTIC:
        return DenseTacticHead
    else:
        raise ValueError(f'{gnn_type} is not a valid tactic head type')


def get_arguments_head_constructor(gnn_type: str) -> Callable[..., tf.keras.layers.Layer]:
    if gnn_type == SIMPLE_RNN:
        return RNNArgumentsHead
    else:
        raise ValueError(f'{gnn_type} is not a valid tactic head type')


def get_definition_head_constructor(definition_head_type: str) -> Callable[..., tf.keras.layers.Layer]:
    if definition_head_type == DENSE_DEFINITION:
        return DenseDefinitionHead
    else:
        raise ValueError(f'{definition_head_type} is not a valid definition head type')
