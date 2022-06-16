"""
Graph neural network layers and models
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import Layer
from tensorflow import keras

from graph2tac.tf2.segments import norm_weights, segment_summax, segment_avgmax
from graph2tac.tf2.datatypes import EdgeClassTensor, EdgeTypeTensor, GraphTensor

class GraphLayer(Layer):
    pass

class MessagePassingLayer2(GraphLayer):
    def __init__(self, num_edge_types, dim, norm_msgs, reduction):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.dim = dim
        self.norm_msgs = norm_msgs
        self.reduction = reduction
        self.dense = Dense(dim)

    def message_passing(self, node_values, edges_by_type): # (s) data [V,dim] -> (d) [V,et,dim_red]
        V = tf.shape(node_values)[0]
        # TODO(jrute): Document normalization
        # TODO(jrute): Is this the right normalization given that we have seperate edge types?
        #              Should we be dividing by number of edges across all types?
        if self.norm_msgs:
            norm = lambda x: norm_weights(x, V)  # square root of node degree (within that edge type)
        else:
            norm = lambda _: 1
        return tf.stack([ # [E,dim_red]
            self.reduction(tf.gather(node_values / norm(src), src), dest, V) / norm(dest) # [e,dim_red]
            for src, dest in edges_by_type
        ], axis = 1)

    def call(
        self,
        graphs_with_embs: GraphTensor  # nodes: [bs, (nodes), dim] (ragged)
    ) -> GraphTensor:  # nodes: [bs, (nodes), dim] (ragged)
        nodes = graphs_with_embs.nodes
        edges = graphs_with_embs.edges
        assert isinstance(edges, EdgeTypeTensor)
        msgs = self.message_passing(nodes.values, edges.edges_by_type)  # [nodes, edge_types, dim_red]
        msgs_shape = (tf.shape(msgs)[0], self.num_edge_types*msgs.shape[-1])
        msgs = tf.reshape(msgs, msgs_shape)  # [nodes, edge_types * dim_red]
        msgs = self.dense(msgs)  # [nodes, dim]
        return graphs_with_embs.replace_nodes(
            graphs_with_embs.nodes.replace_values(msgs)  # [bs, (nodes), dim] (ragged)
        )

class MessagePassingLayerEC(GraphLayer):
    def __init__(self, num_edge_types, dim, norm_msgs, reduction, edge_activation = tf.keras.activations.relu):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.dim = dim
        self.norm_msgs = norm_msgs
        self.reduction = reduction
        self.edge_activation = edge_activation
        self.edge_emb = Embedding(num_edge_types, dim)
        self.dense_src = Dense(dim)
        self.dense_dest = Dense(dim)

    def call(
        self,
        graphs_with_embs: GraphTensor  # nodes: [bs, (nodes), dim] (ragged)
    ) -> GraphTensor:  # nodes: [bs, (nodes), dim] (ragged)
        nodes = graphs_with_embs.nodes
        V = tf.shape(nodes.indices)[0]
        edges = graphs_with_embs.edges  # src: [edges], dest: [edges], classes: [edges]
        assert isinstance(edges, EdgeClassTensor)
        msgs_src = tf.gather(self.dense_src(nodes.values), edges.src)  # [edges, dim]
        msgs_dest = tf.gather(self.dense_dest(nodes.values), edges.dest)  # [edges, dim]
        msgs_edge = self.edge_emb(edges.classes)  # [edges, dim]
        msgs = self.edge_activation(msgs_src + msgs_dest + msgs_edge)  # [edges, dim]

        if self.norm_msgs:
            msgs_weights = (  # [edges]
                tf.gather(norm_weights(edges.src, V), edges.src)
                * tf.gather(norm_weights(edges.dest, V), edges.dest)
            )
            msgs = msgs / msgs_weights  # [edges, dim]

        msgs = self.reduction(msgs, edges.dest, V)  # [nodes, dim]
        return graphs_with_embs.replace_nodes(nodes.replace_values(msgs))

class GraphSequential(GraphLayer):
    def __init__(self, *layers):
        super().__init__()
        for layer in layers:
            assert not isinstance(layer, list), f"Pass lists of layers via GraphSequential(*layers)"
            assert (
                isinstance(layer, GraphLayer)
                or callable(layer)
                or layer in ("residual", "residual_new")
            ), f"Invalid layer passed to GraphSequential : {layer}"
        self.layers = layers
    def call(
        self,
        graphs_with_embs: GraphTensor  # nodes: [bs, (nodes), dim] (ragged)
    ) -> GraphTensor:  # nodes: [bs, (nodes), dim] (ragged)
        residual =  None
        for layer in self.layers:
            if isinstance(layer, GraphLayer):
                graphs_with_embs = layer(graphs_with_embs)  # [bs, (nodes), dim] (ragged)
            elif callable(layer):
                node_values = layer(graphs_with_embs.nodes.values)  # [nodes, dim]
                graphs_with_embs = graphs_with_embs.replace_nodes(
                    graphs_with_embs.nodes.replace_values(node_values)  # [bs, (nodes), dim] (ragged)
                )
            elif layer in ("residual", "residual_new"):
                if residual is not None and layer == "residual":
                    node_values = graphs_with_embs.nodes.values + residual  # [nodes, dim]
                    graphs_with_embs = graphs_with_embs.replace_nodes(
                        graphs_with_embs.nodes.replace_values(node_values)  # [bs, (nodes), dim] (ragged)
                    )
                residual = graphs_with_embs.nodes.values  # [nodes, dim]
            else:
                raise Exception(f"Unexpected layer: {layer}")
        return graphs_with_embs

class GraphBlock(GraphLayer):
    def __init__(self, mp_layer, mp_args, nonlin_type, nonlin_position, residuals, residual_dropout, residual_norm):
        super().__init__()

        # the nonlinearity (we will delay adding it to graph until we know where to put it)
        if nonlin_position is None:
            nonlin = None
        else:
            nonlin = getattr(tf.keras.activations, nonlin_type)

        layers = []

        # residual split
        if residuals:
            layers.append("residual_new")

        # message passing
        layers.append(mp_layer(*mp_args))

        # residual
        if residuals:
            # nonlinearity
            if nonlin_position == "after_mp":
                layers.append(nonlin)

            # dropout
            if residual_dropout:
                layers.append(tf.keras.layers.Dropout(.1))

            # nonlinearity
            if nonlin_position == "after_dropout":
                layers.append(nonlin)

            # add
            layers.append("residual")

            # nonlinearity
            if nonlin_position == "after_add":
                layers.append(nonlin)

            # norm
            if residual_norm is None:
                pass
            elif residual_norm == "layer_norm":
                layers.append(tf.keras.layers.LayerNormalization())
            elif residual_norm == "batch_norm":
                layers.append(tf.keras.layers.BatchNormalization())
            else:
                raise Exception(f"Unexpected value for residual_norm: {residual_norm}")

            # nonlinearity
            if nonlin_position == "after_norm":
                layers.append(nonlin)

            assert nonlin is None, f"Nonlinearity not used.  Incorrect position: {nonlin_position}"

        else:
            # nonlinearity (position doesn't matter since no residual)
            if nonlin is not None:
                layers.append(nonlin)

        self.block_layer = GraphSequential(*layers)

    def call(self, graphs):
        return self.block_layer(graphs)


class GraphNN(GraphLayer):
    def __init__(self, aggreg_max, norm_msgs, network_type, layer_edge_types, node_dim,
        total_hops, nonlin_position, nonlin_type, residuals, residual_dropout, residual_norm):
        super().__init__()

        if aggreg_max:
            if norm_msgs:
                reduction = segment_summax
            else:
                reduction = segment_avgmax
        else:
            if norm_msgs:
                reduction = tf.math.unsorted_segment_sum
            else:
                reduction = tf.math.unsorted_segment_mean

        if network_type == "conv2":
            mp_layer = MessagePassingLayer2
            mp_args = layer_edge_types, node_dim, norm_msgs, reduction
            self.use_edge_classes = False
        elif network_type == "conv_ec":
            mp_layer = MessagePassingLayerEC
            mp_args = layer_edge_types, node_dim, norm_msgs, reduction
            self.use_edge_classes = True
        else:
            raise Exception(f"Unexpected message passing network type {network_type}")

        layers = []
        for _ in range(total_hops):
            layers.append(GraphBlock(
                mp_layer=mp_layer,
                mp_args=mp_args,
                nonlin_type=nonlin_type,
                nonlin_position=nonlin_position,
                residuals=residuals,
                residual_dropout=residual_dropout,
                residual_norm=residual_norm
            ))

        self.gnn = GraphSequential(*layers)

    def call(self, graphs: GraphTensor):
        if self.use_edge_classes:
            edges = graphs.edges
            assert isinstance(edges, EdgeTypeTensor)
            graphs = graphs.replace_edges(edges.to_edge_class_tensor())
        return self.gnn(graphs)
