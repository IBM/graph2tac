"""
Custom datatypes for the TF code
"""
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

class RaggedPair(tf.experimental.ExtensionType):
    """
    A precurser to putting everything in TFs RaggedTensor type

    The idea is that most data is stored in the form (x_i, x) where x_i gives the index along
    the batch dim.  We simply replace those pairs with this extension type.
    """
    indices: tf.Tensor
    values: tf.Tensor
    #batch_size: int

    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape([None, None])

    def replace_values(self, new_values: tf.Tensor):
        return RaggedPair(
            indices=self.indices,
            values=new_values
        )

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types
        """
        def __init__(self, inner_shape: Union[tf.TensorShape, list[int]], dtype: tf.DType):
            self.indices=tf.TensorSpec(shape=tf.TensorShape([None]), dtype=tf.int32)
            self.values=tf.TensorSpec(
                shape=tf.TensorShape([None]).concatenate(inner_shape),
                dtype=dtype
            )

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape([None, None])

class EdgeTensor(tf.experimental.ExtensionType):
    """
    Tensor representation of the edges of a graph.  This is meant to be an abstract class.

    The shape should be [batch_size, None] allowing us to switch to ragged
    tensors later.
    """
    pass


class EdgeClassTensor(EdgeTensor):
    """
    Tensor representation of the edges of a graph.

    The edges classes are represented by labels.
    """

    # list of all edges organized by type (each edge is a source-target tuple)
    # the indices of the source nodes
    src: tf.Tensor  # shape=[# edges]  dtype=int

    # the indices of the target nodes
    dest: tf.Tensor  # shape=[# edges]  dtype=int

    # the class of each edge
    classes: tf.Tensor  # shape=[# edges]  dtype=int


class EdgeTypeTensor(EdgeTensor):
    """
    Tensor representation of the edges of a graph.

    Each edge type has its own tensors.
    """

    # list of all edges organized by type (each edge is a source-target tuple)
    edges_by_type: Tuple[Tuple[tf.Tensor, tf.Tensor], ...] # list for all edge types (len=# of edge types)
                                                   # each array: shape=[# edges]  dtype=int

    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape([None])

    def to_edge_class_tensor(self) -> EdgeClassTensor:
        edge_tuple = self.edges_by_type
        src, dest = zip(*edge_tuple)
        src = tf.concat(src, axis = 0)  # [edges]  dtype=int32
        dest = tf.concat(dest, axis = 0)  # [edges]  dtype=int32
        classes = tf.concat([  # [edges]  dtype=int32
            tf.fill(tf.shape(src), i)  # TODO(jrute): Is this really int32, or is it int64?
            for i, (src,dest) in enumerate(edge_tuple)
        ], axis = 0)
        return EdgeClassTensor(src=src, dest=dest, classes=classes)

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types

        batch_size should be None unless expect all inputs to have constant batch size
        """
        def __init__(self, edge_factor_num: int):
            self.edges_by_type=((tf.TensorSpec(shape=[None], dtype=tf.int32),) * 2,) * edge_factor_num

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape([None])


class GraphTensor(tf.experimental.ExtensionType):
    """
    Tensor representation of a graph.  Each edge type is stored in its own tensor.
    """
    #batch_size: Optional[int]

    # class (int) or embedding (float) for each node
    # classes may include both the starting class and the final class
    nodes: RaggedPair  # shape=[batch_size, (nodes)]  dtype=int|float

    # list of all edges organized by type (each edge is a source-target tuple)
    edges: EdgeTensor

    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape([None, None])

    @property
    def dtype(self)-> tf.DType:
        return self.node_values.dtype

    def replace_nodes(self, new_nodes: RaggedPair):
        # TODO(jrute): Is this memory intensive?
        # I'm worried that TF will hold onto the old node information even after it is done
        # being used to calculate new_nodes (assuming it isn't used elsewhere).
        # If so, we could instead have this method take a function which we apply to new nodes.
        return GraphTensor(
            nodes=new_nodes,
            edges=self.edges
        )

    def replace_edges(self, new_edges: EdgeTensor):
        return GraphTensor(
            nodes=self.nodes,
            edges=new_edges
        )

    class GetNodes(Layer):
        """
        Makes the .nodes accessor into a Keras layer for use in functional API
        """
        def call(self, graphs: "GraphTensor") -> RaggedPair:
            return graphs.nodes

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types

        batch_size should be None unless expect all inputs to have constant batch size
        """
        def __init__(
            self,
            edge_factor_num: int,
            inner_shape: Union[tf.TensorShape, list[int]],
            dtype: tf.DType
        ):
            self.nodes=RaggedPair.Spec(inner_shape=inner_shape, dtype=dtype)
            self.edges=EdgeTypeTensor.Spec(edge_factor_num=edge_factor_num)

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape([None, None])

        @property
        def dtype(self)-> tf.DType:
            return self.nodes.values.dtype


class StateTensor(tf.experimental.ExtensionType):
    """
    Tensor representation of the proof state.  The input for tactic prediction and training.
    """
    # the graphs of nodes and edges
    graphs: GraphTensor

    # index of root node for each batch
    roots: tf.Tensor  # shape=[batch size]  dtype=int32

    context: RaggedPair  # [batch_size, (context)]  dtype=int32

    # shape will be [batch_size, None]
    # 1. It is needed by Keras to use the functional API and is shown in the model summary
    # 2. It will help us convert code to ragged tensors later since we can
    # think of this object as a ragged tensor of that shape
    # 3. It is a way to access the batch_size via .shape[0]
    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape([self.roots.shape[0], None])

    class GetGraphs(Layer):
        """
        Makes the .graphs accessor into a Keras layer for use in functional API
        """
        def call(self, states: "StateTensor") -> GraphTensor:
            return states.graphs

    class GetRoots(Layer):
        """
        Makes the .roots accessor into a Keras layer for use in functional API
        """
        def call(self, states: "StateTensor") -> tf.Tensor:
            return states.roots

    class GetContext(Layer):
        """
        Makes the .context accessor into a Keras layer for use in functional API
        """
        def call(self, states: "StateTensor") -> RaggedPair:
            return states.context

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types

        batch_size should be None unless expect all inputs to have constant batch size
        """
        def __init__(self, batch_size: Optional[int], edge_factor_num: int):
            self.graphs=GraphTensor.Spec(
                edge_factor_num=edge_factor_num,
                inner_shape=[],
                dtype=tf.int32
            )
            self.roots=tf.TensorSpec(shape=[batch_size], dtype=tf.int32)
            self.context=RaggedPair.Spec(inner_shape=[], dtype=tf.int32)

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape([self.roots.shape[0], None])


class ActionTensor(tf.experimental.ExtensionType):
    """
    Tensor representation of the tactic action.  Used for training.
    """
    # index of base tactic for each batch, e.g. "apply _"
    tactic_labels: tf.Tensor  # shape=[batch size]  dtype=int

    # argument label (index of local cxt arg node) for each batch (ignore this value if it is masked)
    arg_labels: tf.Tensor  # shape=[total_args]  dtype=int

    # TODO(jrute): Remove as this is redundant, since it only depends on the arg count,
    #              which can be computed from tactic_labels.
    #              Moreover, for prediction we won't have this mask given,
    #              so we need to compute the arg count or mask seperately anyway.
    # mask for arguments
    # if a tactic doesn't have a argument in this position it is masked
    # if there is some argument in position i for batch item index b, then (b, i) is true
    mask_args: tf.Tensor  # shape=[batch size, max # of arguments]  dtype=bool

    # shape will be [batch_size, None]
    # 1. It is needed by Keras to use the functional API and is shown in the model summary
    # 2. It will help us convert code to ragged tensors later since we can
    # think of this object as a ragged tensor of that shape
    # 3. It is a way to access the batch_size via .shape[0]
    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape([self.tactic_labels.shape[0], None])

    class GetTacticLabels(Layer):
        """
        Makes the .tactic_labels accessor into a Keras layer for use in functional API
        """
        def call(self, actions: "ActionTensor") -> tf.Tensor:
            return actions.tactic_labels

    class GetArgLabels(Layer):
        """
        Makes the .arg_lables accessor into a Keras layer for use in functional API
        """
        def call(self, actions: "ActionTensor") -> tf.Tensor:
            return actions.arg_labels

    class GetMaskArgs(Layer):
        """
        Makes the .mask_args accessor into a Keras layer for use in functional API
        """
        def call(self, actions: "ActionTensor") -> tf.Tensor:
            return actions.mask_args

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types

        batch_size should be None unless expect all inputs to have constant batch size
        """
        def __init__(self, batch_size: Optional[int], max_args: int):
            self.tactic_labels=tf.TensorSpec(shape=[batch_size], dtype=tf.int32)
            self.arg_labels=tf.TensorSpec(shape=[None], dtype=tf.int32)
            self.mask_args=tf.TensorSpec(shape=[None, max_args], dtype=tf.bool)

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape([self.tactic_labels.shape[0], None])

class StateActionTensor(tf.experimental.ExtensionType):
    """
    Input object for the graph combining the proof state input and tactic action output
    """
    states: StateTensor
    actions: ActionTensor

    # shape will be [batch_size, None]
    # 1. It is needed by Keras to use the functional API and is shown in the model summary
    # 2. It will help us convert code to ragged tensors later since we can
    # think of this object as a ragged tensor of that shape
    # 3. It is a way to access the batch_size via .shape[0]
    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape([self.states.shape[0], None])

    @staticmethod
    def from_arrays(
        nodes_i,
        nodes_c,
        edges,
        roots,
        context_i,
        context,
        tactic_labels,
        arg_labels,
        mask_args
    ) -> "StateActionTensor":
        return StateActionTensor(
            states=StateTensor(
                graphs=GraphTensor(
                    nodes=RaggedPair(nodes_i.astype("int32"), nodes_c.astype("int32")),
                    edges=EdgeTypeTensor(
                        edges_by_type=tuple((e.astype("int32")[:,0], e.astype("int32")[:,1]) for e in edges)
                    )
                ),
                roots=roots.astype("int32"),
                context=RaggedPair(context_i.astype("int32"), context.astype("int32"))
            ),
            actions=ActionTensor(
                tactic_labels=tactic_labels.astype("int32"),
                arg_labels=arg_labels.astype("int32"),
                mask_args=mask_args.astype("bool")
            )
        )

    class GetStates(Layer):
        """
        Makes the .states accessor into a Keras layer for use in functional API
        """
        def call(self, state_action: "StateActionTensor") -> StateTensor:
            return state_action.states

    class GetActions(Layer):
        """
        Makes the .actions accessor into a Keras layer for use in functional API
        """
        def call(self, state_action: "StateActionTensor") -> ActionTensor:
            return state_action.actions

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types

        batch_size should be None unless expect all inputs to have constant batch size
        """
        def __init__(self, batch_size: Optional[int], edge_factor_num: int, max_args: int):
            self.states=StateTensor.Spec(batch_size=batch_size, edge_factor_num=edge_factor_num)
            self.actions=ActionTensor.Spec(batch_size=batch_size, max_args=max_args)

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape([self.states.shape[0], None])

class InnerInput(tf.experimental.ExtensionType):
    """
    The input to the second half of the network, including the tactic choses
    as well as the hidden embeddings for the state and context
    """
    tactics: tf.Tensor  # [bs]
    context_emb: RaggedPair  # [bs, (cxt), cxt_dim]
    state_emb: tf.Tensor  # [bs, state_dim]

    # shape will be [batch_size, None]
    # 1. It is needed by Keras to use the functional API and is shown in the model summary
    # 2. It will help us convert code to ragged tensors later since we can
    # think of this object as a ragged tensor of that shape
    # 3. It is a way to access the batch_size via .shape[0]
    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape([self.tactics.shape[0], None])

    class GetTactics(Layer):
        """
        Makes the .tactics accessor into a Keras layer for use in functional API
        """
        def call(self, inner_input: "InnerInput") -> tf.Tensor:
            return inner_input.tactics

    class GetContextEmb(Layer):
        """
        Makes the .context_emb accessor into a Keras layer for use in functional API
        """
        def call(self, inner_input: "InnerInput") -> tf.Tensor:
            return inner_input.context_emb

    class GetStateEmb(Layer):
        """
        Makes the .state_emb accessor into a Keras layer for use in functional API
        """
        def call(self, inner_input: "InnerInput") -> tf.Tensor:
            return inner_input.state_emb

    class Build(Layer):
        """
        Makes the init constructor into a Keras layer for use in the functional API
        """
        def call(self, tactics: tf.Tensor, context_emb: RaggedPair, state_emb: tf.Tensor):
            return InnerInput(
                tactics=tactics,
                context_emb=context_emb,
                state_emb=state_emb
            )

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types

        batch_size should be None unless expect all inputs to have constant batch size
        """
        def __init__(self, batch_size: Optional[int], context_dim: int, state_dim: int):
            self.tactics=tf.TensorSpec(shape=[batch_size], dtype=tf.int32)
            self.context_emb=RaggedPair.Spec(inner_shape=[context_dim], dtype=tf.float32)
            self.state_emb=tf.TensorSpec(shape=[batch_size, state_dim], dtype=tf.float32)

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape([self.tactics.shape[0], None])


class GraphDefTensor(tf.experimental.ExtensionType):
    """
    Tensor representation of the proof state.  The input for tactic prediction and training.
    """
    # the graphs of nodes and edges
    batch_size: tf.TensorShape
    graphs: GraphTensor

    # index of root nodes for each batch
    roots: RaggedPair  # shape=[batch_size, (number of roots)]  dtype=int32
    # classes of root nodes for each batch
    roots_c: RaggedPair  # shape=[batch_size, (number of roots)]  dtype=int32

    # shape will be [batch_size, None]
    # 1. It is needed by Keras to use the functional API and is shown in the model summary
    # 2. It will help us convert code to ragged tensors later since we can
    # think of this object as a ragged tensor of that shape
    # 3. It is a way to access the batch_size via .shape[0]
    @property
    def shape(self)-> tf.TensorShape:
        return tf.TensorShape(self.batch_size+[None])

    @staticmethod
    def from_arrays(
        batch_size,
        nodes_i,
        nodes_c,
        edges,
        roots,
        roots_i,
        roots_c,
    ) -> "GraphDefTensor":
        return GraphDefTensor(
            batch_size = tf.TensorShape(batch_size),
            graphs=GraphTensor(
                nodes=RaggedPair(nodes_i.astype("int32"), nodes_c.astype("int32")),
                edges=EdgeTypeTensor(
                    edges_by_type=tuple((e.astype("int32")[:,0], e.astype("int32")[:,1]) for e in edges)
                )
            ),
            roots=RaggedPair(roots_i.astype("int32"), roots.astype("int32")),
            roots_c=RaggedPair(roots_i.astype("int32"), roots_c.astype("int32")),
        )

    class GetGraphs(Layer):
        """
        Makes the .graphs accessor into a Keras layer for use in functional API
        """
        def call(self, states: "GraphDefTensor") -> GraphTensor:
            return states.graphs

    class GetRoots(Layer):
        """
        Makes the .roots accessor into a Keras layer for use in functional API
        """
        def call(self, states: "GraphDefTensor") -> tf.Tensor:
            return states.roots

    class GetRootClasses(Layer):
        """
        Makes the .context accessor into a Keras layer for use in functional API
        """
        def call(self, states: "GraphDefTensor") -> tf.RaggedTensor:
            return states.roots_c

    class Spec:
        """
        Spec used for TF to ensure correct object shapes and types

        batch_size should be None unless expect all inputs to have constant batch size
        """
        def __init__(self, batch_size: Optional[int], edge_factor_num: int):
            self.batch_size = tf.TensorShape(batch_size)
            self.graphs=GraphTensor.Spec(
                edge_factor_num=edge_factor_num,
                inner_shape=[],
                dtype=tf.int32
            )
            self.roots = RaggedPair.Spec(inner_shape=[], dtype=tf.int32)
            self.roots_c = RaggedPair.Spec(inner_shape=[], dtype=tf.int32)

        @property
        def shape(self)-> tf.TensorShape:
            return tf.TensorShape(self.batch_size+[None])
