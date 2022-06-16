from argparse import ArgumentError
from dataclasses import dataclass
from pathlib import Path
from tabnanny import check
from dataclasses_json import dataclass_json
from typing import Optional
from tensorflow.python.keras.utils.generic_utils import check_for_unexpected_keys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.layers import Layer, Dense, Embedding


from graph2tac.tf2.datatypes import ActionTensor, EdgeTypeTensor, GraphTensor, RaggedPair, StateActionTensor, StateTensor, GraphDefTensor
from graph2tac.tf2.graph_nn import GraphNN
from graph2tac.tf2.graph_nn_batch import FlatBatchNP
from graph2tac.tf2.graph_nn_def_batch import FlatDefBatchNP
from graph2tac.tf2.model_params import ModelParams
from graph2tac.tf2.segments import segment_avgmax, segment_logsoftmax, segment_argmax


class PrepareGraphs(Layer):
    """
    Process the graph input.

    This is non-trainable and doesn't need to be in TF.
    """

    def __init__(
        self,
        limit_constants: Optional[int],  # how many node classes to use (used for testing no constants)
        single_edge_label: bool,  # use only a single edge type
        symmetric_edges: bool,  # use same edge type for edges in both directions
        self_edges: bool,  # add reflexive self edges
    ):
        super().__init__()
        self.limit_constants = limit_constants
        self.single_edge_label = single_edge_label
        self.symmetric_edges = symmetric_edges
        self.self_edges = self_edges

    def get_edge_label_num(self, original_edge_label_num: int) -> int:
        """
        How many edge types will their be after processing
        """
        if self.single_edge_label:
            edge_label_num = 1
        else:
            edge_label_num = original_edge_label_num

        self_edge_num = int(self.self_edges)  # 0 or 1

        if self.symmetric_edges:
            return edge_label_num + self_edge_num
        else:
            return 2 * edge_label_num + self_edge_num

    def get_node_classes_num(self, original_node_classes_num: int) -> int:
        """
        How many node classes will their be after processing
        """
        if self.limit_constants is not None:
            return self.limit_constants
        else:
            return original_node_classes_num

    def prepare_edges(
        self,
        edges: tuple[tuple[tf.Tensor, tf.Tensor], ...],  # src, tgt pairs for each edge type
                                                         # each tensor: [edges]  dtype=int32
        num_nodes: int
    ) -> tuple[tuple[tf.Tensor, tf.Tensor], ...]:  # src, tgt pairs for each edge type
                                                   # each tensor: [edges]  dtype=int32

        res: list[tuple[tf.Tensor, tf.Tensor]] = []

        if self.self_edges:
            res.append((tf.range(num_nodes), tf.range(num_nodes)))

        if self.single_edge_label:
            edges = ((
                tf.concat([e[0] for e in edges], axis = 0),
                tf.concat([e[1] for e in edges], axis = 0)
            ),)

        for a,b in edges:
            if self.symmetric_edges:
                res.append((tf.concat([a,b], axis = 0),
                            tf.concat([b,a], axis = 0)))
            else:
                res.append((a,b))
                res.append((b,a))
        return tuple(res)

    def prepare_nodes(
        self,
        node_classes: RaggedPair  # [bs, (nodes)] (ragged)
    ):
        if self.limit_constants is not None:
            nodes_c = tf.minimum(node_classes.values, self.limit_constants)  # [nodes]
            node_classes = RaggedPair(node_classes.indices, nodes_c)

        return node_classes

    def call(self, graphs #: GraphTensor
             ) -> GraphTensor:
        assert isinstance(graphs.edges, EdgeTypeTensor)
        nodes = graphs.nodes
        edges = graphs.edges.edges_by_type

        # process graph, building edge structure
        edges = self.prepare_edges(
            num_nodes=tf.shape(nodes.indices)[0],
            edges=edges
        )

        # process nodes according to settings
        nodes = self.prepare_nodes(nodes)  # [bs, (nodes)], (ragged)

        graphs = GraphTensor(
            nodes=nodes,
            edges=EdgeTypeTensor(edges_by_type=edges)
        )
        return graphs

class NodeAndDefEmbedding(Layer):
    """
    A wrapper around the embedding for both nodes and definitions.

    This is used both for the initial nodes embeddings
    and as keys for global argument prediction.
    """
    def __init__(self, node_class_num: int, node_dim: int, normalize: bool):
        super().__init__()
        # Note: The embeddings_constraint argument in Embedding has a bug which prevents us
        # from using the unit norm constraint.  (https://stackoverflow.com/questions/63122880/)
        # So we just normalize whenever the embedding layer is used for now.

        # TODO(jrute): Look for a better solution

        self.node_class_num = node_class_num
        self.node_dim = node_dim
        self.normalize = normalize  # whether embdds are unit vectors
        self._embedding_layer = Embedding(
            input_dim=node_class_num,
            output_dim=node_dim,
        )

    def extend(self, new_node_class_num):
        assert new_node_class_num >= self.node_class_num
        new_emb_layer = Embedding(input_dim=new_node_class_num, output_dim=self.node_dim)
        if self._embedding_layer.built:
            new_emb_layer.build((0,))
            new_emb_layer.embeddings[:self.node_class_num].assign(self._embedding_layer.embeddings)

        self.node_class_num = new_node_class_num
        self._embedding_layer = new_emb_layer

    def call(
        self,
        ix  # [...]  dtype=int32
    ):  # -> [..., dim]  dtype=float
        emb = self._embedding_layer(ix)
        if self.normalize:
            return UnitNorm(axis=-1)(emb)
        else:
            return emb


class InitialNodeEmbedding(Layer):
    """
    Turn the node classes (which combine both node classifications and definition ids) into
    embeddings.
    """
    def __init__(self, node_and_def_embedding_layer: NodeAndDefEmbedding):
        super().__init__()
        self.node_and_def_embedding_layer = node_and_def_embedding_layer

    def call(
        self,
        node_class_graphs: GraphTensor  # nodes: [bs, (nodes)] (ragged) dtype=int
    ) -> GraphTensor:  # nodes: [bs, (nodes), dim] (ragged) dtype=float
        node_classes = node_class_graphs.nodes  # [bs, (nodes)] (ragged)  dtype=int32
        node_embs = self.node_and_def_embedding_layer(node_classes.values) # [total_nodes, dim]
        node_embs = node_classes.replace_values(node_embs)  # [bs, (nodes), dim] dtype=float
        return node_class_graphs.replace_nodes(node_embs)


class StateEmbedding(Layer):
    """
    Get the embedding for each proof state in the batch

    Note: The output dimension will depend on the settings of this layer.
    """
    def __init__(self, final_collapse: bool):
        super().__init__()
        self.final_collapse = final_collapse

    def call(
        self,
        node_embs: RaggedPair,  # [bs, (nodes), dim] (ragged)
        roots,  # [bs]  dtype=int32
    ):  # -> [bs, dim]
        bs = tf.shape(roots)[0]
        root_emb = tf.gather(node_embs.values, roots)  # [bs, dim]
        if self.final_collapse:
            state_embs = tf.concat([  # [bs, dim+dim+dim]  # for avg, max, and root_emb
                segment_avgmax(node_embs.values, node_embs.indices, bs),
                root_emb,
            ], axis = 1)
        else:
            state_embs = root_emb  # [bs, dim]

        return state_embs

class StateDefEmbedding(Layer):
    def __init__(self, final_collapse: bool, dim: int, normalize: bool):
        super().__init__()
        self.final_collapse = final_collapse
        self.final_dense = Dense(dim)
        self.normalize = normalize  # constrain output to be unit vector

    def call(self, node_embs: RaggedPair, roots: RaggedPair):

        batch_size_ext = tf.shape(roots.indices)[0] # a bit bigger than necessary but OK

        root_emb = tf.gather(node_embs.values, roots.values)  # [roots, dim]
        if self.final_collapse:
            state_embs = segment_avgmax(node_embs.values, node_embs.indices, batch_size_ext) # [bs, dim+dim]
            state_embs = tf.gather(state_embs, roots.indices) # [roots, dim+dim]
            state_embs = tf.concat([  # [roots, dim+dim+dim]  # for avg, max, and root_emb
                state_embs,
                root_emb,
            ], axis = 1)
        else:
            state_embs = root_emb  # [roots, dim]

        x = self.final_dense(state_embs)  # [roots, dim]
        if self.normalize:
            x = UnitNorm(axis=-1)(x)
        x = roots.replace_values(x)  # [bs, (roots), dim]
        return x

class RootDefEmbedding(Layer):
    def __init__(self, node_and_def_emb_layer: NodeAndDefEmbedding):
        super().__init__()
        self.node_and_def_emb_layer = node_and_def_emb_layer
    def call(self, roots_c: RaggedPair):
        x = self.node_and_def_emb_layer(roots_c.values)
        x = roots_c.replace_values(x)
        return x

class TacticLogits(Layer):
    """
    Predict the tactic from the state
    """

    def __init__(self, num_tactics: int):
        super().__init__()
        self.final_dense = Dense(num_tactics)

    def call(self, state_embs): # [bs, dim] -> [bs, tactics]
        return self.final_dense(state_embs)


class ContextEmbedding(Layer):
    """
    Get an embedding for each element of the context

    Currently works by taking the embeddings of the context nodes in the graph
    """

    def __init__(self, node_dim: int):
        super().__init__()
        self.dim = node_dim

    def call(
        self,
        node_embs: RaggedPair,  # [bs, (nodes), dim] (ragged)
        context_nodes: RaggedPair,  # [bs, (cxt)] (ragged)  dtype=int32
    ) -> RaggedPair:  # [bs, (cxt), dim] (ragged)
        context_emb = tf.gather(node_embs.values, context_nodes.values)  # [bs, (cxt), dim] (ragged)
        return context_nodes.replace_values(context_emb)  # [bs, (cxt), dim] (ragged)


class TacticEmbedding(Layer):
    """
    Embedding for each tactic as an input to the argument prediction.
    """
    # TODO(jrute): We could try making this the same as the tactic embedding (dense layer)
    # used for tactic prediction.

    def __init__(self, num_tactics: int, dim: int):
        super().__init__()
        self.tactic_emb = Embedding(num_tactics, dim)

    def call(self, tactic_labels):  # [bs] -> [bs, dim]
        return self.tactic_emb(tactic_labels)  # [bs, dim]


class ArgumentLogits(Layer):
    """
    Compute the logits for each argument using the base tactic and other embeddings.
    Each argument is (currently) computed independently of the others.
    """

    def __init__(
            self,
            hidden_dim: int,
            context_dim: int,  # dimension of the context embs (which is the node_dim currently)
            node_dim: int, # must match const_emb
            max_args: int,
            const_emb : NodeAndDefEmbedding,
            global_context, # selection of the constants
        ):
        super().__init__()

        self.max_args = max_args
        self.context_dim = context_dim
        self.node_dim = node_dim

        # Dense layer to combine state_embs and tactic_embs
        self.state_tac_emb = Dense(hidden_dim)
        self.local_none_global_emb = Dense(max_args*(context_dim+1+node_dim))

        self.local_logits = ArgumentLocalLogits(context_dim)
        self.global_logits = ArgumentGlobalLogits(const_emb, global_context)

    def call(
        self,
        context_emb: RaggedPair,  # [bs, (nodes), cxt_dim] (ragged)
        state_emb,  # [bs, state_emb_dim]
        tactic_emb,  # [bs, tac_emb_dim]
        arg_cnt,
        ):

        bs = tf.shape(state_emb)[0]
        mask_args = tf.sequence_mask(  # [bs, max_args] dtype=bool
            lengths=arg_cnt,
            maxlen=self.max_args,
            dtype=tf.dtypes.bool
        )

        # combine state and tactics to get embedding
        state_tac_emb = tf.nn.relu(self.state_tac_emb(tf.concat([  # [bs, hidden_dim]
            state_emb,  # [bs, state_emb_dim]
            tactic_emb  # [bs, tac_emb_dim]
        ], axis = 1)))

        # get keys
        dim = self.context_dim+1+self.node_dim
        local_none_global = self.local_none_global_emb(state_tac_emb) # [bs, max_args*dim]
        local_none_global = tf.reshape(local_none_global, [-1, self.max_args, dim]) # [bs, max_args, dim]

        # mask to the current arguments
        local_none_global_m = tf.boolean_mask(local_none_global, mask_args) # [total_args, dim]
        arguments_i = tf.boolean_mask(tf.tile(tf.expand_dims(tf.range(bs), 1), (1, self.max_args)), mask_args) # [total_args]

        # split to keys
        local_queries = local_none_global_m[:,:self.context_dim] # [total_args, context_dim]
        global_queries = local_none_global_m[:,self.context_dim+1:] # [total_args, context_dim]

        # calculate logits
        none_logits = local_none_global_m[:,self.context_dim] # [total_args]
        none_logits = RaggedPair(tf.range(tf.shape(arguments_i)[0]), none_logits)
        local_logits = self.local_logits(bs, RaggedPair(arguments_i, local_queries), context_emb)
        global_logits = self.global_logits(global_queries)

        # join them together
        logits = [local_logits, none_logits, global_logits]
        logits = RaggedPair(
            tf.concat([x.indices for x in logits], axis = 0),
            tf.concat([x.values for x in logits], axis = 0),
        )
        return logits

class ArgumentLocalLogits(Layer):
    """
    Compute the logits for each argument using the base tactic and other embeddings.
    Each argument is (currently) computed independently of the others.
    """

    def __init__(
            self,
            key_dim: int,  # dimension of the context embs (which is the node_dim currently)
        ):
        super().__init__()
        self.key_dim = key_dim
        self.act_key = Dense(self.key_dim)

    def call(
        self,
        bs,
        arg_query: RaggedPair, # [bs, (args), dim]
        context_emb: RaggedPair,  # [bs, (ctx), dim], must be sorted
    ) -> RaggedPair: # [total_args, (ctx)] (ragged)

        # use key-query to compute logits for args

        arg_ids = arg_query.indices # [total_args]
        ctx_ids = context_emb.indices # [total_ctx]
        ctx_lens = tf.math.unsorted_segment_sum(tf.ones_like(ctx_ids), ctx_ids, num_segments=bs) # [bs]
        arg_ctx_lens = tf.gather(ctx_lens, arg_ids) # [total_args]
        arg_ctx_ragged = tf.ragged.range(arg_ctx_lens) # [total_args, (ctx)]
        arg_ctx_rows = arg_ctx_ragged.value_rowids() # [total_args]
        ctx_starts =  tf.cumsum(ctx_lens, exclusive = True) # [bs]
        arg_ctx_cols = tf.gather(tf.gather(ctx_starts, arg_ids), arg_ctx_rows) + arg_ctx_ragged.values # [sum (args*ctx)]
        queries = tf.gather(arg_query.values, arg_ctx_rows) # [sum (args*ctx), dim]
        keys = tf.gather(self.act_key(context_emb.values), arg_ctx_cols) # [sum (args*ctx), dim]
        logits_flat = tf.einsum("ij,ij->i", queries, keys) # [sum (args*ctx)]
        logits = RaggedPair(tf.cast(arg_ctx_rows, tf.int32), logits_flat) # [total_args, (ctx)]

        return logits # [total_args, (ctx)]

class ArgumentGlobalLogits(Layer):
    def __init__(
            self,
            const_emb : NodeAndDefEmbedding,
            global_context, # selection of the constants
        ):
        super().__init__()
        self.emb_layer = const_emb
        self.def_num = len(global_context)
        self.global_context = tf.constant(global_context)
    def call(self,
             global_queries, # [total_args, dim]
        ):
        total_args = tf.shape(global_queries)[0]
        global_keys = self.emb_layer(self.global_context) # [def_num, dim]
        logits = tf.matmul(global_queries, global_keys, transpose_b = True) # [total_args, def_num]
        logits_ids = tf.repeat(tf.range(total_args), self.def_num)
        logits_ragged = RaggedPair(logits_ids, tf.reshape(logits, [-1]))

        return logits_ragged

class NumberOfArgs(Layer):
    """
    Look up the number of args for each tactic from a list.

    This layer stores that list in a tensor.
    """

    def __init__(self, tactic_index_to_numargs: list[int]):
        super().__init__()
        self.tactic_index_to_numargs = tf.constant(  # [bs]  dtype=int32
            tactic_index_to_numargs,
            dtype=tf.int32
        )

    def call(self, tactic_labels):  # [bs] dtype=int32 -> [bs] dtype=int32
        return tf.gather(params=self.tactic_index_to_numargs, indices=tactic_labels)

class TrainingPredictionsAndLosses(Layer):
    """
    Get training predictions and losses
    """
    # TODO: Should this even be a layer?
    def __init__(self, max_args: int):
        super().__init__()
        self.max_args = max_args

    def logits_to_arg_pred(
        self,
        arg_logits: RaggedPair,  # [total_args, (local_global_ctx)]
        arg_ids, # [total_args]
    ) -> RaggedPair: # -> [bs, (args)]  dtype=int32
        total_args = tf.shape(arg_ids)[0]
        predictions = segment_argmax(arg_logits.values, arg_logits.indices, total_args)  # [total_args]
        return RaggedPair(arg_ids, predictions)

    def logits_to_arg_loss(
        self,
        bs : int,
        arg_logits: RaggedPair,  # [total_args, (local_global_ctx)]
        arg_labels,  # [bs, (args)]  dtype=int32
    ): # -> [bs]
        total_args = tf.shape(arg_labels.indices)[0]
        logprobs = segment_logsoftmax(arg_logits.values, arg_logits.indices, total_args) # [sum (args*local_global_ctx)]
        loss_args = -tf.gather(logprobs, arg_labels.values) # [total_args]
        loss = tf.math.unsorted_segment_sum(loss_args, arg_labels.indices, num_segments = bs) # [bs]
        return loss

    def call(
        self,
        tactic_logits,  # [bs, tactics]
        tactic_labels,  # [bs]
        arg_cnt,        # [bs]  dtype=int
        arg_logits: RaggedPair,  # [total_args, (local_global_ctx)]
        arg_labels      # [total_args]
    ):  # ([bs], [bs, (args)]), ([bs], [bs])

        bs = tf.shape(tactic_logits)[0]

        tactic_pred = tf.argmax(tactic_logits, axis = 1, output_type=tf.int32)  # [bs]
        tactic_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(  # [bs]
            tactic_labels, tactic_logits,
        )

        mask_args = tf.sequence_mask(  # [bs, max_args] dtype=bool
            lengths=arg_cnt,
            maxlen=self.max_args,
            dtype=tf.dtypes.bool
        )
        arg_ids = tf.boolean_mask(tf.tile(tf.expand_dims(tf.range(bs), 1), (1, self.max_args)), mask_args)
        arg_labels_ragged = RaggedPair(arg_ids, arg_labels)
        arg_pred = self.logits_to_arg_pred(arg_logits, arg_ids)  # [bs, (args)]
        arg_loss = self.logits_to_arg_loss(bs, arg_logits, arg_labels_ragged)  # [bs]
        return (tactic_pred, arg_pred), (tactic_loss, arg_loss)  # ([bs], [bs, (args)]), ([bs], [bs])

def np_to_tensor(flat_batch_np: FlatBatchNP) -> StateActionTensor:
    return StateActionTensor.from_arrays(
        nodes_i=flat_batch_np.nodes_i.astype("int32"),
        nodes_c=flat_batch_np.nodes_c.astype("int32"),
        edges=tuple(e.astype("int32") for e in flat_batch_np.edges),
        roots=flat_batch_np.roots.astype("int32"),
        context_i=flat_batch_np.context_i.astype("int32"),
        context=flat_batch_np.context.astype("int32"),
        tactic_labels=flat_batch_np.tactic_labels.astype("int32"),
        arg_labels=flat_batch_np.arg_labels.astype("int32"),
        mask_args=flat_batch_np.mask_args.astype("bool")
    )

def np_to_tensor_def(flat_def_batch_np: FlatDefBatchNP) -> GraphDefTensor:
    return GraphDefTensor.from_arrays(
        batch_size=flat_def_batch_np.batch_size,
        nodes_i=flat_def_batch_np.nodes_i.astype("int32"),
        nodes_c=flat_def_batch_np.nodes_c.astype("int32"),
        edges=tuple(e.astype("int32") for e in flat_def_batch_np.edges),
        roots=flat_def_batch_np.roots.astype("int32"),
        roots_i=flat_def_batch_np.roots_i.astype("int32"),
        roots_c=flat_def_batch_np.roots_c.astype("int32"),
    )


class CombinedModel(tf.keras.Model):
    """
    Combine the multiple models used in training into one model

    They will still be called seperately, but this allows us to use
    `.trainable_variables` to get the list of weights.  Since the two
    model's share weights this is the correct way as compared to calling
    `.trainable_variables` for each model seperately.

    """
    def __init__(self, model, model_def):
        super().__init__()
        self.model = model
        self.model_def = model_def
        self.built = True

class ModelWrapper:
    """
    Class which handles building the tf code, saving it, loading it, and using it for various
    settings including training and inference.  (More to be added here later.)
    """

    def __init__(self, params: Optional[ModelParams] = None, checkpoint: Optional[Path] = None):
        """
        Either build the model from a parameter file or use a checkpoint.
        """
        if checkpoint is not None:
            self._build_model_from_checkpoint(checkpoint, params)
        elif params is not None:
            self._build_model_from_params(params)
        else:
            raise Exception("Need one of params or checkpoint to be supplied to construct model")

    def _build_model_from_params(self, params: ModelParams):
        assert params.dataset_consts is not None, "Need to supply dataset constants in model params"

        self.params = params

        self.input_spec = StateActionTensor.Spec(
            batch_size=None,
            edge_factor_num=params.dataset_consts.edge_label_num,
            max_args=max(params.dataset_consts.tactic_index_to_numargs)
        )
        self.def_input_spec = GraphDefTensor.Spec(
            batch_size=None,
            edge_factor_num=params.dataset_consts.edge_label_num,
        )

        if params.ignore_definitions:
            # TODO(jrute): Moved into the building of the model?  (Or are we done with this?)
            limit_constants = params.dataset_consts.base_node_label_num
        else:
            limit_constants = None
        self.model, self.model_def, self.node_and_def_emb = self._build_model(
            input_spec=self.input_spec,
            def_input_spec=self.def_input_spec,
            num_embeddings=params.dataset_consts.node_label_num,
            et=params.dataset_consts.edge_label_num,
            num_tactics=params.dataset_consts.tactic_num,
            tactic_max_args=max(params.dataset_consts.tactic_index_to_numargs),
            tactic_index_to_numargs=params.dataset_consts.tactic_index_to_numargs,
            global_context=params.dataset_consts.global_context,
            total_hops=params.total_hops,
            node_dim=params.node_dim,
            network_type=params.message_passing_layer,
            norm_msgs=params.norm_msgs,
            aggreg_max=params.aggreg_max,
            nonlin_type=params.nonlin_type,
            nonlin_position=params.nonlin_position,
            residuals=params.residuals,
            final_collapse=params.final_collapse,
            single_edge_label=params.single_edge_label,
            symmetric_edges=params.symmetric_edges,
            self_edges=params.self_edges,
            residual_dropout=params.residual_dropout,
            residual_norm=params.residual_norm,
            limit_constants=limit_constants,
            use_same_graph_nn_weights_for_def_training=params.use_same_graph_nn_weights_for_def_training,
            normalize_def_embeddings=params.normalize_def_embeddings
        )
        self.combined_model = CombinedModel(self.model, self.model_def)

        self.get_predictions_and_losses = TrainingPredictionsAndLosses(
            max_args=max(params.dataset_consts.tactic_index_to_numargs)
        )

    @staticmethod
    def _build_model(input_spec, def_input_spec, num_embeddings, et, num_tactics,
                     tactic_max_args, tactic_index_to_numargs, global_context,
                    total_hops, node_dim,
                    network_type, norm_msgs, aggreg_max,
                    nonlin_type, nonlin_position, residuals,
                    final_collapse, single_edge_label, symmetric_edges, self_edges,
                    residual_dropout, residual_norm,
                    limit_constants, use_same_graph_nn_weights_for_def_training,
                    normalize_def_embeddings) -> Model:

        # Using the Functional API (all functions here have to be Layers or Models)

        # input
        state_actions: StateActionTensor = tf.keras.Input(type_spec=input_spec)

        # split states and actions
        states = StateActionTensor.GetStates()(state_actions)
        actions = StateActionTensor.GetActions()(state_actions)

        # split state parts
        graphs = StateTensor.GetGraphs()(states)
        roots = StateTensor.GetRoots()(states)  # [bs]
        context = StateTensor.GetContext()(states)  # [bs, (cxt)] (ragged)

        # process graph and nodes, building edge structure
        # this is non-trainable and could be moved out of TF
        prepare_graph_layer = PrepareGraphs(
            limit_constants=limit_constants,
            single_edge_label=single_edge_label,
            symmetric_edges=symmetric_edges,
            self_edges=self_edges
        )
        # later steps need to know how many edge types and node classes this layer will create
        num_embeddings = prepare_graph_layer.get_node_classes_num(num_embeddings)
        layer_edge_types = prepare_graph_layer.get_edge_label_num(et)
        graphs = prepare_graph_layer(graphs)

        # build initial node embeddings
        node_and_def_emb = NodeAndDefEmbedding(
            node_class_num=num_embeddings,
            node_dim=node_dim,
            normalize=normalize_def_embeddings,
        )
        ini_node_emb = InitialNodeEmbedding(
            node_and_def_embedding_layer=node_and_def_emb
        )
        graphs_with_emb = ini_node_emb(graphs)

        # run GNN to get final node embeddings
        graph_nn_kwargs = dict(
            aggreg_max=aggreg_max,
            norm_msgs=norm_msgs,
            network_type=network_type,
            layer_edge_types=layer_edge_types,
            node_dim=node_dim,
            total_hops=total_hops,
            nonlin_position=nonlin_position,
            nonlin_type=nonlin_type,
            residuals=residuals,
            residual_norm=residual_norm,
            residual_dropout=residual_dropout,
        )
        graph_nn = GraphNN(**graph_nn_kwargs)
        if use_same_graph_nn_weights_for_def_training:
            def_graph_nn = graph_nn
        else:
            def_graph_nn = GraphNN(**graph_nn_kwargs)
        graphs_with_emb = graph_nn(graphs_with_emb)
        node_embs = GraphTensor.GetNodes()(graphs_with_emb)  # [bs, (nodes), dim] (ragged)

        # get state embedding (dim of output not necessarily node dim)
        state_embs = StateEmbedding(final_collapse=final_collapse)(node_embs, roots) # [bs, dim]

        # get embeddings for each element of context
        context_emb_layer = ContextEmbedding(node_dim=node_dim)
        context_dim = context_emb_layer.dim   # future version of this may change the output dim
        context_emb = context_emb_layer(node_embs, context)  # [bs, (cxt), dim] (ragged)

        # split actions
        tactic_labels = ActionTensor.GetTacticLabels()(actions)  # [bs]  dtype=int32

        # predict tactic from state embeddings
        tactic_logits = TacticLogits(num_tactics=num_tactics)(state_embs)  # [bs, tactics]

        # embedd tactic
        tactic_embs = TacticEmbedding(  # [bs, dim]
            num_tactics=num_tactics,
            dim=node_dim
        )(tactic_labels)

        # get number of args
        arg_cnt = NumberOfArgs(  # [bs]  dtype=int32
            tactic_index_to_numargs=tactic_index_to_numargs
        )(tactic_labels)

        # predict argument logits
        arg_logits = ArgumentLogits(  # [total_args, (local_global_ctx)] (ragged)
            hidden_dim=node_dim,
            context_dim=context_dim,
            node_dim = node_dim,
            max_args=tactic_max_args,
            const_emb = node_and_def_emb,
            global_context = global_context,
        )(context_emb, state_embs, tactic_embs, arg_cnt = arg_cnt)

        out = (tactic_logits, arg_cnt, arg_logits)

        model = Model(state_actions, out, name="tactic_arg_training")

        # build the model for definition learning

        # input
        graph_def = tf.keras.Input(type_spec=def_input_spec)

        # split
        graphs = GraphDefTensor.GetGraphs()(graph_def)
        roots = GraphDefTensor.GetRoots()(graph_def)
        roots_c = GraphDefTensor.GetRootClasses()(graph_def)

        # process graph nodes, same as before
        graphs = prepare_graph_layer(graphs)
        graphs_with_emb = ini_node_emb(graphs)
        graphs_with_emb = def_graph_nn(graphs_with_emb)
        node_embs = GraphTensor.GetNodes()(graphs_with_emb)

        # get the root constants in two ways
        root_embs = StateDefEmbedding(
            final_collapse=final_collapse,
            dim=node_dim,
            normalize=normalize_def_embeddings,
        )(node_embs, roots)
        root_c_embs = RootDefEmbedding(node_and_def_emb)(roots_c)

        out = (root_embs, root_c_embs)
        model_def = Model(graph_def, out, name="graph_def_training")

        return model, model_def, node_and_def_emb

    def save_model_checkpoint(self, checkpoint_dir: Path):
        model_weights_path = checkpoint_dir / "model_weights.h5"
        self.combined_model.save_weights(str(model_weights_path))
        #self.model_def.save_weights(str(model_weights_path))

        param_yaml_path = checkpoint_dir / "model_params.yaml"
        param_yaml_path.write_text(self.params.to_yaml())

    @staticmethod
    def get_params_from_checkpoint(checkpoint_dir: Path):
        model_params_file = checkpoint_dir / "model_params.yaml"
        assert model_params_file.exists(), f"Must have file {model_params_file}"
        return ModelParams.from_yaml_file(model_params_file)

    def _build_model_from_checkpoint(self, checkpoint_dir: Path, params: Optional[ModelParams] = None):
        assert checkpoint_dir.is_dir(), f"{checkpoint_dir} must exist and be a directory"

        # get params
        if params is None:
            params = ModelWrapper.get_params_from_checkpoint(checkpoint_dir)

        # build model
        self._build_model_from_params(params)

        # load saved weights
        weights_file = checkpoint_dir / "model_weights.h5"
        self.combined_model.load_weights(weights_file, by_name=True)
        #self.model_def.load_weights(weights_file, by_name=True)

    def extend_embedding_table(self, new_node_label_num):
        self.node_and_def_emb.extend(new_node_label_num)
        self.params.dataset_consts.node_label_num = new_node_label_num

    def get_predict_fn(self):
        @tf.function(input_signature=(self.input_spec,))
        def pred_fn(batch):
            return self.model(batch)

        return pred_fn

    def get_set_def_fn(self):
        @tf.function(input_signature=(self.def_input_spec,))
        def set_def_fn(batch: GraphDefTensor):
            def_body_embs, _ = self.model_def(batch, training=False)
            emb_vars = self.node_and_def_emb._embedding_layer.embeddings
            emb_vars.scatter_update(
                tf.IndexedSlices(
                    def_body_embs.values,
                    batch.roots_c.values,
                )
            )

        return set_def_fn
