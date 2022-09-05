"""
defines batch dataclass used by the specific gnn defined in graph_nn.py
the classes in these modules probably should be packaged together with
GraphToLogits class in graph_nn
"""
from typing import List, Tuple, Any

import numpy as np
from dataclasses import dataclass

from graph2tac.loader.data_server import graph_as, LoaderProofstate
from graph2tac.tf2.model_params import ModelDatasetConstants

# TODO(jrute): This code is unique to the model in model.py,
# and it now depends on the dataset constants saved with that model.
# So we might as well move it there alongside np_to_tensor.
# Then we can avoid writing np_to_tensor(flat_batch_np(...)) everywhere.

@dataclass
class FlatBatchNP:
    """
    Numpy data used for training a batch

    Originally stored as format:
    input_signature = ((
        (
            ( tf.TensorSpec(shape=[None], dtype=tf.int32),  # nodes_i
              tf.TensorSpec(shape=[None], dtype=tf.int32)), # nodes_c
            (
                tf.TensorSpec(shape=[None,2], dtype=tf.int32), # edges
            )*graph_constants.edge_factor_num,
            tf.TensorSpec(shape=[None], dtype=tf.int32), # roots
            ( tf.TensorSpec(shape=[None], dtype=tf.int32),  # context_i
              tf.TensorSpec(shape=[None], dtype=tf.int32)), # context
        ),
        (
            tf.TensorSpec(shape=[None], dtype=tf.int32), # tactic_labels
            tf.TensorSpec(shape=[None, graph_constants.max_args], dtype=tf.int32), # arg_labels
            tf.TensorSpec(shape=[None, graph_constants.max_args], dtype=tf.bool), # mask_args
        )
    ),)
    """
    # this a format to collect batch_size datapoints
    # each datapoint is a graph tactic and argument

    # we concatenate all nodes from graphs in the list
    # batch index for each node from a
    nodes_i: np.ndarray  # shape=[sum(lens of nodes_c)]  dtype=np.int32
    # graph class for each node
    nodes_c: np.ndarray  # shape=[sum(lens of nodes_c)]  dtype=np.int32

    # we concatenate all edges from all graphs
    # the outer index is conflated edges classes
    # list of lists of all edges, external index is the conflated edge class,
    # each edge is (source, target) tuple   dtype=np.int32
    # shape = (graphs_constants.edge_factor.num,  number of edges of that type, 2)
    edges: List[np.ndarray]

    # index of root node for each datapoint
    roots: np.ndarray  # shape=[batch size]  dtype=np.int32

    # batch index for each context element
    context_i: np.ndarray  # shape=[sum(lens of contexts)]  dtype=np.int32
    # node pointers from datapoint contex concatenated
    context: np.ndarray  # shape=[sum(lens of context)]  dtype=np.int32

    # index of base tactic for each batch, e.g. "apply _"
    tactic_labels: np.ndarray  # shape=[batch size]  dtype=np.int32

    # argument label pointer to a context index for each datapoint
    arg_labels: np.ndarray  # shape=[total number of arguments]

    # mask for arguments
    # e.g. [True, True, False , False, .... False] for a tactic with 2 arguments
    mask_args: np.ndarray  # shape=[batch size, max # of arguments]

    def counts(self):
        """
        returns string representation of batch counts
        """
        num_nodes = len(self.nodes_i)
        num_edge_types = len(self.edges)
        num_edges = sum(e.shape[0] for e in self.edges)
        num_roots = len(self.roots)
        return (f"{num_nodes} nodes "
                f"{num_edges} edges of {num_edge_types} classes "
                f"{num_roots} datapoints ")

def build_padded_arguments_and_mask(args: np.array, max_arg_num: int, context_len: int):
    """
    pads the list of arguments up to max_arg_num
    if max_arg_num is not sufficient, throws exception for further refactoring of the code
    """
    tail_args = np.tile(np.array([[0,context_len]]), (max_arg_num-len(args),1))
    a = np.full((len(args),), True)
    b = np.full((max_arg_num - len(args),), False)
    mask = np.concatenate([a,b])
    padded_args = np.concatenate([args, tail_args], axis=0)
    return padded_args, mask


def make_flat_batch_np(batch: List[Tuple[LoaderProofstate, Any, int]],
                       global_context_size: int,
                       max_arg_num: int
                       ) -> FlatBatchNP:
    """
    max_arg_num: is the uniformized length of the list of arguments for tail padding

    this function forms FlatBatchNP of numpy arrays from a non-empty list of datapoints
    for empty input consider a function make_flat_batch_np_empty defined below (with different API)

    # you can use proof_step_id to query DataServer.data_point(proof_step_id) for meta-information

    """
    assert len(batch) > 0
    # each batch element is (state, action, graph_id)
    states, actions, _ = zip(*batch)

    # each state is (loader_graph, root, context, metadata)
    graphs, roots, all_context, _ = zip(*states)

    # each context is (local_context, available_global_context)
    context, _ = zip(*all_context)

    nodes_c, edges = zip(* map(lambda x: graph_as("tf2", x), graphs))


    tactic_labels, args_np_raw = zip(*actions)

    args_and_mask_np = [build_padded_arguments_and_mask(a, max_arg_num, len(context)) for a in args_np_raw]
    args_np, mask_args_np = zip(*args_and_mask_np)


    n_lens = [len(x) for x in nodes_c]
    n_offsets = np.cumsum([0] + n_lens[:-1])
    nodes_c = np.concatenate(nodes_c)
    nodes_i = np.concatenate([np.full(x, i) for i, x in enumerate(n_lens)])
    edges = [
        np.concatenate([e + offset for e, offset in zip(es, n_offsets)], axis=0)
        for es in zip(*edges)
    ]
    roots = np.array([root + offset for root, offset in zip(roots, n_offsets)])
    c_lens = [len(x) for x in context]
    c_offsets = np.cumsum([0] + c_lens[:-1])

    context = np.concatenate([c + offset for c, offset in zip(context, n_offsets)])
    context_i = np.concatenate([np.full(x, i, dtype=np.int32) for i, x in enumerate(c_lens)])

    tactic_labels = np.array(tactic_labels)

    args_np = np.stack(args_np) # [bs, max_args, 2]
    mask_args_np = np.stack(mask_args_np) # [bs, max_args]
    _, max_args = mask_args_np.shape
    al_lens = np.tile(np.expand_dims(c_lens,1), max_args)[mask_args_np] # [total_args]
    al_offsets = np.cumsum(np.concatenate([[0], al_lens])) # [total_args+1]
    al_total = al_offsets[-1]
    al_offsets = al_offsets[:-1] # [total_args]
    [arg_num] = al_lens.shape
    an_offsets = np.arange(arg_num)+al_total
    gctx_size = global_context_size
    ag_offsets = np.arange(arg_num) * gctx_size + (al_total+arg_num)

    arg_is_global = (args_np[:,:,0] == 1)[mask_args_np] # [total_args]
    args_np = args_np[:,:,1] # [bs, max_args]
    arg_is_none = (args_np == np.expand_dims(np.array(c_lens), 1))[mask_args_np] # [total_args]
    arg_is_local = (~arg_is_none) & (~arg_is_global)
    arg_is_none = arg_is_none & ~arg_is_global

    args_flat = args_np[mask_args_np] # [total_args]
    args_loc_flat = (args_flat + al_offsets)*arg_is_local # [total_args]
    args_none_flat = an_offsets*arg_is_none  # [total_args]
    args_glob_flat = (args_flat + ag_offsets)*arg_is_global # [total_args]
    args_flat_final = args_loc_flat + args_none_flat + args_glob_flat # [total_args]

    flat_batch_np = FlatBatchNP(nodes_i=nodes_i,
                                nodes_c=nodes_c,
                                edges=edges,
                                roots=roots,
                                context_i=context_i,
                                context=context,
                                tactic_labels=tactic_labels,
                                arg_labels=args_flat_final,
                                mask_args=mask_args_np)
    return flat_batch_np

def make_flat_batch_np_empty(dataset_constants: ModelDatasetConstants) -> FlatBatchNP:
    """
    this function takes graph_constants as argument and returns empty flat_batch_np
    but of proper tensor shape
    """
    nodes_i = np.zeros([0], dtype=np.int32)
    nodes_c = np.zeros([0], dtype=np.int32)
    edges = [
        np.zeros([0, 2], dtype=np.int32) for _ in range(dataset_constants.edge_label_num)
    ]
    roots = np.zeros([0], dtype=np.int32)
    context_i = np.zeros([0], dtype=np.int32)
    context = np.zeros([0], dtype=np.int32)
    tactic_labels = np.zeros([0], dtype=np.int32)
    args_labels = np.zeros([0], dtype=np.int32)
    mask_args = np.zeros([0, dataset_constants.tactic_max_arg_num], dtype=np.bool)
    return FlatBatchNP(nodes_i=nodes_i,
                       nodes_c=nodes_c,
                       edges=edges,
                       roots=roots,
                       context_i=context_i,
                       context=context,
                       tactic_labels=tactic_labels,
                       arg_labels=args_labels,
                       mask_args=mask_args)
