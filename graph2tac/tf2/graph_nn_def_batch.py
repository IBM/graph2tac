"""
defines batch dataclass used by the specific gnn defined in graph_nn.py
the classes in these modules probably should be packaged together with
GraphToLogits class in graph_nn
"""


from dataclasses import dataclass
from typing import List
import numpy as np

from graph2tac.loader.data_server import graph_as, LoaderDefinition

# TODO(jrute): This code is unique to the model in model.py,
# and it now depends on the dataset constants saved with that model.
# So we might as well move it there alongside np_to_tensor.
# Then we can avoid writing np_to_tensor(flat_batch_np(...)) everywhere.
from graph2tac.tf2.model_params import ModelDatasetConstants

@dataclass
class FlatDefBatchNP:
    """
    Numpy data used for training a batch of definitions
    """
    # this a format to collect batch_size datapoints
    # each datapoint is a graph tactic and argument
    batch_size : int

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

    # root nodes for all datapoints
    roots: np.ndarray  # shape=[batch size]  dtype=np.int32

    # batch index for each context element
    roots_i: np.ndarray  # shape=[sum(lens of roots)]  dtype=np.int32
    # node pointers from datapoint contex concatenated
    roots_c: np.ndarray  # shape=[sum(lens of roots)]  dtype=np.int32

    def counts(self):
        """
        returns string representation of batch counts
        """
        num_nodes = len(self.nodes_i)
        num_edge_types = len(self.edges)
        num_edges = sum(e.shape[0] for e in self.edges)
        num_roots = len(self.roots)
        num_datapoints = self.batch_size
        return (f"{num_nodes} nodes "
                f"{num_edges} edges of {num_edge_types} classes "
                f"{num_roots} roots "
                f"{num_datapoints} datapoints ")


def make_flat_def_batch_np(batch: List[LoaderDefinition]) -> FlatDefBatchNP:
    """
    this function forms FlatDefBatchNP of numpy arrays from a non-empty list of datapoints
    for empty input consider a function make_flat_def_batch_np_empty defined below (with different API)
    """
    assert len(batch) > 0
    batch_size = len(batch)
    # nodes_c, edges, root_nums = zip(*batch)
    graphs, root_nums, _ = zip(*batch)
    nodes_c, edges = zip(* map(lambda x: graph_as("tf2", x), graphs))



    roots = [np.arange(rn) for rn in root_nums]

    n_lens = [len(x) for x in nodes_c]
    n_offsets = np.cumsum([0] + n_lens[:-1])
    nodes_c = np.concatenate(nodes_c)
    nodes_i = np.concatenate([np.full(x, i) for i, x in enumerate(n_lens)])
    edges = [
        np.concatenate([e + offset for e, offset in zip(es, n_offsets)], axis=0)
        for es in zip(*edges)
    ]
    roots_i = np.concatenate([np.full(rn, i) for i, rn in enumerate(root_nums)])
    roots = np.concatenate([np.arange(rn) + offset for rn, offset in zip(root_nums, n_offsets)])
    roots_c = nodes_c[roots]
    nodes_c[roots] = 3

    flat_def_batch_np = FlatDefBatchNP(batch_size=batch_size,
                                       nodes_i=nodes_i,
                                       nodes_c=nodes_c,
                                       edges=edges,
                                       roots=roots,
                                       roots_i=roots_i,
                                       roots_c=roots_c)
    return flat_def_batch_np

def make_flat_def_batch_np_empty(dataset_constants: ModelDatasetConstants) -> FlatDefBatchNP:
    """
    this function takes graph_constants as argument and returns empty flat_def_batch_np
    but of proper tensor shape
    """
    batch_size = 0
    nodes_i = np.zeros([0], dtype=np.int32)
    nodes_c = np.zeros([0], dtype=np.int32)
    edges = [
        np.zeros([0, 2], dtype=np.int32) for _ in range(dataset_constants.edge_label_num)
    ]
    roots = np.zeros([0], dtype=np.int32)
    roots_i = np.zeros([0], dtype=np.int32)
    roots_c = np.zeros([0], dtype=np.int32)

    flat_def_batch_np = FlatDefBatchNP(batch_size=batch_size,
                                       nodes_i=nodes_i,
                                       nodes_c=nodes_c,
                                       edges=edges,
                                       roots=roots,
                                       roots_i=roots_i,
                                       roots_c=roots_c)
    return flat_def_batch_np
