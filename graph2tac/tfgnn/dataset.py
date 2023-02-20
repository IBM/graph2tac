from typing import Tuple, Dict, Optional, Any, Callable

import argparse
import yaml
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path
import numpy as np
import random

from graph2tac.loader.data_classes import GraphConstants, LoaderGraph, LoaderAction, LoaderDefinition, LoaderProofstate
from graph2tac.loader.data_server import DataServer, get_splitter, TRAIN, VALID
from graph2tac.tfgnn.graph_schema import proofstate_graph_spec, vectorized_definition_graph_spec
from graph2tac.common import logger

BIDIRECTIONAL = 'bidirectional'
UNDIRECTED = 'undirected'

class DataServerDataset:
    """
    Class for TF-GNN datasets which are obtained directly from the loader
    """
    MAX_LABEL_TOKENS = 128
    MAX_PROOFSTATES = int(1e7)
    MAX_DEFINITIONS = int(1e7)

    # tf.TensorSpec for the data coming from the loader
    node_labels_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64, name='node_labels')
    edges_spec = tf.TensorSpec(shape=(None,2), dtype=tf.int32, name='edges')
    edge_labels_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64, name='edge_labels')
    edges_offset_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32, name='edges_offset')
    loader_graph_spec = (node_labels_spec, edges_spec, edge_labels_spec, edges_offset_spec)

    root_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='root')

    local_context_ids_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64, name='local_context_ids')
    global_context_ids_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64, name='global_context_ids')
    context_spec = (local_context_ids_spec, global_context_ids_spec)

    name_spec = tf.TensorSpec(shape=(), dtype=tf.string, name='name')
    step_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='step')
    faithful_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='faithful')
    proofstate_info_spec = (name_spec, step_spec, faithful_spec)

    state_spec = (loader_graph_spec, root_spec, context_spec, proofstate_info_spec)

    tactic_id_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='tactic_id')
    arguments_array_spec = tf.TensorSpec(shape=(None, 2), dtype=tf.int64, name='arguments_array')
    action_spec = (tactic_id_spec, arguments_array_spec)

    graph_id_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='graph_id')

    num_definitions_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='num_definitions')
    definition_names_spec = tf.TensorSpec(shape=(None,), dtype=tf.string, name='definition_names')

    proofstate_data_spec = (state_spec, action_spec, graph_id_spec)
    definition_data_spec = (loader_graph_spec, num_definitions_spec, definition_names_spec)

    def __init__(self,
                 data_dir: Path,
                 split_method : str,
                 split,
                 max_subgraph_size: int = 1024,
                 symmetrization: Optional[str] = None,
                 add_self_edges: bool = False,
                 exclude_none_arguments: bool = False,
                 exclude_not_faithful: bool = False,
                 graph_constants = None,
    ):
        """
        @param data_dir: the directory containing the data
        @param split_method: `hash` or `file_prefix` -- the splitting procedure
        @param split: arguments for the appropriate splitting procedure
        @param max_subgraph_size: the maximum size of the returned sub-graphs
        @param kwargs: additional keyword arguments are passed on to the parent class
        @param symmetrization: use BIDIRECTIONAL, UNDIRECTED or None
        @param add_self_edges: whether to add a self-loop to each node
        @param exclude_none_arguments: whether to exclude proofstates with `None` arguments
        @param exclude_not_faithful: whether to exclude proofstates which are not faithful
        """

        if data_dir is not None:
            self.data_server = DataServer(data_dir=data_dir,
                                          max_subgraph_size=max_subgraph_size,
                                          split = get_splitter(split_method, split),
            )

        if graph_constants is None:
            graph_constants = self.data_server.graph_constants()

        if symmetrization is not None and symmetrization != BIDIRECTIONAL and symmetrization != UNDIRECTED:
            raise ValueError(f'{symmetrization} is not a valid graph symmetrization scheme (use {BIDIRECTIONAL}, {UNDIRECTED} or None)')
        self.symmetrization = symmetrization
        self.add_self_edges = add_self_edges
        self.max_subgraph_size = max_subgraph_size
        self.exclude_none_arguments = exclude_none_arguments
        self.exclude_not_faithful = exclude_not_faithful
        self.graph_constants = graph_constants
        vocabulary = [
            chr(i) for i in range(ord('a'), ord('z')+1)
        ] + [
            chr(i) for i in range(ord('A'), ord('Z')+1)
        ] + [
            chr(i) for i in range(ord('0'), ord('9')+1)
        ] + ["_", "'", "."]
        self._label_tokenizer = tf.keras.layers.TextVectorization(standardize=None,
                                                                  split='character',
                                                                  ngrams=None,
                                                                  output_mode='int',
                                                                  max_tokens=self.MAX_LABEL_TOKENS,
                                                                  vocabulary = vocabulary,
                                                                  ragged=True)
        #self._label_tokenizer.adapt(graph_constants.label_to_names)

    def proofstates(self, label, shuffle) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Returns a pair of proof-state datasets for train and validation.

        @param shuffle: whether to shuffle the resulting datasets
        @return: a dataset of (GraphTensor, label) pairs
        """

        # get proof-states
        proofstate_dataset = tf.data.Dataset.from_generator(
            lambda: self._proofstates_generator(label, shuffle),
            output_signature=proofstate_graph_spec
        )

        # filter out proof-states with term arguments
        if self.exclude_not_faithful:
            proofstate_dataset = proofstate_dataset.filter(lambda proofstate_graph: tf.reduce_all(proofstate_graph.context['faithful'] == 1))

        # filter out proof-states with `None` arguments
        if self.exclude_none_arguments:
            proofstate_dataset = proofstate_dataset.filter(self._no_none_arguments)

        return proofstate_dataset

    def definitions(self, label, shuffle) -> tf.data.Dataset:
        """
        Returns the definition dataset.

        @param shuffle: whether to shuffle the resulting dataset
        @return: a dataset with all the definition clusters
        """
        return tf.data.Dataset.from_generator(
            lambda: self._definitions_generator(label, shuffle),
            output_signature=vectorized_definition_graph_spec
        )

    @staticmethod
    def _no_none_arguments(proofstate_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Check whether a proof-state contains no `None` arguments.

        @param proofstate_graph: a scalar GraphTensor for a proof-state
        @return: true if the proof-state does not contain any `None` arguments
        """
        return tf.reduce_all(tf.reduce_max(tf.stack([proofstate_graph.context['local_arguments'], proofstate_graph.context['global_arguments']], axis=-1), axis=-1) != -1)

    def get_config(self) -> dict:
        return {
            'symmetrization': self.symmetrization,
            'add_self_edges': self.add_self_edges,
            'max_subgraph_size': self.max_subgraph_size,
            'exclude_none_arguments': self.exclude_none_arguments,
            'exclude_not_faithful': self.exclude_not_faithful
        }

    @classmethod
    def from_yaml_config(cls, data_dir: Path, yaml_filepath: Path) -> "DataServerDataset":
        """
        Create a DataLoaderDataset from a YAML configuration file

        @param data_dir: the directory containing the data
        @param yaml_filepath: the filepath to the YAML file containing the
        @return: a DataLoaderDataset object
        """
        with yaml_filepath.open() as yaml_file:
            dataset_config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        return cls(data_dir=data_dir, **dataset_config)

    @staticmethod
    def _make_graph_tensor(graph: LoaderGraph, context) -> tfgnn.GraphTensor:
        """
        Converts the data loader's graph representation into a TF-GNN compatible GraphTensor.

        @param node_labels: tf.Tensor of node labels (dtype=tf.int64)
        @param sources: tf.Tensor of edge sources (dtype=tf.int32)
        @param targets: tf.Tensor of edge targets (dtype=tf.int32)
        @param edge_labels: tf.Tensor of edge labels (dtype=tf.int64)
        @return: a GraphTensor object that is compatible with the `bare_graph_spec` in `graph_schema.py`
        """
        node_labels = tf.convert_to_tensor(graph.nodes, dtype = tf.int64)
        edge_labels = tf.convert_to_tensor(graph.edge_labels, dtype = tf.int64)
        sources = tf.convert_to_tensor(graph.edges[:,0], dtype = tf.int32)
        targets = tf.convert_to_tensor(graph.edges[:,1], dtype = tf.int32)

        node_set = tfgnn.NodeSet.from_fields(features={'node_label': node_labels},
                                             sizes=tf.shape(node_labels))

        adjacency = tfgnn.Adjacency.from_indices(source=('node', sources),
                                                 target=('node', targets))

        edge_set = tfgnn.EdgeSet.from_fields(features={'edge_label': edge_labels},
                                             sizes=tf.shape(edge_labels),
                                             adjacency=adjacency)

        return tfgnn.GraphTensor.from_pieces(node_sets={'node': node_set}, edge_sets={'edge': edge_set}, context = context)

    @staticmethod
    def _split_action_arguments(arguments_array: tf.Tensor, local_context_length: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Convert an action's arguments from loader format into the corresponding TF-GNN format.

        @param arguments_array: the argument array, as returned by the DataServer
        @param local_context_length: the size of the local context in the proofstate
        @return: a pair with a tf.Tensor for the local arguments and a tf.Tensor for the global arguments
        """
        is_global_argument, argument_ids = tf.unstack(arguments_array, axis=1)

        # there can still be local arguments that are None
        is_valid_local_argument = tf.where(is_global_argument == 0, argument_ids, int(1e9)) < local_context_length
        local_arguments = tf.where(is_valid_local_argument, argument_ids, -1)

        # global arguments go from 0 to node_label_num-base_node_label_num
        global_arguments = tf.where(is_global_argument == 1, argument_ids, -1)

        return local_arguments, global_arguments

    @staticmethod
    def _loader_to_proofstate_graph_tensor(loader_data: tuple[LoaderProofstate, LoaderAction, int]) -> tfgnn.GraphTensor:
        """Convert loader proofstate and action format to graph tensor"""
        state, action, id = loader_data 

        # state
        available_global_context = tf.convert_to_tensor(state.context.global_context, dtype = tf.int64)
        context_node_ids = tf.convert_to_tensor(state.context.local_context, dtype = tf.int64)

        local_context_length = tf.shape(context_node_ids, out_type=tf.int64)[0]

        tactic_args = tf.convert_to_tensor(action.args, dtype = tf.int64)
        local_arguments, global_arguments = DataServerDataset._split_action_arguments(tactic_args, local_context_length)

        context = tfgnn.Context.from_fields(features={
            'tactic': tf.convert_to_tensor([action.tactic_id], tf.int64),
            'local_context_ids': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(context_node_ids, axis=0),
                                                             row_splits_dtype=tf.int32),
            'global_context_ids': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(available_global_context, axis=0),
                                                              row_splits_dtype=tf.int32),
            'local_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(local_arguments, axis=0),
                                                           row_splits_dtype=tf.int32),
            'global_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(global_arguments, axis=0),
                                                            row_splits_dtype=tf.int32),
            'graph_id': tf.expand_dims(tf.convert_to_tensor(id, dtype = tf.int64), axis=0),
            'name': tf.expand_dims(state.metadata.name, axis=0),
            'step': tf.expand_dims(tf.convert_to_tensor(state.metadata.step, dtype = tf.int64), axis=0),
            'faithful': tf.expand_dims(tf.convert_to_tensor(state.metadata.is_faithful, dtype = tf.int64), axis=0)
        })
        return DataServerDataset._make_graph_tensor(state.graph, context)

    def _loader_to_definition_graph_tensor(self, defn: LoaderDefinition) -> tuple:
        """Convert loader definition format to corresponding format for definition_data_spec"""

        num_definitions = tf.convert_to_tensor(defn.num_definitions, dtype = tf.int64)
        vectorized_definition_names = self._label_tokenizer(defn.definition_names)
        
        context = tfgnn.Context.from_fields(features={
            'num_definitions': tf.expand_dims(num_definitions, axis=0),
            'definition_name_vectors': tf.expand_dims(vectorized_definition_names.with_row_splits_dtype(tf.int32), axis=0)
        })
        return DataServerDataset._make_graph_tensor(defn.graph, context)

    def _proofstates_generator(self, *labels, shuffle = False):
        indices = self.data_server.datapoint_indices(*labels)
        if shuffle: random.shuffle(indices)
        for i in indices:
            loader_proofstate = self.data_server.datapoint_graph(i)
            yield self._loader_to_proofstate_graph_tensor(loader_proofstate)

    def _definitions_generator(self, *labels, shuffle = False):
        indices = self.data_server.def_cluster_indices(*labels)
        if shuffle: random.shuffle(indices)
        for i in indices:
            loader_proofstate = self.data_server.def_cluster_subgraph(i)
            yield self._loader_to_definition_graph_tensor(loader_proofstate)
