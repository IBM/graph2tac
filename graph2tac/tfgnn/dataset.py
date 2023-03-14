from typing import Tuple, Dict, Optional, Any, Callable

import argparse
import yaml
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path
import numpy as np
import random

from graph2tac.loader.data_classes import GraphConstants, LoaderGraph, LoaderAction, LoaderDefinition, LoaderProofstate,\
    LoaderGraphSpec, LoaderActionSpec, LoaderDefinitionSpec, LoaderProofstateSpec
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

    def __init__(self,
                 data_dir: Path,
                 split_method : str,
                 split,
                 graph_constants = None,
                 **kwargs,
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
                                          split = get_splitter(split_method, split),
                                          **kwargs,
            )

        if graph_constants is None:
            graph_constants = self.data_server.graph_constants()

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

    def proofstates(self, label, shuffle) -> tf.data.Dataset:
        """
        Returns a pair of proof-state datasets for train and validation.

        @param shuffle: whether to shuffle the resulting datasets
        @return: a dataset of (GraphTensor, label) pairs
        """

        graph_id_spec = tf.TensorSpec([], tf.int64)
        # get proof-states
        proofstate_dataset = tf.data.Dataset.from_generator(
            lambda: self.data_server.get_datapoints(label, shuffle),
            output_signature=(LoaderProofstateSpec, LoaderActionSpec, graph_id_spec),
        )
        proofstate_dataset = proofstate_dataset.map(self._loader_to_proofstate_graph_tensor)

        return proofstate_dataset

    def definitions(self, label, shuffle) -> tf.data.Dataset:
        """
        Returns the definition dataset.

        @param shuffle: whether to shuffle the resulting dataset
        @return: a dataset with all the definition clusters
        """
        res = tf.data.Dataset.from_generator(
            lambda: self.data_server.def_cluster_subgraphs(label, shuffle),
            output_signature=LoaderDefinitionSpec,
        )
        res = res.map(self._loader_to_definition_graph_tensor)
        return res

    def get_config(self) -> dict:
        return {
            'symmetrization': self.data_server.symmetrization,
            'add_self_edges': self.data_server.add_self_edges,
            'max_subgraph_size': self.data_server.max_subgraph_size,
            'exclude_none_arguments': self.data_server.exclude_none_arguments,
            'exclude_not_faithful': self.data_server.exclude_not_faithful,
            'shuffle_random_seed': self.data_server.shuffle_random_seed,
            'restrict_to_spine': self.data_server.restrict_to_spine,
            'stop_at_definitions': self.data_server.stop_at_definitions,
            'bfs_option': self.data_server.bfs_option,
            'split_method': self.data_server.split.name,
            'split': self.data_server.split.args,
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
    def _loader_to_proofstate_graph_tensor(state: LoaderProofstate, action: LoaderAction, id: int) -> tfgnn.GraphTensor:
        """Convert loader proofstate and action format to graph tensor"""

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

    def _loader_to_definition_graph_tensor(self, defn: LoaderDefinition) -> tfgnn.GraphTensor:
        """Convert loader definition format to corresponding format for definition_data_spec"""

        num_definitions = tf.convert_to_tensor(defn.num_definitions, dtype = tf.int64)
        vectorized_definition_names = self._label_tokenizer(defn.definition_names)
        
        context = tfgnn.Context.from_fields(features={
            'num_definitions': tf.expand_dims(num_definitions, axis=0),
            'definition_name_vectors': tf.expand_dims(vectorized_definition_names.with_row_splits_dtype(tf.int32), axis=0)
        })
        return DataServerDataset._make_graph_tensor(defn.graph, context)
