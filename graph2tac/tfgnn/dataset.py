from typing import Tuple, Dict, Optional, Any, Callable

import argparse
import numpy as np
import yaml
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path

from graph2tac.loader.data_classes import GraphConstants, LoaderAction, LoaderDefinition, LoaderProofstate
from graph2tac.loader.data_server import DataServer, get_splitter, TRAIN, VALID
from graph2tac.tfgnn.graph_schema import proofstate_graph_spec, definition_graph_spec
from graph2tac.common import logger

BIDIRECTIONAL = 'bidirectional'
UNDIRECTED = 'undirected'

class Dataset:
    """
    Base class for TF-GNN datasets, subclasses should define:
        - _proofstates(label)
        - _definitions(label)
    """
    MAX_LABEL_TOKENS = 128
    MAX_PROOFSTATES = int(1e7)
    MAX_DEFINITIONS = int(1e7)
    SHUFFLE_BUFFER_SIZE = int(1e7)
    STATISTICS_BATCH_SIZE = int(1e4)

    _proofstates: Callable[[int], tf.data.Dataset]
    _definitions: Callable[[int], tf.data.Dataset]

    def __init__(self,
                 graph_constants: GraphConstants,
                 symmetrization: Optional[str] = None,
                 add_self_edges: bool = False,
                 max_subgraph_size: int = 1024,
                 exclude_none_arguments: bool = False,
                 exclude_not_faithful: bool = False
                 ):
        """
        This initialization method should be called *after* setting the _graph_constants attribute

        @param symmetrization: use BIDIRECTIONAL, UNDIRECTED or None
        @param add_self_edges: whether to add a self-loop to each node
        @param max_subgraph_size: the maximum size of the returned sub-graphs
        @param exclude_none_arguments: whether to exclude proofstates with `None` arguments
        @param exclude_not_faithful: whether to exclude proofstates which are not faithful
        """
        if symmetrization is not None and symmetrization != BIDIRECTIONAL and symmetrization != UNDIRECTED:
            raise ValueError(f'{symmetrization} is not a valid graph symmetrization scheme (use {BIDIRECTIONAL}, {UNDIRECTED} or None)')
        self.symmetrization = symmetrization
        self.add_self_edges = add_self_edges
        self.max_subgraph_size = max_subgraph_size
        self.exclude_none_arguments = exclude_none_arguments
        self.exclude_not_faithful = exclude_not_faithful
        self._graph_constants = graph_constants
        self._stats = None
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

    def get_config(self) -> dict:
        return {
            'symmetrization': self.symmetrization,
            'add_self_edges': self.add_self_edges,
            'max_subgraph_size': self.max_subgraph_size,
            'exclude_none_arguments': self.exclude_none_arguments,
            'exclude_not_faithful': self.exclude_not_faithful
        }

    def proofstates(self,
                    shuffle: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Returns a pair of proof-state datasets for train and validation.

        @param shuffle: whether to shuffle the resulting datasets
        @return: a dataset of (GraphTensor, label) pairs
        """

        data_parts = []
        for label in (TRAIN, VALID):
            # get proof-states
            proofstate_dataset = self._proofstates(label)

            # filter out proof-states with term arguments
            if self.exclude_not_faithful:
                proofstate_dataset = proofstate_dataset.filter(lambda proofstate_graph: tf.reduce_all(proofstate_graph.context['faithful'] == 1))

            # filter out proof-states with `None` arguments
            if self.exclude_none_arguments:
                proofstate_dataset = proofstate_dataset.filter(self._no_none_arguments)

            # apply the symmetrization and self-edge transformations
            proofstate_dataset = proofstate_dataset.apply(self._preprocess)
            proofstate_dataset = proofstate_dataset.cache()
            data_parts.append(proofstate_dataset)

        train, valid = data_parts

        if shuffle:
            train = train.shuffle(buffer_size=self.SHUFFLE_BUFFER_SIZE)

        return train, valid

    def definitions(self, label, shuffle: bool = False) -> tf.data.Dataset:
        """
        Returns the definition dataset.

        @param shuffle: whether to shuffle the resulting dataset
        @return: a dataset with all the definition clusters
        """
        definitions = self._definitions(label).apply(self._preprocess)
        definitions = definitions.cache()
        if shuffle:
            definitions = definitions.shuffle(buffer_size=self.SHUFFLE_BUFFER_SIZE)
        return definitions

    def tokenize_definition_graph(self, definition_graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
        """
        Tokenizes the definition names in a definition graph, replacing the 'definition_names' context feature
        with another 'vectorized_definition_names' feature.

        @param definition_graph: a scalar GraphTensor conforming to the definition_graph_spec
        @return: a scalar GraphTensor conforming to the vectorized_definition_graph_spec
        """
        context = dict(definition_graph.context.features)
        definition_names = tf.squeeze(context.pop('definition_names'), axis=0)
        vectorized_definition_names = self._label_tokenizer(definition_names)
        context['definition_name_vectors'] = tf.expand_dims(vectorized_definition_names.with_row_splits_dtype(tf.int32), axis=0)
        return definition_graph.replace_features(context=context)

    def graph_constants(self) -> GraphConstants:
        """
        Update the original GraphConstants object according to the graph transformations used.

        :return: a GraphConstants object with a possibly updated number of edge labels
        """
        graph_constants = GraphConstants(**self._graph_constants.__dict__)
        if self.symmetrization == BIDIRECTIONAL:
            graph_constants.edge_label_num *= 2
        if self.add_self_edges:
            graph_constants.edge_label_num += 1
        return graph_constants

    def stats(self) -> dict[str, dict[str, int]]:
        """
        Compute statistics for the proof-state and definition datasets.

        @return: a dictionary with statistics for the proof-state and definition datasets
        """

        if self._stats is None:
            logger.info('computing dataset statistics (this may take a while)...')

            train_proofstates, valid_proofstates = self.proofstates(shuffle=False)

            proofstate_datasets = {'train_proofstates': train_proofstates.batch(self.STATISTICS_BATCH_SIZE),
                                   'valid_proofstates': valid_proofstates.batch(self.STATISTICS_BATCH_SIZE)}
            definition_dataset = self.definitions(TRAIN).batch(self.STATISTICS_BATCH_SIZE) # TODO: what are we using definitions for here?

            stats = {}
            for name, dataset in proofstate_datasets.items():
                num_arguments = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_arguments(proofstate_graph))
                num_local_arguments = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_local_arguments(proofstate_graph))
                num_global_arguments = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_global_arguments(proofstate_graph))
                num_local_proofstates = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_local(proofstate_graph))
                num_global_proofstates = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_global(proofstate_graph))

                stats[name] = self._basic_stats(dataset)
                stats[name].update({
                    'num_arguments': int(num_arguments),
                    'num_local_arguments': int(num_local_arguments),
                    'num_global_arguments': int(num_global_arguments),
                    'num_local_proofstates': int(num_local_proofstates),
                    'num_global_proofstates': int(num_global_proofstates)
                })

            num_definitions = definition_dataset.reduce(0, lambda result, definition_graph: result + self._count_definitions(definition_graph))
            stats['definitions'] = self._basic_stats(definition_dataset)
            stats['definitions'].update({'num_definitions': int(num_definitions)})

            self._stats = stats
        return self._stats

    @staticmethod
    def _no_none_arguments(proofstate_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Check whether a proof-state contains no `None` arguments.

        @param proofstate_graph: a scalar GraphTensor for a proof-state
        @return: true if the proof-state does not contain any `None` arguments
        """
        return tf.reduce_all(tf.reduce_max(tf.stack([proofstate_graph.context['local_arguments'], proofstate_graph.context['global_arguments']], axis=-1), axis=-1) != -1)

    @staticmethod
    def _symmetrize(graph_tensor: tfgnn.GraphTensor,
                    edge_label_num: int,
                    symmetric_edges: bool = False
                    ) -> tfgnn.GraphTensor:
        """
        Duplicates the edges in a graph tensor to make them go in both directions.
        Optionally forces both edges to have the same label.

        @param graph_tensor: the input graph
        @param edge_label_num: the total number of edge labels
        @param symmetric_edges: set to true in order to have forward and backward edges with the same labels
        @return: the symmetrized graph tensor
        """
        sources = graph_tensor.edge_sets['edge'].adjacency.source
        targets = graph_tensor.edge_sets['edge'].adjacency.target
        edge_labels = graph_tensor.edge_sets['edge']['edge_label']

        new_sources = tf.concat([sources, targets], axis=-1)
        new_targets = tf.concat([targets, sources], axis=-1)
        if symmetric_edges:
            new_edge_labels = tf.concat([edge_labels, edge_labels], axis=-1)
        else:
            new_edge_labels = tf.concat([edge_labels, edge_labels + edge_label_num], axis=-1)

        adjacency = tfgnn.Adjacency.from_indices(source=('node', new_sources),
                                                 target=('node', new_targets))

        edge_set = tfgnn.EdgeSet.from_fields(features={'edge_label': new_edge_labels},
                                             sizes=graph_tensor.edge_sets['edge'].sizes * 2,
                                             adjacency=adjacency)

        return tfgnn.GraphTensor.from_pieces(context=graph_tensor.context,
                                             node_sets=graph_tensor.node_sets,
                                             edge_sets={'edge': edge_set})

    @staticmethod
    def _add_self_edges(graph_tensor: tfgnn.GraphTensor,
                        edge_label_num: int,
                        ) -> tfgnn.GraphTensor:
        """
        Creates a self-edge for each node in the graph, assigning it a new edge label.
        This function should be used *after* any calls _symmetrize_graph_tensor, never before.

        @param graph_tensor: the input graph
        @param edge_label_num: the total number of edge labels
        @return: a graph tensor with one additional self-edge per node
        """
        num_nodes = graph_tensor.node_sets['node'].sizes[0]
        sources = graph_tensor.edge_sets['edge'].adjacency.source
        targets = graph_tensor.edge_sets['edge'].adjacency.target
        edge_labels = graph_tensor.edge_sets['edge']['edge_label']

        node_ids = tf.range(num_nodes)
        new_sources = tf.concat([sources, node_ids], axis=-1)
        new_targets = tf.concat([targets, node_ids], axis=-1)
        new_edge_labels = tf.concat([edge_labels, edge_label_num * tf.ones(shape=num_nodes, dtype=tf.int64)], axis=-1)

        adjacency = tfgnn.Adjacency.from_indices(source=('node', new_sources),
                                                 target=('node', new_targets))

        edge_set = tfgnn.EdgeSet.from_fields(features={'edge_label': new_edge_labels},
                                             sizes=graph_tensor.edge_sets['edge'].sizes + num_nodes,
                                             adjacency=adjacency)

        return tfgnn.GraphTensor.from_pieces(context=graph_tensor.context,
                                             node_sets=graph_tensor.node_sets,
                                             edge_sets={'edge': edge_set})

    def _preprocess_single(self, x):
        # applying symmetrization
        edge_label_num = self._graph_constants.edge_label_num
        if self.symmetrization == UNDIRECTED:
            x = self._symmetrize(x, edge_label_num, True)
        elif self.symmetrization == BIDIRECTIONAL:
            x = self._symmetrize(x, edge_label_num, False)
            edge_label_num *= 2
        # introducing self edges
        if self.add_self_edges:
            x = self._add_self_edges(x, edge_label_num)
        return x

    def _preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Applies the symmetrization and adds self-edges to the GraphTensor objects streamed by the input dataset.

        @param dataset: the dataset of GraphTensor objects we want to process
        @return: a tf.data.Dataset object with symmetrization and self-edges added according to the dataset config
        """
        # applying symmetrization
        edge_label_num = self._graph_constants.edge_label_num
        if self.symmetrization == UNDIRECTED:
            dataset = dataset.map(lambda graph_tensor: self._symmetrize(graph_tensor, edge_label_num, True),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        elif self.symmetrization == BIDIRECTIONAL:
            dataset = dataset.map(lambda graph_tensor: self._symmetrize(graph_tensor, edge_label_num, False),
                                  num_parallel_calls=tf.data.AUTOTUNE)
            edge_label_num *= 2

        # introducing self edges
        if self.add_self_edges:
            dataset = dataset.map(lambda graph_tensor: self._add_self_edges(graph_tensor, edge_label_num),
                                  num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    @staticmethod
    def _count_graphs(graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Counts the number of graphs in a batched GraphTensor.
        """
        return graph_tensor.total_num_components

    @staticmethod
    def _count_nodes(graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Count the number of nodes in a batched GraphTensor.
        """
        return tf.reduce_sum(graph_tensor.node_sets['node'].sizes)

    @staticmethod
    def _min_nodes(graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Count the minimum number of nodes in a batched GraphTensor.
        """
        return tf.reduce_min(graph_tensor.node_sets['node'].sizes)

    @staticmethod
    def _max_nodes(graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Count the maximum number of nodes in a batched GraphTensor.
        """
        return tf.reduce_max(graph_tensor.node_sets['node'].sizes)

    @staticmethod
    def _count_edges(graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Count the number of edges in a batched GraphTensor.
        """
        return tf.reduce_sum(graph_tensor.edge_sets['edge'].sizes)

    @staticmethod
    def _min_edges(graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Count the minimum number of edges in a batched GraphTensor.
        """
        return tf.reduce_min(graph_tensor.edge_sets['edge'].sizes)

    @staticmethod
    def _max_edges(graph_tensor: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Count the maximum number of edges in a batched GraphTensor.
        """
        return tf.reduce_max(graph_tensor.edge_sets['edge'].sizes)

    @classmethod
    def _basic_stats(cls, dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Gather basic statistics about a graph dataset (applicable to proof-state and definition graphs).

        @param dataset: the tf.data.Dataset we want statistics for
        @return: a dict with basic statistics about the graphs
        """
        num_graphs = dataset.reduce(0, lambda result, graph_tensor: result + cls._count_graphs(graph_tensor))
        num_nodes = dataset.reduce(0, lambda result, graph_tensor: result + cls._count_nodes(graph_tensor))
        min_num_nodes = dataset.reduce(int(1e9), lambda result, graph_tensor: tf.math.minimum(result, cls._min_nodes(graph_tensor)))
        max_num_nodes = dataset.reduce(0, lambda result, graph_tensor: tf.math.maximum(result, cls._max_nodes(graph_tensor)))
        num_edges = dataset.reduce(0, lambda result, graph_tensor: result + cls._count_edges(graph_tensor))
        min_num_edges = dataset.reduce(int(1e9), lambda result, graph_tensor: tf.math.minimum(result, cls._min_edges(graph_tensor)))
        max_num_edges = dataset.reduce(0, lambda result, graph_tensor: tf.math.maximum(result, cls._max_edges(graph_tensor)))
        return {
            'num_graphs': int(num_graphs),
            'num_nodes': int(num_nodes),
            'min_num_nodes': int(min_num_nodes),
            'max_num_nodes': int(max_num_nodes),
            'num_edges': int(num_edges),
            'min_num_edges': int(min_num_edges),
            'max_num_edges': int(max_num_edges),
            'mean_num_nodes': float(num_nodes / num_graphs),
            'mean_num_edges': float(num_edges / num_graphs)
        }

    @staticmethod
    def _count_proofstate_arguments(proofstate_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Counts the number of arguments in a batched GraphTensor of proof-states.
        """
        return tf.shape(proofstate_graph.context['local_arguments'].flat_values)[0]

    @staticmethod
    def _count_proofstate_local_arguments(proofstate_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Counts the number of local arguments which are not `None` in a batched GraphTensor of proof-states.
        """
        return tf.reduce_sum(tf.where(proofstate_graph.context['local_arguments'].flat_values!=-1, 1, 0))

    @staticmethod
    def _count_proofstate_global_arguments(proofstate_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Counts the number of arguments (local or global) which are not `None` in a batched GraphTensor of proof-states.
        """
        argument_ids = tf.stack([proofstate_graph.context['local_arguments'].flat_values,
                                 proofstate_graph.context['global_arguments'].flat_values], axis=-1)
        return tf.reduce_sum(tf.where(tf.reduce_max(argument_ids, axis=-1)!=-1, 1, 0))

    @staticmethod
    def _count_proofstate_local(proofstate_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Counts the number of proof-states without any `None` local arguments in a batched GraphTensor of proof-states.
        """
        return tf.reduce_sum(tf.cast(tf.reduce_min(proofstate_graph.context['local_arguments'], axis=-1) > -1, dtype=tf.int32))

    @staticmethod
    def _count_proofstate_global(proofstate_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Counts the number of proof-states without any `None` arguments (local or global) in a batched GraphTensor of proof-states.
        """
        global_plus_local = proofstate_graph.context['global_arguments'] + proofstate_graph.context['local_arguments']
        return tf.reduce_sum(tf.cast(tf.reduce_min(global_plus_local, axis = -1).flat_values > -2, dtype=tf.int32))

    @staticmethod
    def _count_definitions(definition_graph: tfgnn.GraphTensor) -> tf.Tensor:
        """
        Counts the number of node labels defined in a batched GraphTensor of definitions.
        """
        return tf.reduce_sum(tf.cast(definition_graph.context['num_definitions'], dtype=tf.int32))

class DataServerDataset(Dataset):
    """
    Subclass for TF-GNN datasets which are obtained directly from the loader
    (this is presumable slower than using pre-processed `TFRecordDataset`).
    UPDATE: `TFRecordDataset` no longer used
    TODO: merge this class with Dataset
    """

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

    def __init__(self, data_dir: Path, split_method : str, split, max_subgraph_size: int = 1024, **kwargs):
        """
        @param data_dir: the directory containing the data
        @param split_method: `hash` or `file_prefix` -- the splitting procedure
        @param split: arguments for the appropriate splitting procedure
        @param max_subgraph_size: the maximum size of the returned sub-graphs
        @param kwargs: additional keyword arguments are passed on to the parent class
        """
        self.data_server = DataServer(data_dir=data_dir,
                                      max_subgraph_size=max_subgraph_size,
                                      split = get_splitter(split_method, split),
        )

        super().__init__(graph_constants=self.data_server.graph_constants(),
                         max_subgraph_size=max_subgraph_size,
                         **kwargs)

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
    def _make_bare_graph_tensor(node_labels: tf.Tensor,
                                sources: tf.Tensor,
                                targets: tf.Tensor,
                                edge_labels: tf.Tensor
                                ) -> tfgnn.GraphTensor:
        """
        Converts the data loader's graph representation into a TF-GNN compatible GraphTensor.

        @param node_labels: tf.Tensor of node labels (dtype=tf.int64)
        @param sources: tf.Tensor of edge sources (dtype=tf.int32)
        @param targets: tf.Tensor of edge targets (dtype=tf.int32)
        @param edge_labels: tf.Tensor of edge labels (dtype=tf.int64)
        @return: a GraphTensor object that is compatible with the `bare_graph_spec` in `graph_schema.py`
        """
        node_set = tfgnn.NodeSet.from_fields(features={'node_label': node_labels},
                                             sizes=tf.shape(node_labels))

        adjacency = tfgnn.Adjacency.from_indices(source=('node', sources),
                                                 target=('node', targets))

        edge_set = tfgnn.EdgeSet.from_fields(features={'edge_label': edge_labels},
                                             sizes=tf.shape(edge_labels),
                                             adjacency=adjacency)

        return tfgnn.GraphTensor.from_pieces(node_sets={'node': node_set}, edge_sets={'edge': edge_set})

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

    @classmethod
    @tf.function(input_signature = proofstate_data_spec)
    def _make_proofstate_graph_tensor(cls,
                                      state: Tuple,
                                      action: Tuple,
                                      graph_id: tf.Tensor) -> tfgnn.GraphTensor:
        """
        Converts the data loader's proof-state representation into a TF-GNN compatible GraphTensor.

        @param state: the tuple containing tf.Tensor objects for the graph structure
        @param action: the tuple containing tf.Tensor objects for the tactic and arguments
        @param graph_id: the id of the graph
        @return: a GraphTensor object that is compatible with the `proofstate_graph_spec` in `graph_schema.py`
        """
        loader_graph, root, context, proofstate_info = state
        context_node_ids, available_global_context = context
        node_labels, edges, edge_labels, _ = loader_graph
        sources = edges[:, 0]
        targets = edges[:, 1]
        proofstate_name, proofstate_step, proofstate_faithful = proofstate_info

        bare_graph_tensor = cls._make_bare_graph_tensor(node_labels, sources, targets, edge_labels)

        local_context_length = tf.shape(context_node_ids, out_type=tf.int64)[0]

        tactic_id, tactic_args = action
        local_arguments, global_arguments = cls._split_action_arguments(tactic_args, local_context_length)

        context = tfgnn.Context.from_fields(features={
            'tactic': tf.expand_dims(tactic_id, axis=0),
            'local_context_ids': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(context_node_ids, axis=0),
                                                             row_splits_dtype=tf.int32),
            'global_context_ids': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(available_global_context, axis=0),
                                                              row_splits_dtype=tf.int32),
            'local_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(local_arguments, axis=0),
                                                           row_splits_dtype=tf.int32),
            'global_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(global_arguments, axis=0),
                                                            row_splits_dtype=tf.int32),
            'graph_id': tf.expand_dims(graph_id, axis=0),
            'name': tf.expand_dims(proofstate_name, axis=0),
            'step': tf.expand_dims(proofstate_step, axis=0),
            'faithful': tf.expand_dims(proofstate_faithful, axis=0)
        })

        return tfgnn.GraphTensor.from_pieces(node_sets=bare_graph_tensor.node_sets,
                                             edge_sets=bare_graph_tensor.edge_sets,
                                             context=context)
    
    @staticmethod
    def _wrap_make_proofstate_graph_tensor(x) -> tfgnn.GraphTensor:
        state, action, graph_id = x
        return DataServerDataset._make_proofstate_graph_tensor(state, action, graph_id)
    
    @classmethod
    @tf.function(input_signature = definition_data_spec)
    def _make_definition_graph_tensor(cls,
                                      loader_graph: Tuple,
                                      num_definitions: tf.Tensor,
                                      definition_names: tf.Tensor
                                      ) -> tfgnn.GraphTensor:
        """
        Converts the data loader's definition cluster representation into a TF-GNN compatible GraphTensor.

        @param loader_graph: Tuple of tf.Tensor objects encoding the graph in the loader's format
        @param num_definitions: tf.Tensor for the number of labels being defined (dtype=tf.int64)
        @return: a GraphTensor object that is compatible with the `definition_graph_spec` in `graph_schema.py`
        """
        node_labels, edges, edge_labels, _ = loader_graph
        sources = edges[:, 0]
        targets = edges[:, 1]

        bare_graph_tensor = cls._make_bare_graph_tensor(node_labels, sources, targets, edge_labels)

        context = tfgnn.Context.from_fields(features={
            'num_definitions': tf.expand_dims(num_definitions, axis=0),
            'definition_names': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(definition_names, axis=0),
                                                            row_splits_dtype=tf.int32)
        })

        return tfgnn.GraphTensor.from_pieces(node_sets=bare_graph_tensor.node_sets,
                                             edge_sets=bare_graph_tensor.edge_sets,
                                             context=context)

    @staticmethod
    def _wrap_make_definition_graph_tensor(x) -> tfgnn.GraphTensor:
        loader_graph, num_definitions, definition_names = x
        return DataServerDataset._make_definition_graph_tensor(loader_graph, num_definitions, definition_names)

    @staticmethod
    def _loader_to_proofstate_data(loader_data: tuple[LoaderProofstate, LoaderAction, int]) -> tuple:
        """Convert loader proofstate and action format to corresponding format for proofstate_data_spec"""
        state, action, id = loader_data 
        # state
        graph = state.graph
        graph_tuple = (graph.nodes, graph.edges, graph.edge_labels, graph.edge_offsets)
        context_tuple = (state.context.local_context, state.context.global_context)
        metadata_tuple = (state.metadata.name, state.metadata.step, state.metadata.is_faithful)
        state_tuple = (graph_tuple, state.root, context_tuple, metadata_tuple)

        # action
        action_tuple = (np.array(action.tactic_id, dtype="int64"), action.args.astype("int64"))

        return (state_tuple, action_tuple, np.array(id, dtype="int64"))

    def _proofstates(self, label : int = TRAIN) -> tf.data.Dataset:
        """
        Returns a dataset with all the proof-states.

        @return: a tf.data.Dataset streaming GraphTensor objects
        """
        proofstates = tf.data.Dataset.from_generator(
            lambda: map(self._wrap_make_proofstate_graph_tensor, map(self._loader_to_proofstate_data, self.data_server.get_datapoints(label, shuffled=False, as_text=False))),
            output_signature=proofstate_graph_spec)
        return proofstates

    @staticmethod
    def _loader_to_definition_data(defn: LoaderDefinition) -> tuple:
        """Convert loader definition format to corresponding format for definition_data_spec"""
        graph = defn.graph
        graph_tuple = (graph.nodes, graph.edges, graph.edge_labels, graph.edge_offsets)

        return (graph_tuple, defn.num_definitions, defn.definition_names)

    def _definitions(self, label) -> tf.data.Dataset:
        """
        Returns a dataset with all the definitions.

        @return: a tf.data.Dataset streaming GraphTensor objects
        """
        dataset = tf.data.Dataset.from_generator(
            lambda: map(self._wrap_make_definition_graph_tensor, map(self._loader_to_definition_data, self.data_server.def_cluster_subgraphs(label))),
            output_signature=definition_graph_spec
        )
        return dataset
