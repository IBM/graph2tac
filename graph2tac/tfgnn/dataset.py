from typing import Tuple, Dict, Iterable, Optional, Any, Callable

import argparse
import yaml
import tensorflow as tf
import tensorflow_gnn as tfgnn
from pathlib import Path
from math import ceil

from graph2tac.loader.data_server import DataServer, GraphConstants
from graph2tac.tfgnn.graph_schema import proofstate_graph_spec, definition_graph_spec


BIDIRECTIONAL = 'bidirectional'
UNDIRECTED = 'undirected'


class Dataset:
    """
    Base class for TF-GNN datasets, subclasses should populate the _graph_constants
    attribute and implement the following methods:
        - _train_proofstates
        - _valid_proofstates
        - _definitions
    """
    SHUFFLE_BUFFER_SIZE = int(1e4)
    STATISTICS_BATCH_SIZE = int(1e4)

    _train_proofstates: Callable[[], tf.data.Dataset]
    _valid_proofstates: Callable[[], tf.data.Dataset]
    _definitions: Callable[[], tf.data.Dataset]
    _graph_constants: GraphConstants

    def __init__(self,
                 symmetrization: Optional[str] = None,
                 add_self_edges: bool = False,
                 split_random_seed: int = 0,
                 max_subgraph_size: int = 1024):
        """
        @param symmetrization: use BIDIRECTIONAL, UNDIRECTED or None
        @param add_self_edges: whether to add a self-loop to each node
        @param split_random_seed: the seed for random splitting
        @param max_subgraph_size: the maximum size of the returned sub-graphs
        """
        if symmetrization is not None and symmetrization != BIDIRECTIONAL and symmetrization != UNDIRECTED:
            raise ValueError(f'{symmetrization} is not a valid graph symmetrization scheme (use {BIDIRECTIONAL}, {UNDIRECTED} or None)')
        self.symmetrization = symmetrization
        self.add_self_edges = add_self_edges
        self.split_random_seed = split_random_seed
        self.max_subgraph_size = max_subgraph_size
        self.options = tf.data.Options()
        self._stats = None

    def get_config(self) -> dict:
        return {
            'symmetrization': self.symmetrization,
            'add_self_edges': self.add_self_edges,
            'split_random_seed': self.split_random_seed,
            'max_subgraph_size': self.max_subgraph_size
        }

    def train_proofstates(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Returns the training proof-state dataset.
        """
        return self._preprocess(self._train_proofstates(), shuffle=shuffle)

    def valid_proofstates(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Returns the validation proof-state dataset.
        """
        return self._preprocess(self._valid_proofstates(), shuffle=shuffle)

    def definitions(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Returns the definition dataset.
        """
        return self._preprocess(self._definitions(), shuffle=shuffle)

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

    def dump(self, tfrecord_prefix: Path) -> None:
        """
        Create TFRecord files for this dataset.
        WARNING: this function will overwrite files if they already exist!

        @param tfrecord_prefix: the prefix for the TFRecord files we want to create
        """
        train_tfrecord_filepath = self._train_tfrecord_filepath(tfrecord_prefix)
        valid_tfrecord_filepath = self._valid_tfrecord_filepath(tfrecord_prefix)
        definition_tfrecord_filepath = self._definition_tfrecord_filepath(tfrecord_prefix)
        graph_constants_filepath = self._graph_constants_filepath(tfrecord_prefix)

        # Make sure we dump the original data, not the pre-processed data!
        datasets = {
            train_tfrecord_filepath: self._train_proofstates(),
            valid_tfrecord_filepath: self._valid_proofstates(),
            definition_tfrecord_filepath: self._definitions(),
        }
        for tfrecord_filepath, dataset in datasets.items():
            print(f'exporting {tfrecord_filepath}')
            with tf.io.TFRecordWriter(str(tfrecord_filepath.absolute())) as writer:
                for graph_tensor in iter(dataset):
                    example = tfgnn.write_example(graph_tensor)
                    writer.write(example.SerializeToString())

        print(f'exporting {graph_constants_filepath}')
        graph_constants_filepath.write_text(yaml.dump(self._graph_constants.__dict__))

    def stats(self) -> dict[str, dict[str, int]]:
        """
        Compute statistics for the various datasets.
        """
        if self._stats is None:
            print('computing dataset statistics (this may take a while)...')

            proofstate_datasets = {'train_proofstates': self.train_proofstates().batch(self.STATISTICS_BATCH_SIZE),
                                   'valid_proofstates': self.valid_proofstates().batch(self.STATISTICS_BATCH_SIZE)}
            definition_dataset = self.definitions().batch(self.STATISTICS_BATCH_SIZE)

            self._stats = {}
            for name, dataset in proofstate_datasets.items():
                num_arguments = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_arguments(proofstate_graph))
                num_local_arguments = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_local_arguments(proofstate_graph))
                num_global_arguments = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_global_arguments(proofstate_graph))
                num_local_proofstates = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_local(proofstate_graph))
                num_global_proofstates = dataset.reduce(0, lambda result, proofstate_graph: result + self._count_proofstate_global(proofstate_graph))

                self._stats[name] = self._basic_stats(dataset)
                self._stats[name].update({
                    'num_arguments': int(num_arguments),
                    'num_local_arguments': int(num_local_arguments),
                    'num_global_arguments': int(num_global_arguments),
                    'num_local_proofstates': int(num_local_proofstates),
                    'num_global_proofstates': int(num_global_proofstates)
                })

            num_definitions = definition_dataset.reduce(0, lambda result, definition_graph: result + self._count_definitions(definition_graph))
            self._stats['definitions'] = self._basic_stats(definition_dataset)
            self._stats['definitions'].update({'num_definitions': int(num_definitions)})
        return self._stats

    def train_proofstates_steps_per_epoch(self, batch_size: int) -> int:
        """
        Return the number of train steps in an epoch, for the given file size.
        """
        return ceil(self.stats().get('train_proofstates').get('num_graphs')/batch_size)

    def valid_proofstates_steps_per_epoch(self, batch_size: int) -> int:
        """
        Return the number of validation steps in an epoch, for the given file size.
        """
        return ceil(self.stats().get('valid_proofstates').get('num_graphs')/batch_size)

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

    def _preprocess(self, dataset: tf.data.Dataset, shuffle: bool = True) -> tf.data.Dataset:
        """
        @param dataset: the dataset we want to prepare
        @param shuffle: whether to shuffle the data or leave it as is
        @return: a tf.data.Dataset object with symmetrization and self-edges added according to the dataset config
        """
        # applying symmetrization
        edge_label_num = self._graph_constants.edge_label_num
        if self.symmetrization == UNDIRECTED:
            dataset = dataset.map(lambda graph_tensor: self._symmetrize(graph_tensor, edge_label_num, True))
        elif self.symmetrization == BIDIRECTIONAL:
            dataset = dataset.map(lambda graph_tensor: self._symmetrize(graph_tensor, edge_label_num, False))
            edge_label_num *= 2

        # introducing self edges
        if self.add_self_edges:
            dataset = dataset.map(lambda graph_tensor: self._add_self_edges(graph_tensor, edge_label_num))

        # shuffling
        if shuffle:
            dataset = dataset.shuffle(self.SHUFFLE_BUFFER_SIZE)

        # applying options
        dataset = dataset.with_options(self.options)

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

    def _train_tfrecord_filepath(self, tfrecord_prefix: Path) -> Path:
        """
        Returns the filepath for the train proof-states TFRecord file with prefix `tfrecord_prefix`.
        """
        return tfrecord_prefix.with_stem(f'{tfrecord_prefix.stem}-train-{self.split_random_seed}-{self.max_subgraph_size}').with_suffix('.tfrecord')

    def _valid_tfrecord_filepath(self, tfrecord_prefix: Path) -> Path:
        """
        Returns the filepath for the validation proof-states TFRecord file with prefix `tfrecord_prefix`.
        """
        return tfrecord_prefix.with_stem(f'{tfrecord_prefix.stem}-valid-{self.split_random_seed}-{self.max_subgraph_size}').with_suffix('.tfrecord')

    def _definition_tfrecord_filepath(self, tfrecord_prefix: Path) -> Path:
        """
        Returns the filepath for the definition TFRecord file with prefix `tfrecord_prefix`.
        """
        return tfrecord_prefix.with_stem(f'{tfrecord_prefix.stem}-definition-{self.max_subgraph_size}').with_suffix('.tfrecord')

    @staticmethod
    def _graph_constants_filepath(tfrecord_prefix: Path) -> Path:
        """
        Returns the filepath for the graph constants YAML file with prefix `tfrecord_prefix`.
        """
        return tfrecord_prefix.with_stem(f'{tfrecord_prefix.stem}-constants').with_suffix('.yml')


class TFRecordDataset(Dataset):
    """
    Subclass for TF-GNN datasets which are read from TFRecord files (no GraphTensor construction here,
    so this is presumably more efficient than using `DataLoaderDataset`).
    """
    def __init__(self, tfrecord_prefix: Path, **kwargs):
        """
        @param tfrecord_prefix: the prefix for the .tfrecord files containing the data
        @param kwargs: additional keyword arguments are passed on to the parent class
        """
        super().__init__(**kwargs)
        self.tfrecord_prefix = tfrecord_prefix

        graph_constants_filepath = self._graph_constants_filepath(tfrecord_prefix)
        if not graph_constants_filepath.exists():
            raise FileNotFoundError(f'{graph_constants_filepath} does not exist')

        with graph_constants_filepath.open('r') as yml_file:
            self._graph_constants = GraphConstants(**yaml.load(yml_file, Loader=yaml.UnsafeLoader))

    @classmethod
    def from_yaml_config(cls, tfrecord_prefix: Path, yaml_filepath: Path) -> "TFRecordDataset":
        """
        Create a TFRecordDataset from a YAML configuration file

        @param tfrecord_prefix: the prefix for the .tfrecord files containing the data
        @param yaml_filepath: the filepath to the YAML file containing the
        @return: a TFRecordDataset object
        """
        with yaml_filepath.open() as yaml_file:
            dataset_config = yaml.load(yaml_file, Loader=yaml.SafeLoader)
        return cls(tfrecord_prefix=tfrecord_prefix, **dataset_config)

    @classmethod
    def _tfdataset_from_tfrecord(cls, tfrecord_filepath: Path, graph_spec: tfgnn.GraphTensorSpec) -> tf.data.Dataset:
        """
        Create a TFDataset from a TFRecord file.

        @param tfrecord_filepath: the path to the .tfrecord file containing the data
        @param graph_spec: the graph spec for the graphs contained in the dataset
        @return: a tf.data.Dataset containing all the graphs
        """
        # get the data from the TFRecord
        dataset = tf.data.TFRecordDataset(filenames=[str(tfrecord_filepath.absolute())])

        # parse it into GraphTensor objects
        dataset = dataset.map(lambda graph_tensor: tfgnn.parse_single_example(graph_spec, graph_tensor))

        return dataset

    def _train_proofstates(self) -> tf.data.Dataset:
        train_tfrecord_filepath = self._train_tfrecord_filepath(self.tfrecord_prefix)
        if not train_tfrecord_filepath.exists():
            raise FileNotFoundError(f'{train_tfrecord_filepath} does not exist')
        return self._tfdataset_from_tfrecord(train_tfrecord_filepath, graph_spec=proofstate_graph_spec)

    def _valid_proofstates(self) -> tf.data.Dataset:
        valid_tfrecord_filepath = self._valid_tfrecord_filepath(self.tfrecord_prefix)
        if not valid_tfrecord_filepath.exists():
            raise FileNotFoundError(f'{valid_tfrecord_filepath} does not exist')
        return self._tfdataset_from_tfrecord(valid_tfrecord_filepath, graph_spec=proofstate_graph_spec)

    def _definitions(self) -> tf.data.Dataset:
        definition_tfrecord_filepath = self._definition_tfrecord_filepath(self.tfrecord_prefix)
        if not definition_tfrecord_filepath.exists():
            raise FileNotFoundError(f'{definition_tfrecord_filepath} does not exist')
        return self._tfdataset_from_tfrecord(definition_tfrecord_filepath, graph_spec=definition_graph_spec)


class DataLoaderDataset(Dataset):
    """
    Subclass for TF-GNN datasets which are obtained directly from the loader (this is presumable slower
    than using pre-processed `TFRecordDataset`).
    """

    # tf.TensorSpec for the data coming from the loader
    node_labels_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64, name='node_labels')
    sources_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32, name='sources')
    targets_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32, name='targets')
    edge_labels_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64, name='edge_labels')
    root_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='root')
    context_node_ids_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64, name='context_node_ids')
    state_spec = (node_labels_spec, sources_spec, targets_spec, edge_labels_spec, root_spec, context_node_ids_spec)

    tactic_id_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='tactic_id')
    arguments_array_spec = tf.TensorSpec(shape=(None, 2), dtype=tf.int64, name='arguments_array')
    action_spec = (tactic_id_spec, arguments_array_spec)

    graph_id_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='graph_id')

    num_definitions_spec = tf.TensorSpec(shape=(), dtype=tf.int64, name='num_definitions')

    proofstate_data_spec = (state_spec, action_spec, graph_id_spec)
    definition_data_spec = (node_labels_spec, sources_spec, targets_spec, edge_labels_spec, num_definitions_spec)

    def __init__(self, data_dir: Path, **kwargs):
        """
        @param data_dir: the directory containing the data
        @param kwargs: additional keyword arguments are passed on to the parent class
        """
        super().__init__(**kwargs)
        self.data_server = DataServer(data_dir=data_dir,
                                      work_dir=Path('.'),
                                      split_random_seed=self.split_random_seed,
                                      max_subgraph_size=self.max_subgraph_size,
                                      restrict_to_spine=False)

        self._graph_constants = self.data_server.graph_constants()

    @classmethod
    def from_yaml_config(cls, data_dir: Path, yaml_filepath: Path) -> "DataLoaderDataset":
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
    def _action_to_arguments(action: Tuple, local_context_length: tf.Tensor) -> Tuple[int, tf.Tensor, tf.Tensor]:
        (tactic_id, arguments_array) = action
        is_global_argument, argument_ids = tf.unstack(arguments_array, axis=1)

        # there can still be local arguments that are None
        is_valid_local_argument = tf.where(is_global_argument == 0, argument_ids, int(1e9)) < local_context_length
        local_arguments = tf.where(is_valid_local_argument, argument_ids, -1)

        # global arguments go from 0 to node_label_num-base_node_label_num
        global_arguments = tf.where(is_global_argument == 1, argument_ids, -1)

        return tactic_id, local_arguments, global_arguments

    @classmethod
    def _make_proofstate_graph_tensor(cls, state: Tuple, action: Tuple, graph_id: tf.Tensor) -> tfgnn.GraphTensor:
        """
        Converts the data loader's proof-state representation into a TF-GNN compatible GraphTensor.

        @param state: the tuple containing tf.Tensor objects for the graph structure
        @param action: the tuple containing tf.Tensor objects for the tactic and arguments
        @param graph_id: the id of the graph
        @return: a GraphTensor object that is compatible with the `proofstate_graph_spec` in `graph_schema.py`
        """
        (node_labels, sources, targets, edge_labels, root, context_node_ids) = state

        bare_graph_tensor = cls._make_bare_graph_tensor(node_labels, sources, targets, edge_labels)

        local_context_length = tf.shape(context_node_ids, out_type=tf.int64)[0]
        tactic_id, local_arguments, global_arguments = cls._action_to_arguments(action, local_context_length)

        context = tfgnn.Context.from_fields(features={
            'tactic': tf.expand_dims(tactic_id, axis=0),
            'context_node_ids': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(context_node_ids, axis=0),
                                                            row_splits_dtype=tf.int32),
            'local_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(local_arguments, axis=0),
                                                           row_splits_dtype=tf.int32),
            'global_arguments': tf.RaggedTensor.from_tensor(tensor=tf.expand_dims(global_arguments, axis=0),
                                                            row_splits_dtype=tf.int32)
        })

        return tfgnn.GraphTensor.from_pieces(node_sets=bare_graph_tensor.node_sets,
                                             edge_sets=bare_graph_tensor.edge_sets,
                                             context=context)

    @classmethod
    def _make_definition_graph_tensor(cls,
                                      node_labels: tf.Tensor,
                                      sources: tf.Tensor,
                                      targets: tf.Tensor,
                                      edge_labels: tf.Tensor,
                                      num_definitions: tf.Tensor
                                      ) -> tfgnn.GraphTensor:
        """
        Converts the data loader's definition cluster representation into a TF-GNN compatible GraphTensor.

        @param node_labels: tf.Tensor of node labels (dtype=tf.int64)
        @param sources: tf.Tensor of edge sources (dtype=tf.int32)
        @param targets: tf.Tensor of edge targets (dtype=tf.int32)
        @param edge_labels: tf.Tensor of edge labels (dtype=tf.int64)
        @param num_definitions: tf.Tensor for the number of labels being defined (dtype=tf.int64)
        @return: a GraphTensor object that is compatible with the `definition_graph_spec` in `graph_schema.py`
        """
        bare_graph_tensor = cls._make_bare_graph_tensor(node_labels, sources, targets, edge_labels)

        context = tfgnn.Context.from_fields(features={'num_definitions': tf.expand_dims(num_definitions, axis=0)})

        return tfgnn.GraphTensor.from_pieces(node_sets=bare_graph_tensor.node_sets,
                                             edge_sets=bare_graph_tensor.edge_sets,
                                             context=context)

    def _train_proofstates(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(lambda: self.data_server.data_train(tf_gnn=True),
                                                 output_signature=self.proofstate_data_spec)
        return dataset.map(self._make_proofstate_graph_tensor, num_parallel_calls=tf.data.AUTOTUNE)

    def _valid_proofstates(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(lambda: self.data_server.data_valid(tf_gnn=True),
                                                 output_signature=self.proofstate_data_spec)
        return dataset.map(self._make_proofstate_graph_tensor, num_parallel_calls=tf.data.AUTOTUNE)

    def _definitions(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(lambda: self.data_server.def_cluster_subgraphs(tf_gnn=True),
                                                 output_signature=self.definition_data_spec)
        return dataset.map(self._make_definition_graph_tensor, num_parallel_calls=tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description="Create TFRecord datasets from the capnp data loader")
    parser.add_argument("--data-dir", metavar="DIRECTORY", type=Path, required=True,
                        help="Location of the capnp dataset")
    parser.add_argument("--tfrecord-prefix", metavar="TFRECORD_PREFIX", type=Path, required=True,
                        help="Prefix for the .tfrecord and .yml files to generate")
    parser.add_argument("--dataset-config", metavar="DATASET_YAML", type=Path, required=True,
                        help="YAML file with the configuration for the dataset")
    args = parser.parse_args()

    # check inputs
    if not args.data_dir.is_dir():
        parser.error(f'--data-dir {args.data_dir} must be an existing directory')
    if not args.dataset_config.is_file():
        parser.error(f'--dataset-config {args.dataset_config} must be an existing file')
    if args.tfrecord_prefix.stem is None:
        parser.error(f'--tfrecord-prefix {args.tfrecord_prefix} must have a filename prefix')

    # create dataset
    dataset = DataLoaderDataset.from_yaml_config(data_dir=args.data_dir, yaml_filepath=args.dataset_config)

    # check we don't overwrite existing files
    train_tfrecord_filepath = dataset._train_tfrecord_filepath(tfrecord_prefix=args.tfrecord_prefix)
    if train_tfrecord_filepath.exists():
        parser.error(f'{train_tfrecord_filepath} already exists')

    valid_tfrecord_filepath = dataset._valid_tfrecord_filepath(tfrecord_prefix=args.tfrecord_prefix)
    if valid_tfrecord_filepath.exists():
        parser.error(f'{valid_tfrecord_filepath} already exists')

    definition_tfrecord_filepath = dataset._definition_tfrecord_filepath(tfrecord_prefix=args.tfrecord_prefix)
    if definition_tfrecord_filepath.exists():
        parser.error(f'{definition_tfrecord_filepath} already exists')

    graph_constants_filepath = dataset._graph_constants_filepath(tfrecord_prefix=args.tfrecord_prefix)
    if graph_constants_filepath.exists():
        parser.error(f'{graph_constants_filepath} already exists')

    # dump datasets to .tfrecord files
    dataset.dump(tfrecord_prefix=args.tfrecord_prefix)


if __name__ == "__main__":
    main()
