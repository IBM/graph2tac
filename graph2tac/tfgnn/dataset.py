from typing import Tuple, Dict, Optional, Any, Callable

import argparse
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

class DataServerDataset:
    """
    Class for TF-GNN datasets which are obtained directly from the loader
    """
    MAX_LABEL_TOKENS = 128
    MAX_PROOFSTATES = int(1e7)
    MAX_DEFINITIONS = int(1e7)
    SHUFFLE_BUFFER_SIZE = int(1e7)

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
        definitions = self._definitions(label)
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

    @classmethod
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
        action_tuple = (action.tactic_id, action.args)

        return (state_tuple, action_tuple, id)

    def _proofstates(self, label : int = TRAIN) -> tf.data.Dataset:
        """
        Returns a dataset with all the proof-states.

        @return: a tf.data.Dataset streaming GraphTensor objects
        """
        proofstates = tf.data.Dataset.from_generator(
            lambda: map(self._loader_to_proofstate_data, self.data_server.get_datapoints(label, shuffled=False, as_text=False)),
            output_signature=self.proofstate_data_spec
        )
        return proofstates.map(self._make_proofstate_graph_tensor, num_parallel_calls=tf.data.AUTOTUNE)

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
            lambda: map(self._loader_to_definition_data, self.data_server.def_cluster_subgraphs(label)),
            output_signature=self.definition_data_spec
        )
        return dataset.map(self._make_definition_graph_tensor, num_parallel_calls=tf.data.AUTOTUNE)
