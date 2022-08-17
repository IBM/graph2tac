import tensorflow as tf
import tensorflow_gnn as tfgnn

_bare_node_schema = """
node_sets {
  key: "node"
  value {
    description: "A node in the bare graph."

    features {
      key: "node_label"
      value: {
        description: "[DATA] The id of the node label."
        dtype: DT_INT64
      }
    }
  }
}
"""

_bare_edge_schema = """
edge_sets {
  key: "edge"
  value {
    description: "An edge between nodes in the bare graph."
    source: "node"
    target: "node"

    features {
      key: "edge_label"
      value: {
        description: "[DATA] The id of the edge label."
        dtype: DT_INT64
      }
    }
  }
}
"""

_proofstate_context_schema = """
context {
  features {
    key: "local_context_ids"
    value: {
      description: "[DATA] The ids of the nodes in the local context."
      dtype: DT_INT64
      shape { dim { size: -1 } }
    }
  }
  
  features {
    key: "global_context_ids"
    value: {
      description: "[DATA] Pointers into the global context for the definitions that are actually available."
      dtype: DT_INT64
      shape { dim { size: -1 } }
    }
  }

  features {
    key: "tactic"
    value: {
      description: "[LABEL] The id of the tactic we are trying to predict."
      dtype: DT_INT64
    }
  }

  features {
    key: "local_arguments"
    value: {
      description: "[LABEL] The local arguments we are trying to predict (or -1 for global/None)."
      dtype: DT_INT64
      shape { dim { size: -1 } }
    }
  }

  features {
    key: "global_arguments"
    value: {
      description: "[LABEL] The global arguments we are trying to predict (or -1 for local/None)."
      dtype: DT_INT64
      shape { dim { size: -1 } }
    }
  }

  features {
    key: "graph_id"
    value: {
      description: "[METADATA] The id of the graph, as returned by the loader."
      dtype: DT_INT64
    }
  }

  features {
    key: "name"
    value: {
      description: "[METADATA] The name of the lemma this proofstate belongs to."
      dtype: DT_STRING
    }
  }

  features {
    key: "step"
    value: {
      description: "[METADATA] The step number in the lemma proof."
      dtype: DT_INT64
    }
  }
  
  features {
    key: "faithful"
    value: {
      description: "[METADATA] 1 if the proofstate action is faithful (action_text == action_interm_text), 0 otherwise."
      dtype: DT_INT64
    }
  }
}
"""

_definition_context_schema = """
context {
  features {
    key: "definition_names"
    value: {
      description: "[DATA] The names of the node labels being defined by this graph."
      dtype: DT_STRING
      shape { dim { size: -1 } }
    }
  }

  features {
    key: "num_definitions"
    value: {
      description: "[LABEL] The number of node labels defined by this graph."
      dtype: DT_INT64
    }
  }
}
"""

_vectorized_definition_context_schema = """
context {
  features {
    key: "definition_name_vectors"
    value: {
      description: "[DATA] The tokenized names of the node labels being defined by this graph."
      dtype: DT_INT64
      shape { dim { size: -1 } dim { size: -1 } }
    }
  }

  features {
    key: "num_definitions"
    value: {
      description: "[LABEL] The number of node labels defined by this graph."
      dtype: DT_INT64
    }
  }
}
"""

_hidden_node_schema = """
node_sets {
  key: "node"
  value {
    description: "A node in the hidden graph."

    features {
      key: "hidden_state"
      value: {
        description: "[HIDDEN_STATE] The hidden state of the node."
        dtype: DT_FLOAT
        shape { dim { size: -1 } }
      }
    }
  }
}
"""

_hidden_edge_schema = """
edge_sets {
  key: "edge"
  value {
    description: "An edge between nodes in the hidden graph."
    source: "node"
    target: "node"

    features {
      key: "edge_embedding"
      value: {
        description: "[HIDDEN_STATE] The edge embedding corresponding to the edge label."
        dtype: DT_FLOAT
        shape { dim { size: -1 } }
      }
    }
  }
}
"""

_hidden_context_schema = """
context {
  features {
    key: "hidden_state"
    value: {
      description: "[HIDDEN_STATE] The hidden state for the whole graph."
      dtype: DT_FLOAT
      shape { dim { size: -1 } }
    }
  }
}
"""

_bare_graph_schema = tfgnn.parse_schema(_bare_node_schema+_bare_edge_schema)
bare_graph_spec = tfgnn.create_graph_spec_from_schema_pb(_bare_graph_schema)

_proofstate_graph_schema = tfgnn.parse_schema(_bare_node_schema+_bare_edge_schema+_proofstate_context_schema)
proofstate_graph_spec = tfgnn.create_graph_spec_from_schema_pb(_proofstate_graph_schema)

_definition_graph_schema = tfgnn.parse_schema(_bare_node_schema+_bare_edge_schema+_definition_context_schema)
definition_graph_spec = tfgnn.create_graph_spec_from_schema_pb(_definition_graph_schema)

_vectorized_definition_graph_schema = tfgnn.parse_schema(_bare_node_schema+_bare_edge_schema+_vectorized_definition_context_schema)
vectorized_definition_graph_spec = tfgnn.create_graph_spec_from_schema_pb(_vectorized_definition_graph_schema)

_hidden_graph_schema = tfgnn.parse_schema(_hidden_node_schema+_hidden_edge_schema+_hidden_context_schema)
hidden_graph_spec = tfgnn.create_graph_spec_from_schema_pb(_hidden_graph_schema)


def strip_graph(input_graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    return input_graph.replace_features(
        node_sets={'node': {'node_label': input_graph.node_sets['node'].features['node_label']}},
        edge_sets={'edge': {'edge_label': input_graph.edge_sets['edge'].features['edge_label']}},
        context={}
    )


def batch_graph_spec(graph_spec: tfgnn.GraphTensorSpec) -> tfgnn.GraphTensorSpec:
    return tf.data.Dataset.from_generator(lambda: iter([]), output_signature=graph_spec).batch(1).element_spec
