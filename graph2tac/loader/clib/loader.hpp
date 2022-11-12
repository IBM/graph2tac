#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <set>
#include <map>
typedef uint32_t c_FileIndex;
typedef uint32_t c_NodeIndex;
typedef uint32_t c_DefIndex;
typedef uint32_t c_NodeLabel;
typedef uint32_t c_ClassIdx;  // NodeClass - basenode offset
typedef uint32_t c_EdgeLabel;
typedef uint64_t c_NodeHash;
typedef uint64_t c_StepIndex;
typedef uint64_t c_TacticHash;
typedef uint32_t c_TacticIndex;
typedef size_t   c_ClusterIndex;
typedef uint16_t c_NodePrimitive;
typedef uint16_t c_ChildrenCount;
typedef uint32_t c_ChildrenBegin;


struct c_GlobalNode {
    c_FileIndex file_idx;
    c_NodeIndex node_idx;
};



struct c_Node {
    c_NodePrimitive label;
    c_ChildrenCount children_count;
    c_ChildrenBegin children_begin;
};





inline bool operator<(c_GlobalNode a, c_GlobalNode b) {
    return (((uint64_t(a.file_idx)) << 32 | uint64_t(a.node_idx)) <
	    ((uint64_t(b.file_idx)) << 32 | uint64_t(b.node_idx)));
};

inline bool operator==(c_GlobalNode a, c_GlobalNode b) {
  return (a.file_idx == b.file_idx && a.node_idx == b.node_idx);
}



namespace std {
    template<> struct hash<c_GlobalNode> {
	inline size_t operator()(const c_GlobalNode& a) const {
	    size_t result;
	    std::memcpy(&result, &a, sizeof(result));
	    return result;
	}
    };
}


struct c_EdgeTarget {
    c_EdgeLabel label;
    c_FileIndex dep_index;
    c_NodeIndex node_index;
};


struct c_Link {
    c_EdgeLabel sort;
    c_GlobalNode target;
};

struct c_Edge {
    c_NodeIndex source;
    c_GlobalNode target;
    c_EdgeLabel sort;
};


struct c_SubEdge {
    c_NodeIndex source;
    c_NodeIndex target;
    c_EdgeLabel sort;
};






struct c_ProofStep {
    c_GlobalNode root;
    c_NodeHash def_hash;
    c_StepIndex step_idx;
    std::vector<c_GlobalNode> context;
    c_TacticHash  tactic_hash;
    std::vector<c_GlobalNode> tactic_args;
    std::string state_text;
    std::string tactic_text;
    std::string tactic_base_text;
    std::string tactic_interm_text;
    c_NodeHash def_hash_for_split;
};


struct c_Def {
    c_NodeIndex node;
    c_NodeHash hash;
    std::string name;
    std::vector<c_ProofStep> proof_steps;
};

struct c_FileGraph {
    std::vector<c_Node> nodes;
    std::vector<std::vector<c_Link>> links;
    std::vector<c_Def> defs;
    std::unordered_map<c_NodeIndex, c_DefIndex> def_node_to_idx;
    std::set<c_NodeIndex> definitions;
};




struct c_ConflateLabels {
    std::vector<size_t> map;
    std::size_t size;
};

struct c_SubGraph {
    std::vector<c_GlobalNode> sub_to_glob;
    std::unordered_map<c_GlobalNode, size_t> glob_to_sub;
    std::vector<c_NodeHash> node_labels;  // basic node type OR original node hash: TODO: danger of collision
    std::vector<std::array<c_NodeIndex, 2>> edges_split_flat;
    std::vector<c_EdgeLabel> edges_label_flat;
    std::vector<c_NodeIndex> edge_label_offsets;
    std::set<c_NodeHash> depends;
    std::size_t number_of_roots;
};


struct c_Tactics {
    std::vector<c_TacticHash> index_to_hash;
    std::map<c_TacticHash, c_TacticIndex> hash_to_index;
    std::map<c_TacticHash, size_t> hash_to_numargs;
    std::map<c_TacticHash, std::string> hash_to_string;
    std::size_t max_tactic_args_size;
};


struct c_DefTable {
    std::unordered_map<c_GlobalNode, c_NodeHash> node_to_hash;
    std::map<c_NodeHash, c_GlobalNode> hash_to_node;
    std::map<c_NodeHash, std::string> hash_to_name;
    std::vector<c_NodeHash> available_def_hashes;
    std::unordered_map<c_NodeHash, c_NodeLabel> hash_to_node_label;
    std::vector<c_NodeHash> node_label_to_hash;
    std::vector<c_NodeLabel> available_def_classes;
};


struct c_Data {
    std::vector<c_FileGraph> graph;
    c_DefTable def_table;
    std::vector<std::vector<c_ProofStep>> proof_steps;
    std::vector<std::pair<size_t, size_t>> flatten_proof_steps_index;
    std::unordered_map<c_GlobalNode, c_SubGraph> subgraphs;
    c_ConflateLabels conflate;
    std::vector<std::vector<size_t>> tasks;
    c_Tactics tactics;
    std::vector<c_NodeLabel> eval_label_to_train_label;
    std::map<c_NodeLabel,c_NodeLabel> train_label_to_eval_label;
    std::map<c_NodeHash, c_ClusterIndex> hash_to_cluster_idx;
    std::map<c_ClusterIndex, std::set<c_NodeHash>> cluster_idx_to_hashset;
    std::map<c_ClusterIndex, c_SubGraph> cluster_subgraphs;
    std::vector<c_ClassIdx> global_context;
};
