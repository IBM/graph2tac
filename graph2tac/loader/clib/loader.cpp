/* this code supports only capnp v8 and capnp v9 versions; the previous version deprecated */
/* if you want dataloader for earlier versions, go to earlier tagged git commit */

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCH_LEVEL__)

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>


#include "graph_api_v11.capnp.h"



#include <sys/mman.h>

#include <algorithm>
#include <capnp/serialize-packed.h>
#include <capnp/schema.h>
#include <capnp/dynamic.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <numeric>
#include <queue>
#include <deque>
#include <set>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <array>
#include "loader.hpp"
#include <filesystem>
#include <thread>
#include <algorithm>
#include "numpy_conversion.hpp"
#include "simple_directed_graph.hpp"




constexpr uint64_t ENCODE_ALL = 1;

constexpr c_NodePrimitive NodeLabelDefinition = Graph::Node::Label::DEFINITION;
const size_t  base_node_label_num = capnp::Schema::from<Graph::Node::Label>().getFields().size();



// TODO: check for hash collisions between node hashes and basic labels, globally

/* ---------------------------- HELPERS ---------------------------------- */
/* ----------------------------------------------------------------------- */




char __data_capsule_name[]{"__c_data_capsule"};
char __data_online_capsule_name[]{"__c_data_online_capsule"};
char __glob_to_sub_capsule_name[]{"__c_data_glob_to_sub"};


constexpr c_FileIndex __MISSING_FILE = std::numeric_limits<c_FileIndex>::max();
constexpr c_NodeIndex __MISSING_NODE = std::numeric_limits<c_NodeIndex>::max();
constexpr auto G2T_LOG_LEVEL  = "G2T_LOG_LEVEL";
constexpr int CRITICAL = 50;
constexpr int ERROR = 40;
constexpr int WARNING = 30;
constexpr int INFO = 20;
constexpr int VERBOSE = 15;
constexpr int DEBUG = 10;
constexpr int NOTSET = 0;


void log_level(
    const std::string &msg,
    const int msg_level) {
    char * c_val = std::getenv(G2T_LOG_LEVEL);
    int log_level = INFO;
    if (c_val != nullptr) {
	auto val = std::string(c_val);
	log_level = std::stoi(val);
    }
    if (msg_level >= log_level) {
	if (msg_level >= CRITICAL) {
	    std::cerr << "C_LOADER CRITICAL | " <<  msg << std::endl;
	} else if (msg_level >= ERROR) {
	    std::cerr << "C_LOADER ERROR | " <<  msg << std::endl;
	} else if (msg_level >= WARNING) {
	    std::cerr << "C_LOADER WARNING | " <<  msg << std::endl;
	} else if (msg_level >= INFO) {
	    std::cerr << "C_LOADER INFO | " <<  msg << std::endl;
	} else if (msg_level >= VERBOSE) {
	    std::cerr << "C_LOADER VERBOSE | " <<  msg << std::endl;
	} else if (msg_level >= DEBUG) {
	    std::cerr << "C_LOADER DEBUG | " <<  msg << std::endl;
	} else if (msg_level >= NOTSET) {
	    std::cerr << "C_LOADER | " <<  msg << std::endl;
	}
    }
}


void log_level(const std::vector<c_FileGraph>& graph,
		 const int msg_level) {
    size_t counter_nodes {0};
    size_t counter_edges {0};
    for (const auto& file_graph: graph) {
	counter_nodes += file_graph.nodes.size();
	for (const auto& link: file_graph.links) {
	    counter_edges += link.size();
	}
    }
    log_level("total nodes " + std::to_string(counter_nodes), msg_level);
    log_level("total edges " + std::to_string(counter_edges), msg_level);
}




template <typename T1, typename T2>
void log_level(std::map<T1, T2> map_object,
	       const int msg_level) {
    std::string result;
    for (const auto & [k, v]: map_object) {
	result += std::to_string(k) + " " + std::to_string(v) + std::endl;
    }
    log_level(result, msg_level);
}

std::string to_string(c_GlobalNode node) {
    return std::to_string(node.file_idx) + ", " + std::to_string(node.node_idx);
}


template <typename T1, typename T2>
void log_level(std::unordered_map<T1, T2> map_object,
	       const int msg_level) {
    std::string result;
    using std::to_string;
    using std::endl;
    for (const auto & [k, v]: map_object) {
	result += to_string(k) + " " + to_string(v) + "\n";
    }
    log_level(result, msg_level);
}


template <typename T>
void log_level(std::vector<T> a,
	       const int msg_level) {
    std::stringstream result;

    for (const auto & e: a) {
	result << e << ' ';
    }
    log_level(result.str(), msg_level);
}




template <typename T>
void log_debug(const T  &msg) {
    log_level(msg, DEBUG);
}

template <typename T>
void log_verbose(const T& msg) {
    log_level(msg, VERBOSE);
}

template <typename T>
void log_info(const T  &msg) {
    log_level(msg, INFO);
}

template <typename T>
void log_warning(const T  &msg) {
    log_level(msg, WARNING);
}

template <typename T>
void log_error(const T  &msg) {
    log_level(msg, ERROR);
}

template <typename T>
void log_critical(const T  &msg) {
    log_level(msg, CRITICAL);
}




FILE *fopen_safe(const std::string &fname, const std::string &mode) {
    FILE *pFile = fopen(fname.c_str(), mode.c_str());
    if (pFile == nullptr) {
	log_critical("error opening file " + fname);
    }
    return pFile;
}



class MMapFileMessageReader {
public:
    capnp::FlatArrayMessageReader * msg;
    size_t file_size;
    std::string fname;
    FILE * file;
    char * ptr;
    MMapFileMessageReader(const std::string &fname) :
	fname (fname) {
	file = fopen_safe(fname, "rb");
	auto filed = fileno(file);

	file_size  = std::filesystem::file_size(fname);
	//log_debug("file_size is " + std::to_string(file_size));
	//log_debug("mmaping the file " + fname);
	ptr = (char *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, filed, 0);
	//log_debug("mmap call returned " + fname);

	if (ptr == MAP_FAILED) {
	    throw std::runtime_error("mapping of " + fname + " of size " + std::to_string(file_size) + " failed");
	    log_critical("mapping of " + fname + " of size " + std::to_string(file_size) + " failed");
	}
	auto array_ptr = kj::ArrayPtr<const capnp::word>( reinterpret_cast<capnp::word*>(ptr),
							  file_size / sizeof(capnp::word));
	msg = new capnp::FlatArrayMessageReader(array_ptr,  capnp::ReaderOptions{SIZE_MAX});
	//log_debug("finished constructor of MMapReader " + fname);
    }
    ~MMapFileMessageReader() {
	//log_debug("started destructor of MMapReader " + fname);
	auto err = munmap((void *) ptr, file_size);
	//log_debug("munamp finished " + fname);
	if (err != 0) {
	    log_critical("unmapping of " + fname + " of size " + std::to_string(file_size) + " failed in the destructor of MMapFileMessageReader");
	}
	fclose(file);
	delete msg;
    }
};




struct c_DataOnline {
    std::vector<c_GlobalNode> def_idx_to_node;
    std::unordered_map<c_GlobalNode, c_NodeIndex> def_node_to_idx;
    c_ConflateLabels conflate;
    std::unordered_map<c_TacticHash, c_TacticIndex> tactic_hash_to_idx;
    std::unordered_map<c_TacticIndex, c_TacticHash> tactic_idx_to_hash;
    std::map<uint64_t, uint64_t> subgraph_node_counter;

    std::vector<std::string> fnames;
    std::vector<Py_buffer> views; //to remove them nicely
    std::vector<capnp::FlatArrayMessageReader*> areaders;
    std::vector<std::vector<c_FileIndex>> rel_file_idx;
    std::vector<Graph::Reader> graphs;
    std::vector<::capnp::List<::Graph::Node, ::capnp::Kind::STRUCT>::Reader> global_nodes;
    std::vector<::capnp::List<::Graph::EdgeTarget, ::capnp::Kind::STRUCT>::Reader> global_edges;
};


class Profiler {
private:
    const std::string name;
    const std::chrono::time_point<std::chrono::high_resolution_clock> t_start;

public:
    Profiler(const std::string &name)
	: name(name),
	  t_start(std::chrono::high_resolution_clock::now()) {}
    ~Profiler() {
	auto t_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration<double>(t_end - t_start);
	log_info(name + ": " + std::to_string(duration.count()));
    }
};


class PackedFileMessageReader {
public:
    capnp::PackedFdMessageReader msg;
    PackedFileMessageReader(const std::string &fname)
	: msg(kj::AutoCloseFd(fileno(fopen_safe(fname, "rb"))),
	      capnp::ReaderOptions{SIZE_MAX}) {}
};

class FileMessageReader {
public:
    capnp::StreamFdMessageReader msg;
    FileMessageReader(const std::string &fname)
	: msg(kj::AutoCloseFd(fileno(fopen_safe(fname, "rb"))),
	      capnp::ReaderOptions{SIZE_MAX}) {}
};



std::vector<std::vector<size_t>> get_tasks(
    unsigned int num_proc,
    const std::vector<std::string> &fnames) {

    std::vector<std::pair<size_t, size_t>> f_sizes;
    for (size_t file_idx = 0; file_idx < fnames.size(); ++file_idx) {
	f_sizes.emplace_back(
	    std::make_pair(file_idx, std::filesystem::file_size(fnames[file_idx])));
    }

    std::sort(f_sizes.begin(), f_sizes.end(),
	      [](const std::pair<size_t, size_t> &a, const std::pair<size_t, size_t> &b) {
		  return (a.second > b.second ||
			  (a.second == b.second && a.first < b.first));
	      });

    std::vector<std::vector<size_t>> tasks(num_proc);
    std::vector<size_t> task_sizes(num_proc);

    for (auto [file_idx, file_size] : f_sizes) {
	int i =
	    min_element(task_sizes.begin(), task_sizes.end()) - task_sizes.begin();
	task_sizes[i] += file_size;
	tasks[i].emplace_back(file_idx);
    }
    return tasks;
}



std::vector<c_NodeIndex> get_labels_over_hashes(
    const std::unordered_map<c_NodeHash, c_NodeLabel> &encode_table,
    const std::vector<c_NodeHash> &node_labels,
    const std::vector<c_NodeLabel> &eval_label_to_train_label) {

    std::vector<c_NodeIndex> labels_over_hashes(node_labels.size());

    if (eval_label_to_train_label.size() == 0) {
    std::transform(node_labels.begin(),   node_labels.end(),
		   labels_over_hashes.begin(),
		   [&encode_table](auto label) {
		       return encode_table.at(label);
		   });
    } else {
    std::transform(node_labels.begin(),   node_labels.end(),
		   labels_over_hashes.begin(),
		   [&encode_table, &eval_label_to_train_label](auto label) {
		       return eval_label_to_train_label.at(encode_table.at(label));
		   });

    }
    return labels_over_hashes;
}



class FileTable {
    const std::vector<std::string> _fnames;
    const std::string data_dir;
    std::unordered_map<std::string, c_FileIndex> i_fnames;

private:
    std::string convert(const std::string dep_fname) const {
	std::string new_dep_fname = dep_fname;

	//if (dep_fname.size() > 4 &&
	//    dep_fname.substr(dep_fname.size() - 4, 4) == ".bin") {
	//    new_dep_fname = dep_fname.substr(0, dep_fname.size() - 4) + ".binx";
	//}

	std::string result;
	if (new_dep_fname.size() > 0 && new_dep_fname[0] == '/') {
	    result = new_dep_fname;
	} else {
	    result = data_dir + "/" + new_dep_fname;
	}
	log_debug("original dep name " + dep_fname);
	log_debug("converted file name " + result);
	return result;
    }

public:
    FileTable(const std::vector<std::string> &fnames, std::string data_dir) :
	_fnames (fnames),
	data_dir (data_dir) {

	for (size_t i = 0; i < fnames.size(); ++i) {
	    i_fnames[_fnames[i]] = i;
	    log_debug("index " + std::to_string(i) + " file " + _fnames[i]);
	}
	if (i_fnames.size() != _fnames.size()) {
	    std::cerr << "WARNING: repeating filenames, there are  "
		      << i_fnames.size() << " unique names in the provided list of "
		      << _fnames.size() << " names" << std::endl;
	}
    }
    c_FileIndex index(const std::string fname) const {
	auto converted_name = FileTable::convert(fname);
	if (i_fnames.find(converted_name) == i_fnames.end()) {
	    log_critical("WARNING! " + converted_name + " not found");
	    return __MISSING_FILE;
	} else {
	    return i_fnames.at(FileTable::convert(fname));
	}
    }

    std::string fname(const c_FileIndex file_idx) const {
	return _fnames.at(file_idx);
    }
    std::vector<std::string> fnames() const {
	return _fnames;
    }
};


/* --------------------------- READING SCHEMA--------------------------- */
/* --------------------------------------------------------------------- */




const c_ConflateLabels build_edge_conflate_map() {
    const auto edge_labels = capnp::Schema::from<EdgeClassification>();
    const size_t edge_labels_size = edge_labels.getEnumerants().size();
    std::vector<size_t> conflate_map(edge_labels_size, static_cast<size_t>(-1));

    size_t conf_idx = 0;

    for (const auto& e: *CONFLATABLE_EDGES) {
	const auto group = e.getConflatable();
	for (const auto &x : group) {
	  conflate_map[static_cast<size_t>(x)] = conf_idx;
	}
	++conf_idx;
    }

    for (size_t edge_idx = 0; edge_idx < edge_labels_size; ++edge_idx) {
	if (conflate_map[edge_idx] == static_cast<size_t>(-1)) {
	conflate_map[edge_idx] = conf_idx++;
      }
    }
    return c_ConflateLabels {conflate_map, conf_idx};
}



/* --------------------------- READING CAPNP --------------------------- */
/* --------------------------------------------------------------------- */


void encode_hashes(const  std::map<c_NodeHash, c_GlobalNode> &hash_to_node,
		   std::unordered_map<c_NodeHash, c_NodeLabel> &hash_to_node_label,
		   std::vector<c_NodeHash> &node_label_to_hash) {
    for (const auto & [hash, node]: hash_to_node) {
	if (hash_to_node_label.find(hash) == hash_to_node_label.end()) {
	    hash_to_node_label[hash] = node_label_to_hash.size();
	    node_label_to_hash.emplace_back(hash);
	}
    }

}




std::vector<c_FileIndex> build_dependencies(
    const ::capnp::List<::capnp::Text, ::capnp::Kind::BLOB>::Reader  *dependencies,
    const FileTable & file_table) {

    std::vector<c_FileIndex> dep_local_to_global;
    for (auto const &fname: *dependencies) {
	dep_local_to_global.emplace_back(file_table.index(fname));
    }
    return dep_local_to_global;
}




void check_adjacency(
    const std::vector<c_FileGraph> &global_graph) {

    log_info("pass check_adjacency: checking adjacency list counts... ");

    size_t total_edges = 0;
    size_t total_nodes = 0;
    for (size_t file_idx = 0; file_idx < global_graph.size(); ++file_idx) {
	total_nodes += global_graph[file_idx].links.size();
	for (size_t node = 0; node < global_graph[file_idx].links.size(); ++node) {
	    total_edges += global_graph[file_idx].links[node].size();
	}
    }

    log_info("nodes " + std::to_string(total_nodes));
    log_info("edges " + std::to_string(total_edges));
}

void check_cross_edges(
    std::vector<c_FileGraph> &global_graph,
    bool debug_option) {

    log_info("pass check_cross_edges: cross edges count by edge type, target type: ");

    std::map<std::pair<c_EdgeLabel, c_NodePrimitive>, int> counter_cross_edges;

    for (size_t file_idx = 0; file_idx < global_graph.size(); ++file_idx) {
	for (size_t source = 0; source < global_graph[file_idx].links.size();
	     ++source) {
	    for (auto const &edge : global_graph[file_idx].links[source]) {
		if (edge.target.file_idx != c_FileIndex(file_idx)) {
		    auto target_c =
			global_graph[edge.target.file_idx].nodes[edge.target.node_idx].label;
		    counter_cross_edges[std::make_pair(edge.sort, target_c)] += 1;
		}
	    }
	}
    }

    if (debug_option) {
	for (auto [k, v] : counter_cross_edges) {
	    log_debug(std::to_string(k.first) + " " + std::to_string(k.second) + " " +
		  std::to_string(v));
	}
    }
}


std::vector<c_GlobalNode> build_context(
    c_FileIndex file_idx,
    const ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader *context_obj) {
    std::vector<c_GlobalNode> context;
    for (const auto &ctxt_node_idx : *context_obj) {
	context.emplace_back(c_GlobalNode{file_idx, ctxt_node_idx});
    }
    return context;
}



std::vector<c_GlobalNode> build_tactic_args(
    const ::capnp::List<Argument, capnp::Kind::STRUCT>::Reader tactic_args_obj,
    const std::vector<c_FileIndex> &dep_local_to_global) {
    std::vector<c_GlobalNode> tactic_args;
    for (const auto &obj_arg : tactic_args_obj) {
	if (obj_arg.which() == Argument::UNRESOLVABLE) {
	    tactic_args.emplace_back(
		c_GlobalNode{__MISSING_FILE, __MISSING_NODE});
	    continue;
	} else {
	    auto term = obj_arg.getTerm();
	    c_FileIndex local_file_idx = term.getDepIndex();
	    if (!(local_file_idx < dep_local_to_global.size())) {
		log_critical("WARNING: global tactic argument local_file_idx " +
		      std::to_string(local_file_idx) + " not found");
		tactic_args.emplace_back(
		    c_GlobalNode{__MISSING_FILE, __MISSING_NODE});
	    } else {
		c_FileIndex other_file_idx = dep_local_to_global[local_file_idx];
		c_NodeIndex node_idx = term.getNodeIndex();
		tactic_args.emplace_back(
		    c_GlobalNode{other_file_idx, node_idx});
	    }
	}
    }
    return tactic_args;
}






c_ProofStep build_proof_step_v9(
    c_NodeHash def_hash,
    c_NodeHash def_hash_for_split,
    c_StepIndex step_idx,
    const Tactic::Reader &tactic,
    const Outcome::Reader &outcome,
    const std::vector<c_FileIndex> &dep_local_to_global
    ) {
    auto state = outcome.getBefore();
    auto local_root_idx = state.getRoot();
    auto context_obj = state.getContext();
    auto context = build_context(dep_local_to_global[0], &context_obj);
    auto state_text = state.getText();
    auto tactic_hash = tactic.getIdent();
    auto tactic_args_obj = outcome.getTacticArguments();
    auto tactic_args = build_tactic_args(tactic_args_obj,
					 dep_local_to_global);
    auto tactic_text = tactic.getText();
    auto tactic_base_text = tactic.getBaseText();
    auto tactic_interm_text = tactic.getIntermText();
    return c_ProofStep{c_GlobalNode{dep_local_to_global[0], local_root_idx},
	    def_hash,
		step_idx,
		std::move(context),
		tactic_hash,
		std::move(tactic_args),
		std::move(state_text),
		std::move(tactic_text),
		std::move(tactic_base_text),
		std::move(tactic_interm_text),
		def_hash_for_split
		};
}




void get_flatten_proof_steps_index(
    const std::vector<std::vector<c_ProofStep>> &global_proof_steps,
    std::vector<std::pair<size_t, size_t>> &flatten_proof_steps_index) {

    for (size_t file_idx = 0; file_idx < global_proof_steps.size(); ++file_idx) {
	for (size_t file_step_idx = 0;
	     file_step_idx < global_proof_steps[file_idx].size(); ++file_step_idx) {
	    flatten_proof_steps_index.emplace_back(std::make_pair(file_idx, file_step_idx));
	}
    }
}

typedef std::unordered_map<c_GlobalNode, unsigned long long> DistanceMap;

DistanceMap forward_closure_distance(
    const std::vector<c_FileGraph>&  global_graph,
    const c_GlobalNode source,
    const c_NodePrimitive stop_label,
    const c_EdgeLabel stop_edge_label) {

    // forward closure with bfs and distance; don't go beyond stop_label

    std::queue<c_GlobalNode> q;
    DistanceMap visited;

    visited[source] = 0;
    q.push(source);

    while (q.size() != 0) {
	auto node = q.front();	q.pop();
	for (auto edge: global_graph[node.file_idx].links[node.node_idx]) {
	    if (edge.sort != stop_edge_label) {
		auto other = edge.target;
		auto other_label = global_graph[other.file_idx].nodes[other.node_idx].label;

		if (visited.find(other) == visited.end()) {
		    visited[other] = visited[node] + 1;
		    if (other_label != stop_label) {
			q.push(other);
		    }
		}
	    }
	}
    }
    return visited;
}





std::pair<std::vector<c_GlobalNode>, std::unordered_map<c_GlobalNode, size_t>*> mmaped_forward_closure(
    const std::vector<Graph::Reader>& graphs,
    const std::vector<c_GlobalNode> & roots,
    const std::vector<std::vector<c_FileIndex>> & rel_file_idx,
    const EdgeClassification stop_edge_label,
    const bool bfs,
    const size_t max_subgraph_size,
    const std::unordered_map<c_GlobalNode, c_NodeIndex>& def_node_to_idx,
    const std::vector<::capnp::List<::Graph::Node, ::capnp::Kind::STRUCT>::Reader>& global_nodes,
    const std::vector<::capnp::List<::Graph::EdgeTarget, ::capnp::Kind::STRUCT>::Reader>& global_edges)
{
    std::vector<c_GlobalNode> shallow_deps;

    std::deque<c_GlobalNode> q;
    // std::unordered_map<c_GlobalNode, size_t>* visited;
    auto visited = new std::unordered_map<c_GlobalNode, size_t>;


    size_t counter = 0;
    for (const auto& root: roots) {
	(*visited)[root] = counter++;
	q.push_back(root);
    }

    while (q.size() != 0 && visited->size() < max_subgraph_size) {
	c_GlobalNode node_ptr;
	if (bfs) {
	    node_ptr = q.front(); q.pop_front();
	} else {
	    node_ptr = q.back(); q.pop_back();
	}
	auto this_nodes = global_nodes.at(node_ptr.file_idx);
	auto this_edges = global_edges.at(node_ptr.file_idx);
	const auto& node = this_nodes[node_ptr.node_idx];
	uint32_t edge_begin = node.getChildrenIndex();
	uint32_t edge_end = edge_begin + node.getChildrenCount();
	for (uint32_t edge_idx = edge_begin;
	     edge_idx < edge_end;  ++edge_idx) {
	    const auto& edge = this_edges[edge_idx];
	    if (edge.getLabel() != stop_edge_label) {
		const auto& target = edge.getTarget();
		c_GlobalNode other {rel_file_idx.at(node_ptr.file_idx).at(target.getDepIndex()), target.getNodeIndex()};

		if (visited->find(other) == visited->end()) {
		    (*visited)[other] = counter++;
		    //if (global_nodes.at(other.file_idx)[other.node_idx].getLabel().which() != Graph::Node::Label::DEFINITION ) {
		    if (def_node_to_idx.find(other) == def_node_to_idx.end()) {
			q.push_back(other);
		    } else {
			shallow_deps.emplace_back(other);
		    }
		}
	    }
	    if (visited->size() == max_subgraph_size) {
		break;
	    }
	}
    }

    std::sort(shallow_deps.begin(), shallow_deps.end());


    return std::make_pair(shallow_deps,  visited);
}

std::unordered_map<c_GlobalNode, size_t> forward_closure(
    const std::vector<c_FileGraph>&  global_graph,
    const std::vector<c_GlobalNode>& sources,
    const c_NodePrimitive stop_node_label,
    const c_EdgeLabel stop_edge_label,
    const bool bfs,
    const size_t max_subgraph_size) {

    // forward closure with bfs and distance;
    // do not expand after stop_node_label
    // do not traverse stop_edge_label
    // do not expand beyond max_subgraph_size

    std::deque<c_GlobalNode> q;
    std::unordered_map<c_GlobalNode, size_t> visited;

    size_t counter = 0;
    for (const auto& source: sources) {
	visited[source] = counter++;
	q.push_back(source);
    }

    while (q.size() != 0 && visited.size() < max_subgraph_size) {
	c_GlobalNode node;
	if (bfs) {
	    node = q.front(); q.pop_front();
	} else {
	    node = q.back(); q.pop_back();
	}
	for (auto edge: global_graph[node.file_idx].links[node.node_idx]) {
	    if (edge.sort != stop_edge_label) {
		auto other = edge.target;
		auto other_label = global_graph[other.file_idx].nodes[other.node_idx].label;

		if (visited.find(other) == visited.end()) {
		    visited[other] = counter++;
		    if (other_label != stop_node_label) {
			q.push_back(other);
		    }
		}
	    }
	    if (visited.size() == max_subgraph_size) {
		break;
	    }
	}
    }
    return visited;
}

std::vector<std::vector<std::array<c_NodeIndex, 2>>> get_subgraph_edges_split_by_label(
    const c_ConflateLabels & conflate,
    const c_Data & data,
    const std::unordered_map<c_GlobalNode, size_t> & glob_to_sub,
    const std::vector<c_GlobalNode> & roots,
    const c_NodePrimitive stop_node_label,
    const c_EdgeLabel stop_edge_label) {
    // returns edges in split_by_label format
    std::vector<std::vector<std::array<c_NodeIndex, 2>>> edges_split_by_label(conflate.size);

    for (const auto & [glob, source_sub]: glob_to_sub) {
	if (std::find(roots.begin(), roots.end(), glob) != roots.end()
	    || data.graph[glob.file_idx].nodes[glob.node_idx].label != stop_node_label) {
	    for (const auto & link: data.graph[glob.file_idx].links[glob.node_idx]) {
		if (link.sort != stop_edge_label) {
		    if (glob_to_sub.find(link.target) != glob_to_sub.end()) {
			c_NodeIndex target_sub = static_cast<c_NodeIndex>(glob_to_sub.at(link.target));
			edges_split_by_label[conflate.map[link.sort]].emplace_back(
			    std::array<c_NodeIndex, 2> {static_cast<c_NodeIndex>(source_sub), target_sub});
		    }
		}
	    }
	}
    }
    return edges_split_by_label;
}

std::tuple< std::vector<c_NodeIndex>, std::vector<c_EdgeLabel>,
	    std::vector<std::array<c_NodeIndex, 2>>>
compute_offsets(const std::vector<std::vector<std::array<c_NodeIndex, 2>>>&
		edges_split_by_label) {
    std::vector<c_NodeIndex> edges_label_offsets;
    std::vector<c_EdgeLabel> edges_label_flat;
    std::vector<std::array<c_NodeIndex, 2>> edges_split_flat;
    size_t cnt = 0;
    c_EdgeLabel class_idx = 0;
    if (edges_split_by_label.size() > 0) {
	for (const auto &row: edges_split_by_label) {
	    for (const auto &edge : row) {
		edges_split_flat.emplace_back(edge);
		edges_label_flat.emplace_back(class_idx);
		cnt++;
	    }
	    edges_label_offsets.emplace_back(cnt);
	    class_idx++;
	}
	edges_label_offsets.pop_back();  //np.split function requires for the number of splits - 1 positions
    }
    return make_tuple(
	std::move(edges_label_offsets), std::move(edges_label_flat), std::move(edges_split_flat));
}



c_SubGraph build_subgraph (
    const c_Data * const p_data,
    const std::vector<c_GlobalNode> & roots,
    const bool bfs_option,
    const size_t max_subgraph_size) {

    //log_debug("entered forward closure");
    //for (const auto root: roots) {
    //log_debug("root " + std::to_string(root.file_idx) + "," + std::to_string(root.node_idx));
    //}

    std::unordered_map<c_GlobalNode, size_t> glob_to_sub =
	forward_closure(p_data->graph, roots,
			NodeLabelDefinition,
			static_cast<c_EdgeLabel>(EdgeClassification::CONST_OPAQUE_DEF),
			bfs_option,
			max_subgraph_size);
    //log_debug("exit forward closure");


    std::vector<c_GlobalNode> sub_to_glob(glob_to_sub.size());
    for (const auto &[glob, sub] : glob_to_sub) {
	sub_to_glob[sub] = glob;
    }


    auto conflate = p_data->conflate;
    auto edges_split_by_label = get_subgraph_edges_split_by_label(
	conflate, *p_data, glob_to_sub, roots,
	NodeLabelDefinition,
	static_cast<c_NodeLabel>(EdgeClassification::CONST_OPAQUE_DEF));

    //log_debug("get subgraph edges");


    const auto [edges_label_offsets, edges_label_flat, edges_split_flat] = compute_offsets(edges_split_by_label);

    std::vector <c_NodeHash> node_labels(sub_to_glob.size());
    std::set <c_NodeHash> depends;
    //log_debug("entering building deps ");

    // this code inserts into the dependencies the original node as well
    std::transform(sub_to_glob.begin(), sub_to_glob.end(), node_labels.begin(),
		   [&p_data, &depends](const auto & glob) {
		       c_NodePrimitive basic_label = p_data->graph[glob.file_idx].nodes[glob.node_idx].label;
		       if (basic_label == NodeLabelDefinition) {
			   //log_debug("global node "  + std::to_string(glob.file_idx) + " " + std::to_string(glob.node_idx));
			   /// TODO!!! FIX ME: definition node is present in graph 1
			   /// but is not available in the definition table
			   /// for the reason that definition table is empty in graph 1
			   /// fails /tests/bug_server_name_overload_c.v
			   depends.insert(p_data->def_table.node_to_hash.at(glob));
			   return  p_data->def_table.node_to_hash.at(glob);
		       }  else {
			   return static_cast<c_NodeHash>(basic_label);
		       }
		   });
    //log_debug("exiting building deps ");


    return c_SubGraph {
	std::move(sub_to_glob),
	    std::move(glob_to_sub),
	    std::move(node_labels),
	    std::move(edges_split_flat),
	    std::move(edges_label_flat),
	    std::move(edges_label_offsets),
	    std::move(depends),
	    roots.size()
	    };

}


std::unordered_map<c_GlobalNode, c_SubGraph>  get_file_subgraphs(
    c_Data * const p_data,
    const bool bfs_option,
    const size_t max_subgraph_size,
    const c_FileIndex file_idx) {
    std::unordered_map<c_GlobalNode, c_SubGraph> file_subgraphs;

    for (const auto & proof_step : p_data->proof_steps[file_idx]) {
	if (file_subgraphs.find(proof_step.root) == file_subgraphs.end()) {
	    file_subgraphs[proof_step.root] = build_subgraph(
		p_data,
		std::vector<c_GlobalNode> {proof_step.root}, bfs_option, max_subgraph_size);
	} else {
	    log_critical("WARNING: problem in the dataset: repeated proofstep root "
		  + std::to_string(proof_step.root.file_idx) + " "
		  + std::to_string(proof_step.root.node_idx));
	}
    }
    log_debug("finished proof steps subgraphs of " + std::to_string(file_idx));
    for (const auto & def: p_data->graph[file_idx].defs) {
	c_GlobalNode global_node {file_idx, def.node};

	if (p_data->graph[file_idx].definitions.find(def.node) !=
	    p_data->graph[file_idx].definitions.end())

 {
	if (file_subgraphs.find(global_node) == file_subgraphs.end()) {
	    //log_debug("defined subgraph " + std::to_string(global_node.file_idx) + " " +
	    //      std::to_string(global_node.node_idx));
	    file_subgraphs[global_node] = build_subgraph(
		p_data,	std::vector<c_GlobalNode> {global_node}, bfs_option, max_subgraph_size);
	} else {
	    log_critical("WARNING: problem in the dataset: repeated def node/prootstep root detected "
		  + std::to_string(global_node.file_idx) + " "
		  + std::to_string(global_node.node_idx));
	}
	}
    }
    return file_subgraphs;
}


void pass_subgraphs_task(
    c_Data * const p_data,
    const bool bfs_option,
    const size_t max_subgraph_size,
    const std::vector<size_t> &file_idx_list,
    std::unordered_map<c_GlobalNode, c_SubGraph> * task_proof_steps_subgraphs) {
    for (const auto file_idx: file_idx_list) {
	auto file_subgraphs = get_file_subgraphs(p_data,  bfs_option,  max_subgraph_size, file_idx);
	task_proof_steps_subgraphs->merge(file_subgraphs);
    }
}



// TODO: make subgraphs stored in file_graph individually



void    pass_subgraphs(
    c_Data * const p_data,
    const std::vector<std::vector<size_t>>& tasks,
    const bool bfs_option,
    const size_t max_subgraph_size) {

    size_t counter = 0;

    std::vector<std::unordered_map<c_GlobalNode, c_SubGraph>> result (tasks.size());

    const size_t num_proc = tasks.size();

    std::vector<std::thread> my_threads(num_proc);

    for (size_t task_idx = 0; task_idx < num_proc; ++task_idx) {
	my_threads[task_idx] = std::thread {
	    pass_subgraphs_task,
	    std::cref(p_data),
	    std::cref(bfs_option),
	    std::cref(max_subgraph_size),
	    std::cref(tasks[task_idx]),
	    &(result[task_idx]) };
    }

    for (size_t task_idx = 0; task_idx < num_proc; ++task_idx) {
	my_threads[task_idx].join();
    }

    for (auto & task_result: result) {
	p_data->subgraphs.merge(task_result);
    }


    for (const auto & [root, subgraph]: p_data->subgraphs) {
	    counter += subgraph.node_labels.size();
    }

    log_verbose("subgraphs nodes: " + std::to_string(counter));

}


c_Tactics build_index_tactic_hashes(
    const std::vector<std::vector<c_ProofStep>> &proof_steps) {

    c_Tactics tactics;

    tactics.max_tactic_args_size = 0;
    for (const auto & file_proof_steps: proof_steps) {

	for (const auto & proof_step: file_proof_steps) {
	    tactics.hash_to_numargs[proof_step.tactic_hash] = proof_step.tactic_args.size();
	    tactics.max_tactic_args_size = std::max(tactics.max_tactic_args_size, proof_step.tactic_args.size());
	    tactics.hash_to_string[proof_step.tactic_hash] = proof_step.tactic_base_text;
	}
    }
    c_TacticIndex tactic_idx = 0;
    for (const auto& [tactic_hash, num_args]: tactics.hash_to_numargs) {
	tactics.index_to_hash.emplace_back(tactic_hash);
	tactics.hash_to_index[tactic_hash] = tactic_idx++;
    }
    return tactics;
}


void init_hash_to_node_label(
    std::unordered_map<c_NodeHash, c_NodeLabel> &hash_to_node_label,
    std::vector<c_NodeHash> &node_label_to_hash,
    std::map<c_NodeHash, std::string> &hash_to_name) {

    c_NodeLabel label = 0;

    auto basic_classes = capnp::Schema::from<Graph::Node::Label>().getFields();

    while (label < basic_classes.size()) {
	hash_to_node_label[label] = label;
	node_label_to_hash.emplace_back(label);
	hash_to_name[label] = basic_classes[label].getProto().getName();
	label++;
    }
}



c_FileGraph read_graph(const Graph::Reader * graph,
		       const std::vector<c_FileIndex> & dep_local_to_global,
		       const c_FileIndex file_idx,
		       const bool ignore_def_hash) {
    const auto nodes = graph->getNodes();
    c_FileGraph file_graph;

    // get nodes and tactical definitions
    for (c_NodeIndex node_idx = 0; node_idx < nodes.size(); ++node_idx) {
	auto node_reader = nodes[node_idx];
	c_NodePrimitive label = static_cast<c_NodePrimitive>(node_reader.getLabel().which());
	c_ChildrenBegin children_begin = node_reader.getChildrenIndex();
	c_ChildrenCount children_count = node_reader.getChildrenCount();

	file_graph.nodes.emplace_back(c_Node {label, children_count, children_begin });

	if (label == NodeLabelDefinition) {

	    file_graph.def_node_to_idx[node_idx] = file_graph.defs.size();
	    log_debug("read graph file " + std::to_string(file_idx) + " def_node_to_idx inserting "  + std::to_string(node_idx));
	    uint64_t def_hash;
	    if (ignore_def_hash) {
		def_hash = base_node_label_num  +  uint64_of_node(c_GlobalNode{file_idx, node_idx});
	    } else {
		def_hash = node_reader.getLabel().getDefinition().getHash();
	    }
	    uint64_t def_hash_for_split = node_reader.getLabel().getDefinition().getHash();
	    auto def_name = node_reader.getLabel().getDefinition().getName();

	    std::vector<c_ProofStep> proof_steps;
	    auto def = node_reader.getLabel().getDefinition();


	    if (def.which() == Definition::TACTICAL_CONSTANT) {
		auto def_proof = def.getTacticalConstant();
		for (const auto &step_obj: def_proof) {
		    auto tactic = step_obj.getTactic();
		    if (tactic.isKnown()) {
			for (const auto &outcome: step_obj.getOutcomes()) {
			    proof_steps.emplace_back(build_proof_step_v9(
							 def_hash,
							 def_hash_for_split,
							 proof_steps.size(),
							 tactic.getKnown(),
							 outcome,
							 dep_local_to_global
							 ));
			}
		    }
		}
	    }


	    file_graph.defs.emplace_back(
		c_Def {node_idx, def_hash, def_name, std::move(proof_steps)});


	}
    }

    // get edges

    auto edges = graph->getEdges();
    file_graph.links = std::vector<std::vector<c_Link>> (file_graph.nodes.size());

    for (size_t node_idx = 0; node_idx < file_graph.links.size(); ++node_idx) {
	c_Node node {file_graph.nodes[node_idx]};
	for (size_t edge_idx = node.children_begin;
	     edge_idx < node.children_begin + node.children_count && edge_idx < edges.size();
	     ++edge_idx) {
	    Graph::EdgeTarget::Reader edge = edges[edge_idx];
	    c_EdgeLabel e_sort = static_cast<c_EdgeLabel>(edge.getLabel());
	    auto target = edge.getTarget();
	    c_FileIndex depIndex = target.getDepIndex();
	    c_NodeIndex target_node_idx = target.getNodeIndex();

	    if (dep_local_to_global.at(depIndex) != __MISSING_FILE) {
		file_graph.links[node_idx].emplace_back(c_Link{e_sort, {
			    dep_local_to_global[depIndex], target_node_idx}});
	    } else {
		    log_critical("ERROR: file index " + std::to_string(dep_local_to_global[0]) + " points to missing file " +
				 std::to_string(depIndex) + " " + std::to_string(target_node_idx) + " ");
	    }
	}
    }

    return file_graph;

}


std::vector<c_FileGraph> pass_graphs(
    unsigned int num_proc,
    const std::vector<std::vector<size_t>>& tasks,
    const FileTable& file_table,
    const bool ignore_def_hash) {


    std::vector<c_FileGraph> file_graphs(file_table.fnames().size());


    auto process_task = [&file_table, &file_graphs, &ignore_def_hash](const std::vector<size_t> &task) {
		       for (const auto file_idx: task) {
			   FileMessageReader pf(file_table.fname(file_idx));
			   const auto dataset  = pf.msg.getRoot<Dataset>();
			   const auto graph = dataset.getGraph();
			   const auto dependencies = dataset.getDependencies();
			   const auto dep_local_to_global = build_dependencies(&dependencies, file_table);

			   file_graphs[file_idx] = read_graph(&graph, dep_local_to_global, file_idx, ignore_def_hash);

			   //for (const auto &node_idx: dataset.getDefinitions()) {
			   //    definitions.insert(node_idx);
			   //}

			   std::set<c_NodeIndex> definitions;
			   auto node_idx = dataset.getRepresentative();
			   const auto nodes = graph.getNodes();
			   const auto nodes_size = nodes.size();
			   while (node_idx < nodes_size) {
			       definitions.insert(node_idx);
			       log_debug("file " + std::to_string(file_idx) + " inserting definition " + std::to_string(node_idx));
			       auto other_idx = nodes[node_idx].getLabel().getDefinition().getPrevious();
			       if (other_idx == node_idx) {
				   break;
			       }
			       node_idx = other_idx;
			   }

			   file_graphs[file_idx].definitions = definitions;

		       }
		   };

    std::vector<std::thread> my_threads(num_proc);

   for (unsigned int i = 0; i < num_proc; ++i) {
	my_threads[i] = std::thread {process_task, tasks[i]};
    }

    for (unsigned int i = 0; i < num_proc; ++i) {
	my_threads[i].join();
    }
    return file_graphs;
}



void update_def_table(
    const std::vector<c_FileGraph> &global_graph,
    const std::vector<std::vector<size_t>> tasks,
    c_DefTable &def_table) {

    for (const auto& task: tasks) {
	for (c_FileIndex file_idx: task) {
	    for (c_DefIndex def_idx = 0; def_idx < global_graph[file_idx].defs.size(); ++def_idx) {
		c_NodeIndex node_idx = global_graph[file_idx].defs[def_idx].node;
		log_debug("inspecting def at node_idx " + std::to_string(node_idx));
		if (global_graph[file_idx].definitions.find(node_idx) !=
		    global_graph[file_idx].definitions.end()) {
		    log_debug("def node is in definitions " + std::to_string(node_idx));
		    c_GlobalNode global_node {file_idx, global_graph[file_idx].defs[def_idx].node};

		    c_NodeHash def_hash = global_graph[file_idx].defs[def_idx].hash;
		    if (def_table.hash_to_node.find(def_hash) ==
			def_table.hash_to_node.end()) {
			log_debug("inserting new definition with hash " + std::to_string(def_hash) +
		      + " with name " + global_graph[file_idx].defs[def_idx].name);
		    }

		    def_table.hash_to_node[def_hash] = global_node;
		    def_table.hash_to_name[def_hash] = global_graph[file_idx].defs[def_idx].name;
		    def_table.node_to_hash[global_node] = def_hash;
		}
	    }
	}
    }

    def_table.available_def_hashes.clear();
    for (const auto & [hash, node]: def_table.hash_to_node) {
	def_table.available_def_hashes.emplace_back(hash);
    }

   encode_hashes(def_table.hash_to_node, def_table.hash_to_node_label, def_table.node_label_to_hash);
   def_table.available_def_classes = get_labels_over_hashes(def_table.hash_to_node_label, def_table.available_def_hashes, std::vector<c_NodeLabel> {});
   log_debug("now we have " + std::to_string(def_table.available_def_classes.size()));
}

std::vector<std::vector<c_ProofStep>> make_global_proof_steps(
    const std::vector<c_FileGraph> &global_graph,
    const std::vector<std::vector<size_t>> &tasks) {

    std::vector<std::vector<c_ProofStep>> global_proof_steps(global_graph.size());


    for (const auto& task: tasks) {
	for (c_FileIndex file_idx: task) {
		// we try to collect proof steps in V9 from all definitions
		// we don't care if those definitions are tactical or not
		// if those definitions don't have proof steps they have empty set of proof_steps
	    for (const c_NodeIndex node_idx: global_graph[file_idx].definitions) {
		//log_debug("node_idx is " + std::to_string(node_idx));
		size_t def_idx = global_graph[file_idx].def_node_to_idx.at(node_idx);
		global_proof_steps[file_idx].insert(global_proof_steps[file_idx].end(),
						global_graph[file_idx].defs[def_idx].proof_steps.begin(),
						global_graph[file_idx].defs[def_idx].proof_steps.end());
	    }
	}
    }
    return global_proof_steps;
}




static PyObject * get_step_state(const c_GlobalNode& root,
				 const std::vector<c_GlobalNode>  context,
				 const c_SubGraph& subgraph,
				 const std::unordered_map<c_NodeHash, c_NodeLabel>& hash_to_node_label,
				 const std::vector<c_NodeLabel>& eval_label_to_train_label) {
    std::vector<c_NodeIndex> context_local(context.size());
    std::transform(context.begin(), context.end(), context_local.begin(),
		   [&subgraph](c_GlobalNode node) {
		       if (subgraph.glob_to_sub.find(node) != subgraph.glob_to_sub.end()) {
			   return static_cast<c_NodeIndex>(subgraph.glob_to_sub.at(node));
		       } else {
			   return __MISSING_NODE;
		       }
		   });
    log_debug("working labels over hashes");
    auto labels_over_hashes = get_labels_over_hashes(hash_to_node_label, subgraph.node_labels, eval_label_to_train_label);

    return Py_BuildValue("NNNNIN",
			 numpy_ndarray1d(labels_over_hashes),
			 numpy_ndarray2d(subgraph.edges_split_flat),
			 numpy_ndarray1d(subgraph.edges_label_flat),
			 numpy_ndarray1d(subgraph.edge_label_offsets),
			 subgraph.glob_to_sub.at(root),
			 numpy_ndarray1d(context_local));
}


static PyObject * load_msg(c_Data * p_data,
	      const char* msg_data,
	      const size_t msg_data_size,
	      const unsigned long long msg_idx,
	      const bool bfs_option,
	      const unsigned long long max_subgraph_size) {
    log_debug("in load_msg");

    auto array_ptr = kj::ArrayPtr<const capnp::word>(
	reinterpret_cast<const capnp::word*>(msg_data), msg_data_size / sizeof(capnp::word));
    capnp::FlatArrayMessageReader msg(
	array_ptr,
	capnp::ReaderOptions{SIZE_MAX});

    const auto request = msg.getRoot<PredictionProtocol::Request>();
    const auto predict = request.getPredict();
    const auto graph = predict.getGraph();

    std::vector<c_FileIndex> dep_local_to_global {static_cast<c_FileIndex>(msg_idx), 0};
    p_data->graph.resize(msg_idx + 1);
    p_data->proof_steps.resize(msg_idx + 1);
    p_data->graph[msg_idx] = read_graph(&graph, dep_local_to_global, msg_idx, false);
    std::vector<std::vector<size_t>> tasks {{msg_idx}};

    log_debug("entering update_def_table " + std::to_string(p_data->def_table.hash_to_node.size()));

    update_def_table(p_data->graph, tasks, p_data->def_table);

    log_debug("again, again after  def_table hash_to_node size " + std::to_string(p_data->def_table.hash_to_node.size()));
    log_debug("but def_table.node_label_to_hash.size is " +
	  std::to_string(p_data->def_table.node_label_to_hash.size()));

    log_debug("entering  pass subgraphs with " + std::to_string(p_data->graph.size()));

    pass_subgraphs(p_data, tasks, bfs_option, max_subgraph_size);

    auto state = predict.getState();
    auto local_root_idx = state.getRoot();
    auto root = c_GlobalNode {dep_local_to_global[0], local_root_idx};
    auto context_obj = state.getContext();
    auto context = build_context(dep_local_to_global[0], &context_obj);
    log_verbose("received context of length " + std::to_string(context.size()));
    auto subgraph = build_subgraph(p_data, std::vector<c_GlobalNode> {root}, bfs_option, max_subgraph_size);

    log_debug("subgraph built ");

    p_data->proof_steps[msg_idx].emplace_back(c_ProofStep {root,
							   c_NodeHash {UINT64_MAX},
							   c_StepIndex {msg_idx},
							   context,  c_TacticHash {UINT64_MAX}});

    log_debug("working on step state ");

    return get_step_state(root, context, subgraph, p_data->def_table.hash_to_node_label, p_data->eval_label_to_train_label);
}


void send_response(const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& predictions,
		   const std::vector<double>& confidences,
		   const int fd) {
    ::capnp::MallocMessageBuilder message;
    auto response = message.initRoot<PredictionProtocol::Response>();
    auto capnp_prediction = response.initPrediction(predictions.size());
    for (size_t pred_idx = 0; pred_idx < predictions.size(); ++pred_idx) {
	const auto &prediction = predictions.at(pred_idx);
	auto tactic = capnp_prediction[pred_idx].initTactic();
	capnp_prediction[pred_idx].setConfidence(confidences.at(pred_idx));
	if (prediction.size() > 0) {
	    uint32_t hi = prediction.at(0).first;
	    uint32_t lo = prediction.at(0).second;
	    tactic.setIdent(uint64_t(hi) << 32 | uint64_t(lo));
	} else {
	    log_critical("ERROR: prediction is empty");
	}
	if (prediction.size() > 1) {

	     auto arguments = capnp_prediction[pred_idx].initArguments(prediction.size() - 1);

	    for (size_t arg_idx = 1; arg_idx < prediction.size(); ++arg_idx) {
		auto term = arguments[arg_idx - 1].initTerm();
		term.setDepIndex(prediction.at(arg_idx).first);
		term.setNodeIndex(prediction.at(arg_idx).second);
	    }
	}
    }
    log_debug("writing message to fd " + std::to_string(fd));
    writePackedMessageToFd(fd, message);
    log_debug("writing message to fd finished");
}


static PyObject * c_encode_prediction_online(PyObject *self, PyObject *args) {
    PyObject * caps_data_online;
    PyListObject * pylist_actions;
    PyArrayObject * pynp_confidences;
    PyArrayObject * pynp_context;
    int fd;
    PyObject * pylist_names;

    if (!PyArg_ParseTuple(args, "O!O!O!O!iO!",
			  &PyCapsule_Type, &caps_data_online,
			  &PyList_Type, &pylist_actions,
			  &PyArray_Type, &pynp_confidences,
			  &PyArray_Type, &pynp_context,
			  &fd,
			  &PyList_Type, &pylist_names)) {
	return NULL;
    }
    try {
	c_DataOnline * p_data_online = static_cast<c_DataOnline*>(PyCapsule_GetPointer(caps_data_online, __data_online_capsule_name));
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> predictions = PyList2dArray_AsVectorVector_uint32_pair((PyObject*)pylist_actions);
	std::vector<double> confidences = PyArray_AsVector_double(pynp_confidences);
	std::vector<c_NodeIndex> context = PyArray_AsVector_uint32_t(pynp_context);
	std::vector<std::string> def_names = PyListBytes_AsVectorString(pylist_names);
	log_info("received predictions of size " + std::to_string(predictions.size()));
	std::vector<PyObject *> np_predictions_encoded;
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> predictions_encoded;

	for (const auto& prediction: predictions) {
	    std::vector<std::pair<uint32_t, uint32_t>> prediction_encoded;
	    if (prediction.size() > 0) {
		uint64_t tactic_hash = p_data_online->tactic_idx_to_hash.at(prediction.at(0).first);
		prediction_encoded.emplace_back(std::make_pair(
						    static_cast<uint32_t>(tactic_hash >> 32),
						    static_cast<uint32_t>(tactic_hash)));
	    }
	    for (size_t arg_idx = 1; arg_idx < prediction.size(); ++arg_idx) {
		auto pred_arg = prediction.at(arg_idx);
		if (pred_arg.first == 0) {
		    // local
		    if (pred_arg.second < context.size()) {
			prediction_encoded.emplace_back(
			    std::pair<uint32_t, uint32_t>{0, context.at(pred_arg.second)});
		    } else {
			log_critical("error: local context of size " + std::to_string(context.size()) + " is: ");
			log_critical(context);
			log_critical("network prediction is (0," + std::to_string(pred_arg.second));
			throw std::logic_error("local argument predicted by network is not in range(len(context))");
		    }
		} else if (pred_arg.first == 1) {
		    // global
		    const auto& eval_global_context = p_data_online->def_idx_to_node;
		    if (pred_arg.second < p_data_online->def_idx_to_node.size()) {
			log_debug("arg idx " + std::to_string(arg_idx - 1) + ", def idx " + std::to_string(pred_arg.second) +  "def name " + def_names.at(pred_arg.second));
			prediction_encoded.emplace_back(
			    std::pair<uint32_t, uint32_t>{1, p_data_online->def_idx_to_node.at(pred_arg.second).node_idx});
		    } else {
			log_critical("error: global context of size " + std::to_string(eval_global_context.size()));
			log_critical(eval_global_context);
			log_critical("network prediction is (1," + std::to_string(pred_arg.second));
			throw std::logic_error("global argument predicted by network is not in range(len(global_context))");
		    }
		} else {
		    log_critical("error: network prediction is " + std::to_string(pred_arg.first) + "," +
				 std::to_string(pred_arg.second));
		    throw std::logic_error("first position in the argument predicted by network is not 0 (local) or 1 (global)");
		}
	    }
	    predictions_encoded.emplace_back(prediction_encoded);
	    np_predictions_encoded.emplace_back(numpy_ndarray2d(prediction_encoded));
	}
	send_response(predictions_encoded, confidences, fd);

	return Py_BuildValue("N",
			     PyList_FromVector(np_predictions_encoded));
    } catch (const std::exception &ia) {
	log_critical("exception in c_encode_prediction_online");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}




static PyObject * c_encode_prediction(PyObject *self, PyObject *args) {
    PyObject * capsObj;
    PyListObject * listObj_act;
    // PyArrayObject * p_PyArrayObj; <--- todo: use for confidences

    unsigned long long msg_idx;
    int fd;

    if (!PyArg_ParseTuple(args, "O!O!Ki", &PyCapsule_Type, &capsObj,
			  &PyList_Type, &listObj_act, &msg_idx, &fd)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    try {
	c_Data *p_data =static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));
	//auto predictions = PyListArray_AsVectorVector_uint64_t((PyObject*)listObj_act);
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> predictions = PyList2dArray_AsVectorVector_uint32_pair((PyObject*)listObj_act);
	log_verbose("received predictions of size " + std::to_string(predictions.size()));
	std::vector<PyObject *> np_predictions_encoded;
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> predictions_encoded;

	for (const auto& prediction: predictions) {
	    std::vector<std::pair<uint32_t, uint32_t>> prediction_encoded;

	    if (prediction.size() > 0) {
		uint64_t tactic_hash = p_data->tactics.index_to_hash.at(prediction.at(0).first);
		log_debug("tactic hash "  + std::to_string(tactic_hash));

		prediction_encoded.emplace_back(std::make_pair(
						    static_cast<uint32_t>(tactic_hash >> 32),
						    static_cast<uint32_t>(tactic_hash)));
	    }

	    log_debug("looping over " + std::to_string(prediction.size()) + " arguments");
	    for (size_t arg_idx = 1; arg_idx < prediction.size(); ++arg_idx) {
		log_debug("preparing to get context at msg_idx " + std::to_string(msg_idx));
		auto &context = p_data->proof_steps.at(msg_idx).back().context;
		log_debug("got context of size " + std::to_string(context.size()));

		if (prediction.at(arg_idx).first == 0) {
		    log_debug("the arg_idx " + std::to_string(arg_idx) + " is local" + std::to_string(prediction.at(arg_idx).second));

		    if (prediction.at(arg_idx).second < context.size()) {
			prediction_encoded.emplace_back(
			    std::make_pair(0, context.at(prediction.at(arg_idx).second).node_idx));
		    } else {
			throw std::logic_error("predict from python has (0, out of context)");
		    }
		} else if (prediction.at(arg_idx).first == 1) {
		    log_debug("the arg_idx " + std::to_string(arg_idx) + " is global " + std::to_string(prediction.at(arg_idx).second));


		    if (prediction.at(arg_idx).second < p_data->def_table.node_label_to_hash.size()) {
			auto global_node_train_class = p_data->global_context.at(prediction.at(arg_idx).second);
			log_info("global_node_train_class " + std::to_string(global_node_train_class));
			auto global_node_eval_class = p_data->train_label_to_eval_label.at(global_node_train_class);
			log_info("mapped to global_node_eval_class " + std::to_string(global_node_train_class));
			auto def_hash = p_data->def_table.node_label_to_hash.at(global_node_eval_class);
			log_info("mapped to def hash " + std::to_string(def_hash));

			//auto def_hash = p_data->def_table.node_label_to_hash.at(prediction.at(arg_idx).second);
			auto def_name = p_data->def_table.hash_to_name.at(def_hash);
			log_info("mapped to def name " +  def_name);
			auto global_node = p_data->def_table.hash_to_node.at(def_hash);
			log_info("global node " + std::to_string(global_node.file_idx) + "," +
			      std::to_string(global_node.node_idx));
			prediction_encoded.emplace_back(
			    std::make_pair(1, global_node.node_idx));
		    } else {
			throw std::logic_error("ERROR: python predict returnd to us (1, " +
					       std::to_string(prediction.at(arg_idx).second) + ") " +
					       "but def_table.node_label_to_hash.size is " +
					       std::to_string(p_data->def_table.node_label_to_hash.size()));
		    }
		} else {
		    log_debug("at arg_idx" + std::to_string(arg_idx));
		    throw std::logic_error("ERROR: python predict returned to us (X, ?) where X is " +
					   std::to_string(prediction.at(arg_idx).first) + " instead of 0|1");
		}
	    }

	    predictions_encoded.emplace_back(prediction_encoded);
	    np_predictions_encoded.emplace_back(numpy_ndarray2d(prediction_encoded));

	}
	log_critical("ATTENTION, as this legacy function is scheduled to deprecated it does not send any message to server");
	log_critical("USING ONLY FOR DEBUG");
	log_critical("normally, use c_encode_prediction_online");

	//send_response(predictions_encoded, fd);


	return Py_BuildValue("N",
			     PyList_FromVector(np_predictions_encoded));
    } catch (const std::exception &ia) {
	log_critical("exception in c_encode_prediction");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}




c_Data load_init_msg(const char* data,
		     const size_t data_size,
		     const std::vector<c_TacticHash> &network_tactic_index_to_hash,
		     const std::vector<uint64_t> &network_tactic_index_to_num_args,
		     const std::vector<std::string> &train_node_label_to_name,
		     const std::vector<uint8_t> &train_node_label_in_spine,

		     const bool bfs_option,
		     const uint64_t max_subgraph_size) {
    log_verbose("in load_init_msg");
    log_info("train node labels: " + std::to_string(train_node_label_to_name.size()));


    std::map<std::string, c_NodeLabel> def_name_to_train_label;
    for (size_t idx = 0; idx < train_node_label_to_name.size(); ++idx) {
	log_debug("idx " + std::to_string(idx));
	const auto& name = train_node_label_to_name.at(idx);
	if (train_node_label_in_spine.at(idx)) {
	    auto res = def_name_to_train_label.find(name);
	    if (res == def_name_to_train_label.end()) {
		def_name_to_train_label[name] = idx;
		log_debug("name, train_label: " + name + " , " + std::to_string(idx));
	    } else {
		throw std::invalid_argument("error: name alignment table has repeating entry "
					    + name + " at positions " + std::to_string(res->second) +
					    "," + std::to_string(idx));
	    }
	}
    }
    log_info("train names in spine: " + std::to_string(def_name_to_train_label.size()));

    auto array_ptr = kj::ArrayPtr<const capnp::word>(reinterpret_cast<const capnp::word*>(data), data_size / sizeof(capnp::word));
    capnp::FlatArrayMessageReader msg(
	array_ptr,
	capnp::ReaderOptions{SIZE_MAX});

    const auto request = msg.getRoot<PredictionProtocol::Request>();
    const auto init = request.getInitialize();
    const auto graph = init.getGraph();

    c_DefTable def_table;
    log_debug("before entering init_hash_to_node_label");
    init_hash_to_node_label(def_table.hash_to_node_label,
			    def_table.node_label_to_hash,
			    def_table.hash_to_name);


    log_verbose("current hash_to_node size " + std::to_string(def_table.hash_to_node.size()));

    std::vector<c_FileIndex> dep_local_to_global {0};

    std::vector<c_FileGraph> global_graph(1);
    global_graph[0] = read_graph(&graph, dep_local_to_global, 0, false);


    std::vector<std::vector<size_t>> tasks {{0}};

    const auto visible_definitions = init.getDefinitions();
    for (const auto& def: visible_definitions) {
	global_graph[0].definitions.insert(def);
    }

    log_info("loaded initialization graph " + std::to_string(global_graph[0].nodes.size()) + " nodes "
	     + std::to_string(global_graph[0].definitions.size()) + " definitions ");



    update_def_table(global_graph, tasks, def_table);
    log_info("evaluation node labels " + std::to_string(def_table.node_label_to_hash.size()));

    std::vector<c_NodeLabel> eval_label_to_train_label;
    std::map<c_NodeLabel, c_NodeLabel> train_label_to_eval_label;

    c_NodeLabel original_train_node_labels = train_node_label_to_name.size();
    c_NodeLabel new_label = original_train_node_labels;
    c_NodeLabel align_counter = 0;
    for (c_NodeLabel eval_label=0; eval_label < def_table.node_label_to_hash.size(); ++eval_label) {
	log_debug("eval_label " + std::to_string(eval_label));
	const auto& def_hash = def_table.node_label_to_hash.at(eval_label);
	log_debug("def hash " + std::to_string(def_hash));

	const auto& def_name = def_table.hash_to_name.at(def_hash);
	log_debug("def name " + def_name);

	auto res = def_name_to_train_label.find(def_name);
	if (res != def_name_to_train_label.end()) {
	    const auto train_label = res->second;
	    log_verbose("aligninig evaluation def with training def " + def_name + " at train label " + std::to_string(train_label));
	    auto prev_res = train_label_to_eval_label.find(train_label);
	    if (prev_res != train_label_to_eval_label.end()) {
		throw std::invalid_argument("repeated definition name in the evaluation graph " +
					    def_name + " at position " + std::to_string(eval_label - base_node_label_num) +
					    "references to train label " + std::to_string(train_label) + " that already is aligned with another eval label " +
					    std::to_string(prev_res->second));
	    } else {
		++align_counter;
		train_label_to_eval_label[train_label] = eval_label_to_train_label.size();
		eval_label_to_train_label.emplace_back(train_label);
		log_debug("eval -> train: " + std::to_string(eval_label) + " " + std::to_string(train_label));
	    }
	} else {
		train_label_to_eval_label[new_label] = eval_label_to_train_label.size();
		eval_label_to_train_label.emplace_back(new_label++);
		log_debug("eval -> train: " + std::to_string(eval_label) + " " + std::to_string(new_label));
	}
    }

    log_info("the network node embedding table has increased from " + std::to_string(original_train_node_labels) +
	     " to " + std::to_string(new_label) + " with " + std::to_string(align_counter) + " defs aligned");

    auto global_proof_steps = make_global_proof_steps(global_graph, tasks);
    std::vector<std::pair<size_t, size_t>> flatten_proof_steps_index;

    get_flatten_proof_steps_index(global_proof_steps, flatten_proof_steps_index);


    std::unordered_map<c_GlobalNode, c_SubGraph> subgraphs;

    c_Tactics tactics {network_tactic_index_to_hash};

    log_verbose("updated def_table hash_to_node size " + std::to_string(def_table.hash_to_node.size()));

    c_Data c_data {
	std::move(global_graph),
	    std::move(def_table),
	    std::move(global_proof_steps),
	    std::move(flatten_proof_steps_index),
	    std::move(subgraphs),
	    build_edge_conflate_map(),
	    std::move(tasks),
	    std::move(tactics),
	    std::move(eval_label_to_train_label),
	    std::move(train_label_to_eval_label)
	    };

    pass_subgraphs(&c_data, c_data.tasks, bfs_option, max_subgraph_size);

    for (c_NodeLabel idx = base_node_label_num; idx < c_data.eval_label_to_train_label.size(); ++idx) {
	c_data.global_context.emplace_back(c_data.eval_label_to_train_label.at(idx));
    }

    log_verbose("populated visible context of size " + std::to_string(c_data.global_context.size()));
    return c_data;
}

/*
// SCHEDULED TO DEPRECATE
std::map<c_FileIndex, std::set<c_FileIndex>> get_local_to_global_file_idx(    const std::string &data_dir,
								      const std::vector<std::string> &fnames) {

    FileTable file_table (fnames, data_dir);
    auto task = get_tasks(1, fnames).front();

    std::map<c_FileIndex, std::set<c_FileIndex>> global_dep_links;

    for (const auto file_idx: task) {
	FileMessageReader pf(file_table.fname(file_idx));
	const auto dataset =  pf.msg.getRoot<Dataset>();
	const auto dependencies = dataset.getDependencies();
	const auto dep_local_to_global = build_dependencies(&dependencies,
							    file_table);

	global_dep_links[file_idx] = std::set<c_FileIndex> (
	    dep_local_to_global.begin(),
	    dep_local_to_global.end());
    }

    return global_dep_links;
}

*/

std::map<c_FileIndex, std::vector<c_FileIndex>> get_local_to_global_file_idx(    const std::string &data_dir,
										 const std::vector<std::string> &fnames) {

    std::map<c_FileIndex, std::vector<c_FileIndex>> global_dep_links {};

    FileTable file_table (fnames, data_dir);

    auto tasks = get_tasks(1, fnames);

    if (tasks.size() == 1) {
	auto task = tasks[0];

	for (const auto file_idx: task) {

	    auto fname = file_table.fname(file_idx);
	    MMapFileMessageReader freader(fname);
	    const auto dataset =  freader.msg->getRoot<Dataset>();
	    const auto dependencies = dataset.getDependencies();
	    const auto dep_local_to_global = build_dependencies(&dependencies,
								file_table);
	    global_dep_links[file_idx] = std::vector<c_FileIndex> (
		dep_local_to_global.begin(),
		dep_local_to_global.end());
	}
    }

    log_info("return global_dep_links");
    return global_dep_links;
}


c_Data get_data(
    const unsigned int num_proc,
    const std::string &data_dir,
    const std::vector<std::string> &fnames,
    const bool bfs_option,
    const unsigned long long max_subgraph_size,
    const bool ignore_def_hash) {
    log_info("get num_proc " + std::to_string(num_proc));
    log_info("get data_dir " + data_dir);
    log_info("number of files  " + std::to_string(fnames.size()));
    log_info("max_subgraph_size  " + std::to_string(max_subgraph_size));


    std::unordered_map<c_GlobalNode, c_SubGraph> subgraphs;

    FileTable file_table (fnames, data_dir);
    auto tasks = get_tasks(num_proc, fnames);

    auto global_graph = pass_graphs(num_proc, tasks, file_table, ignore_def_hash);

    log_info("after pass graph ");
    log_info(global_graph);

    c_DefTable def_table;
    init_hash_to_node_label(def_table.hash_to_node_label, def_table.node_label_to_hash, def_table.hash_to_name);

    update_def_table(global_graph, tasks, def_table);
    log_info("after update def table ");

    auto global_proof_steps = make_global_proof_steps(global_graph, tasks);

    log_info("after global proof steps ");

    std::vector<std::pair<size_t, size_t>> flatten_proof_steps_index;

    c_Tactics tactics;
    {
	Profiler p("global_proof_steps");
	get_flatten_proof_steps_index(global_proof_steps,
				      flatten_proof_steps_index);
	tactics = build_index_tactic_hashes(global_proof_steps);
    }
    log_debug("before c_data  ");

    c_Data data {
	std::move(global_graph),
	    std::move(def_table),
	    std::move(global_proof_steps),
	    std::move(flatten_proof_steps_index),
	    std::move(subgraphs),
	    build_edge_conflate_map(),
	    std::move(tasks),
	    std::move(tactics),
	    };

    log_info("data computed");

    pass_subgraphs(&data,  data.tasks,   bfs_option,  max_subgraph_size);

    return data;
}



/*-------------------------PYTHON API ----------------------------------- */
/*------------------------ PYTHON API ---------------------------------- */


void data_destructor(PyObject *capsObj) {

    log_verbose("c_data destructor is called: freeing memory");

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));
    delete p_data;
    log_verbose("c_data destruction complete");
}

static PyObject *c_get_file_dep(PyObject *self, PyObject *args) {
    PyObject *bytesObj_0;
    if (!PyArg_ParseTuple(args, "S", &bytesObj_0)) {
	return NULL;
    }
    try {
	auto fname = PyBytes_AsSTDString(bytesObj_0);

	MMapFileMessageReader freader(fname);

	const auto dataset =  freader.msg->getRoot<Dataset>();

	const auto dependencies = dataset.getDependencies();
	std::vector<std::string> local_names;
	for (auto const &local_name: dependencies) {
	    local_names.emplace_back(local_name);
	}
	return PyListBytes_FromVectorString(local_names);


    }
    catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}


static PyObject *get_def(const Graph::Reader & graph,
			 std::vector<c_NodeIndex> & def_nodes) {

    std::sort(def_nodes.begin(), def_nodes.end());

    std::vector<uint64_t> def_hashes;
    std::vector<std::string> def_names;
    auto nodes = graph.getNodes();
    for (const auto &node_idx: def_nodes) {
	const auto node = nodes[node_idx];
	if (!node.getLabel().hasDefinition()) {
	    throw std::invalid_argument("the node " + std::to_string(node_idx) +
					"from Dataset.definitions "
					"is not a definition but of type " + std::to_string(node.getLabel().which()));
	}
	const auto def = node.getLabel().getDefinition();
	def_hashes.emplace_back(def.getHash());
	def_names.emplace_back(def.getName());
	log_debug("at node_idx " + std::to_string(node_idx) + " " +  std::string(def.getName()));
    }

    return Py_BuildValue("NNN",
			 numpy_ndarray1d(std::vector<uint64_t>(def_nodes.begin(), def_nodes.end())),
			 numpy_ndarray1d(def_hashes),
			 PyListBytes_FromVectorString(def_names));
}

static PyObject *get_msg_def(capnp::FlatArrayMessageReader *msg,
			     const std::string message_type, int restrict_to_spine) {

    std::vector<c_NodeIndex> def_nodes;
    const auto graph = (message_type == "dataset" ? msg->getRoot<Dataset>().getGraph() : (message_type == "request.initialize" ? msg->getRoot<PredictionProtocol::Request>().getInitialize().getGraph() : msg->getRoot<PredictionProtocol::Request>().getCheckAlignment().getGraph()));
    const auto definitions = (message_type == "dataset" ? msg->getRoot<Dataset>().getDefinitions() : (message_type == "request.initialize" ? msg->getRoot<PredictionProtocol::Request>().getInitialize().getDefinitions() :  msg->getRoot<PredictionProtocol::Request>().getInitialize().getDefinitions()));

    if (!restrict_to_spine) {
	log_info("reading the definitions from message type " + message_type);
	for (const auto &node_idx: definitions) {
	    def_nodes.emplace_back(node_idx);
	}
    } else {
	if (message_type != "dataset") {
	    throw std::invalid_argument("you have called get definitions with the message type that is not dataset");
	}
	auto node_idx = msg->getRoot<Dataset>().getRepresentative();
	auto nodes = graph.getNodes();
	auto nodes_size = nodes.size();
	while (node_idx < nodes_size) {
	    def_nodes.emplace_back(node_idx);
	    auto other_idx = nodes[node_idx].getLabel().getDefinition().getPrevious();
	    if (other_idx == node_idx) {
		break;
	    } else {
		node_idx = other_idx;
	    }
	}
    }
    return get_def(graph, def_nodes);
}


static PyObject *c_get_buf_def(PyObject *self, PyObject *args) {
    PyObject *buf_object;
    char * c_message_type;
    int restrict_to_spine {false};

    if (!PyArg_ParseTuple(args, "Osp",
			  &buf_object, &c_message_type, &restrict_to_spine)) {
	return NULL;
    }
    auto message_type = std::string(c_message_type);
    try {
	if (!(message_type == "dataset" || message_type == "request.initialize" || message_type == "request.checkAlignment")) {
	    throw std::invalid_argument("the message_type parameter must be one of dataset | request.initialize | request.checkAlignment");
	}
	Py_buffer view;
	PyObject_GetBuffer(buf_object, &view, PyBUF_SIMPLE);
	auto array_ptr = kj::ArrayPtr<const capnp::word>(
	    reinterpret_cast<const capnp::word*>(view.buf), view.len / sizeof(capnp::word));
	capnp::FlatArrayMessageReader msg(array_ptr, capnp::ReaderOptions{SIZE_MAX});
	auto result = get_msg_def(&msg, message_type, restrict_to_spine);
	PyBuffer_Release(&view);
	return result;
    }  catch (const std::invalid_argument &ia) {
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}








static PyObject *c_get_local_to_global_file_idx(PyObject *self, PyObject *args) {
    PyObject *bytesObj_0;
    PyObject *listObj;

    if (!PyArg_ParseTuple(args, "SO!", &bytesObj_0,  &PyList_Type,  &listObj)) {
	return NULL;
    }
    try {
	auto data_dir = PyBytes_AsSTDString(bytesObj_0);
	auto fnames = PyListBytes_AsVectorString(listObj);
	log_info("processing " + std::to_string(fnames.size()) + " files");
	std::map<c_FileIndex, std::vector<c_FileIndex>> global_dep_links = get_local_to_global_file_idx(data_dir, fnames);

	std::vector<PyObject*> temp_result;
	for (const auto & [e,v]: global_dep_links) {
	    temp_result.emplace_back(numpy_ndarray1d(v));
	}

	return PyList_FromVector(temp_result);
    }
    catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}




static PyObject *c_get_scc_components(PyObject *self, PyObject *args) {
    PyObject *listObj;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type,  &listObj)) {
	return NULL;
    }
    try {
	auto vectored_links = PyListArray_AsVectorVector_uint64_t(listObj);
	std::map<uint64_t, std::vector<uint64_t>> mapped_links;
	for (uint64_t idx = 0; idx < vectored_links.size(); ++idx) {
	    mapped_links[idx] = vectored_links[idx];
	}
	SimpleDirectedGraph<uint64_t> def_dep_graph {mapped_links};
	auto node_to_component = def_dep_graph.find_strongly_connected_componets();
	std::vector<uint64_t> components;
	std::map<uint64_t, std::vector<uint64_t>> component_to_nodes {};
	for (const auto & [node, component]: node_to_component) {
	    component_to_nodes[component].push_back(node);
	}
	std::vector<PyObject*> vect_component_to_nodes_numpy;
	for (const auto & [component, nodes]: component_to_nodes) {
	    vect_component_to_nodes_numpy.emplace_back(numpy_ndarray1d(nodes));
	}
	return Py_BuildValue("N",PyList_FromVector(vect_component_to_nodes_numpy));

    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}



static PyObject *c_files_scc_components(PyObject *self, PyObject *args) {
    PyObject *bytesObj_0;
    PyObject *listObj;

    if (!PyArg_ParseTuple(args, "SO!", &bytesObj_0,  &PyList_Type,  &listObj)) {
	return NULL;
    }
    try {
	{
	    auto data_dir = PyBytes_AsSTDString(bytesObj_0);
	    auto fnames = PyListBytes_AsVectorString(listObj);
	    log_info("processing " + std::to_string(fnames.size()) + " files");
	    auto global_dep_links = get_local_to_global_file_idx(data_dir, fnames);
	    SimpleDirectedGraph<c_FileIndex> module_dependency_graph {global_dep_links};
	    log_verbose("launching scc");
	    auto scc_components =  module_dependency_graph.find_strongly_connected_componets();
	    std::vector<size_t> scc_components_v;
	    for (const auto& [k, v]: scc_components) {
		scc_components_v.emplace_back(v);
	    }

	    auto number_of_scc_components = scc_components_v.size()==0 ? 0:(* std::max_element(scc_components_v.begin(), scc_components_v.end())) + 1;
	    if (number_of_scc_components == fnames.size()) {
		log_verbose("using priority top sort on modules");
		std::vector<std::vector<uint32_t>> global_dep_links_v(fnames.size());
		std::vector<uint64_t> file_sizes(fnames.size());
		for (size_t file_idx = 0; file_idx < global_dep_links_v.size(); ++file_idx) {
		    global_dep_links_v[file_idx].insert(global_dep_links_v[file_idx].end(),
							global_dep_links.at(file_idx).begin(),
							global_dep_links.at(file_idx).end());
		    file_sizes[file_idx] = std::filesystem::file_size(fnames[file_idx]);
		}
		auto sorted_file_indices = top_sort(global_dep_links_v, file_sizes);
		return numpy_ndarray1d(sorted_file_indices);
	    }
	    return numpy_ndarray1d(scc_components_v);
	}
    }
    catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}

void data_online_destructor(PyObject *capsObj) {
    log_verbose("c_online_data destructor is called: freeing memory");
    c_DataOnline *p_data_online =
	static_cast<c_DataOnline *>(PyCapsule_GetPointer(capsObj, __data_online_capsule_name));
    for (auto &ptr : p_data_online->areaders) {
	delete ptr;
    }
    for (auto &view: p_data_online->views) {
	PyBuffer_Release(&view);
    }

    delete p_data_online;
    log_verbose("c_data online destruction complete");
}

void glob_to_sub_destructor(PyObject *capsObj) {
    std::unordered_map<c_GlobalNode, size_t> * p_glob_to_sub =
	static_cast<std::unordered_map<c_GlobalNode, size_t> *>(
	    PyCapsule_GetPointer(capsObj, __glob_to_sub_capsule_name));
    delete p_glob_to_sub;
}


static PyObject *c_get_buf_tactics(PyObject *self, PyObject *args) {
    PyObject *buf_object;
    PyArrayObject * p_PyArray_def_nodes;
    PyObject *error_annotation;

    if (!PyArg_ParseTuple(args, "OO!S", &buf_object, &PyArray_Type, &p_PyArray_def_nodes, &error_annotation)) {
	return NULL;
    }
    try {
	Py_buffer view;
	PyObject_GetBuffer(buf_object, &view, PyBUF_SIMPLE);
	auto array_ptr = kj::ArrayPtr<const capnp::word>(
	    reinterpret_cast<const capnp::word*>(view.buf), view.len / sizeof(capnp::word));
	capnp::FlatArrayMessageReader msg(array_ptr, capnp::ReaderOptions{SIZE_MAX});

	auto fname = PyBytes_AsSTDString(error_annotation);


	auto def_nodes = PyArray_AsVector_uint64_t(p_PyArray_def_nodes);


	const auto dataset =  msg.getRoot<Dataset>();

	// const auto definitions = dataset.getDefinitions();
	const auto graph = dataset.getGraph();
	const auto nodes = graph.getNodes();
	std::vector<uint64_t> tactic_hashes;
	std::vector<uint64_t> number_arguments;
	std::vector<uint64_t> def_node_indexes;
	std::vector<uint64_t> proof_step_indexes;
	std::vector<uint64_t> outcome_indexes;
	std::vector<uint64_t> context_local_roots;

	uint64_t def_idx = 0;
	for (const auto &node_idx: def_nodes) {
	    const auto node = nodes[node_idx];
	    if (!node.getLabel().hasDefinition()) {
		throw std::invalid_argument("in file " + fname +
					    "the node " + std::to_string(node_idx) +
					    " from Dataset.definitions "
					    "is not a definition!");
	    }
	    const auto def = node.getLabel().getDefinition();
	    if (def.which() == Definition::TACTICAL_CONSTANT) {
		auto def_proof = def.getTacticalConstant();
		uint64_t proof_step_idx = 0;
		for (const auto &step_obj: def_proof) {
		    auto tactic = step_obj.getTactic();
		    if (tactic.isKnown()) {
			auto tactic_known = tactic.getKnown();
			auto tactic_hash = tactic_known.getIdent();
			uint64_t outcome_idx = 0;
			for (const auto &outcome: step_obj.getOutcomes()) {
			    auto num_args = outcome.getTacticArguments().size();
			    auto proof_state = outcome.getBefore();
			    uint64_t local_root = proof_state.getRoot();
			    tactic_hashes.emplace_back(tactic_hash);
			    number_arguments.emplace_back(num_args);
			    def_node_indexes.emplace_back(node_idx);
			    proof_step_indexes.emplace_back(proof_step_idx);
			    outcome_indexes.emplace_back(outcome_idx);
			    context_local_roots.emplace_back(local_root);
			    ++outcome_idx ;
			}
		    }
		    ++proof_step_idx;
		}

	    }
	    ++def_idx;
	}
	PyBuffer_Release(&view);

	return Py_BuildValue("NNNNNN",
			     numpy_ndarray1d(tactic_hashes),
			     numpy_ndarray1d(number_arguments),
			     numpy_ndarray1d(def_node_indexes),
			     numpy_ndarray1d(proof_step_indexes),
			     numpy_ndarray1d(outcome_indexes),
			     numpy_ndarray1d(context_local_roots));
    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}

void check_size(const c_DataOnline * p_data_online) {
	std::vector<size_t> check_sizes {p_data_online->views.size(),
					 p_data_online->fnames.size(),
					 p_data_online->areaders.size(),
					 p_data_online->graphs.size(),
					 p_data_online->rel_file_idx.size(),
					 p_data_online->global_nodes.size(),
					 p_data_online->global_edges.size()};
	if (*(std::max_element(check_sizes.begin(), check_sizes.end())) !=
	    *(std::max_element(check_sizes.begin(), check_sizes.end()))) {
	    log_critical("the sizes of views,fnames, areaders,rel_file_idx,graphs,global_nodes,global_edges:" +
			 std::to_string(p_data_online->views.size()) + "," +
			 std::to_string(p_data_online->fnames.size()) + "," +
			 std::to_string(p_data_online->areaders.size()) + "," +
			 std::to_string(p_data_online->rel_file_idx.size()) + "," +
			 std::to_string(p_data_online->graphs.size()) + "," +
			 std::to_string(p_data_online->global_nodes.size()) + "," +
			 std::to_string(p_data_online->global_edges.size()));
	    throw std::invalid_argument("error: the sizes of views/areaders/graphs/global_nodes/global_edges are not equal, data corrupted, report bug");
	}
}


static PyObject * c_data_online_resize(PyObject *self, PyObject *args) {
    PyObject * capsObj_data_online;
    uint64_t new_size;
    if (!PyArg_ParseTuple(args, "O!K",
			  &PyCapsule_Type, &capsObj_data_online, &new_size)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj_data_online, __data_online_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_online_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }
    try {
	c_DataOnline * const p_data_online = static_cast<c_DataOnline *>(
	    PyCapsule_GetPointer(capsObj_data_online, __data_online_capsule_name));

	check_size(p_data_online);

	if (new_size > p_data_online->graphs.size()) {
	    throw std::invalid_argument("error: the new size is larger then existing");
	}


	// destruction in reverse order
	p_data_online->global_edges.resize(new_size);
	p_data_online->global_nodes.resize(new_size);
	p_data_online->graphs.resize(new_size);
	p_data_online->rel_file_idx.resize(new_size);
	for (auto file_idx = new_size; file_idx < p_data_online->areaders.size(); ++file_idx) {
	    delete p_data_online->areaders.at(file_idx);
	}
	p_data_online->areaders.resize(new_size);

	for (auto file_idx = new_size; file_idx < p_data_online->views.size(); ++file_idx) {
	    PyBuffer_Release(&(p_data_online->views.at(file_idx)));
	}
	p_data_online->views.resize(new_size);
	p_data_online->fnames.resize(new_size);

	check_size(p_data_online);

	return Py_BuildValue("K", p_data_online->graphs.size());
    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}


static PyObject * c_data_online_extend(PyObject *self, PyObject *args) {
    PyObject * capsObj_data_online;
    PyObject * list_buf_objects;
    PyObject * list_Obj_fnames;
    PyObject * p_PyArray_rel_file_idx;
    char * p_message_type;

    if (!PyArg_ParseTuple(args, "O!O!O!O!s",
			  &PyCapsule_Type, &capsObj_data_online,
			  &PyList_Type, &list_buf_objects,
			  &PyList_Type,  &list_Obj_fnames,
			  &PyList_Type,  &p_PyArray_rel_file_idx,
			  &p_message_type)) {
	return NULL;
    }
    std::string message_type(p_message_type);

    if ((message_type != "dataset") &&
	(message_type != "request.initialize") &&
	(message_type != "request.predict")) {
	auto msg = "error: message type must be dataset | request.initialize | request.predict";
	PyErr_SetString(PyExc_TypeError, msg);
	return NULL;
    }



    if (!PyCapsule_IsValid(capsObj_data_online, __data_online_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_online_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    try {
	c_DataOnline * const p_data_online = static_cast<c_DataOnline *>(
	    PyCapsule_GetPointer(capsObj_data_online, __data_online_capsule_name));

	check_size(p_data_online);


	// CHECKS
	if (!PyList_Check(list_buf_objects)) {
	      throw std::invalid_argument("expected list of buf objects in first argument");
	}

	if (!PyList_Check(list_Obj_fnames)) {
	    throw std::invalid_argument("expected list of fnames (bytes) annotations in second argument");
	}

	if (!PyList_Check(p_PyArray_rel_file_idx)) {
	    throw std::invalid_argument("expected list 1d uint32 np.arrays enconding the local_to_glocal dep map in third argument");
	}

	if (!(PyList_Size(list_buf_objects) == PyList_Size(list_Obj_fnames))) {
	    throw std::invalid_argument("the length of list of fnames must be equal to the length of list of buf objects");
	}

	if (!(PyList_Size(list_buf_objects) == PyList_Size(p_PyArray_rel_file_idx))) {
	    throw std::invalid_argument("the length of list of dep references must be equal to the length of list of buf objects");
	}


	// BUFFER GRAPHS
	auto rel_file_idx_ext = PyListArray_AsVectorVector_uint32_t(p_PyArray_rel_file_idx);
	auto names_ext = PyListBytes_AsVectorString(list_Obj_fnames);


	for (Py_ssize_t cnt = 0; cnt < PyList_Size(list_buf_objects); ++cnt) {
	    PyObject *buf_object = PyList_GetItem(list_buf_objects, cnt);
	    if (!PyObject_CheckBuffer(buf_object)) {
		throw std::invalid_argument("all elements in the list passed as a first argument must support buffer protocol (aka memoryview)");
	    }
	    Py_buffer view;
	    PyObject_GetBuffer(buf_object, &view, PyBUF_SIMPLE);
	    auto array_ptr = kj::ArrayPtr<const capnp::word>(
		reinterpret_cast<const capnp::word*>(view.buf), view.len / sizeof(capnp::word));
	    capnp::FlatArrayMessageReader * ptr = new capnp::FlatArrayMessageReader(array_ptr, capnp::ReaderOptions{SIZE_MAX});


	    p_data_online->views.emplace_back(view);
	    p_data_online->areaders.emplace_back(ptr);
	    if (message_type == "dataset") {
		p_data_online->graphs.emplace_back(ptr->getRoot<Dataset>().getGraph());
	    } else if (message_type == "request.initialize") {
		p_data_online->graphs.emplace_back(ptr->getRoot<PredictionProtocol::Request>().getInitialize().getGraph());
	    } else if (message_type == "request.predict") {
		p_data_online->graphs.emplace_back(ptr->getRoot<PredictionProtocol::Request>().getPredict().getGraph());
	    } else { // Python Type Exception
		auto msg = "error: expected in the last argument mesage type dataset | request.initialize | request.predict  ";
		PyErr_SetString(PyExc_TypeError, msg);
		return NULL;
	    }

	    p_data_online->global_nodes.emplace_back(p_data_online->graphs.back().getNodes());
	    p_data_online->global_edges.emplace_back(p_data_online->graphs.back().getEdges());
	    p_data_online->rel_file_idx.emplace_back(rel_file_idx_ext.at(cnt));
	    p_data_online->fnames.emplace_back(names_ext.at(cnt));
	}
	check_size(p_data_online);

	return Py_BuildValue("K", p_data_online->graphs.size());
    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}




static PyObject * c_build_data_online_from_buf(PyObject *self, PyObject *args) {
    PyArrayObject * p_PyArray_def_idx_to_node;
    PyArrayObject * p_PyArray_tactic_hashes;
    PyArrayObject * p_PyArray_tactic_indexes;
    if (!PyArg_ParseTuple(args, "O!O!O!",
			  &PyArray_Type,  &p_PyArray_def_idx_to_node,
			  &PyArray_Type, &p_PyArray_tactic_hashes,
			  &PyArray_Type, &p_PyArray_tactic_indexes)) {
	return NULL;
    }
    try {
	auto p_data_online = new c_DataOnline;

	// GLOBAL
	p_data_online->conflate = build_edge_conflate_map();


	// TACTICS
	const std::vector<uint64_t> tactic_hashes = PyArray_AsVector_uint64_t(p_PyArray_tactic_hashes);
	const std::vector<uint32_t> tactic_indexes = PyArray_AsVector_uint32_t(p_PyArray_tactic_indexes);
	if (tactic_hashes.size() != tactic_indexes.size()) {
	    throw std::invalid_argument("error: the size of provide tactic_hashes and tactic_indexes arrays are different");
	}
	for (size_t idx = 0; idx < std::min(tactic_hashes.size(), tactic_indexes.size()); ++idx) {
	    p_data_online->tactic_hash_to_idx[tactic_hashes.at(idx)] = tactic_indexes.at(idx);
	    p_data_online->tactic_idx_to_hash[tactic_indexes.at(idx)] = tactic_hashes.at(idx);
	}

	// DEF INDEX
	auto def_idx_to_node_uint64_t = PyArray_AsVector_uint64_t(p_PyArray_def_idx_to_node);
	for (const auto &x: def_idx_to_node_uint64_t) {
	    c_GlobalNode node = node_of_uint64(x);
	    p_data_online->def_node_to_idx[node] = p_data_online->def_idx_to_node.size();
	    p_data_online->def_idx_to_node.emplace_back(node);
	}



	return PyCapsule_New(p_data_online, __data_online_capsule_name, data_online_destructor);
    }	catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}






static PyObject *c_get_def_deps_online(PyObject *self, PyObject *args) {
    PyObject *capsObj;
    int bfs_option {false};
    uint64_t max_subgraph_size;


    if (!PyArg_ParseTuple(args, "O!pK", &PyCapsule_Type, &capsObj, &bfs_option, &max_subgraph_size)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_online_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_online_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }
    try {
	c_DataOnline *p_data_online = static_cast<c_DataOnline *>(
	    PyCapsule_GetPointer(capsObj, __data_online_capsule_name));

	std::vector<PyObject*> all_def_deps;



	for (const auto& def_node: p_data_online->def_idx_to_node) {
	    auto global_roots = std::vector<c_GlobalNode> {def_node};
	    auto [global_deps, global_visited] =  mmaped_forward_closure(
		p_data_online->graphs, global_roots,
		p_data_online->rel_file_idx,
		EdgeClassification::CONST_OPAQUE_DEF,
		bfs_option,
		max_subgraph_size,
		p_data_online->def_node_to_idx,
		p_data_online->global_nodes,
		p_data_online->global_edges
		);
	    std::vector<uint64_t> def_deps;
	    delete global_visited;
	    for (const auto global_dep: global_deps) {
		def_deps.emplace_back(p_data_online->def_node_to_idx.at(global_dep));
	    }
	    all_def_deps.emplace_back(numpy_ndarray1d(def_deps));
	}
	return Py_BuildValue("N", PyList_FromVector(all_def_deps));
    }  catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}






static PyObject *c_get_subgraph_online(PyObject *self, PyObject *args) {
    PyObject *capsObj;
    PyArrayObject *p_roots_uint64;
    int bfs_option {false};
    uint64_t max_subgraph_size;
    int with_node_counter {false};

    if (!PyArg_ParseTuple(args, "O!O!pKp", &PyCapsule_Type, &capsObj,
			  &PyArray_Type, &p_roots_uint64,
			  &bfs_option, &max_subgraph_size, &with_node_counter)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_online_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_online_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    try {
	c_DataOnline *p_data_online = static_cast<c_DataOnline *>(
	    PyCapsule_GetPointer(capsObj, __data_online_capsule_name));

	uint32_t min_node_idx = UINT32_MAX;
	uint32_t max_node_idx = 0;


	std::vector<c_NodeIndex> local_roots;
	std::vector<c_GlobalNode> roots;
	for (const auto &e:  PyArray_AsVector_uint64_t(p_roots_uint64)) {
	    auto this_node = node_of_uint64(e);
	    if (roots.size() > 0) {
		if (roots.back().file_idx != this_node.file_idx) {
		    throw std::invalid_argument(
			"sampling subgraph online for local_roots of different files is not supported");
		}
	    }
	    roots.emplace_back(this_node);
	    local_roots.emplace_back(this_node.node_idx);
	    min_node_idx = std::min(min_node_idx, this_node.node_idx);
	    max_node_idx = std::max(max_node_idx, this_node.node_idx);


	}
	if (roots.size() == 0) {
	    throw std::invalid_argument("sampling subgraphs from empty set of roots is not supported");
	}

	c_FileIndex this_file_idx = roots.back().file_idx;
	auto fname = p_data_online->fnames[this_file_idx];

	// FORWARD CLOSURE
	auto [global_deps, global_visited] =  mmaped_forward_closure(
	    p_data_online->graphs, roots,
	    p_data_online->rel_file_idx,
	    EdgeClassification::CONST_OPAQUE_DEF,
	    bfs_option,
	    max_subgraph_size,
	    p_data_online->def_node_to_idx,
	    p_data_online->global_nodes,
	    p_data_online->global_edges
	    );

	// NODES
	std::vector<c_GlobalNode> sub_to_glob(global_visited->size());
	std::vector<c_NodeLabel> node_labels(global_visited->size());

	for (const auto& [global_node, idx]: (*global_visited)) {

	    sub_to_glob[idx] = global_node;
	    c_NodeLabel node_label;

	    if (p_data_online->def_node_to_idx.find(global_node) != p_data_online->def_node_to_idx.end()) {
		node_label = p_data_online->def_node_to_idx.at(global_node) + base_node_label_num;
	    } else {
		node_label = static_cast<c_NodeLabel>(p_data_online->global_nodes.at(global_node.file_idx)[global_node.node_idx].getLabel().which());
		//min_node_idx = std::min(min_node_idx, global_node.node_idx);
		//max_node_idx = std::max(max_node_idx, global_node.node_idx);
	    }
	    node_labels[idx] = node_label;
	}
	const auto &conflate = p_data_online->conflate;

	std::vector<std::vector<std::array<c_NodeIndex, 2>>> edges_split_by_label(conflate.size);

	uint32_t min_edge_idx = UINT32_MAX;
	uint32_t max_edge_idx = 0;

	for (const auto & [source, source_sub]: (*global_visited)) {
	    auto node = p_data_online->global_nodes.at(source.file_idx)[source.node_idx];
	    if (std::find(global_deps.begin(), global_deps.end(), source) == global_deps.end()) {
		auto edge_idx_begin = node.getChildrenIndex();
		auto edge_idx_end = edge_idx_begin + node.getChildrenCount();
		auto edges = p_data_online->global_edges.at(source.file_idx);
		for (uint32_t edge_idx = edge_idx_begin; edge_idx < edge_idx_end; ++edge_idx) {
		    const auto& capnp_edge = edges[edge_idx];
		    if (capnp_edge.getLabel() != EdgeClassification::CONST_OPAQUE_DEF) {
			const auto& target = capnp_edge.getTarget();
			auto rel_target_file_idx = target.getDepIndex();
			auto target_node_idx = target.getNodeIndex();
			auto target_file_idx = p_data_online->rel_file_idx.at(source.file_idx).at(rel_target_file_idx);
			c_GlobalNode target_node {target_file_idx, target_node_idx};
			if (global_visited->find(target_node) != global_visited->end()) {
			    min_edge_idx = std::min(min_edge_idx, edge_idx);
			    max_edge_idx = std::max(max_edge_idx, edge_idx);

			    c_NodeIndex target_sub = static_cast<c_NodeIndex>(global_visited->at(target_node));
			    edges_split_by_label[conflate.map.at(static_cast<c_EdgeLabel>(capnp_edge.getLabel()))].emplace_back(
				std::array<c_NodeIndex, 2> {static_cast<c_NodeIndex>(source_sub), target_sub});
			}
		    }
		}
	    }
	}
	const auto [edges_label_offsets, edges_label_flat, edges_split_flat] = compute_offsets(edges_split_by_label);

	std::map<uint64_t, uint64_t> report_node_counter;
	//if (with_node_counter) {
	//   report_node_counter = p_data_online->subgraph_node_counter;
	//}
	return Py_BuildValue("NNNNNKKII",
			     numpy_ndarray1d(node_labels),
			     numpy_ndarray2d(edges_split_flat),
			     numpy_ndarray1d(edges_label_flat),
			     numpy_ndarray1d(edges_label_offsets),
			     PyCapsule_New(global_visited, __glob_to_sub_capsule_name, glob_to_sub_destructor),
			     static_cast<uint64_t>(roots.size()),
			     global_deps.size(),
			     max_node_idx - min_node_idx,
			     max_edge_idx - min_edge_idx );
//			     numpy_ndarray1d_uint64_of_node(sub_to_glob),
//			     numpy_ndarray2d(report_node_counter)
//	    );
    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}

static PyObject *c_get_proof_step_online_text(PyObject *self, PyObject *args) {
    PyObject *capsObj_data_online;
    PyArrayObject *p_file_def_step_outcome;

    if (!PyArg_ParseTuple(args, "O!O!", &PyCapsule_Type, &capsObj_data_online,
			  &PyArray_Type, &p_file_def_step_outcome)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj_data_online, __data_online_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_online_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }
    try {
	const c_DataOnline *p_data_online = static_cast<c_DataOnline *>(
	    PyCapsule_GetPointer(capsObj_data_online, __data_online_capsule_name));


	const auto params = PyArray_AsVector_uint64_t(p_file_def_step_outcome);


	c_FileIndex file_idx = static_cast<c_FileIndex>(params[0]);


	const auto def_node_idx = params[1];
	const auto step_idx = params[2];
	const auto outcome_idx = params[3];

	const auto def = p_data_online->global_nodes.at(file_idx)[def_node_idx].getLabel().getDefinition();
	if (!(def.which() == Definition::TACTICAL_CONSTANT)) {
	    throw std::invalid_argument("the definition at provided location must be tactical constant");
	}
	const auto capnp_step = def.getTacticalConstant()[step_idx];
	const auto tactic = capnp_step.getTactic();


	const auto capnp_outcome = capnp_step.getOutcomes()[outcome_idx];
	const auto capnp_proof_state = capnp_outcome.getBefore();


	const std::string state_text = capnp_proof_state.getText();
	const std::string action_base_text = tactic.getKnown().getBaseText();
	const std::string action_interm_text = tactic.getKnown().getIntermText();
	const std::string action_text = tactic.getKnown().getText();

	return Py_BuildValue("yyyy",
			     state_text.c_str(),
			     action_base_text.c_str(),
			     action_interm_text.c_str(),
			     action_text.c_str());
    }
    catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}




static PyObject *c_get_proof_step_online(PyObject *self, PyObject *args) {
    PyObject *capsObj_data_online;
    PyArrayObject *p_file_def_step_outcome;
    PyObject *capsObj_glob_to_sub;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyCapsule_Type, &capsObj_data_online,
			  &PyArray_Type, &p_file_def_step_outcome,
			  &PyCapsule_Type, &capsObj_glob_to_sub)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj_data_online, __data_online_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_online_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj_glob_to_sub, __glob_to_sub_capsule_name)) {
	auto msg = "error: expected " + std::string(__glob_to_sub_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    try {
	const c_DataOnline *p_data_online = static_cast<c_DataOnline *>(
	    PyCapsule_GetPointer(capsObj_data_online, __data_online_capsule_name));

	const std::unordered_map<c_GlobalNode, size_t>  * p_glob_to_sub = static_cast<std::unordered_map<c_GlobalNode, size_t>*>(
	    PyCapsule_GetPointer(capsObj_glob_to_sub,  __glob_to_sub_capsule_name));

	const auto params = PyArray_AsVector_uint64_t(p_file_def_step_outcome);
	c_FileIndex file_idx = static_cast<c_FileIndex>(params[0]);
	const auto def_node_idx = params[1];
	const auto step_idx = params[2];
	const auto outcome_idx = params[3];
	const auto def = p_data_online->global_nodes.at(file_idx)[def_node_idx].getLabel().getDefinition();
	if (!(def.which() == Definition::TACTICAL_CONSTANT)) {
	    throw std::invalid_argument("the definition at provided location must be tactical constant");
	}
	const auto capnp_step = def.getTacticalConstant()[step_idx];

	uint32_t tactic_index;
	if (!capnp_step.getTactic().isKnown()) {
	    throw std::invalid_argument("error: the tactic is not known to capnp protocol -- please refactor loader to remove unknown tactics");
	} else {
	    const auto tactic_hash = capnp_step.getTactic().getKnown().getIdent();
	    auto res = p_data_online->tactic_hash_to_idx.find(tactic_hash);
	    if (res == p_data_online->tactic_hash_to_idx.end()) {
		throw std::invalid_argument("error: the tactic hash is not in the capnp tactic hash table; failure to encode");
	    }
	    tactic_index = (*res).second;
	}
	const auto capnp_outcome = capnp_step.getOutcomes()[outcome_idx];
	const auto capnp_proof_state = capnp_outcome.getBefore();
	const auto capnp_context = capnp_proof_state.getContext();
	std::vector<uint32_t> encoded_context;
	encoded_context.reserve(capnp_context.size());

	std::unordered_map<c_GlobalNode, size_t> global_to_context_idx;
	for (const auto &node_idx: capnp_context) {
	    auto res = p_glob_to_sub->find(c_GlobalNode{file_idx, node_idx});
	    if (res != p_glob_to_sub->end()) {
		global_to_context_idx[c_GlobalNode{file_idx, node_idx}] = encoded_context.size();
		encoded_context.emplace_back(static_cast<uint32_t>((*res).second));
	    } else {
		log_warning("skipped the context node " + std::to_string(node_idx) +  " is not in the sampled subgraph at"
			    + p_data_online->fnames.at(file_idx) + " def_node_idx=" + std::to_string(def_node_idx) +
			    + ", step_idx=" + std::to_string(step_idx) + ", outcome_idx=" + std::to_string(outcome_idx) +
			    " : max_subgraph_size is too small");
	    }
	}
	uint32_t encoded_root;
	auto res = p_glob_to_sub->find(c_GlobalNode{file_idx, capnp_proof_state.getRoot()});
	if (res == p_glob_to_sub->end()) {
	    throw std::invalid_argument("the root is not in the subgraph and can't be encoded (this shouldn't happend for a subgraph of a positive size and indicates a severed bug");
	} else {
	    encoded_root = (*res).second;
	}

	std::vector<c_GlobalNode> tactic_args = build_tactic_args(
	    capnp_outcome.getTacticArguments(), p_data_online->rel_file_idx.at(file_idx));



	std::vector<std::array<c_NodeIndex,2>> arg_context_index;
	for (const c_GlobalNode tactic_node: tactic_args) {
	    auto res = global_to_context_idx.find(tactic_node);
	    if (res != global_to_context_idx.end()) {
		arg_context_index.emplace_back(std::array<c_NodeIndex,2>{0,static_cast<c_NodeIndex>((*res).second)});
	    } else {
		auto res = p_data_online->def_node_to_idx.find(tactic_node);
		if (res != p_data_online->def_node_to_idx.end()) {
		    arg_context_index.emplace_back(std::array<c_NodeIndex,2>{1,static_cast<c_NodeIndex>((*res).second)});
		} else {
		    arg_context_index.emplace_back(std::array<c_NodeIndex,2>{0,static_cast<c_NodeIndex>(encoded_context.size())});
		}
	    }
	}
	return Py_BuildValue("NIIN", numpy_ndarray1d(encoded_context), encoded_root, tactic_index, numpy_ndarray2d(arg_context_index));
   } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}




static PyObject *c_capnp_unpack(PyObject *self, PyObject *args) {
    PyObject *bytesObj;
    if (!PyArg_ParseTuple(args, "S", &bytesObj)) {
	return NULL;
    }
    try {
	auto fname = PyBytes_AsSTDString(bytesObj);
	log_info("unpacking " + fname);

	PackedFileMessageReader pf(fname);
	capnp::MallocMessageBuilder capnp_msg;
	capnp_msg.setRoot(pf.msg.getRoot<Dataset>());
	auto file_out = fileno(fopen_safe(fname + "x", "wb"));
	capnp::writeMessageToFd(file_out,capnp_msg);
	return Py_BuildValue("I", 0);
    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}


static PyObject *c_check(PyObject *self, PyObject *args) {
    PyObject *bytesObj;
    if (!PyArg_ParseTuple(args, "S", &bytesObj)) {
	return NULL;
    }
    try {
	auto fname = PyBytes_AsSTDString(bytesObj);

	FileMessageReader pf(fname);
	log_info("get dataset root");
	auto dataset = pf.msg.getRoot<Dataset>();
	log_info("get dataset graph");
	auto graph = dataset.getGraph();

	log_info("get dataset nodes");
	auto nodes = graph.getNodes();
	log_info("nodes size");
	log_info(std::to_string(nodes.size()));


	log_info("get dataset edges: ");
	auto edges = graph.getEdges();
	log_info("edges size: ");
	log_info(std::to_string(edges.size()));


	return Py_BuildValue("I", 0);
    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}



static PyObject *c_get_data(PyObject *self, PyObject *args) {

    PyObject *bytesObj_0;
    PyObject *bytesObj_1;
    PyObject *listObj;
    unsigned int num_proc;
    int bfs_option {false};
    unsigned long long max_subgraph_size;
    int ignore_def_hash {false};
    if (!PyArg_ParseTuple(args, "SSO!IpKp", &bytesObj_0, &bytesObj_1, &PyList_Type,
			  &listObj, &num_proc, &bfs_option, &max_subgraph_size, &ignore_def_hash)) {
	return NULL;
    }
    try {
	auto data_dir = PyBytes_AsSTDString(bytesObj_0);
	auto project_dir = PyBytes_AsSTDString(bytesObj_1);
	auto fnames = PyListBytes_AsVectorString(listObj);
	log_info("fnames computed");

	auto data = get_data(num_proc, data_dir, fnames, bfs_option, max_subgraph_size, ignore_def_hash);
	log_info("data computed");
	auto p_data = new c_Data(std::move(data));
	return PyCapsule_New(p_data, __data_capsule_name, data_destructor);
    } catch (const std::invalid_argument &ia) {
	log_critical("exception caught");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}

static PyObject *c_load_init_msg(PyObject *self, PyObject *args) {
    PyObject * bytesObj;
    PyObject * p_ListObj;
    PyArrayObject * p_PyArrayObject_tactic;
    PyArrayObject * p_PyArrayObject_tactic_num_args;
    PyArrayObject * p_PyArrayObject_in_spine;
    int bfs_option {false};
    uint64_t max_subgraph_size;
    if (!PyArg_ParseTuple(args, "SO!O!O!O!pK", &bytesObj,
			  &PyArray_Type, &p_PyArrayObject_tactic,
			  &PyArray_Type, &p_PyArrayObject_tactic_num_args,
			  &PyList_Type, &p_ListObj,
			  &PyArray_Type, &p_PyArrayObject_in_spine,
			  &bfs_option, &max_subgraph_size)) {
	return NULL;
    }
    try {
	auto msg_data = PyBytes_AsString(bytesObj);
	auto msg_data_size = PyBytes_Size(bytesObj);
	log_verbose("c_load_init runs on data of size " + std::to_string(msg_data_size));

	auto network_tactic_index_to_hash = PyArray_AsVector_uint64_t(p_PyArrayObject_tactic);
	log_verbose("network tactic_index_to_hash table of size " + std::to_string(network_tactic_index_to_hash.size()));

	auto network_tactic_index_to_num_args = PyArray_AsVector_uint64_t(p_PyArrayObject_tactic);
	log_verbose("network tactic_index_to_numargs " + std::to_string(network_tactic_index_to_num_args.size()));

	auto node_label_in_spine = PyArray_AsVector_uint8_t(p_PyArrayObject_in_spine);
	log_verbose("received node_label_in_spine table of size " + std::to_string(node_label_in_spine.size()));


	auto node_label_to_name = PyListBytes_AsVectorString(p_ListObj);

		    c_Data data = load_init_msg(msg_data, msg_data_size,
				    network_tactic_index_to_hash,
				    network_tactic_index_to_num_args,
				    node_label_to_name,
				    node_label_in_spine,
				    bfs_option,
				    max_subgraph_size);
	auto p_data = new c_Data(std::move(data));


	return PyCapsule_New(p_data, __data_capsule_name,
			 data_destructor);

    } catch (const std::exception &ia) {
	log_critical("exception in c_load_init_msg");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}


static PyObject * c_load_msg_online(PyObject *self, PyObject *args) {
    PyObject * p_msg_data;
    PyObject * capsObj_glob_to_sub;
    c_FileIndex file_idx;
    if (!PyArg_ParseTuple(args, "OO!I", &p_msg_data,
			  &PyCapsule_Type, &capsObj_glob_to_sub, &file_idx)) {
	return NULL;
    }
    try {
	if (!PyObject_CheckBuffer(p_msg_data)) {
	    throw std::invalid_argument("the first argument must be a message supporting buffer protocol");
	}
	if (!PyCapsule_IsValid(capsObj_glob_to_sub, __glob_to_sub_capsule_name)) {
	    auto msg = "error: expected " + std::string(__glob_to_sub_capsule_name);
	    PyErr_SetString(PyExc_TypeError, msg.c_str());
	    return NULL;
	}


	Py_buffer view;
	PyObject_GetBuffer(p_msg_data, &view, PyBUF_SIMPLE);
	const auto array_ptr = kj::ArrayPtr<const capnp::word>(
		reinterpret_cast<const capnp::word*>(view.buf), view.len / sizeof(capnp::word));
	capnp::FlatArrayMessageReader msg(array_ptr, capnp::ReaderOptions{SIZE_MAX});
	const auto capnp_state = msg.getRoot<PredictionProtocol::Request>().getPredict().getState();


	const std::unordered_map<c_GlobalNode, size_t>  * p_glob_to_sub = static_cast<std::unordered_map<c_GlobalNode, size_t>*>(
	    PyCapsule_GetPointer(capsObj_glob_to_sub,  __glob_to_sub_capsule_name));



	uint32_t encoded_root;
	auto res = p_glob_to_sub->find(c_GlobalNode{file_idx, capnp_state.getRoot()});
	if (res != p_glob_to_sub->end()) {
	    encoded_root = (*res).second;
	} else {
	    throw std::invalid_argument("the root is not in the subgraph and can't be encoded (this shouldn't happend for a subgraph of a positive size and indicates a severed bug");
	}



	std::vector<uint32_t> encoded_context;
	std::vector<uint32_t> context;
	for (const auto &node_idx: capnp_state.getContext()) {
	    auto res = p_glob_to_sub->find(c_GlobalNode{file_idx, node_idx});
	    if (res != p_glob_to_sub->end()) {
		context.emplace_back(node_idx);
		encoded_context.emplace_back(static_cast<uint32_t>((*res).second));
	    } else {
	    log_warning("skipped the context node " + std::to_string(node_idx) + " is not in the sampled subgraph at "
			+ " msg idx " + std::to_string(file_idx) );
	    }
	}

	return Py_BuildValue("INN",
			     encoded_root,
			     numpy_ndarray1d(encoded_context),
			     numpy_ndarray1d(context));
    }  catch (const std::exception &ia) {
	log_critical("exception in load_msg");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}




static PyObject *c_load_msg(PyObject *self, PyObject *args) {
    PyObject * capsObj;
    PyObject * bytesObj;
    unsigned long long msg_idx;
    int bfs_option;
    unsigned long long max_subgraph_size;
    if (!PyArg_ParseTuple(args, "O!SKpK", &PyCapsule_Type, &capsObj, &bytesObj,
			  &msg_idx, &bfs_option, &max_subgraph_size)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    try {
	auto msg_data = PyBytes_AsString(bytesObj);
	auto msg_data_size = PyBytes_Size(bytesObj);
	c_Data *p_data =static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

	return load_msg(p_data, msg_data, msg_data_size, msg_idx, bfs_option, max_subgraph_size);
    } catch (const std::exception &ia) {
	log_critical("exception in load_msg");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}





static PyObject *c_get_proof_steps_size(PyObject *self, PyObject *args) {

    PyObject *capsObj;
    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &capsObj)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    size_t counter = 0;

    for (const auto &file_proof_steps : p_data->proof_steps) {
	counter += file_proof_steps.size();
    }
    return PyLong_FromSize_t(counter);
}





static PyObject *c_get_subgraph(PyObject *self, PyObject *args) {

    PyObject *capsObj;
    uint64_t root_uint64;
    int bfs_option {false};
    uint64_t max_subgraph_size;

    if (!PyArg_ParseTuple(args, "O!KpK", &PyCapsule_Type, &capsObj,
			  &root_uint64,  &bfs_option, &max_subgraph_size   )) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    const c_Data *p_data = static_cast<c_Data *>(
	PyCapsule_GetPointer(capsObj, __data_capsule_name));

    const c_GlobalNode root = node_of_uint64(root_uint64);

    c_SubGraph subgraph;
    if (p_data->subgraphs.find(root) == p_data->subgraphs.end()) {
	log_error("WARNING: requested subgraph not found, building again");
	subgraph = build_subgraph (p_data, std::vector<c_GlobalNode> {root}, bfs_option, max_subgraph_size);
    } else {
	subgraph = p_data->subgraphs.at(root);
    }


    auto labels_over_hashes =	get_labels_over_hashes(p_data->def_table.hash_to_node_label, subgraph.node_labels, p_data->eval_label_to_train_label);

    return Py_BuildValue("NNNN",
			 numpy_ndarray1d_uint64_of_node(subgraph.sub_to_glob),
			 numpy_ndarray1d(labels_over_hashes),
			 numpy_ndarray2d(subgraph.edges_split_flat),
			 numpy_ndarray1d(subgraph.edge_label_offsets));

}


static PyObject *c_build_subgraphs(PyObject *self, PyObject *args) {

    PyObject *capsObj;
    int bfs_option {false};
    unsigned long long max_subgraph_size;
    if (!PyArg_ParseTuple(args, "O!pK", &PyCapsule_Type, &capsObj,  &bfs_option, &max_subgraph_size)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data = static_cast<c_Data *>(
	PyCapsule_GetPointer(capsObj, __data_capsule_name));



    { Profiler p("pass build proof_step and definition subgraphs");
	//auto [total_nodes, all_hashes_encode_count]
    pass_subgraphs(p_data,  p_data->tasks,   bfs_option,  max_subgraph_size);

    return Py_BuildValue("(O)", Py_None);
    }

}






static PyObject *c_get_proof_step(PyObject *self, PyObject *args) {

    PyObject *capsObj;
    Py_ssize_t proof_step_idx;

    if (!PyArg_ParseTuple(args, "O!n", &PyCapsule_Type, &capsObj,
			  &proof_step_idx)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(proof_step_idx >= 0 && static_cast<size_t>(proof_step_idx) <
	  p_data->flatten_proof_steps_index.size())) {
	auto msg = ("error: proof_step_idx " + std::to_string(proof_step_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }

    auto [i0, i1] = p_data->flatten_proof_steps_index[proof_step_idx];

    const c_ProofStep & proof_step = p_data->proof_steps[i0][i1];

    return Py_BuildValue(
	"KKsIKNNsssI",
	uint64_of_node(proof_step.root),
	proof_step.def_hash,
	p_data->def_table.hash_to_name.at(proof_step.def_hash).c_str(),
	proof_step.step_idx,
	proof_step.tactic_hash,
	numpy_ndarray1d_uint64_of_node(proof_step.tactic_args),
	numpy_ndarray1d_uint64_of_node(proof_step.context),
	proof_step.state_text.c_str(),
	proof_step.tactic_text.c_str(),
	proof_step.tactic_base_text.c_str(),
	proof_step.root.file_idx);
}







static PyObject *c_get_step_state_text(PyObject *self, PyObject *args) {
	PyObject *capsObj;
	Py_ssize_t proof_step_idx;

    if (!PyArg_ParseTuple(args, "O!n", &PyCapsule_Type, &capsObj,
			  &proof_step_idx)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(proof_step_idx >= 0 && static_cast<size_t>(proof_step_idx) <
	  p_data->flatten_proof_steps_index.size())) {
	auto msg = ("error: proof_step_idx " + std::to_string(proof_step_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }

    auto [i0, i1] = p_data->flatten_proof_steps_index[proof_step_idx];

    const c_ProofStep &proof_step = p_data->proof_steps[i0][i1];

    return Py_BuildValue("y", proof_step.state_text.c_str());
}



static PyObject *c_get_step_state(PyObject *self, PyObject *args) {
	PyObject *capsObj;
	Py_ssize_t proof_step_idx;

    if (!PyArg_ParseTuple(args, "O!n", &PyCapsule_Type, &capsObj,  &proof_step_idx)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(proof_step_idx >= 0 && static_cast<size_t>(proof_step_idx) <
	  p_data->flatten_proof_steps_index.size())) {
	auto msg = ("error: proof_step_idx " + std::to_string(proof_step_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }

    auto [i0, i1] = p_data->flatten_proof_steps_index[proof_step_idx];

    const c_ProofStep& proof_step = p_data->proof_steps[i0][i1];

    const auto root = proof_step.root;
    const auto &context = proof_step.context;
    if (p_data->subgraphs.find(root) == p_data->subgraphs.end()) {
	PyErr_SetString(PyExc_RuntimeError, "report bug in the loader: subgraph root not found");
	return NULL;
    }
    const auto &subgraph = p_data->subgraphs.at(root);

    return get_step_state(root, context, subgraph, p_data->def_table.hash_to_node_label, p_data->eval_label_to_train_label);
}


static PyObject * c_get_node_label_subgraph(PyObject *self, PyObject *args) {
	PyObject *capsObj;
	uint32_t node_label_idx;


    if (!PyArg_ParseTuple(args, "O!I", &PyCapsule_Type, &capsObj,
			  &node_label_idx)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    const c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(static_cast<size_t>(node_label_idx) <
	  p_data->def_table.node_label_to_hash.size())) {
	auto msg = ("error: node_label_idx " + std::to_string(node_label_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }


    c_NodeHash hash = p_data->def_table.node_label_to_hash[node_label_idx];

    c_GlobalNode node = p_data->def_table.hash_to_node.at(hash);

    auto const & subgraph = p_data->subgraphs.at(node);

    auto labels_over_hashes =  get_labels_over_hashes(p_data->def_table.hash_to_node_label, subgraph.node_labels, p_data->eval_label_to_train_label);

    return Py_BuildValue("NNNN",
			 numpy_ndarray1d(labels_over_hashes),
			 numpy_ndarray2d(subgraph.edges_split_flat),
			 numpy_ndarray1d(subgraph.edges_label_flat),
			 numpy_ndarray1d(subgraph.edge_label_offsets));
}




static PyObject * c_build_def_clusters(PyObject * self, PyObject *args) {
    PyObject *capsObj;
    int bfs_option;
    unsigned long long max_subgraph_size;
    log_info("entering build_def_clusters");
    if (!PyArg_ParseTuple(args, "O!pK", &PyCapsule_Type, &capsObj,
			  &bfs_option,
			  &max_subgraph_size)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data * const p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));


    std::map<c_NodeHash, std::set<c_NodeHash>> fwd_links;
    log_info("fwd_links started");


    // DEPRECATE TEMPORARY LOGS
    //std::ofstream log_out;
    //log_out.open("dep_log.txt");
    for (const auto &[hash, root]: p_data->def_table.hash_to_node) {
	std::vector<std::string> temp_log;
	//temp_log.clear();
	fwd_links[hash] = p_data->subgraphs.at(root).depends;
	std::string line = "#" + p_data->def_table.hash_to_name.at(hash);
	//for (const auto &e: fwd_links[hash]) {
	//    temp_log.emplace_back(p_data->def_table.hash_to_name.at(e));
	//}
	//std::sort(temp_log.begin(), temp_log.end());
	//for (const auto &e: temp_log) {
	//    line += (" " + e);
	//}
	//log_out << line << std::endl;
    }
    //log_out.close();


    log_info("fwd_links finished");


    SimpleDirectedGraph<c_NodeHash> def_dependency_graph {fwd_links};
    p_data->hash_to_cluster_idx = def_dependency_graph.find_strongly_connected_componets();

    log_info("strongly connected components finished");

    for (const auto& [hash, cluster_idx]: p_data->hash_to_cluster_idx) {
	p_data->cluster_idx_to_hashset[cluster_idx].insert(hash);
    }

    log_info("hash to cluster idx finished");

    {
	size_t counter = 0;
	Profiler p("pass cluster definition graphs");
	for (const auto & [cluster_idx, hash_cluster]: p_data->cluster_idx_to_hashset) {
	    if (hash_cluster.size() == 1) {
		p_data->cluster_subgraphs[cluster_idx] = p_data->subgraphs.at(
		    p_data->def_table.hash_to_node.at(*hash_cluster.begin()));
	    } else {
		std::vector<c_GlobalNode> roots;
		for (const auto & hash: hash_cluster) {
		    roots.emplace_back(p_data->def_table.hash_to_node.at(hash));
		}
		p_data->cluster_subgraphs[cluster_idx] = build_subgraph(
		    p_data, roots, bfs_option, max_subgraph_size);
		counter += hash_cluster.size() - 1;
	    }
	}
	log_info("merged " + std::to_string(counter) +
	      " shallow subgraphs to build strongly component definition subgraphs");
    }


    std::vector<PyObject*> vect_of_clusters_np;
    for (const auto& [cluster_idx, hash_cluster]: p_data->cluster_idx_to_hashset) {
	vect_of_clusters_np.emplace_back(
	    numpy_ndarray1d(
		std::vector<c_NodeHash> (hash_cluster.begin(),
					 hash_cluster.end())));
    }


    std::vector<c_NodeHash> keys;
    std::vector<size_t> values;

    for (const auto& [node_hash, cluster_idx]: p_data->hash_to_cluster_idx) {
	keys.emplace_back(node_hash);
	values.emplace_back(cluster_idx);
    }

    return Py_BuildValue("NNN",
			 numpy_ndarray1d(keys),
			 numpy_ndarray1d(values),
			 PyList_FromVector(vect_of_clusters_np));
}



static PyObject * c_get_def_cluster_subgraph(PyObject *self, PyObject *args) {
	PyObject *capsObj;
	uint32_t cluster_idx;


    if (!PyArg_ParseTuple(args, "O!I", &PyCapsule_Type, &capsObj,
			  &cluster_idx)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    const c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(static_cast<size_t>(cluster_idx) <
	  p_data->cluster_subgraphs.size())) {
	auto msg = ("error: cluster_idx " + std::to_string(cluster_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }

    auto const & subgraph = p_data->cluster_subgraphs.at(cluster_idx);


    auto labels_over_hashes = get_labels_over_hashes(p_data->def_table.hash_to_node_label, subgraph.node_labels, p_data->eval_label_to_train_label);


    return Py_BuildValue("NNNNK",
			 numpy_ndarray1d(labels_over_hashes),
			 numpy_ndarray2d(subgraph.edges_split_flat),
			 numpy_ndarray1d(subgraph.edges_label_flat),
			 numpy_ndarray1d(subgraph.edge_label_offsets),
			 static_cast<uint64_t>(subgraph.number_of_roots));
}


static PyObject *c_get_step_label_text(PyObject *self, PyObject *args) {
	PyObject *capsObj;
	Py_ssize_t proof_step_idx;

    if (!PyArg_ParseTuple(args, "O!n", &PyCapsule_Type, &capsObj,
			  &proof_step_idx)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(proof_step_idx >= 0 && static_cast<size_t>(proof_step_idx) <
	  p_data->flatten_proof_steps_index.size())) {
	auto msg = ("error: proof_step_idx " + std::to_string(proof_step_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }

    auto [i0, i1] = p_data->flatten_proof_steps_index[proof_step_idx];
    const c_ProofStep &proof_step = p_data->proof_steps[i0][i1];

    return Py_BuildValue("yyy", proof_step.tactic_base_text.c_str(),
			 proof_step.tactic_interm_text.c_str(),
			 proof_step.tactic_text.c_str());
}





static PyObject *c_get_step_label(PyObject *self, PyObject *args) {
	PyObject *capsObj;
	Py_ssize_t proof_step_idx;
	int global_args {false};

    if (!PyArg_ParseTuple(args, "O!np", &PyCapsule_Type, &capsObj,
			  &proof_step_idx, &global_args)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(proof_step_idx >= 0 && static_cast<size_t>(proof_step_idx) <
	  p_data->flatten_proof_steps_index.size())) {
	auto msg = ("error: proof_step_idx " + std::to_string(proof_step_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }


    auto [i0, i1] = p_data->flatten_proof_steps_index[proof_step_idx];
    const c_ProofStep& proof_step = p_data->proof_steps[i0][i1];
    const uint32_t tactic_index = p_data->tactics.hash_to_index.at(proof_step.tactic_hash);


    std::vector<bool>        arg_mask(p_data->tactics.max_tactic_args_size, false);
    std::fill(arg_mask.begin(), arg_mask.begin() + proof_step.tactic_args.size(), true);

    if (!global_args) {
	std::vector<c_NodeIndex> arg_context_index(p_data->tactics.max_tactic_args_size, proof_step.context.size());
	std::transform(proof_step.tactic_args.begin(),
		       proof_step.tactic_args.end(),
		       arg_context_index.begin(),
		       [&proof_step](c_GlobalNode tactic_node) {
			   return (std::find(proof_step.context.begin(),
					     proof_step.context.end(), tactic_node) - proof_step.context.begin());
		       });

	return Py_BuildValue("INN",
			 tactic_index,
			 numpy_ndarray1d(arg_context_index),
			 numpy_ndarray1d(arg_mask));
    } else {
	std::vector<std::array<c_NodeIndex,2>> arg_context_index(p_data->tactics.max_tactic_args_size,
								 {0, static_cast<c_NodeIndex>(proof_step.context.size())});

	std::transform(proof_step.tactic_args.begin(),
		       proof_step.tactic_args.end(),
		       arg_context_index.begin(),
		       [&proof_step, &p_data](c_GlobalNode tactic_node) {
			   size_t local_context_idx = std::find(proof_step.context.begin(),
								proof_step.context.end(), tactic_node) - proof_step.context.begin();
			   if (local_context_idx < proof_step.context.size()) {
			       return std::array<c_NodeIndex, 2> {0, static_cast<c_NodeIndex>(local_context_idx)};
			   }
			   if (p_data->def_table.node_to_hash.find(tactic_node) != p_data->def_table.node_to_hash.end()) {
			       c_NodeHash tactic_node_hash = p_data->def_table.node_to_hash.at(tactic_node);
			       size_t global_context_idx = (std::find(p_data->def_table.available_def_hashes.begin(),
								      p_data->def_table.available_def_hashes.end(), tactic_node_hash) -
							    p_data->def_table.available_def_hashes.begin());
			       if (global_context_idx < p_data->def_table.available_def_hashes.size()) {
				   return std::array<c_NodeIndex, 2> {1, static_cast<c_NodeIndex>(global_context_idx)};
			       }
			   }
			   return std::array<c_NodeIndex, 2> {0, static_cast<c_NodeIndex>(proof_step.context.size())};
		       }
	    );

	return Py_BuildValue("INN",
			 tactic_index,
			 numpy_ndarray2d(arg_context_index),
			 numpy_ndarray1d(arg_mask));

    }
}

static PyObject *c_get_step_hash_and_size(PyObject *self, PyObject *args) {
	PyObject *capsObj;
	Py_ssize_t proof_step_idx;

    if (!PyArg_ParseTuple(args, "O!n", &PyCapsule_Type, &capsObj,
			  &proof_step_idx)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (!(proof_step_idx >= 0 && static_cast<size_t>(proof_step_idx) <
	  p_data->flatten_proof_steps_index.size())) {
	auto msg = ("error: proof_step_idx " + std::to_string(proof_step_idx) +
		    " is out of boundary");
	PyErr_SetString(PyExc_IndexError, msg.c_str());
	return NULL;
    }

    auto [i0, i1] = p_data->flatten_proof_steps_index[proof_step_idx];
    const c_ProofStep& proof_step = p_data->proof_steps[i0][i1];

    return Py_BuildValue("KK", proof_step.def_hash_for_split, p_data->subgraphs[proof_step.root].node_labels.size());

}



static PyObject *c_get_node_label_to_hash(PyObject *self, PyObject *args) {
    PyObject *capsObj;

    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &capsObj)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    return Py_BuildValue("N", numpy_ndarray1d(p_data->def_table.node_label_to_hash));
}


static PyObject *c_get_hash_to_name(PyObject *self, PyObject *args) {

    PyObject *capsObj;
    c_NodeHash hash;

    if (!PyArg_ParseTuple(args, "O!K", &PyCapsule_Type, &capsObj, &hash)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    if (p_data->def_table.hash_to_name.find(hash) != p_data->def_table.hash_to_name.end()) {
	return Py_BuildValue("s", p_data->def_table.hash_to_name[hash].c_str());
    } else {
	auto msg = "error: name of hash not found ";
	PyErr_SetString(PyExc_IndexError, msg);
	return NULL;
    }
}



static PyObject *c_get_files_num_nodes(PyObject *self, PyObject *args) {

    PyObject *capsObj;
    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &capsObj)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    std::vector<uint64_t> c_num_nodes;
    for (const auto &f_graph : p_data->graph) {
	c_num_nodes.push_back(f_graph.nodes.size());
    }
    return Py_BuildValue("N", numpy_ndarray1d(c_num_nodes));
}


static PyObject *c_get_files_num_edges(PyObject *self, PyObject *args) {

    PyObject *capsObj;
    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &capsObj)) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    std::vector<uint64_t> c_num_nodes;
    for (const auto &f_graph : p_data->graph) {
	size_t res = 0;
	for (auto l: f_graph.links) {
	    res += l.size();
	}
	c_num_nodes.push_back(res);
    }
    return Py_BuildValue("N", numpy_ndarray1d(c_num_nodes));
}


static PyObject * c_get_global_context(PyObject *self, PyObject *args) {
    PyObject *capsObj;

    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &capsObj )) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));


    return Py_BuildValue("N", numpy_ndarray1d(p_data->global_context));
}


static PyObject *c_get_graph_constants_online(PyObject *self, PyObject *args) {

    if (!PyArg_ParseTuple(args, "")) {
	return NULL;
    }

    const auto edge_conflate_map = build_edge_conflate_map();
    const size_t edge_conflate_map_size = edge_conflate_map.size;

    auto basic_classes = capnp::Schema::from<Graph::Node::Label>().getFields();
    std::vector<std::string> basic_classes_names;
    for (const auto& basic_class: basic_classes) {
	basic_classes_names.emplace_back(basic_class.getProto().getName());
    }

    return Py_BuildValue("NK", PyListBytes_FromVectorString(basic_classes_names), edge_conflate_map_size);
}



static PyObject *c_get_graph_constants(PyObject *self, PyObject *args) {
    PyObject *capsObj;

    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &capsObj )) {
	return NULL;
    }

    if (!PyCapsule_IsValid(capsObj, __data_capsule_name)) {
	auto msg = "error: expected " + std::string(__data_capsule_name);
	PyErr_SetString(PyExc_TypeError, msg.c_str());
	return NULL;
    }

    c_Data *p_data =
	static_cast<c_Data *>(PyCapsule_GetPointer(capsObj, __data_capsule_name));

    std::vector<c_NodeIndex> tactic_index_to_numargs;
    for (const auto & [tactic_hash, tactic_idx]: p_data->tactics.hash_to_index) {
	tactic_index_to_numargs.emplace_back(
	    static_cast<c_NodeIndex>(p_data->tactics.hash_to_numargs[tactic_hash]));
    }

    std::vector<std::string> tactic_index_to_string;
    for (const auto& tactic_idx: p_data->tactics.index_to_hash) {
	tactic_index_to_string.emplace_back(p_data->tactics.hash_to_string.at(tactic_idx));
    }


    const size_t label_encode_table_size = p_data->def_table.hash_to_node_label.size();

    std::vector<std::string> node_label_to_name;

    for (const auto hash: p_data->def_table.node_label_to_hash) {
	    node_label_to_name.emplace_back(p_data->def_table.hash_to_name.at(hash));
    }

    return Py_BuildValue(
	"NNNNNNNNNNN",
	PyLong_FromSize_t(p_data->tactics.index_to_hash.size()),
	// PyLong_FromSize_t(p_data->tactics.max_tactic_args_size),   #DEPRECATE
	PyLong_FromSize_t(p_data->conflate.size),
	PyLong_FromSize_t(static_cast<size_t>(base_node_label_num)),
	PyLong_FromSize_t(label_encode_table_size),
	PyLong_FromSize_t(p_data->cluster_subgraphs.size()),
	numpy_ndarray1d(tactic_index_to_numargs),
	PyListBytes_FromVectorString(tactic_index_to_string),
	numpy_ndarray1d(p_data->tactics.index_to_hash),
	numpy_ndarray1d(p_data->def_table.available_def_classes),
	numpy_ndarray1d(p_data->def_table.node_label_to_hash),
	PyListBytes_FromVectorString(node_label_to_name));
}



static PyObject *c_test2d(PyObject *self, PyObject *args) {

    uint64_t dim_0;
    constexpr uint64_t dim_1 = 3;
    if (!PyArg_ParseTuple(args, "K", &dim_0)) {
	return NULL;
    }

    std::vector<std::array<uint32_t, dim_1>> test(dim_0, std::array<uint32_t, dim_1> {});
    for (uint64_t i = 0; i < dim_0; ++i) {
	for (uint64_t j = 0; j < dim_1; ++j) {
	    test[i][j] = dim_0 * i + j;
	}
    }
    return numpy_ndarray2d(test);
}


/*
typedef enum
{
    ACCESS_DEFAULT,
    ACCESS_READ,
    ACCESS_WRITE,
    ACCESS_COPY
} access_mode;


typedef struct {
    PyObject_HEAD
    char *      data;
    Py_ssize_t  size;
    Py_ssize_t  pos;
    off_t       offset;
    Py_ssize_t  exports;
    int fd;
    PyObject *weakreflist;
    access_mode access;
} mmap_object;



static PyObject *c_test_mmap(PyObject *self, PyObject *args) {
    PyObject * bytesObj;
    if (!PyArg_ParseTuple(args, "O", &bytesObj)) {
	return NULL;
    }
    else {
	mmap_object* res = (mmap_object*)(bytesObj);
	std::cout << res->data << std::endl;
	return Py_BuildValue("k", res->size);
    }
}
*/


static PyObject* parse_memory_view(PyObject *self, PyObject *args) {
    PyMemoryViewObject * mem_view;
    if (!PyArg_ParseTuple(args, "O!", &PyMemoryView_Type, &mem_view)) {
	return NULL;
    }
    auto buf = (char*) mem_view->view.buf;
    auto length = mem_view->view.len;
    std::cout << "hello I see buffer[" << buf << "]" << std::endl;
    std::cout << "of length" << length << std::endl;

    return Py_BuildValue("kk", mem_view->exports, mem_view->ob_array[0]);
}



static PyMethodDef GraphMethods[] = {

    /* python function name, c function name, _ , doc string" */
    {"parse_memory_view", parse_memory_view, METH_VARARGS,
     "memoryview -> int"},
    {"get_data", c_get_data, METH_VARARGS,
     "data_dir: bytes, proj_dir: bytes, fnames: list[bytes], num_proc --> data_capsule"},
    {"get_file_dep", c_get_file_dep, METH_VARARGS,
     "fname -> file deps"},
    {"get_buf_def", c_get_buf_def,  METH_VARARGS,
     "buf_object: buffer, restrict_to_spine: bool = False, as_request: bool = False -> def table"},
    {"get_buf_tactics", c_get_buf_tactics, METH_VARARGS,
     "fname, numpy 1d uint64 array of def nodes -> file tactics  numpy_ndarray1d(tactic_hashes),"
			     "numpy_ndarray1d(number_arguments), "
			     "numpy_ndarray1d(def_indexes),"
			     "numpy_ndarray1d(proof_step_indexes),"
			     "numpy_ndarray1d(outcome_indexes)"
    },
    {"get_local_to_global_file_idx", c_get_local_to_global_file_idx, METH_VARARGS,
     "data_dir, fnames -> list of numpy arrays of local to global file idx map"},
    {"get_scc_components", c_get_scc_components, METH_VARARGS,
     "adj_list: list[numpy1darray] -> components list: list[numpy1darray]"},
    {"files_get_scc_components", c_files_scc_components, METH_VARARGS,
     "data_dir: bytes, fnames: list[bytes] -> list of ssc top sorted labels"},
    {"build_data_online_from_buf", c_build_data_online_from_buf, METH_VARARGS,
     "fnames, def_idx_node -> c_capsule"},
    {"data_online_resize", c_data_online_resize, METH_VARARGS,
     "c_capsule, new_size -> new_size"},
    {"data_online_extend", c_data_online_extend, METH_VARARGS,
     "c_capuse, buffers, names, local_to_global_file  -> new_size"},
    {"get_def_deps_online", c_get_def_deps_online, METH_VARARGS,
     "c_data_online, bfs_option, max_subgraph_size: def_deps table "},
    {"get_subgraph_online", c_get_subgraph_online, METH_VARARGS,
     "c_data_online, roots, bfs_option, max_subgraph_size: numpy subgraph"},
    {"get_proof_step_online", c_get_proof_step_online, METH_VARARGS,
     "c_data_online, [file_idx, def_local_node, step_idx, outcome_idx]: numpy1d, glob_to_sub capsule -> proof state"},
    {"get_proof_step_online_text", c_get_proof_step_online_text, METH_VARARGS,
     "c_data_online, [file_idx, def_local_node, step_idx, outcome_idx]: numpy1d -> action_base_text, action_interm_text, action_text"},
    {"load_init_msg", c_load_init_msg, METH_VARARGS,
     "fname: bytes -> cdata"},
    {"capnp_unpack", c_capnp_unpack, METH_VARARGS,
     "fname: bytes -> void; unpacks capnp file"},
    {"check", c_check, METH_VARARGS,
     "fname: bytes -> void; unpacks capnp file"},
    {"load_msg", c_load_msg, METH_VARARGS,
     "fname: bytes, cdata -> void"},
    {"load_msg_online", c_load_msg_online, METH_VARARGS,
     "msg_data: buffer, glob_to_sub: capsule, file_idx: integer -> encoded_context, encoded_root"},
    {"encode_prediction", c_encode_prediction, METH_VARARGS,
     "prediction -> encoded_prediction"},
    {"encode_prediction_online", c_encode_prediction_online, METH_VARARGS,
     "capsule_data_online, predictions: list[nparray of shape (1+n_args, 2)], context: nparray uint32, fd -> encodes and writes prediction to fd"},
    {"get_files_num_nodes", c_get_files_num_nodes, METH_VARARGS,
     "data_capsule -> num_nodes: np.array shape=(num_files), dtype=np.uint64"},
    {"get_files_num_edges", c_get_files_num_edges, METH_VARARGS,
     "data_capsule -> num_edges: np.array shape=(num_files), dtype=np.uint64"},
    {"get_proof_steps_size", c_get_proof_steps_size, METH_VARARGS,
     "data_capsule -> total number of proof steps"},
    {"get_proof_step", c_get_proof_step, METH_VARARGS,
     "data_capsule, proof_step_idx -> a tuple of proof_step (root, def_hash, "
     "step_idx, "
     "tactic_hash, tactic_args, context, state_text, tactic_text, "
     "tactic_base_text"},
    {"get_hash_to_name", c_get_hash_to_name, METH_VARARGS,
     "data_capsule, hash of definition -> name of definition"},
    {"get_node_label_to_hash", c_get_node_label_to_hash, METH_VARARGS,
     "data_capsule,  vector of hashes of definiton"},
    {"get_subgraph", c_get_subgraph, METH_VARARGS,
     "data_capsule, root: c_GlobalNode as pyint, bfs_option -> subgraph: (nodes, labels, edges3, edges_flat, edges_offset)"},
    {"build_subgraphs", c_build_subgraphs, METH_VARARGS,
     "data_capsule, bfs_option -> total size of all proof_step_subgraphs"},
    {"get_step_state", c_get_step_state, METH_VARARGS,
     "data_capsule, step_idx -> (labels, edges_flat, edges_offset, root, context)"},
    {"get_step_state_text", c_get_step_state_text, METH_VARARGS,
     "data_capsule, step_idx -> state_text"},
    {"get_node_label_subgraph", c_get_node_label_subgraph, METH_VARARGS,
     "data_capsule -> (labels, edges_flat, edges_label_flat, edges_offset)"},
    {"get_def_cluster_subgraph", c_get_def_cluster_subgraph, METH_VARARGS,
     "data_capsule -> (labels, edges_flat, edges_label_flat, edges_offset, number_of_roots)"},
    {"get_step_label", c_get_step_label, METH_VARARGS,
     "data_capsule -> tactic index, tactic args, mask (args is index in a local context or one element beyond)"},
    {"get_step_label_text", c_get_step_label_text, METH_VARARGS,
     "data_capsule, step_idx -> tactic_base_text, tactic_interm_text, tactic_text"},
    {"get_step_hash_and_size", c_get_step_hash_and_size, METH_VARARGS,
     "data_capsule -> step_hash (from defid), step_nodes_size"},
    {"get_graph_constants", c_get_graph_constants, METH_VARARGS,
     "data_capsule -> step_hash (from defid), step_nodes_size, ..., cluster_subgraphs_size, available_def_classes"},
    {"get_graph_constants_online", c_get_graph_constants_online, METH_VARARGS,
     "c_data_online -> base_node_label_num, edge_conflate_map.size()"},
    {"build_def_clusters", c_build_def_clusters, METH_VARARGS,
     "data_capsule -> nparray def_idx, comp_idx"},
    {"get_global_context", c_get_global_context, METH_VARARGS,
     "data_capsule -> visible context def class"},
    {"test2d", c_test2d, METH_VARARGS,
     "data_capsule, python int as uint64 node -> np.array shape=(num_nodes) the list of nodes"},

    {NULL, NULL, 0, NULL}};

static struct PyModuleDef graphmodule = {

    PyModuleDef_HEAD_INIT, "graph", /* name of module */
    "graph is C-extension module implementing fast loader of graphs .bin files",
    -1, /*  size of per-interpreter state of the module of -1 if the module
	    keeps state in global variables */
    GraphMethods};


/* -----------------GLOBAL INITIALISATION-------------------------- */
/* ---------------------------------------------------------------- */


bool test() {

    if (!test_uint64_of_node()) {
	log_critical("uint64 test conversion failed");
	return false;
    }
    return true;
}

PyMODINIT_FUNC PyInit_loader(void) {

    import_array();

    if (!test()) {
	PyErr_SetString(
	    PyExc_AssertionError,
	    "c-extension unit tests failed, report the bug and the conditions to reproduce");
	return NULL;
    }

    return PyModule_Create(&graphmodule);
}
