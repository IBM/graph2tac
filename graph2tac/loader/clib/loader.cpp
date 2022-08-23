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
	throw std::runtime_error("error opening file " + fname);
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
	ptr = (char *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, filed, 0);

	if (ptr == MAP_FAILED) {
	    log_critical("mapping of " + fname + " of size " + std::to_string(file_size) + " failed");
	    throw std::runtime_error("mapping of " + fname + " of size " + std::to_string(file_size) + " failed");
	}
	auto array_ptr = kj::ArrayPtr<const capnp::word>( reinterpret_cast<capnp::word*>(ptr),
							  file_size / sizeof(capnp::word));
	msg = new capnp::FlatArrayMessageReader(array_ptr,  capnp::ReaderOptions{SIZE_MAX});
    }
    ~MMapFileMessageReader() {
	auto err = munmap((void *) ptr, file_size);
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
    /* this function is convenient to split
     a list of files into num_proc buckets
     by sorting file sizes in decreasing order
     and then placing next file to the minimum total
     size bucket

     currently is not used but may be suitable if go on multithreaded mode
    */

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




class FileTable {
    /*
      This class handles indexes a list of filenames and provides
      forward and reverse map
     */
    const std::vector<std::string> _fnames;
    const std::string data_dir;
    std::unordered_map<std::string, c_FileIndex> i_fnames;

private:
    std::string convert(const std::string dep_fname) const {
	// take care of absoluted dependencies
	std::string result;
	if (dep_fname.size() > 0 && dep_fname[0] == '/') {
	    result = dep_fname;
	} else {
	    result = data_dir + "/" + dep_fname;
	}
	log_debug("original dep name " + dep_fname + " converted to " + result);
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
	    log_critical("error indexing file list: duplicate names detected");
	    throw std::runtime_error("error indexing file list: duplicate names detected");
	}
    }
    c_FileIndex index(const std::string fname) const {
	/*
	  maps fname to index
	 */
	auto converted_name = FileTable::convert(fname);
	if (i_fnames.find(converted_name) == i_fnames.end()) {
	    log_critical("WARNING! " + converted_name + " not found");
	    return __MISSING_FILE;
	} else {
	    return i_fnames.at(FileTable::convert(fname));
	}
    }

    std::string fname(const c_FileIndex file_idx) const {
	/*
	  maps index to fname
	 */
	return _fnames.at(file_idx);
    }
    std::vector<std::string> fnames() const {
	return _fnames;
    }
};


/* --------------------------- READING SCHEMA--------------------------- */
/* --------------------------------------------------------------------- */




const c_ConflateLabels build_edge_conflate_map() {
    // conflation map: Edge Classifications  --> Edge Labels

    const auto edge_classes = capnp::Schema::from<EdgeClassification>();
    const size_t edge_classes_size = edge_classes.getEnumerants().size();
    std::vector<size_t> conflate_map(edge_classes_size, static_cast<size_t>(-1));

    size_t edge_label = 0;

    for (const auto& e: *CONFLATABLE_EDGES) {
	const auto group = e.getConflatable();
	for (const auto &x : group) {
	  conflate_map[static_cast<size_t>(x)] = edge_label;
	}
	++edge_label;
    }

    for (size_t edge_class = 0; edge_class < edge_classes_size; ++edge_class) {
	if (conflate_map[edge_class] == static_cast<size_t>(-1)) {
	conflate_map[edge_class] = edge_label++;
      }
    }
    return c_ConflateLabels {conflate_map, edge_label};
}



/* --------------------------- READING CAPNP --------------------------- */
/* --------------------------------------------------------------------- */


std::vector<c_FileIndex> build_dependencies(
    const ::capnp::List<::capnp::Text, ::capnp::Kind::BLOB>::Reader  *dependencies,
    const FileTable & file_table) {
    /*
      dep_local_to_global[file_i] is the vector of absolute file indices referenced by file_i
     */

    std::vector<c_FileIndex> dep_local_to_global;
    for (auto const &fname: *dependencies) {
	dep_local_to_global.emplace_back(file_table.index(fname));
    }
    return dep_local_to_global;
}


/*
std::vector<c_GlobalNode> build_context(
    c_FileIndex file_idx,
    const ::capnp::List<::uint32_t, ::capnp::Kind::PRIMITIVE>::Reader *context_obj) {
    //
    //  reads capnp context and returns it as STL vector
    //
    std::vector<c_GlobalNode> context;
    for (const auto &ctxt_node_idx : *context_obj) {
	context.emplace_back(c_GlobalNode{file_idx, ctxt_node_idx});
    }
    return context;
}
*/



std::vector<c_GlobalNode> build_tactic_args(
    const ::capnp::List<Argument, capnp::Kind::STRUCT>::Reader tactic_args_obj,
    const std::vector<c_FileIndex> &dep_local_to_global) {
    /*
      reads capnp Argument and returns vector of GlobalNode as arguments
      use sentinels for not found nodes
     */
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
    /*
      the core sampling function from the collection of graphs
      represented by vector of capnp global node graph readers
      with pre-dereferenced graph.nodes readers and graph.edges readers (speedup!)

      roots is the list of entry points to instantiate forward closure

      stop_edge_label specifies the label of edge that is not allowed to travel
      (could generalize) to a set of such edge labels

      bfs is the boolean flag for BFS expansion if true otherwise DFS expansion

      rel_file_idx is the adjacency list of file dependencies as file indices

      def_node_to_idx is an index of all special nodes on which to stop expansion
      this set of special nodes is currently the set of definition nodes
      but we can change that

    */
{
    // this will contains the list of nodes on which expansion will be stopped
    std::vector<c_GlobalNode> shallow_deps;

    // deque to server bfs or dfs expansion
    std::deque<c_GlobalNode> q;

    // to record the order in which nodes have been visited maintain the
    // hash map from visited node to its time of visit
    auto visited = new std::unordered_map<c_GlobalNode, size_t>;

    size_t visit_time = 0;

    for (const auto& root: roots) {
	(*visited)[root] = visit_time++;
	q.push_back(root);
    }

    // standard bfs/dfs expansion using the hash map visited
    // do not travel on stop_edge_label
    // do not expand past the nodes from the list of def_node_to_idx (but include them to visited and to shallow deps)
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
		    (*visited)[other] = visit_time++;
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


std::tuple< std::vector<c_NodeIndex>, std::vector<c_EdgeLabel>,
	    std::vector<std::array<c_NodeIndex, 2>>>
compute_offsets(const std::vector<std::vector<std::array<c_NodeIndex, 2>>>&
		edges_split_by_label)
/*
  this is a helper function to sort edges into buckets by their labels,
  return concatenated (flattened) list of edges, and return offsets where labels start
*/
{
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



void send_response(const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& predictions,
		   const std::vector<double>& confidences,
		   const int fd)
/*
   builds capnp message from list of predictions and list of confidences and sends it to file descriptor fd
*/

{
    ::capnp::MallocMessageBuilder message;
    auto response = message.initRoot<PredictionProtocol::Response>();
    auto capnp_prediction = response.initPrediction(predictions.size());
    for (size_t pred_idx = 0; pred_idx < predictions.size(); ++pred_idx) {
	const auto &prediction = predictions.at(pred_idx);
	auto tactic = capnp_prediction[pred_idx].initTactic();
	capnp_prediction[pred_idx].setConfidence(confidences.at(pred_idx));
	// tactic
	if (prediction.size() > 0) {
	    uint32_t hi = prediction.at(0).first;
	    uint32_t lo = prediction.at(0).second;
	    tactic.setIdent(uint64_t(hi) << 32 | uint64_t(lo));
	} else {
	    log_critical("error:  prediction has zero size, but the hash of tactic hash must be present ");
	    throw std::runtime_error("error:  prediction has zero size, but the hash of tactic hash must be present ");
	}
	// arguments
	if (prediction.size() > 1) {
	     auto arguments = capnp_prediction[pred_idx].initArguments(prediction.size() - 1);
	    for (size_t arg_idx = 1; arg_idx < prediction.size(); ++arg_idx) {
		auto term = arguments[arg_idx - 1].initTerm();
		term.setDepIndex(prediction.at(arg_idx).first);
		term.setNodeIndex(prediction.at(arg_idx).second);
	    }
	}
    }
    writePackedMessageToFd(fd, message);
}


static PyObject * c_encode_prediction_online(PyObject *self, PyObject *args) {
    PyObject * caps_data_online;
    PyListObject * pylist_actions;
    PyArrayObject * pynp_confidences;
    PyArrayObject * pynp_local_context;
    int fd;
    PyObject * py_defnames;
    /*
      pylist_actions: list of predictions

      each prediction has format of numpy vector of shape=(n,2), of dtype=np.uint32 with the content like

      [(tac_hash, tac_hash), (1, global_arg), (0,local_arg), (1, global_arg)

      where local_arg is index into the local context and global_arg is index into the global context
    */


    if (!PyArg_ParseTuple(args, "O!O!O!O!iO!",
			  &PyCapsule_Type, &caps_data_online,
			  &PyList_Type, &pylist_actions,
			  &PyArray_Type, &pynp_confidences,
			  &PyArray_Type, &pynp_local_context,
			  &fd,
			  &PyList_Type, &py_defnames)) {
	return NULL;
    }
    try {
	c_DataOnline * p_data_online = static_cast<c_DataOnline*>(PyCapsule_GetPointer(caps_data_online, __data_online_capsule_name));
	std::vector<std::vector<std::pair<uint32_t, uint32_t>>> predictions = PyList2dArray_AsVectorVector_uint32_pair((PyObject*)pylist_actions);
	std::vector<double> confidences = PyArray_AsVector_double(pynp_confidences);
	std::vector<c_NodeIndex> local_context = PyArray_AsVector_uint32_t(pynp_local_context);
	std::vector<std::string> def_names = PyListBytes_AsVectorString(py_defnames);
	log_verbose("received predictions of size " + std::to_string(predictions.size()));
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
		    // local arg
		    if (pred_arg.second < local_context.size()) {
			prediction_encoded.emplace_back(
			    std::pair<uint32_t, uint32_t>{0, local_context.at(pred_arg.second)});
		    } else {
			log_critical("error: local context of size " + std::to_string(local_context.size()) + " is: ");
			log_critical(local_context);
			log_critical("network prediction is (0," + std::to_string(pred_arg.second));
			throw std::logic_error("local argument predicted by network is not in range(len(local_context))");
		    }
		} else if (pred_arg.first == 1) {
		    // global arg
		    // we send back the actual global node of the definition in position idx of the context
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
	//
	// TODO: refactor out send_response to be independent function called from Python
	// using the PyList_FromVector(np_predictions_encoded) that we return
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




std::map<c_FileIndex, std::vector<c_FileIndex>> get_local_to_global_file_idx(    const std::string &data_dir,
										 const std::vector<std::string> &fnames) {
    /*
      build the adjacency list of dependencies for the list of files
     */
    std::map<c_FileIndex, std::vector<c_FileIndex>> global_dep_links {};

    FileTable file_table (fnames, data_dir);

    for (size_t file_idx = 0; file_idx < file_table.fnames().size(); ++file_idx) {

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
    return global_dep_links;
}


/*-------------------------PYTHON API ----------------------------------- */


static PyObject *c_get_file_dep(PyObject *self, PyObject *args) {
    PyObject *bytesObj_0;
    if (!PyArg_ParseTuple(args, "S", &bytesObj_0)) {
	return NULL;
    }
    /*
      given a filename in os.fsencode (bytes in python) return the list of filenames referenced by this file
     */
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
    catch (const std::exception &ia) {
	log_critical("exception in loader.cpp c_get_file_dep: " + std::string(ia.what()));
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}


static PyObject *get_def(const Graph::Reader & graph,
			 std::vector<c_NodeIndex> & def_nodes) {
    /*
      return (nodes, hashes, names) for  list of definitions described by local node index in def_nodes
      in a given capnp Graph reader graph
     */

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

static PyObject *get_def_previouses(capnp::FlatArrayMessageReader *msg,
                                 const std::string message_type,
                                 const c_NodeIndex def_node_idx) {
    const auto graph = (message_type == "dataset" ? msg->getRoot<Dataset>().getGraph() : (message_type == "request.initialize" ? msg->getRoot<PredictionProtocol::Request>().getInitialize().getGraph() : msg->getRoot<PredictionProtocol::Request>().getCheckAlignment().getGraph()));
    const auto definitions = (message_type == "dataset" ? msg->getRoot<Dataset>().getDefinitions() : (message_type == "request.initialize" ? msg->getRoot<PredictionProtocol::Request>().getInitialize().getDefinitions() :  msg->getRoot<PredictionProtocol::Request>().getInitialize().getDefinitions()));
    auto nodes = graph.getNodes();
    const auto node = nodes[def_node_idx];
    if (!node.getLabel().hasDefinition()) {
	    throw std::invalid_argument("in get_def_previouses the node " + std::to_string(def_node_idx) +
					"from Dataset.definitions "
					"is not a definition but of type " + std::to_string(node.getLabel().which()));
    }
    const auto def = node.getLabel().getDefinition();
    const auto def_previous = def.getPrevious();
    PyObject * local_previouses;
    PyObject * external_previouses;
    if (def_previous < nodes.size()) {
      local_previouses = numpy_ndarray1d(std::vector<uint32_t> {def_previous});
    } else {
      local_previouses = numpy_ndarray1d(std::vector<uint32_t> {});
    }
    std::vector<uint32_t> v_external_previouses;
    for (const auto& dep: def.getExternalPrevious()) {
      v_external_previouses.emplace_back(dep);
      }
    external_previouses = numpy_ndarray1d(v_external_previouses);
    return Py_BuildValue("NN", local_previouses, external_previouses);
}

static PyObject *get_msg_def(capnp::FlatArrayMessageReader *msg,
			     const std::string message_type, int restrict_to_spine) {
    /*
      this is a wrapper to describe definitions contained in capnp message
      of type Dataset / PredictionProtocol::Request() / PredictionProtocol::checkAlignment
     */

    /* if boolean option restrict_to_spine then do not open sections using the traversing back mechanism
       from the representative index
    */

    std::vector<c_NodeIndex> def_nodes;
    const auto graph = (message_type == "dataset" ? msg->getRoot<Dataset>().getGraph() : (message_type == "request.initialize" ? msg->getRoot<PredictionProtocol::Request>().getInitialize().getGraph() : msg->getRoot<PredictionProtocol::Request>().getCheckAlignment().getGraph()));
    const auto definitions = (message_type == "dataset" ? msg->getRoot<Dataset>().getDefinitions() : (message_type == "request.initialize" ? msg->getRoot<PredictionProtocol::Request>().getInitialize().getDefinitions() :  msg->getRoot<PredictionProtocol::Request>().getInitialize().getDefinitions()));

    if (!restrict_to_spine) {
	for (const auto &node_idx: definitions) {
	    def_nodes.emplace_back(node_idx);
	}
    } else {
	if (message_type != "dataset") {
	    throw std::invalid_argument("restrict to spine is allowed only in dataset message type(otherwise represenative node doesn't exist)");
	}
	auto node_idx = msg->getRoot<Dataset>().getRepresentative();
	auto nodes = graph.getNodes();
	auto nodes_size = nodes.size();
	while (node_idx < nodes_size) {
	    def_nodes.emplace_back(node_idx);
	    auto other_idx = nodes[node_idx].getLabel().getDefinition().getPrevious();
	    node_idx = other_idx;
	}
    }
    c_NodeIndex representative;
    if (message_type == "dataset") {
      representative = msg->getRoot<Dataset>().getRepresentative();
    } else {
      representative = UINT32_MAX;
    }
    return Py_BuildValue("NI", get_def(graph, def_nodes), representative);
}




static PyObject *c_get_buf_def(PyObject *self, PyObject *args) {
    PyObject *buf_object;
    char * c_message_type;
    int restrict_to_spine {false};

    /*
      get definitions from an unpacked capnp message provided in byte buffer (python memory view object)
      the memory view (or mmap) gives us the physical pointer to the buffer and no copy is necessary !
     */

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
    }
    catch (const std::exception &ia) {
	log_critical("exception in loader.cpp c_get_buf_def: " + std::string(ia.what()));
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}

static PyObject *c_get_def_previouses(PyObject *self, PyObject *args) {
  PyObject *buf_object;
  char * c_message_type;
  c_NodeIndex def_node_idx;



  if (!PyArg_ParseTuple(args, "OsI", &buf_object, &c_message_type, &def_node_idx)) {
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
    auto result = get_def_previouses(&msg, message_type, def_node_idx);
    PyBuffer_Release(&view);
    return result;
  }
  catch (const std::exception &ia) {
    log_critical("exception in loader.cpp c_get_def_previouses: " +
                 std::string(ia.what()));
    PyErr_SetString(PyExc_TypeError, ia.what());
    return NULL;
  }
}



static PyObject *c_get_local_to_global_file_idx(PyObject *self, PyObject *args) {
    PyObject *py_datadir_bytes;
    PyObject *py_fnames_list_bytes;

    if (!PyArg_ParseTuple(args, "SO!", &py_datadir_bytes,  &PyList_Type,  &py_fnames_list_bytes)) {
	return NULL;
    }
    try {
	auto data_dir = PyBytes_AsSTDString(py_datadir_bytes);
	auto fnames = PyListBytes_AsVectorString(py_fnames_list_bytes);
	log_info("processing " + std::to_string(fnames.size()) + " files");
	std::map<c_FileIndex, std::vector<c_FileIndex>> global_dep_links = get_local_to_global_file_idx(data_dir, fnames);

	std::vector<PyObject*> temp_result;
	for (const auto & [e,v]: global_dep_links) {
	    temp_result.emplace_back(numpy_ndarray1d(v));
	}

	return PyList_FromVector(temp_result);
    }
    catch (const std::exception &ia) {
	log_critical("exception in loader.cpp c_get_local_to_global_file_idx: " + std::string(ia.what()));
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
}



static PyObject *c_get_scc_components(PyObject *self, PyObject *args) {
    // computes strongly connected components from provided
    // adjacency list representation of directed graph
    // at k-th position this list contains 1d np.array of dtype=np.uint64
    // listing the target node indices from the source k

    // returns a list of cluster of nodes topologically ordered
    // (the nodes with no children are first)
    PyObject *py_adjacency_list;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type,  &py_adjacency_list)) {
	return NULL;
    }
    try {
	auto vectored_links = PyListArray_AsVectorVector_uint64_t(py_adjacency_list);
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

    }
    catch (const std::exception &ia) {
	log_critical("exception in loader.cpp c_get_local_to_global_file_idx: " + std::string(ia.what()));
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
    PyArrayObject * p_PyArray_def_idx_to_node_32x2;
    PyArrayObject * p_PyArray_tactic_hashes;
    PyArrayObject * p_PyArray_tactic_indexes;
    if (!PyArg_ParseTuple(args, "O!O!O!",
			  &PyArray_Type,  &p_PyArray_def_idx_to_node_32x2,
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
	auto def_idx_to_node_uint32x2 = Py2dArray_AsVector_uint32_pair(p_PyArray_def_idx_to_node_32x2);
	for (const auto &x: def_idx_to_node_uint32x2) {
	    c_GlobalNode node {x.first, x.second};
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
    PyArrayObject *p_roots_uint32x2;
    int bfs_option {false};
    uint64_t max_subgraph_size;
    int with_node_counter {false};

    if (!PyArg_ParseTuple(args, "O!O!pKp", &PyCapsule_Type, &capsObj,
			  &PyArray_Type, &p_roots_uint32x2,
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
	for (const auto &e:  Py2dArray_AsVector_uint32_pair(p_roots_uint32x2)) {
	    auto this_node = c_GlobalNode {e.first, e.second};
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

	for (c_NodeIndex source_sub = 0; source_sub < sub_to_glob.size(); ++source_sub) {
	    c_GlobalNode source = sub_to_glob.at(source_sub);
//	for (const auto & [source, source_sub]: (*global_visited)) {
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
	log_critical("exception in load_msg_online");
	log_critical(ia.what());
	PyErr_SetString(PyExc_TypeError, ia.what());
	return NULL;
    }
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
    {"get_file_dep", c_get_file_dep, METH_VARARGS,
     "fname -> file deps"},
    {"get_buf_def", c_get_buf_def,  METH_VARARGS,
     "buf_object: buffer, restrict_to_spine: bool = False, as_request: bool = False -> def table"},
    {"get_def_previouses", c_get_def_previouses,  METH_VARARGS,
     "buf_object: buffer, def_node_idx -> def previouses"},
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
    {"capnp_unpack", c_capnp_unpack, METH_VARARGS,
     "fname: bytes -> void; unpacks capnp file"},
    {"check", c_check, METH_VARARGS,
     "fname: bytes -> void; unpacks capnp file"},
    {"load_msg_online", c_load_msg_online, METH_VARARGS,
     "msg_data: buffer, glob_to_sub: capsule, file_idx: integer -> encoded_context, encoded_root"},
    {"encode_prediction_online", c_encode_prediction_online, METH_VARARGS,
     "capsule_data_online, predictions: list[nparray of shape (1+n_args, 2)], context: nparray uint32, fd -> encodes and writes prediction to fd"},
      {"get_graph_constants_online", c_get_graph_constants_online, METH_VARARGS,
       "c_data_online -> base_node_label_num, edge_conflate_map.size()"},
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


PyMODINIT_FUNC PyInit_loader(void) {

    import_array();

    return PyModule_Create(&graphmodule);
}
