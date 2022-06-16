typedef size_t CompIndex;

template <typename Node>
class SimpleDirectedGraph {
public:
    SimpleDirectedGraph(const std::map<Node, std::set<Node>> &fwd_links);
    SimpleDirectedGraph(const std::map<Node, std::vector<Node>> &fwd_links);
    const std::map<Node, CompIndex> find_strongly_connected_componets();
private:
    std::map<Node, std::set<Node>> fwd_links;
    std::map<Node, std::set<Node>> bwd_links;
    std::vector<Node> post_order;

    void search1(const Node node,
		 const std::map<Node,std::set<Node>>& links,
		 std::map<Node, bool> &visited);
    void dfs1(
	const std::map<Node, std::set<Node>>& links);

    void search2(
	const Node node,
	const std::map<Node, std::set<Node>>& links,
	std::map<Node, CompIndex>& node_to_comp,
	const CompIndex comp_idx);

    std::map<Node, CompIndex>  dfs2(
	const std::map<Node,std::set<Node>>& links);

};

template <typename Node>
SimpleDirectedGraph<Node>::SimpleDirectedGraph(const std::map<Node, std::vector<Node>> &fwd_links_vectorized)  {
    for (const auto& [k, v]: fwd_links_vectorized) {
	fwd_links[k] = std::set<Node>(v.begin(), v.end());
    }
    for (const auto& [node, node_links]: fwd_links) {
	bwd_links[node] = std::set<Node> {};
    }

    for (const auto& [node, node_links]: fwd_links) {
	for (const auto& other_node: node_links) {
	    if (fwd_links.find(other_node) == fwd_links.end()) {
		throw std::invalid_argument(
		    "at node " + std::to_string(node) + ", at other node " + std::to_string(other_node) +
		    " the links in the input to SimpleDirectedGraph contain unknown node ");
	    }
	    bwd_links[other_node].insert(node);
	}
    }
};


template <typename Node>
SimpleDirectedGraph<Node>::SimpleDirectedGraph(const std::map<Node, std::set<Node>> &fwd_links)
    : fwd_links(fwd_links) {

    for (const auto& [node, node_links]: fwd_links) {
	bwd_links[node] = std::set<Node> {};
    }

    for (const auto& [node, node_links]: fwd_links) {
	for (const auto& other_node: node_links) {
	    if (fwd_links.find(other_node) == fwd_links.end()) {
		throw std::invalid_argument(
		    "at node " + std::to_string(node) + ", at other node " + std::to_string(other_node) +
		    " the links in the input to SimpleDirectedGraph contain unknown node ");
	    }
	    bwd_links[other_node].insert(node);
	}
    }
};



template <typename Node>
const std::map<Node, CompIndex> SimpleDirectedGraph<Node>::find_strongly_connected_componets() {
    dfs1(bwd_links);
    auto result = dfs2(fwd_links);
    return result;
};


template <typename Node>
void SimpleDirectedGraph<Node>::dfs1(
    const std::map<Node, std::set<Node>>& links) {
    std::map<Node, bool> visited;
    for (const auto & [node, value]: links) {
	visited[node] = false;
    }
    for (const auto& [node, node_links]: links) {
	if (!visited.at(node)) {
	    search1(node, links, visited);
	}
    }
}


template <typename Node>
void SimpleDirectedGraph<Node>::search1(
	const Node node,
	const std::map<Node, std::set<Node>>& links,
	std::map<Node, bool> & visited) {

	visited[node] = true;
	for (const auto& other_node: links.at(node)) {
	    if (!visited.at(other_node)) {
		search1(other_node, links, visited);
	    }
	}
	post_order.emplace_back(node);
}



template <typename Node>
std::map<Node, CompIndex>SimpleDirectedGraph<Node>::dfs2(
    const std::map<Node,std::set<Node>>& links) {

    std::map<Node, CompIndex> node_to_comp;

    CompIndex comp_idx {0};
    for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
	if (node_to_comp.find(*it)==node_to_comp.end()) {
	    search2(*it, links, node_to_comp, comp_idx);
	    ++comp_idx;
	}
    }
    return node_to_comp;
}

template <typename Node>
void SimpleDirectedGraph<Node>::search2(
    const Node node,
    const std::map<Node, std::set<Node>>& links,
    std::map<Node, CompIndex>& node_to_comp,
    const CompIndex comp_idx) {
    node_to_comp[node] = comp_idx;
    for (const auto& other_node: links.at(node)) {
	if (node_to_comp.find(other_node) == node_to_comp.end()) {
	    search2(other_node, links, node_to_comp, comp_idx);
	}
    }
}






std::vector<uint32_t> top_sort(const std::vector<std::vector<uint32_t>> &dep_list,
			       const std::vector<uint64_t> &node_counts) {
    // priority topological sort of dependency
    // prioritize the smallest possible first
    std::vector<std::set<uint32_t>> dep_set (dep_list.size());

    for (uint32_t i = 0; i < dep_list.size(); ++i) {
	dep_set[i].insert(dep_list[i].begin(), dep_list[i].end());
	if (dep_set[i].size() != dep_list[i].size()) {
	    std::cerr << "WARNING: the node " << i
		      << " had out edges with multiplicity that was discarded"
		      << std::endl;
	}
	dep_set[i].erase(i);
    }

    std::vector<std::vector<uint32_t>> rev_dep_list(dep_list.size(), std::vector<uint32_t> {});
    for (uint32_t i = 0; i < dep_list.size(); ++i) {
	for (uint32_t j : dep_set[i]) {
	    rev_dep_list[j].push_back(i);
	}
    }

    std::priority_queue<std::pair<uint64_t,uint32_t>,
			std::vector<std::pair<uint64_t, uint32_t>>,
			std::greater<std::pair<uint64_t, uint32_t>>> pq;

    std::vector<bool> visited (dep_list.size(), false);

    for (uint32_t i = 0; i < dep_set.size(); ++i) {
	if (dep_set[i].size() == 0) {
	    pq.push(std::make_pair(node_counts[i], i));
	}
    }
    std::vector<uint32_t> result;

    while (pq.size() != 0) {
	auto [n_count, i] = pq.top();
	pq.pop();
	visited[i] = true;
	//std::cerr << i << std::endl;
	result.push_back(i);
	for (auto j: rev_dep_list[i]) {
	    if (dep_set[j].erase(i) != 1) {
		std::cerr << "bug in priority top_sort, forward link from "
			  << j << " to " << i << " not found" << std::endl;
		throw std::logic_error("forward link following back link not found");
	    } else {
		//std::cerr << "forward link from " << j << " to " << i << " processed" << std::endl;;
		if (dep_set[j].size() == 0) {
		    pq.push(std::make_pair(node_counts[j], j));
		    //std::cerr << "node " << j << " is scheduled " << std::endl;
		}
	    }
	}
    }
    result.begin(), result.end();
    if (result.size() != dep_list.size()) {
	std::cerr << "WARNING: the file dependency list does not form a dag, " << std::endl;
	std::cerr << "only " << result.size() << " nodes have been resolved" << std::endl;
    }
    return result;
}
