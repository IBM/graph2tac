#include <iostream>
#include <vector>
#include <map>
#include <set>
#include "simple_directed_graph.hpp"

template <typename Node>
void print(const std::map<Node, std::set<Node>> example) {
    for (const auto &[k, conn]: example) {
	std::cout << k;
	for (const auto &e: conn) {
	    std::cout << " " << e;
	}
	std::cout << std::endl;
    }
}

int main() {
    std::map<char, std::set<char>>  forward_links {
	{'a', {'b'}},
	{'b', {'d', 'c', 'e'}},
	{'c', {'f'}},
	{'d', {}},
	{'e', {'b', 'f', 'g'}},
	{'f', {'c', 'h'}},
	{'g', {'h', 'j'}},
	{'h', {'k'}},
	{'i', {'g'}},
	{'j', {'i'}},
	{'k', {'l'}},
	{'l', {'j'}},
	{'p', {'r'}},
	{'r', {}}};




    SimpleDirectedGraph<char> graph(forward_links);
    auto result = graph.find_strongly_connected_componets();

    for (const auto& [node, comp_index]: result) {
	std::cout << node << " " << comp_index << std::endl;
    }


}
