"""
python prediction server to interact with coq-tactician-reinforce
"""
from typing import NewType

import sys
import time
import os
import yaml
import socket
import pickle
import pkg_resources
import tqdm
import capnp
import argparse
import numpy as np
from pathlib import Path

from graph2tac.common import uuid, logger
from graph2tac.predict import Predict
from graph2tac.loader.data_server import DataServer, build_def_index, get_global_def_table, LoaderGraph, LoaderProofstate, LoaderDefinition


Tactic = NewType('Tactic', object)

capnp.remove_import_hook()
graph_api_filename = pkg_resources.resource_filename('graph2tac.loader','clib/graph_api_v11.capnp')
graph_api_capnp = capnp.load(graph_api_filename)


from graph2tac.loader.clib.loader import (
    get_scc_components,
    data_online_extend,
    data_online_resize,
    get_def_deps_online,
    get_subgraph_online,
    load_msg_online,
    encode_prediction_online,
    get_graph_constants_online,
    build_data_online_from_buf)


BASE_NAMES, EDGE_CONFLATIONS = get_graph_constants_online()


def debug_record(msg, fname: str):
    msg_copy = msg.as_builder()
    logger.debug(f"debug dumping msg to {fname}")
    with open(fname, 'wb') as f_out:
        msg_copy.write_packed(f_out)






def process_synchronize(sock, msg):
    logger.debug(msg)
    response = graph_api_capnp.PredictionProtocol.Response.new_message(synchronized=msg.synchronize)
    logger.debug(f"sending synchronize response in the initialize loop {response}")
    response.write_packed(sock)

def process_initialize(sock, msg):
    graph1 = msg.initialize.graph
    definitions = msg.initialize.definitions
    logger.verbose("initialize tactics")
    logger.debug(f"tactics list {list(msg.initialize.tactics)}")

    logger.verbose("initialize definitions")
    logger.debug(f"definitions list {list(msg.initialize.definitions)}")
    tacs = []
    tac_numargs  = []
    for tac_reader in msg.initialize.tactics:
        tacs.append(tac_reader.ident)
        tac_numargs.append(tac_reader.parameters)
    response = graph_api_capnp.PredictionProtocol.Response.new_message(initialized=None)
    logger.verbose("sending initialize response")
    logger.debug(f"response is {response}")
    response.write_packed(sock)

    logger.verbose(f"tactics {tacs}")
    logger.verbose(f"tactics num args {tac_numargs}")
    return tacs, tac_numargs



def check_consistency(evaluation_tactic_hash_to_numargs, network_tactic_hash_to_numargs):
    for network_hash, network_numargs in network_tactic_hash_to_numargs.items():
        if network_hash in evaluation_tactic_hash_to_numargs.keys():
            evaluation_numargs = evaluation_tactic_hash_to_numargs[network_hash]
            logger.verbose(f"network_hash: {network_hash}")
            logger.verbose(f"network_numargs: {network_numargs}")
            logger.verbose(f"evaluation_numargs: {evaluation_numargs}")
            if network_numargs != evaluation_numargs:
                raise ValueError(f"Error! The network thinks that tactic {network_hash} "
                                 f"has {network_numargs} arguments, "
                                 f"but the evaluation client requests {evaluation_numargs} argument. ")


def get_train_eval_alignment(train_name_to_label: dict[bytes,int], eval_names: list[bytes], original_train_node_labels: int):
    cur_label = original_train_node_labels

    train_label_to_eval_label = dict()
    eval_label_to_train_label = []

    for eval_label, eval_name in enumerate(eval_names):
        if eval_name in train_name_to_label:
            train_label = train_name_to_label.get(eval_name)
            logger.verbose(f"aligning evaluation {eval_name} with training label {train_label}")
            if train_label is train_label_to_eval_label:
                raise ValueError(f"repeated definition {eval_name} with eval_label f{eval_label}")
            train_label_to_eval_label[train_label] = len(eval_label_to_train_label)
            eval_label_to_train_label.append(train_label)
        else:
            train_label_to_eval_label[cur_label] = len(eval_label_to_train_label)
            eval_label_to_train_label.append(cur_label)
            cur_label += 1

    logger.info(f"the network node embedding table has increased from {original_train_node_labels} to {cur_label} with {(len(eval_names) - (cur_label - original_train_node_labels))} aligned ")
    return train_label_to_eval_label, eval_label_to_train_label




def get_train_name_to_label(train_node_label_to_name, train_node_label_in_spine):
    train_name_to_label = dict()
    for label, (name, in_spine) in enumerate(zip(train_node_label_to_name, train_node_label_in_spine)):
        if in_spine:
            if name in train_name_to_label:
                raise ValueError(f"the train data contains name collisions restricted to spine:"
                                 f" {name}, label, train_name_to_label.get(name) ")
            train_name_to_label[name] = label
    logger.verbose(f"train names in spine: {len(train_name_to_label)}")
    return train_name_to_label

def get_def_idx_to_node__train_to_eval__eval_to_train__eval_names(msg_data, train_node_label_to_name, train_node_label_in_spine, message_type):
    train_name_to_label = get_train_name_to_label(train_node_label_to_name, train_node_label_in_spine)
    (node_indexes, def_hashes, def_idx_to_name), _  = get_global_def_table([msg_data], message_type, restrict_to_spine=False)[0]
    eval_names = BASE_NAMES + def_idx_to_name
    def_index_table = build_def_index([(node_indexes, def_hashes, eval_names)], set())
    def_idx_to_node =  np.array(def_index_table.idx_to_global_node, dtype=np.uint32)
    train_label_to_eval_label, eval_label_to_train_label = get_train_eval_alignment(train_name_to_label, eval_names, len(train_node_label_to_name))
    return def_idx_to_node, train_label_to_eval_label, eval_label_to_train_label, eval_names

def wrap_debug_record(debug_dir, msg, context_cnt):
    if debug_dir is not None:
        fname = os.path.join(debug_dir, f'msg_init.{context_cnt}.bin')
        debug_record(msg, fname=fname)

def train_names_in_cluster(def_scc_cluster, eval_label_to_train_label, train_node_label_to_name):
    result = []
    for eval_def_idx in def_scc_cluster:
            eval_label = len(BASE_NAMES) + eval_def_idx.item()
            train_label = eval_label_to_train_label[eval_label]
            train_name = train_node_label_to_name[train_label] if train_label < len(train_node_label_to_name) else b".MISSING"
            result.append(train_name.decode())
    return result


def log_clusters_verbose(def_scc_clusters, eval_label_to_train_label, train_node_label_to_name):
    for i, cluster in enumerate(def_scc_clusters):
        x = train_names_in_cluster(cluster, eval_label_to_train_label, train_node_label_to_name)
        logger.verbose(x)
        string_of_cluster_names = " ".join(x)
        logger.verbose(f"cluster {i}: {string_of_cluster_names}")



def get_unaligned_nodes(def_idx_to_node, def_idx_to_name, eval_label_to_train_label, original_train_len):
    unaligned_nodes = []
    for (node_idx, train_label, eval_name) in zip(def_idx_to_node, eval_label_to_train_label[len(BASE_NAMES):], def_idx_to_name):
        if train_label >= original_train_len:
            unaligned_nodes.append((node_idx, eval_name))
    return unaligned_nodes

def process_alignment_request(
        network_tactic_hash_to_numargs,
        al_msg_data,
        msg,
        train_node_label_to_name,
        train_node_label_in_spine):
    al_def_idx_to_node, al_train_label_to_eval_label, al_eval_label_to_train_label, al_eval_names = get_def_idx_to_node__train_to_eval__eval_to_train__eval_names(
        al_msg_data, train_node_label_to_name, train_node_label_in_spine, "request.checkAlignment")
    al_def_idx_to_name = al_eval_names[len(BASE_NAMES):]
    unaligned_nodes = get_unaligned_nodes(al_def_idx_to_node,
                                          al_def_idx_to_name,
                                          al_eval_label_to_train_label,
                                          len(train_node_label_to_name))
    logger.verbose(f"checkAlignment unaligned nodes: {unaligned_nodes}")


    evaluation_tactic_hash_to_numargs = dict()
    for tac_reader in msg.checkAlignment.tactics:
        evaluation_tactic_hash_to_numargs[tac_reader.ident] = tac_reader.parameters


    unaligned_tactics = list(sorted(set(evaluation_tactic_hash_to_numargs.items())
                                    - set(network_tactic_hash_to_numargs.items())))
    logger.verbose(f"checkAlignment unaligned tactics: {unaligned_tactics}")
    return unaligned_tactics, unaligned_nodes


def main_loop(reader, sock, predict: Predict, debug_dir, session_idx=0,
              bfs_option=True,
              with_meter=False,
              tactic_expand_bound=8,
              total_expand_bound=2048,
              search_expand_bound=8,
              update_all_definitions=False,
              update_new_definitions=False,
              progress_bar=False,
              temperature=1.0):
    max_subgraph_size = predict.get_max_subgraph_size()

    if debug_dir is not None:
        debug_dir_session = os.path.join(debug_dir, f"session_{session_idx}")
        os.makedirs(debug_dir_session, exist_ok=True)
    else:
        debug_dir_session = None


    evaluation_tactic_hash_to_numargs = dict()

    train_node_label_to_name = [bytes(s, 'utf8') for s in predict.get_label_to_name()]
    train_node_label_in_spine = np.array(predict.get_label_in_spine(), dtype=np.uint8)  # bool 0/1
    network_tactic_index_to_hash = np.array(predict.get_tactic_index_to_hash(), dtype=np.uint64)
    network_tactic_index_to_numargs = np.array(predict.get_tactic_index_to_numargs(), dtype=np.uint64)

    network_tactic_hash_to_numargs = {k:v for k,v in zip(network_tactic_index_to_hash, network_tactic_index_to_numargs)}
    logger.debug(f"network tactic hash to numargs {network_tactic_hash_to_numargs}")
    network_tactic_hash_to_index = {tactic_hash : idx for idx, tactic_hash in enumerate(network_tactic_index_to_hash)}

    context_cnt = -1
    n_predict = 0
    t_predict = 0
    t_coq = 0
    n_coq = 0
    decorated_reader = tqdm.tqdm(reader) if with_meter else reader


    msg_idx = -1
    total_data_online_size = 0
    build_network_time = 0
    update_def_time = 0
    n_def_clusters_updated = 0

    for msg in decorated_reader:
        msg_type = msg.which()
        logger.verbose(f"message: {msg.which()}")
        if msg_type == "synchronize":
            process_synchronize(sock, msg)
        elif msg_type == "checkAlignment":
            logger.info("checkAlignment request")
            al_msg_data = msg.as_builder().to_bytes()
            unaligned_tactics, unaligned_nodes = process_alignment_request(
                network_tactic_hash_to_numargs,
                al_msg_data,
                msg,
                train_node_label_to_name,
                train_node_label_in_spine)

            response = graph_api_capnp.PredictionProtocol.Response.new_message(
                alignment={'unalignedTactics': [x for (x,v) in unaligned_tactics],
                           'unalignedDefinitions': [node_idx.item() for ((file_idx, node_idx),n) in unaligned_nodes]})
            logger.info("sending checkAlignment response to coq")
            response.write_packed(sock)


        elif msg_type == "initialize":
            if context_cnt >= 0:
                logger.info(f'theorem {context_cnt} had {msg_idx} messages received of total size (unpacked) {total_data_online_size} bytes, with network compiled in {build_network_time:.6f} s, with {n_def_clusters_updated} def clusters updated in {update_def_time:.6f} s')


            context_cnt += 1

            logger.info(f'session {session_idx} theorem idx={context_cnt}, annotation={msg.initialize.logAnnotation} started.')

            wrap_debug_record(debug_dir, msg, context_cnt)

            msg_data = msg.as_builder().to_bytes()

            def_idx_to_node, train_label_to_eval_label, eval_label_to_train_label, eval_names = get_def_idx_to_node__train_to_eval__eval_to_train__eval_names(
                msg_data, train_node_label_to_name, train_node_label_in_spine, "request.initialize")

            logger.verbose(f"train_label_to_eval_label {train_label_to_eval_label}")
            logger.verbose(f"eval_label_to_train_label {eval_label_to_train_label}")

            def_idx_to_name = eval_names[len(BASE_NAMES):]
            unaligned_nodes = get_unaligned_nodes(def_idx_to_node,
                                                  def_idx_to_name,
                                                  eval_label_to_train_label,
                                                  len(train_node_label_to_name))
            logger.verbose(f"initialization unaligned nodes {unaligned_nodes}")

            c_data_online = build_data_online_from_buf(def_idx_to_node,
                                                       network_tactic_index_to_hash,
                                                       np.arange(len(network_tactic_index_to_hash), dtype=np.uint32))


            local_to_global = [np.array([0], dtype=np.uint32)]
            n_msg_recorded = data_online_extend(c_data_online,
                                                [msg_data],
                                                [b'.initalization_graph'],
                                                local_to_global,
                                                "request.initialize",
                                                )
            total_data_online_size = len(msg_data)

            logger.info(f"c_data_online references {n_msg_recorded} msg")

            def_deps_ids =  get_def_deps_online(c_data_online, bfs_option, max_subgraph_size)
            # log_info(f"computed def_deps_ids =  {def_deps_ids}")
            def_scc_clusters = get_scc_components(def_deps_ids)
            logger.info(f"computed def_scc_clusters  of size {len(def_scc_clusters)}")

            log_clusters_verbose(def_scc_clusters, eval_label_to_train_label, train_node_label_to_name)


            logger.info(f"generating clusters from initialization message")
            map_eval_label_to_train_label = np.array(eval_label_to_train_label, dtype=np.int32)

            global_context = map_eval_label_to_train_label[len(BASE_NAMES):].astype(np.int32)
            # available_global = np.arange(0, len(global_context), 1, dtype=np.uint64)

            logger.verbose(f"online global_context (indexed from 0): {global_context}")


            def_clusters_for_update = [def_cluster for def_cluster in def_scc_clusters
                                       if (update_all_definitions or
                                           (update_new_definitions and
                                            (set(tuple(node.tolist())
                                                 for node, _ in unaligned_nodes).intersection(
                                                         set(tuple(node.tolist()) for node in
                                                             def_idx_to_node[def_cluster])))))]
            if update_all_definitions:
                logger.info(f"Prepared for update all {len(def_clusters_for_update)} definition clusters")
            elif update_new_definitions:
                logger.info(f"Prepared for update {len(def_clusters_for_update)} definition clusters containing unaligned definitions")
            else:
                logger.info(f"No update of the definition clusters requested")


            t0 = time.time()
            logger.info(f"initializing network with {len(global_context)} defs in global context")


            predict.initialize(global_context)
            t1 = time.time()
            build_network_time = t1-t0
            logger.info(f"Building network model completed in {build_network_time:.6f} seconds")

            logger.info(f"Updating  definition clusters...")


            decorated_iterator = tqdm.tqdm(def_clusters_for_update) if progress_bar else def_clusters_for_update
            t0 = time.time()
            for def_cluster in decorated_iterator:
                cluster_names = [def_idx_to_name[def_idx] for def_idx in def_cluster]
                # print(cluster_names)

                res = get_subgraph_online(c_data_online, def_idx_to_node[def_cluster], bfs_option, max_subgraph_size, False)
                eval_node_labels, edges, edge_labels, edges_offset, global_visited, _, _, _, _ = res

                train_node_labels = map_eval_label_to_train_label[eval_node_labels]
                # edges_grouped_by_label = np.split(edges, edges_offset)

                cluster_graph = train_node_labels, edges, edge_labels, edges_offset

                # cluster_state = (train_node_labels, edges_grouped_by_label, len(def_cluster))
                cluster_state = LoaderDefinition( (cluster_graph, len(def_cluster), cluster_names) )
                # print(cluster_state)
                predict.compute_new_definitions([cluster_state])
            t1 = time.time()
            n_def_clusters_updated = len(def_clusters_for_update)
            update_def_time = t1 - t0

            tacs, tac_numargs = process_initialize(sock, msg)
            evaluation_tactic_hash_to_numargs = {k:v for k,v in zip(tacs, tac_numargs)}
            check_consistency(evaluation_tactic_hash_to_numargs, network_tactic_hash_to_numargs)
            allowed_model_tactics = []
            for tac_hash in tacs:
                if tac_hash in network_tactic_hash_to_index.keys():
                    allowed_model_tactics.append(network_tactic_hash_to_index[tac_hash])

            msg_idx = 0
            t0_coq = time.time()
        elif msg_type == "predict":
            t_coq += (time.time() - t0_coq)
            n_coq += 1
            if n_coq != 0 and n_coq % 10 == 0:
                logger.info(f'Coq requests {t_coq/n_coq:.6f} second/request')



            msg_idx += 1

            if debug_dir is not None:
                fname = os.path.join(debug_dir, f'msg_predict.{context_cnt}.{msg_idx}.bin')
                debug_record(msg, fname=fname)

            msg_data = msg.as_builder().to_bytes()

            data_online_resize(c_data_online, 1)
            local_to_global = [np.array([1, 0], dtype=np.uint32)]

            num_messages_stored = data_online_extend(c_data_online,
                                                [msg_data],
                                                [b'.proof_state_graph'],
                                                local_to_global,
                                                "request.predict",
                                                )
            total_data_online_size += len(msg_data)
            logger.info(f"num_messages_stored {num_messages_stored}")
            data_msg_idx = num_messages_stored - 1
            roots = np.array( [(data_msg_idx, msg.predict.state.root)], dtype=np.uint32)

            res = get_subgraph_online(c_data_online,
                                      roots,
                                      bfs_option,
                                      max_subgraph_size,
                                      False)    # debugging information about subgraph allocation
            eval_node_labels, edges, edge_labels, edges_offset, global_visited, _, _, _, _ = res

            train_node_labels = map_eval_label_to_train_label[eval_node_labels]
            # edges_grouped_by_label = np.split(edges, edges_offset)
            this_encoded_root, this_encoded_context, this_context = load_msg_online(msg_data, global_visited, data_msg_idx)

            logger.debug(f"this encoded root {this_encoded_root}")
            logger.debug(f"this encoded context {this_encoded_context}")
            logger.debug(f"this context {this_context}")

            logger.debug(f"state from python: {msg.predict.state}")
            logger.debug("root children from python:")
            child_start = msg.predict.graph.nodes[msg.predict.state.root].childrenIndex
            child_stop  = child_start + msg.predict.graph.nodes[msg.predict.state.root].childrenCount
            for edge_idx in range(child_start, child_stop):
                logger.debug(f"root 0 child {msg.predict.graph.edges[edge_idx]}")

            online_graph = LoaderGraph( (train_node_labels, edges, edge_labels, edges_offset) )

            # FIDEL: is this information present? For now, just fill it in with dummy data
            dummy_proofstate_info = ('dummy_proofstate_name', -1, True)

            logger.verbose(f"online_state: {online_graph}")

            t0 = time.time()

            online_actions, online_confidences = predict.ranked_predictions(
                LoaderProofstate( (online_graph, this_encoded_root, this_encoded_context, dummy_proofstate_info) ),
                tactic_expand_bound=tactic_expand_bound,
                total_expand_bound=total_expand_bound,
                allowed_model_tactics=allowed_model_tactics,
                available_global=None
            )
            # FIDEL: why was this here? Does not conform to the API!
            #                 annotation = msg_idx - 1,
            #                 debug = (LOG_LEVEL <= logging.INFO)

            # apply the temperature (assuming the truncated tail of the probability distribution is clustered in one unseen element)
            new_online_confidences = np.array(online_confidences, np.float64)**(1/temperature)
            new_online_confidences /= new_online_confidences.sum() + (1-sum(online_confidences))**(1/temperature)

            top_online_actions = online_actions[:search_expand_bound]
            top_online_confidences = new_online_confidences[:search_expand_bound]
            top_online_encoded_actions = encode_prediction_online(c_data_online,
                                                                  top_online_actions,
                                                                  top_online_confidences,
                                                              this_context,  sock.fileno(), eval_names[len(BASE_NAMES):])
            for action_idx, (online_encoded_action, online_confidence) in enumerate(
                    zip(top_online_encoded_actions, top_online_confidences)):
                logger.info(f"sending to coq the action {action_idx}, prob = {online_confidence:.6f}, {online_encoded_action}")

            t1 = time.time()
            t_predict += (t1 - t0)
            n_predict += 1

            logger.verbose(f'Process predict this call {t1-t0:.6f} seconds')
            if n_predict != 0 and n_predict % 10 == 0:
                logger.info(f'Network predicts {t_predict/n_predict:.6f} second/call')
            t0_coq = time.time()
        else:
            raise Exception("Capnp protocol error in prediction_loop: "
                            "msg type is not 'predict', 'synchronize', or 'initialize'")


    logger.info(f'(final): theorem {context_cnt} had {msg_idx} messages received of total size (unpacked) {total_data_online_size} bytes, with network compiled in {build_network_time:.6f} s, with {n_def_clusters_updated} def clusters updated in {update_def_time:.6f} s')




def test_record(data_dir):
    os.makedirs(uuid(data_dir), exist_ok=True)
    fname = Path(uuid(data_dir)).joinpath('check_dataset.out')
    d = DataServer(data_dir=data_dir, bfs_option=True, max_subgraph_size=512)
    dataset_recorded = [d.data_point(idx) for idx in range(d._debug_data().proof_steps_size())]
    cluster_subgraphs = d.def_cluster_subgraphs()
    pickle.dump((dataset_recorded, cluster_subgraphs), open(fname,'wb'))

def test_check(data_dir):
    os.makedirs(uuid(data_dir), exist_ok=True)
    fname = Path(uuid(data_dir)).joinpath('check_dataset.out')
    d = DataServer(data_dir=data_dir, bfs_option=True, max_subgraph_size=512)
    dataset = [d.data_point(idx) for idx in range(d._debug_data().proof_steps_size())]
    cluster_subgraphs = d.def_cluster_subgraphs()
    loaded = pickle.load(open(fname,'rb'))
    if repr(loaded) == repr((dataset, cluster_subgraphs)):
        print('PASSED')
    else:
        print('FAILED 0 ')
        for idx in range(len(dataset)):
            if not repr(loaded[0][idx]) == repr(dataset[idx]):
                breakpoint()
        print('FAILED 1 ')
        if repr(loaded[1]) != repr(cluster_subgraphs):
            breakpoint()
        print('FAILED 2 ')
        breakpoint()


def main():
    parser = argparse.ArgumentParser(
        description='graph2tac Predict python tensorflow server')

    parser.add_argument('--tcp', action='store_true',
                        help='start python server on tcp/ip socket')

    parser.add_argument('--port', type=int,
                        default=33333,
                        help='run python server on this port')
    parser.add_argument('--host', type=str,
                        default='127.0.0.1',
                        help='run python server on this local ip')

    parser.add_argument('--model', type=str,
                        default=None,
                        help='checkpoint directory of the model')

    parser.add_argument('--arch', type=str,
                        default='tf2',
                        help='the model architecture tfgnn or tf2 (current default is tf2)')

    parser.add_argument('--test', type=str,
                        default=None,
                        help='run test of loader on a given dataset')

    parser.add_argument('--record', type=str,
                        default=None,
                        help='run test of loader on a given dataset')

    parser.add_argument('--log_level', type=str,
                        default='info',
                        help='debug | verbose | info | warning | error | critical')

    parser.add_argument('--tf_log_level', type=str,
                        default='info',
                        help='debug | verbose | info | warning | error | critical')

    parser.add_argument('--debug_dir', type=str,
                        default = None,
                        help=("If you want to preserve messages for later inspection "
                              "please provide the name of the directory --debug_dir to keep the messages. "
                              "The directory will be created if it doesn't exists and the files may be overwritten. "
                              "Do not provide --debug_dir argument if you do not want to record all capnp messages "
                              "during the interactive session to  --debug_dir. "
                              "You supply --debug_dir under your responsibility to ensure sufficient free space "
                              "and purging when you need. "
                              "If you want quick and  performant recording of debug messages, "
                              "you may wish to use RAM based mounted filesystem like tmpfs "
                              "for example you can create and provide a directory in /run/user/<your user id> "
                              "that is typically RAM based tmpfs on Linux"))

    parser.add_argument('--with_meter',
                        default=False,
                        action='store_true',
                        help="Display throughput of predict calls per second")

    parser.add_argument('--total_expand_bound',
                        type=int,
                        default=2048,
                        help="total_expand_bound for ranked argument search")

    parser.add_argument('--tactic_expand_bound',
                        type=int,
                        default=8,
                        help="tactic_expand_bound for ranked argument search")

    parser.add_argument('--search_expand_bound',
                        type=int,
                        default=8,
                        help="maximal number of predictions to be sent to search algorithm in coq evaluation client ")

    parser.add_argument('--update_new_definitions',
                        default=False,
                        action='store_true',
                        help="call network update embedding on clusters containing new definitions ")

    parser.add_argument('--update_all_definitions',
                        default=False,
                        action='store_true',
                        help="call network update embedding on cluster containing all definitions (overwrites update new definitions)")

    parser.add_argument('--progress_bar',
                        default=False,
                        action='store_true',
                        help="show the progress bar of update definition clusters")


    parser.add_argument('--tf_eager',
                        default=False,
                        action='store_true',
                        help="with tf_eager=True activated network tf2 may initialize faster but run slower, use carefully if you need")

    parser.add_argument('--temperature',
                        type=float,
                        default=1.0,
                        help="temperature to apply to the probability distributions returned by the model")

    parser.add_argument('--debug-predict',
                        type=Path,
                        default=None,
                        help="set this flag to run Predict in debug mode")

    parser.add_argument('--checkpoint-number',
                        type=int,
                        default=None,
                        help="choose the checkpoint number to use (defaults to latest available checkpoint)")

    parser.add_argument('--exclude-tactics',
                        type=Path,
                        default=None,
                        help="a list of tactic names to exclude from predictions")





    args = parser.parse_args()

    if args.debug_dir is not None:
        os.makedirs(args.debug_dir, exist_ok=True)
        logger.warning(f"WARNING!!!! the directory {args.debug_dir} was provided to record messages for debugging purposes. The capnp bin messages will be recorded to {args.debug_dir}. Please do not provide --debug_dir if you do not want to record messages for later debugging and inspection purposes!")


    if not args.test is None:
        test_check(Path(args.test).expanduser().absolute())

    if not args.record is None:
        test_record(Path(args.record).expanduser().absolute())

    log_levels = {'debug':'10',
                  'verbose':'15',
                  'info':'20',
                  'warning':'30',
                  'error':'40',
                  'critical':'50'}

    tf_log_levels={'debug':'0',
                   'verbose':'0',
                   'info':'0',
                   'warning':'1',
                   'error':'2',
                   'critical':'3'}


    os.environ['G2T_LOG_LEVEL'] = log_levels[args.log_level]
    logger.setLevel(int(log_levels[args.log_level]))

    predict = None
    if not args.model is None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_levels[args.tf_log_level]
        if args.arch == 'tf2':
            logger.info("importing tensorflow...")
            import tensorflow as tf
            tf.get_logger().setLevel(int(tf_log_levels[args.log_level]))
            tf.config.run_functions_eagerly(args.tf_eager)

            logger.info("importing TF2Predict class..")
            from graph2tac.tf2.predict import TF2Predict
            predict = TF2Predict(checkpoint_dir=Path(args.model).expanduser().absolute(), debug_dir=args.debug_predict)
        elif args.arch == 'tfgnn':
            logger.info("importing tensorflow...")
            import tensorflow as tf
            tf.get_logger().setLevel(int(tf_log_levels[args.log_level]))
            tf.config.run_functions_eagerly(args.tf_eager)

            if args.exclude_tactics is not None:
                with Path(args.exclude_tactics).open('r') as yaml_file:
                    exclude_tactics = yaml.load(yaml_file, Loader=yaml.SafeLoader)
                logger.info(f'excluding tactics {exclude_tactics}')
            else:
                exclude_tactics = None

            logger.info("importing TFGNNPredict class...")
            from graph2tac.tfgnn.predict import TFGNNPredict
            predict = TFGNNPredict(log_dir=Path(args.model).expanduser().absolute(),
                                   debug_dir=args.debug_predict,
                                   checkpoint_number=args.checkpoint_number,
                                   exclude_tactics=exclude_tactics)
        elif args.arch == 'hmodel':
            logger.info("importing HPredict class..")
            from graph2tac.loader.hmodel import HPredict
            predict = HPredict(checkpoint_dir=Path(args.model).expanduser().absolute(), debug_dir=args.debug_predict)
        else:
            Exception(f'the provided model architecture {args.arch} is not supported')

        logger.info(f"initializing predict network from {Path(args.model).expanduser().absolute()}")

    if not args.tcp:
        logger.info("starting stdin server")
        sock = socket.socket(fileno=sys.stdin.fileno())
        reader = graph_api_capnp.PredictionProtocol.Request.read_multiple_packed(sock, traversal_limit_in_words=2**64-1)
        main_loop(reader, sock, predict, args.debug_dir,
                  session_idx=0,
                  with_meter=args.with_meter,
                  tactic_expand_bound=args.tactic_expand_bound,
                  total_expand_bound=args.total_expand_bound,
                  search_expand_bound=args.search_expand_bound,
                  update_all_definitions=args.update_all_definitions,
                  update_new_definitions=args.update_new_definitions,
                  progress_bar=args.progress_bar,
                  temperature=args.temperature
        )
    else:
        logger.info(f"starting tcp/ip server on port {args.port}")
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((args.host, args.port))
        try:
            server_sock.listen(1)
            logger.info(f"tcp/ip server is listening on {args.port}")
            session_idx = 0
            while True:
                sock, remote_addr = server_sock.accept()
                logger.info(f"coq client connected {remote_addr}")
                reader = graph_api_capnp.PredictionProtocol.Request.read_multiple_packed(sock, traversal_limit_in_words=2**64-1)
                main_loop(reader, sock, predict,
                          args.debug_dir,
                          session_idx,
                          with_meter=args.with_meter,
                          tactic_expand_bound=args.tactic_expand_bound,
                          total_expand_bound=args.total_expand_bound,
                          search_expand_bound=args.search_expand_bound,
                          update_all_definitions=args.update_all_definitions,
                          update_new_definitions=args.update_new_definitions,
                          progress_bar=args.progress_bar,
                )
                logger.info(f"coq client disconnected {remote_addr}")

                session_idx += 1
        finally:
            logger.info(f'closing the server on port {args.port}')
            server_sock.close()


if __name__ == '__main__':
    main()
