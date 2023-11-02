import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import sys
import socket
from pathlib import Path
from typing import BinaryIO, Optional, Union, Dict
import numpy as np
from numpy.typing import NDArray, ArrayLike
import tqdm
import os
import uuid
import time
import psutil
import logging
import contextlib
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from pytact.data_reader import (capnp_message_generator, capnp_message_generator_from_file,
                                TacticPredictionGraph, TacticPredictionsGraph,
                                GlobalContextMessage, CheckAlignmentMessage, CheckAlignmentResponse,
                                ProofState, OnlineDefinitionsReader)
from graph2tac.common import logger
from graph2tac.loader.data_classes import GraphConstants, LoaderProofstate, ProofstateContext, ProofstateMetadata
from graph2tac.loader.data_server import AbstractDataServer
from graph2tac.predict import Predict

def apply_temperature(confidences: ArrayLike, temperature: float) -> NDArray[np.float64]:
    """
    Apply the temperature,
    assuming the truncated tail of the probability distribution
    is clustered in one unseen element
    """
    # TODO(jrute): Temperature should probably be applied earlier in the model before beam search
    
    # confidences are log probabilities associated with a subset of all possible responses
    log_probs = np.array(confidences, np.float64)
    with np.errstate(divide='ignore'):  # ok if taking np.log(0.0) here.  It will be -inf.
        remainder_log_prob = np.log(np.maximum(1.0 - np.exp(log_probs).sum(), 0.0))

    # apply temperature
    log_probs = log_probs / temperature
    remainder_log_prob = remainder_log_prob / temperature

    # normalize log probabilities
    if len(log_probs):
        max_log_prob = max(log_probs.max(), remainder_log_prob)
        log_probs = log_probs - max_log_prob
        remainder_log_prob = remainder_log_prob - max_log_prob
    
    offset = np.log(np.sum(np.exp(log_probs)) + np.exp(remainder_log_prob))
    return log_probs - offset

class ResponseHistory:
    """Records response history for testing."""
    data: dict
    """JSON compatible dictionary which stores all the history."""
    def __init__(self, recording_on: bool):
        """
        :param recording_on: Record results if True, otherwise do nothing.
        """
        self._recording_on = recording_on
        self.data = {"responses": []}

    @staticmethod
    def pred_arg_repr(a):
        if d := a.definition:
            str = f"{repr(a)}-{d.name}"
        else:
            str = repr(a)
        return str

    @staticmethod
    def convert_msg_to_dict(msg: Union[CheckAlignmentResponse, TacticPredictionsGraph]) -> dict:
        if isinstance(msg, TacticPredictionsGraph):
            return {
                "_type": type(msg).__name__,
                "contents": {
                    "predictions": [
                        {"ident": p.ident, "arguments": [ResponseHistory.pred_arg_repr(a) for a in p.arguments], "confidence": p.confidence}
                        for p in msg.predictions
                    ]
                },
            }
        elif isinstance(msg, CheckAlignmentResponse):
            return {
                "_type": type(msg).__name__,
                "contents": {
                    "unknown_definitions": [f"{repr(d.node)}-{d.name}" for d in msg.unknown_definitions],
                    "unknown_tactics": [t for t in msg.unknown_tactics]
                },
            }
        else:
            raise NotImplementedError(f"f{type(msg)} messages not yet supported")
    
    def record_response(self, msg: Union[CheckAlignmentResponse, TacticPredictionsGraph]):
        if self._recording_on:
            self.data["responses"].append(self.convert_msg_to_dict(msg))

class Profiler:
    """
    Controls the tensorflow profiler for both profiling predictions and definitions
    """
    logdir: Dict[str, Path]
    start: Dict[str, int]
    end: Dict[str, int]
    cnt: Dict[str, int]

    def __init__(self, logdir: Dict[str, Optional[Path]], start: Dict[str, int], end: Dict[str, int]):
        self.logdir = {k: v for k,v in logdir.items() if v is not None}
        self.start = start
        self.end = end
        self.cnt = {k: 0 for k in self.logdir.keys()}

    def step(self, key: str):
        """
        Increment counter and start/stop profiler.
        """
        if key not in self.logdir:
            # not profiling this key
            return 

        if self.cnt[key] == self.start[key]:
            options = tf.profiler.experimental.ProfilerOptions(
                host_tracer_level = 3,
                python_tracer_level = 1,
                device_tracer_level = 1
            )
            tf.profiler.experimental.start(str(self.logdir[key]), options)
        elif self.cnt[key] == self.end[key]:
            tf.profiler.experimental.stop()
        
        self.cnt[key] += 1
        

@dataclass
class LoggingCounters:
    """
    The counters recorded in the logs and debug output.
    """
    process_uuid: uuid.UUID
    """
    UUID for the process.
    
    Used to distinguish logs when multiple processes get logged to the same file.
    """
    session_idx: int = -1
    """Index for the current session"""
    thm_idx: int = -1
    """Index of the current theorem (based on number of `initialization` messages seen)"""    
    thm_annotation: str = ""
    """A theorem identifier (including file and linenumber if Coq is using coqc)"""
    msg_idx: int = 0
    """Number of 'prediction' messages recieved since last 'intitialize' message"""
    t_predict: float = 0.0
    """Total time (in seconds) spent on predict since last 'initialize'"""
    max_t_predict: float = 0.0
    """Max time (in seconds) spent on single predict since last 'initialize'"""
    argmax_t_predict: int = -1
    """Index of last maximum time single predict since last 'initialize'"""
    t_coq_send: float = 0.0
    """Total time (in seconds) spent waiting on Coq since last 'initialize'"""
    t_coq_receive: float = 0.0
    """Total time (in seconds) spent waiting on Coq since last 'initialize'"""
    t0_coq_receive: float = -1.0
    """last (absolute) time we called coq"""
    build_network_time: float = 0.0
    """Total time (in seconds) of all network initializations"""
    update_def_time: float = 0.0
    """Time (in seconds) to update the definitions processing most recent 'intitialize' message"""
    n_def_clusters_updated: int = 0
    """Number of definition clusters updated when processing most recent 'initialize' message"""
    proc = psutil.Process()
    """This process, used to get memory and cpu statistics."""
    total_mem: float = 0.0
    """Total memory (in bytes) at time of most recent 'initialize' message"""
    physical_mem: float = 0.0
    """Total physical memory (in bytes) at time of most recent 'initialize' message"""
    total_mem_diff: float = 0.0
    """Difference in memory (in bytes) since last 'initialize' message"""
    physical_mem_diff: float = 0.0
    """Difference in physical memory (in bytes) since last 'initialize' message"""
    cpu_pct = 0.0
    """
    CPU utilization since the last 'initialize' message.

    Can be > 100% if using multiple cores (same as in htop).
    May incorrectly be 0.0 if there isn't enough time between 'initialize' calls.
    """

    @contextmanager
    def measure_build_network_time(self):
        t0 = time.time()
        yield
        t1 = time.time()
        self.build_network_time += t1 - t0

    @contextmanager
    def measure_update_def_time(self, n_def_clusters_updated):
        t0 = time.time()
        yield
        t1 = time.time()
        self.n_def_clusters_updated += n_def_clusters_updated
        self.update_def_time += t1 - t0

    @contextmanager
    def measure_t_coq_send(self):
        t0 = time.time()
        yield
        t1 = time.time()
        self.t_coq_send += t1-t0

    def measure_t_coq_receive_start(self):
        if self.t0_coq_receive >= 0:
            logger.warning("(coq_start) Nesting theorems in LoggingCounters, likely causing to misleading summaries")
        self.t0_coq_receive = time.time()
    def measure_t_coq_receive_finish(self):
        t1_coq_receive = time.time()
        if self.t0_coq_receive < 0:
            logger.warning("(coq_finish) Nesting theorems in LoggingCounters, likely causing misleading summaries")
            return
        self.t_coq_receive += t1_coq_receive - self.t0_coq_receive
        self.t0_coq_receive = -1.0

    @contextmanager
    def measure_t_predict(self):

        t0 = time.time()
        yield
        t1 = time.time()

        if t1 - t0 > self.max_t_predict:
            self.max_t_predict = t1 - t0
            self.argmax_t_predict = self.msg_idx

        self.t_predict += (t1 - t0)

    def start_session(self):
        self.session_idx += 1
        self.thm_idx = -1
        self.reset_definition_counters()

    def start_theorem(self, log_annotation):
        self.thm_idx += 1
        self.thm_annotation = log_annotation
        self.msg_idx = 0
        self.t_predict = 0.0
        self.max_t_predict = 0.0
        self.argmax_t_predict = -1
        self.t_coq_receive = 0.0

    def update_process_stats(self):
        """
        Update the CPU and memory statistics recording the difference 
        """
        self.cpu_pct = self.proc.cpu_percent()
        new_mem = self.proc.memory_info()
        self.total_mem_diff = new_mem.vms - self.total_mem
        self.physical_mem_diff = new_mem.rss - self.physical_mem
        self.total_mem = new_mem.vms
        self.physical_mem = new_mem.rss

    def reset_definition_counters(self):
        self.update_def_time = 0.0
        self.n_def_clusters_updated = 0

    def summary_log_message(self) -> str:
        n_steps = self.msg_idx
        summary_data = {
            "UUID" : f"{self.process_uuid}",
            "Session" : f"{self.session_idx}",
            "Theorem" : f"{self.thm_idx}",
            "Annotation" : f"{self.thm_annotation}",
            "Initialize|Network build time (s)" : f"{self.build_network_time:.6f}",
            "Initialize|Def clusters to update" : f"{self.n_def_clusters_updated}",
            "Initialize|Def update time (s)" : f"{self.update_def_time:.6f}",
            "Predict|Messages cnt" : f"{n_steps}",
            "Predict|Avg predict time (s/msg)" : f"{self.t_predict/n_steps:.6f}",
            "Predict|Max predict time (s/msg)" : f"{self.max_t_predict:.6f}",
            "Predict|Max time predict msg ix" : f"{self.argmax_t_predict}",
            "Predict|Avg Coq time (send) (s/msg)": f"{self.t_coq_send/n_steps:.6f}",
            "Predict|Avg Coq time (iter) (s/msg)": f"{self.t_coq_receive/n_steps:.6f}",
            "Memory|Total memory (MB)": f"{self.total_mem/10**6:.3f}",
            "Memory|Physical memory (MB)": f"{self.physical_mem/10**6:.3f}",
            "Memory|Total mem diff (MB)": f"{self.total_mem_diff/10**6:.3f}",
            "Memory|Physical mem diff (MB)": f"{self.physical_mem_diff/10**6:.3f}",
            "Avg CPU (%)": f"{self.cpu_pct}"
        }
        return (
            f"Thm Summary:\n" +
            "\n".join(f"[g2t-sum] {k}\t{v}" for k, v in summary_data.items())
        )

class DynamicDataServer(AbstractDataServer):
    def __init__(self, graph_constants: GraphConstants, paranoic):
        super().__init__(graph_constants.data_config)
        self.paranoic = paranoic # checks consistency on each update

        self._tactic_i_to_numargs = list(graph_constants.tactic_index_to_numargs)
        self._tactic_i_to_hash = list(graph_constants.tactic_index_to_hash)
        self._tactic_to_i = {
            h : i
            for i,h in enumerate(self._tactic_i_to_hash)
        }

        self._node_i_to_name = graph_constants.label_to_names
        self._node_i_to_ident = graph_constants.label_to_ident
        self._node_i_in_spine = graph_constants.label_in_spine
        self._train_num_nodes = len(self._node_i_to_name)
        assert len(self._node_i_to_ident) == self._train_num_nodes
        assert len(self._node_i_in_spine) == self._train_num_nodes
        self._last_num_nodes = None
        self._context_stack = []

        # extra data
        
        self._ident_to_node_i = {
            ident : i
            for i,ident in enumerate(self._node_i_to_ident)
            if i >= self._base_node_label_num
        }

        self._node_to_node_i = dict()
        self._active_i_to_node = []
        self._active_i_to_node_i = []
        self._node_i_to_active_i = dict()

        if self.paranoic: self.consistency_check()

    @property
    def num_nodes_total(self): # trained & aligned nodes together
        return len(self._node_i_to_name)
    @property
    def num_nodes_active(self): # only aligned nodes
        return len(self._active_i_to_node_i)

    def align_definitions(self, definitions):

        self._context_stack.append((
            len(self._active_i_to_node_i),
            len(self._node_i_to_name)
        ))

        for d in definitions:
            node_i = self._ident_to_node_i.get(d.node.identity, None)
            if node_i is None:
                node_i = self._register_definition(d)
                self._ident_to_node_i[d.node.identity] = node_i
            else:
                self._node_to_node_i[d.node] = node_i

            assert node_i not in self._node_i_to_active_i, "Two nodes of the same identity in current global context"
            self._node_i_to_active_i[node_i] = len(self._active_i_to_node_i)
            self._active_i_to_node.append(d.node)
            self._active_i_to_node_i.append(node_i)

        if self.paranoic: self.consistency_check()

    def pop(self):
        (
            num_nodes_active,
            num_nodes_total,
        ) = self._context_stack.pop()

        for ident in self._node_i_to_ident[num_nodes_total:]:
            del self._ident_to_node_i[ident]
        del self._node_i_to_name[num_nodes_total:]
        del self._node_i_to_ident[num_nodes_total:]
        del self._node_i_in_spine[num_nodes_total:]

        for node in self._active_i_to_node[num_nodes_active:]:
            del self._node_to_node_i[node]
        for node_i in self._active_i_to_node_i[num_nodes_active:]:
            del self._node_i_to_active_i[node_i]

        del self._active_i_to_node[num_nodes_active:]
        del self._active_i_to_node_i[num_nodes_active:]

        if self.paranoic: self.consistency_check()

    def consistency_check(self):
        """
        (debugging) Checks concictency of:
            self._node_i_to_name
            self._node_i_to_ident
            self._node_i_in_spine
            self._ident_to_node_i
            self._node_to_node_i
            self._node_i_to_active_i
            self._active_i_to_node
            self._active_i_to_node_i
        Another purpose of this function is documenting the invariants
        we want to preserve.
        """

        # check node arrays sizes
        num_nodes_total = self.num_nodes_total
        num_nodes_active = self.num_nodes_active
        assert len(self._node_i_to_name) == num_nodes_total
        assert len(self._node_i_to_ident) == num_nodes_total
        assert len(self._node_i_in_spine) == num_nodes_total
        assert len(self._active_i_to_node_i) == num_nodes_active
        assert len(self._active_i_to_node) == num_nodes_active

        # check self._node_i_to_active_i
        assert len(self._node_i_to_active_i) == num_nodes_active
        assert self._node_i_to_active_i == {
            node_i : active_i
            for active_i, node_i in enumerate(self._active_i_to_node_i)
        }

        # check self._active_i_to_node_i, self._active_i_to_node
        assert len(self._node_to_node_i) == num_nodes_active
        assert self._node_to_node_i == dict(zip(
            self._active_i_to_node, self._active_i_to_node_i
        ))

        # check self._ident_to_node_i
        assert self._ident_to_node_i == {
            ident : i
            for i,ident in enumerate(self._node_i_to_ident)
            if i >= self._base_node_label_num
        }

        # check identities of active nodes
        for node, node_i in self._node_to_node_i.items():
            assert node.identity == self._node_i_to_ident[node_i]

    def proofstate(self, root, local_context):
        graph, node_to_i = self._downward_closure([root])
        root_i = 0
        local_context_i = [node_to_i[n] for n in local_context]

        context = ProofstateContext(
            local_context=np.array(local_context_i, dtype = np.uint32),
            global_context=np.array(self._active_i_to_node_i, dtype = np.uint32),
        )
        dummy_proofstate_info = ProofstateMetadata(b'dummy_proofstate_name', -1, True)

        return LoaderProofstate(
            graph=graph,
            root=root_i,
            context=context,
            metadata=dummy_proofstate_info
        )

    def check_alignment(self, tactics, definitions):
        unaligned_tactics = [
            tactic.ident for tactic in tactics
            if tactic.ident not in self._tactic_to_i
        ]
        unaligned_definitions = [
            d for d in definitions
            if self._ident_to_node_i.get(d.node.identity, self._train_num_nodes) >= self._train_num_nodes
        ]
        return CheckAlignmentResponse(
            unknown_tactics = unaligned_tactics,
            unknown_definitions = unaligned_definitions,
        )

    def decode_action(self, action: NDArray[np.int_], confidence: np.float_, proof_state: ProofState) -> TacticPredictionGraph:
        tactic = action[0,0]
        arguments = []
        for arg_type, arg_index in action[1:]:
            if arg_type == 0: # local argument
                arguments.append(proof_state.context[arg_index])
            else:
                arguments.append(self._active_i_to_node[arg_index])

        return TacticPredictionGraph(
            int(self._tactic_i_to_hash[tactic]),
            arguments,
            float(confidence),
        )
    
    def node_i_was_trained(self, node_i : int):
        return node_i < self._train_num_nodes
    def node_was_trained(self, node):
        return self.node_i_was_trained(self._node_to_node_i[node])

    def node_i_was_defined(
            self,
            node_i : int,
            history : int, # how many pop's away is the studied timepoint
    ):
        if node_i < self._base_node_label_num: return True
        if not history:
            num_nodes_active = self.num_nodes_active
        elif history <= len(self._context_stack):
            num_nodes_active = self._context_stack[-history][0]
        else:
            return False

        active_i = self._node_i_to_active_i.get(node_i, num_nodes_active)
        return active_i < num_nodes_active

    def tactic_to_i(self, tactic):
        return self._tactic_to_i.get(tactic.ident, None)

class PredictServer:
    def __init__(self,
                 model: Predict,
                 config: argparse.Namespace,
                 log_cnts: LoggingCounters,
                 profiler: Profiler,
                 response_history: ResponseHistory,
    ):
        self.data_server = DynamicDataServer(
            graph_constants = model.graph_constants,
            paranoic = config.paranoic_data_server,
        )
        self.model = model
        self.config = config
        self.log_cnts = log_cnts
        self.profiler = profiler
        self.current_allowed_tactics = None
        self.response_history = response_history
        self.msg_stack = []

    def _calculate_current_allowed_tactics(self, msg : GlobalContextMessage):
        self.current_allowed_tactics = []
        for tactic in msg.tactics:
            tactic_i = self.data_server.tactic_to_i(tactic)
            if tactic_i is not None:
                self.current_allowed_tactics.append(tactic_i)

    def _enter_coq_context(self, msg : GlobalContextMessage):

        self.msg_stack.append(msg)
        self._calculate_current_allowed_tactics(msg)

        # definition alignment
        
        self.data_server.align_definitions(msg.definitions.definitions(full = False))

        num_labels = self.data_server.num_nodes_total
        logger.info(f"allocating space for {num_labels} defs")
        with self.log_cnts.measure_build_network_time():
            self.model.allocate_definitions(num_labels)

        # definition recalculation
        if self.config.update == "all":
            def_clusters_for_update = list(msg.definitions.clustered_definitions(full = False))
            logger.info(f"Prepared for update all {len(def_clusters_for_update)} definition clusters")
        elif self.config.update == "new":
            def_clusters_for_update = [
                cluster for cluster in msg.definitions.clustered_definitions(full = False)
                if not self.data_server.node_was_trained(cluster[0].node)
            ]
            logger.info(f"Prepared for update {len(def_clusters_for_update)} definition clusters containing unaligned definitions")
        else:
            assert self.config.update is None
            def_clusters_for_update = []
            logger.info(f"No update of the definition clusters requested")

        # definitions.clustered_definitions is in reverse order of dependencies, so we reverse our list
        def_clusters_for_update.reverse()

        defined_nodes = set()
        def was_defined(node_i): # debugging function
            if node_i in defined_nodes:
                return True
            res = self.data_server.node_i_was_defined(node_i, history = 1)
            if self.config.update == "new":
                res = res or self.data_server.node_i_was_trained(node_i)
            if res: defined_nodes.add(node_i)
            return res

        def sanity_check(cluster_graph): # debugging function
            new_defined_nodes = cluster_graph.graph.nodes[:cluster_graph.num_definitions]
            used_nodes = cluster_graph.graph.nodes[cluster_graph.num_definitions:]
            for n in set(used_nodes):
                assert was_defined(n), (
                    f"Definition clusters out of order. "
                    f"Attempting to compute definition embedding for node labels {new_defined_nodes} "
                    f"({cluster_graph.definition_names}) without first computing "
                    f"the definition embedding for node label {n} used in that definition."
                )   
            for n in new_defined_nodes:
                assert not was_defined(n), (
                    f"Something is wrong with the definition clusters. "
                    f"Attempting to compute definition embedding for node labels {new_defined_nodes} "
                    f"({cluster_graph.definition_names}) "
                    f"for which node label {n} has already been computed."
                )
                defined_nodes.add(n)

        with self.log_cnts.measure_update_def_time(len(def_clusters_for_update)):
            if def_clusters_for_update:
                logger.info(f"Updating definition clusters...")
                if self.config.progress_bar:
                    def_clusters_for_update = tqdm.tqdm(def_clusters_for_update)

                for cluster in def_clusters_for_update:
                    self.profiler.step("def")
                    cluster_graph = self.data_server.cluster_to_graph(cluster)
                    sanity_check(cluster_graph)
                    self.model.compute_new_definitions([cluster_graph])

                logger.info(f"Definition clusters updated.")
            else:
                logger.info(f"No cluster to update.")

    def _exit_coq_context(self):

        self.data_server.pop()
        msg = self.msg_stack.pop()
        self._calculate_current_allowed_tactics(msg)

    @contextmanager
    def coq_context(self, msg: GlobalContextMessage):
        self._enter_coq_context(msg)
        yield
        self._exit_coq_context()

    def predict(self, proof_state: ProofState) -> TacticPredictionsGraph:
        if self.current_allowed_tactics is None:
            raise Exception("Cannot predict outside 'with predict_server.coq_context()'")

        proof_state_graph = self.data_server.proofstate(proof_state.root, proof_state.context)

        actions, confidences = self.model.ranked_predictions(
            state=proof_state_graph,
            allowed_model_tactics=self.current_allowed_tactics,
            available_global=None
        )
        confidences = apply_temperature(confidences, self.config.temperature)

        # use only top-k
        actions = actions[:self.config.search_expand_bound]
        confidences = confidences[:self.config.search_expand_bound]

        return TacticPredictionsGraph([
            self.data_server.decode_action(action, confidence, proof_state)
            for action, confidence in zip(actions, confidences)
        ])

    def prediction_loop(self, context: GlobalContextMessage):

        if self.config.with_meter:
            message_iterator = tqdm.tqdm(context.prediction_requests)
        else:
            message_iterator = context.prediction_requests

        is_theorem = False
        
        with self.coq_context(context):

            for msg in message_iterator:

                if isinstance(msg, ProofState):

                    if not is_theorem:
                        is_theorem = True
                        self.log_cnts.start_theorem(context.log_annotation)
                    else:
                        self.log_cnts.measure_t_coq_receive_finish()

                    with self.log_cnts.measure_t_predict():
                        self.profiler.step("pred")
                        response = self.predict(msg) # prediction with a network
                    self.log_cnts.msg_idx += 1

                    self.response_history.record_response(response)

                    with self.log_cnts.measure_t_coq_send():
                        # sending response back to coq
                        context.prediction_requests.send(response)
                    self.log_cnts.measure_t_coq_receive_start()

                elif isinstance(msg, CheckAlignmentMessage):
                    logger.info("checkAlignment request")
                    response = self.data_server.check_alignment(
                        context.tactics,
                        context.definitions.definitions(),
                    )
                    logger.info("sending checkAlignment response to coq")
                    self.response_history.record_response(response)
                    context.prediction_requests.send(response)

                elif isinstance(msg, GlobalContextMessage):
                    self.prediction_loop(msg)

                else:
                    raise Exception("Capnp protocol error")

        if is_theorem:
            self.log_cnts.measure_t_coq_receive_finish()
            self.log_cnts.update_process_stats()
            logger.summary(self.log_cnts.summary_log_message())
            self.log_cnts.reset_definition_counters()

    def start_prediction_loop(self, capnp_socket: socket.socket, record_file: Optional[BinaryIO]):
        self.log_cnts.start_session()
        self.prediction_loop(capnp_message_generator(capnp_socket, record_file))

    def start_prediction_loop_with_replay(self, replay_file: Path, record_file: Optional[BinaryIO]):
        with open(replay_file, "rb") as f:
            self.log_cnts.start_session()
            self.prediction_loop(capnp_message_generator_from_file(f, record=record_file))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='graph2tac Predict python tensorflow server')

    comm_group = parser.add_mutually_exclusive_group()
    comm_group.add_argument('--stdin',
                            action='store_true', default=False,
                            help='communicate via stdin/stdout (default)'
    )
    comm_group.add_argument('--tcp',
                            action='store_true', default=False,
                            help='start python server on tcp/ip socket'
    )
    comm_group.add_argument('--replay',
                            type = Path, dest='replay_file',
                            help='replay previously recorded record file'
    )

    parser.add_argument('--port', type=int,
                        default=33333,
                        help='run python server on this port')
    parser.add_argument('--host', type=str,
                        default='',
                        help='run python server on this local ip')
    parser.add_argument('--model', type=str, required=True,
                        help='checkpoint directory of the model')

    parser.add_argument('--arch', type=str,
                        default='tfgnn',
                        choices=['tfgnn', 'hmodel'],
                        help='the model architecture tfgnn or hmodel (current default is tfgnn)')

    parser.add_argument('--log-level', '--log_level', type=str,
                        default='info',
                        help='debug | verbose | info | summary | warning | error | critical')

    parser.add_argument('--tf-log-level', '--tf_log_level', type=str,
                        default='info',
                        help='debug | verbose | info | summary | warning | error | critical')

    parser.add_argument('--record',
                        type = str,
                        default = None,
                        help='Record all exchanged messages to the specified file, so that they can later be ' +
                        'replayed through "pytact-fake-coq" or --replay')

    parser.add_argument('--with-meter', '--with_meter',
                        default=False,
                        action='store_true',
                        help="Display throughput of predict calls per second")

    parser.add_argument('--total-expand-bound', '--total_expand_bound',
                        type=int,
                        default=2048,
                        help="(deprecated)")

    parser.add_argument('--tactic-expand-bound', '--tactic_expand_bound',
                        type=int,
                        default=8,
                        help="tactic_expand_bound for ranked argument search")

    parser.add_argument('--search-expand-bound', '--search_expand_bound',
                        type=int,
                        default=8,
                        help="maximal number of predictions to be sent to search algorithm in coq evaluation client ")

    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument('--update-no-definitions', '--update_no_definitions',
        action='store_const', dest='update', const=None, default=None,
        help='for new definitions (not learned during training) use default embeddings (default)'
    )
    update_group.add_argument('--update-new-definitions', '--update_new_definitions',
        action='store_const', dest='update', const='new',
        help='for new definitions (not learned during training) use embeddings calculated from the model'
    )
    update_group.add_argument('--update-all-definitions', '--update_all_definitions',
        action='store_const', dest='update', const='all',
        help='overwrite all definition embeddings with new embeddings calculated from the model'
    )

    parser.add_argument('--progress-bar', '--progress_bar',
                        default=False,
                        action='store_true',
                        help="show the progress bar of update definition clusters")

    parser.add_argument('--tf-eager', '--tf_eager',
                        default=False,
                        action='store_true',
                        help="with tf_eager=True activated network may initialize faster but run slower, use carefully if you need")

    parser.add_argument('--temperature',
                        type=float,
                        default=1.0,
                        help="temperature to apply to the probability distributions returned by the model")

    parser.add_argument('--debug-predict', '--debug_predict',
                        type=Path,
                        default=None,
                        help="set this flag to run Predict in debug mode")

    parser.add_argument('--checkpoint-number', '--checkpoint_number',
                        type=int,
                        default=None,
                        help="choose the checkpoint number to use (defaults to latest available checkpoint)")

    parser.add_argument('--exclude-tactics', '--exclude_tactics',
                        type=Path,
                        default=None,
                        help="a list of tactic names to exclude from predictions")

    parser.add_argument('--paranoic-data-server', '--paranoic_data_server',
                        default=False,
                        action='store_true',
                        help="Makes data_server check its inner consistency on each update")

    parser.add_argument('--cpu-thread-count', '--cpu_thread_count',
                        type=int,
                        default=0,
                        help="number of cpu threads to use tensorflow to use (automatic by default)")
    
    parser.add_argument('--pred-profiler-logdir', '--pred_profiler_logdir',
                        type=Path, default=None,
                        help='Supply logdir to profile the predict steps')

    parser.add_argument('--pred-profiler-start', '--pred_profiler_start',
                        type=int, default=10,
                        help='Prediction step to start profiling (default: 10).')
    
    parser.add_argument('--pred-profiler-end', '--pred_profiler_end',
                        type=int, default=15,
                        help='Prediction step to stop profiling (exclusive) (default: 15).')

    parser.add_argument('--def-profiler-logdir', '--def_profiler_logdir',
                        type=Path, default=None,
                        help='Supply logdir to profile the definition steps')

    parser.add_argument('--def-profiler-start', '--def_profiler_start',
                        type=int, default=10,
                        help='Defintion step to start profiling (default: 10).')
    
    parser.add_argument('--def-profiler-end', '--def_profiler_end',
                        type=int, default=15,
                        help='Definition step to stop profiling (exclusive) (default: 15).')
    
    return parser.parse_args()

def load_model(config: argparse.Namespace, log_levels: dict) -> Predict:

    tf_env_log_levels={'debug':'0',
                   'verbose':'0',
                   'info':'0',
                   'summary':'1',
                   'warning':'1',
                   'error':'2',
                   'critical':'3'}

    # TF sometimes uses the warnings module.
    # Here we redirect them to the "py.warnings" logger.
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(int(log_levels[config.tf_log_level]))
    # Turn off (or on) TF logging BEFORE importing tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_env_log_levels[config.tf_log_level]
    if config.arch == 'tfgnn':
        logger.info("importing tensorflow...")
        import tensorflow as tf
        tf.get_logger().setLevel(int(log_levels[config.tf_log_level]))
        tf.config.run_functions_eagerly(config.tf_eager)
        tf.config.threading.set_inter_op_parallelism_threads(
            config.cpu_thread_count
        )

        if config.exclude_tactics is not None:
            with Path(config.exclude_tactics).open('r') as yaml_file:
                exclude_tactics = yaml.load(yaml_file, Loader=yaml.SafeLoader)
            logger.info(f'excluding tactics {exclude_tactics}')
        else:
            exclude_tactics = None

        logger.info("importing TFGNNPredict class...")
        from graph2tac.tfgnn.predict import TFGNNPredict
        model = TFGNNPredict(log_dir=Path(config.model).expanduser().absolute(),
                             tactic_expand_bound=config.tactic_expand_bound,
                             search_expand_bound=config.search_expand_bound,
                             debug_dir=config.debug_predict,
                             checkpoint_number=config.checkpoint_number,
                             exclude_tactics=exclude_tactics)
    elif config.arch == 'hmodel':
        logger.info("importing HPredict class..")
        from graph2tac.loader.hmodel import HPredict
        model = HPredict(
            checkpoint_dir=Path(config.model).expanduser().absolute(),
            tactic_expand_bound=config.tactic_expand_bound,
            search_expand_bound=config.search_expand_bound,
            debug_dir=config.debug_predict
        )
    else:
        raise Exception(f'the provided model architecture {config.arch} is not supported')

    logger.info(f"initializing predict network from {Path(config.model).expanduser().absolute()}")
    return model

def main_with_return_value() -> ResponseHistory:
    sys.setrecursionlimit(10000)
    config = parse_args()

    if config.record is not None:
        logger.warning(f"WARNING!!!! the file {config.record} was provided to record messages for debugging purposes. The capnp bin messages will be recorded to {config.record}. Please do not provide --record if you do not want to record messages for later debugging and inspection purposes!")
    
    log_levels = {
        'debug':'10',
        'verbose':'15',
        'info':'20',
        'summary':'25',
        'warning':'30',
        'error':'40',
        'critical':'50',
    }

    os.environ['G2T_LOG_LEVEL'] = log_levels[config.log_level]
    logger.setLevel(int(log_levels[config.log_level]))

    # This process uuid is used in logging as an identifier for this process.
    # There is a small probability of a collision with another process, but it is unlikely.
    process_uuid = uuid.uuid1()
    logger.info(f"UUID: {process_uuid}")

    log_cnts = LoggingCounters(process_uuid=process_uuid)
    profiler = Profiler(
        logdir={"pred": config.pred_profiler_logdir, "def": config.def_profiler_logdir},
        start={"pred": config.pred_profiler_start, "def": config.def_profiler_start},
        end={"pred": config.pred_profiler_end, "def": config.def_profiler_end},
    )
    with log_cnts.measure_build_network_time():
        model = load_model(config, log_levels)
    response_history = ResponseHistory(recording_on=(config.replay_file is not None))  # only record response history if replay is on
    predict_server = PredictServer(model, config, log_cnts, profiler, response_history)

    if config.record is not None:
        record_context = open(config.record, 'wb')
    else:
        record_context = contextlib.nullcontext()

    with record_context as record_file:
        if config.replay_file is not None:
            predict_server.start_prediction_loop_with_replay(config.replay_file, record_file)
        elif config.tcp:
            logger.info(f"starting tcp/ip server on port {config.port}")
            addr = (config.host, config.port)
            if socket.has_dualstack_ipv6():
                try:
                    server_sock = socket.create_server(
                        addr,
                        family=socket.AF_INET6, dualstack_ipv6=True
                    )
                except OSError:
                    server_sock = socket.create_server(addr)
            else:
                server_sock = socket.create_server(addr)
            try:
                server_sock.listen(1)
                logger.info(f"tcp/ip server is listening on {config.port}")
                while True:
                    capnp_socket, remote_addr = server_sock.accept()
                    logger.info(f"coq client connected {remote_addr}")

                    predict_server.start_prediction_loop(capnp_socket, record_file)
                    logger.info(f"coq client disconnected {remote_addr}")
            finally:
                logger.info(f'closing the server on port {config.port}')
                server_sock.close()
        else:
            logger.info("starting stdin server")
            capnp_socket = socket.socket(fileno=sys.stdin.fileno())
            predict_server.start_prediction_loop(capnp_socket, record_file)
    
    return response_history  # return for testing purposes

def main():
    main_with_return_value()

if __name__ == '__main__':
    main()
