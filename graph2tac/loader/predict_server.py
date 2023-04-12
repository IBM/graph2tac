import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import sys
import socket
from pathlib import Path
from typing import BinaryIO, Optional
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
    confidences = np.array(confidences, np.float64)
    res = confidences**(1/temperature)
    res /= res.sum() + (1-confidences.sum())**(1/temperature)
    return res

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
    def convert_msg_to_dict(msg: CheckAlignmentResponse | TacticPredictionsGraph) -> dict:
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
    
    def record_response(self, msg: CheckAlignmentResponse | TacticPredictionsGraph):
        if self._recording_on:
            self.data["responses"].append(self.convert_msg_to_dict(msg))

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
    msg_idx: int = -1
    """Number of 'prediction' messages recieved since last 'intitialize' message"""
    t_predict: float = 0.0
    """Total time (in seconds) spent on predict since last 'initialize'"""
    n_predict: int = 0
    """Number of 'prediction' messages recieved since last 'intialize'"""
    max_t_predict: float = 0.0
    """Max time (in seconds) spent on single predict since last 'initialize'"""
    argmax_t_predict: int = -1
    """Index of last maximum time single predict since last 'initialize'"""
    t_coq: float = 0.0
    """Total time (in seconds) spent waiting on Coq since last 'initialize'"""
    n_coq: int = 0
    """Number of 'prediction' requests made by Coq"""
    build_network_time: float = 0.0
    """Time (in seconds) to build network when processing most recent 'intitialize' message"""
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

    def summary_log_message(self) -> str:
        summary_data = {
            "UUID" : f"{self.process_uuid}",
            "Session" : f"{self.session_idx}",
            "Theorem" : f"{self.thm_idx}",
            "Annotation" : f"{self.thm_annotation}",
            "Initialize|Network build time (s)" : f"{self.build_network_time:.6f}",
            "Initialize|Def clusters to update" : f"{self.n_def_clusters_updated}",
            "Initialize|Def update time (s)" : f"{self.update_def_time:.6f}",
            "Predict|Messages cnt" : f"{self.msg_idx}",
            "Predict|Avg predict time (s/msg)" : 
                f"{self.t_predict/self.n_predict:.6f}" if self.n_predict else "NA",
            "Predict|Max predict time (s/msg)" : f"{self.max_t_predict:.6f}",
            "Predict|Max time predict msg ix" : f"{self.argmax_t_predict}",
            "Predict|Avg Coq wait time (s/msg)":
                f"{self.t_coq/self.n_coq:.6f}" if self.n_coq else "NA",
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
    def __init__(self, graph_constants: GraphConstants):
        super().__init__(graph_constants.data_config)

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
        self._last_num_nodes = None
        self._context_stack = []

        # extra data
        
        self._def_ident_to_i = {
            name : i
            for i,name in enumerate(self._node_i_to_ident)
            if i >= self._base_node_label_num
        }

        self._def_node_to_i = None
        self._globarg_i_to_node = None
        self.global_defs = None

    def align_definitions(self, definitions):

        self._context_stack.append((
            self.global_defs,
            self._globarg_i_to_node,
            self._def_node_to_i,
            self._last_num_nodes
        ))
        self._last_num_nodes = len(self._node_i_to_name)

        self._globarg_i_to_node = []
        global_defs = []
        self._def_node_to_i = dict()
        for d in definitions:
            i = self._def_ident_to_i.get(d.node.identity, None)
            if i is None:
                i = self._register_definition(d)
            else:
                self._def_node_to_i[d.node] = i
            self._globarg_i_to_node.append(d.node)
            #self._globarg_i_to_node[i] = d.node
            global_defs.append(i)

        global_defs = np.array(global_defs, dtype = np.int32)
        self.global_defs = global_defs

    def is_untrained_definition(self, node):
        return self._def_node_to_i[node] >= self._train_num_nodes

    def pop(self):
        del self._node_i_to_name[self._last_num_nodes:]
        del self._node_i_to_ident[self._last_num_nodes:]
        del self._node_i_in_spine[self._last_num_nodes:]

        (
            self.global_defs,
            self._globarg_i_to_node,
            self._def_node_to_i,
            self._last_num_nodes
        ) = self._context_stack.pop()

    def proofstate(self, root, local_context):
        graph, node_to_i = self._downward_closure([root])
        root_i = 0
        local_context_i = [node_to_i[n] for n in local_context]
        # dynamic_global_context = np.arange(len(self.global_defs), dtype=np.uint32)

        context = ProofstateContext(
            local_context=np.array(local_context_i, dtype = np.uint32),
            global_context=np.array(self.global_defs, dtype = np.uint32),
        )
        dummy_proofstate_info = ProofstateMetadata(b'dummy_proofstate_name', -1, True)

        return LoaderProofstate(
            graph=graph,
            root=root_i,
            context=context,
            metadata=dummy_proofstate_info
        )

class PredictServer:
    def __init__(self,
                 model: Predict,
                 config: argparse.Namespace,
                 log_cnts: LoggingCounters,
                 response_history: ResponseHistory,
    ):
        self.data_server = DynamicDataServer(model.graph_constants)
        self.model = model
        self.config = config
        self.log_cnts = log_cnts
        self.current_allowed_tactics = None
        self.response_history = response_history
        self.allowed_tactics_stack = []

    def _enter_coq_context(self, definitions: OnlineDefinitionsReader, tactics):

        # tactics

        self.allowed_tactics_stack.append(self.current_allowed_tactics)
        self.current_allowed_tactics = []
        for t in tactics:
            i = self.data_server._tactic_to_i.get(t.ident, None)
            if i is not None:
                self.current_allowed_tactics.append(i)

        # definitions

        self.data_server.align_definitions(definitions.definitions())

        if len(self.data_server.global_defs) > 0:
            static_global_context = np.arange(max(self.data_server.global_defs)+1, dtype=np.uint32)
        else:
            static_global_context = np.arange(1, dtype=np.uint32)
        logger.info(f"initializing network with {len(static_global_context)} defs in global context")
        t0 = time.time()

        self.model.initialize(static_global_context)
        t1 = time.time()
        self.log_cnts.build_network_time = t1-t0
        logger.info(f"Building network model completed in {self.log_cnts.build_network_time:.6f} seconds")

        # definition recalculation
        if self.config.update == "all":
            def_clusters_for_update = list(definitions.clustered_definitions(full = False))
            print("updating", len(def_clusters_for_update), "all definitions")
            prev_defined_nodes = self.data_server._base_node_label_num
            logger.info(f"Prepared for update all {len(def_clusters_for_update)} definition clusters")
        elif self.config.update == "new":
            def_clusters_for_update = [
                cluster for cluster in definitions.clustered_definitions()
                if self.data_server.is_untrained_definition(cluster[0].node)
            ]
            prev_defined_nodes = self.data_server._last_num_nodes
            logger.info(f"Prepared for update {len(def_clusters_for_update)} definition clusters containing unaligned definitions")
        else:
            assert self.config.update is None
            def_clusters_for_update = []
            prev_defined_nodes = None
            logger.info(f"No update of the definition clusters requested")
        # definitions.clustered_definitions is in reverse order of dependencies, so we reverse our list
        def_clusters_for_update.reverse()

        t0 = time.time()
        if def_clusters_for_update:
            logger.info(f"Updating definition clusters...")
            if self.config.progress_bar:
                def_clusters_for_update = tqdm.tqdm(def_clusters_for_update)

            defined_nodes = set()
            for cluster in def_clusters_for_update:
                cluster_state = self.data_server.cluster_to_graph(cluster)
                new_defined_nodes = cluster_state.graph.nodes[:cluster_state.num_definitions]
                used_nodes = cluster_state.graph.nodes[cluster_state.num_definitions:]
                # for n in used_nodes:
                #     assert n < prev_defined_nodes or n in defined_nodes, (
                #         f"Definition clusters out of order. "
                #         f"Attempting to compute definition embedding for node labels {new_defined_nodes} "
                #         f"({cluster_state.definition_names}) without first computing "
                #         f"the definition embedding for node label {n} used in that definition."
                #     )   
                # for n in new_defined_nodes:
                #     assert n >= prev_defined_nodes and n not in defined_nodes, (
                #         f"Something is wrong with the definition clusters. "
                #         f"Attempting to compute definition embedding for node labels {new_defined_nodes} "
                #         f"({cluster_state.definition_names}) "
                #         f"for which node label {n} has already been computed."
                #     )
                #     defined_nodes.add(n)

                print("Update:", new_defined_nodes)
                self.model.compute_new_definitions([cluster_state])

            logger.info(f"Definition clusters updated.")
        else:
            logger.info(f"No cluster to update.")

        t1 = time.time()
        self.log_cnts.n_def_clusters_updated = len(def_clusters_for_update)
        self.log_cnts.update_def_time = t1 - t0

    def _exit_coq_context(self):

        self.current_allowed_tactics = self.allowed_tactics_stack.pop()
        self.data_server.pop()

    @contextmanager
    def coq_context(self, msg: GlobalContextMessage):
        self._enter_coq_context(msg.definitions, msg.tactics)
        yield
        self._exit_coq_context()

    def _decode_action(self, action: NDArray[np.int_], confidence: np.float_, proof_state: ProofState) -> TacticPredictionGraph:
        tactic = action[0,0]
        arguments = []
        for arg_type, arg_index in action[1:]:
            if arg_type == 0: # local argument
                arguments.append(proof_state.context[arg_index])
            else:
                arguments.append(self.data_server._globarg_i_to_node[arg_index])

        return TacticPredictionGraph(
            int(self.data_server._tactic_i_to_hash[tactic]),
            arguments,
            float(confidence),
        )

    def predict(self, proof_state: ProofState) -> TacticPredictionsGraph:
        if self.current_allowed_tactics is None:
            raise Exception("Cannot predict outside 'with predict_server.coq_context()'")

        actions, confidences = self.model.ranked_predictions(
            state=self.data_server.proofstate(proof_state.root, proof_state.context),
            tactic_expand_bound=self.config.tactic_expand_bound,
            total_expand_bound=self.config.total_expand_bound,
            allowed_model_tactics=self.current_allowed_tactics,
            available_global=None
        )
        confidences = apply_temperature(confidences, self.config.temperature)
        # use only top-k
        actions = actions[:self.config.search_expand_bound]
        confidences = confidences[:self.config.search_expand_bound]
        return TacticPredictionsGraph([
            self._decode_action(action, confidence, proof_state)
            for action, confidence in zip(actions, confidences)
        ])

    def check_alignment(self, msg: GlobalContextMessage) -> CheckAlignmentResponse:
        unaligned_tactics = [
            tactic.ident for tactic in msg.tactics
            if tactic.ident not in self.data_server._tactic_to_i
        ]
        unaligned_definitions = [
            d for d in msg.definitions.definitions()
            if d.node.identity not in self.data_server._def_ident_to_i
        ]

        return CheckAlignmentResponse(
            unknown_tactics = unaligned_tactics,
            unknown_definitions = unaligned_definitions,
        )

def prediction_loop(predict_server: PredictServer, context: GlobalContextMessage):

    prediction_requests = context.prediction_requests
    if predict_server.config.with_meter:
        message_iterator = tqdm.tqdm(prediction_requests)
    else:
        message_iterator = prediction_requests
    log_cnts = predict_server.log_cnts
    log_cnts.session_idx += 1
    log_cnts.thm_idx = -1

    with predict_server.coq_context(context):

        for msg in message_iterator:

            if isinstance(msg, ProofState):
                log_cnts.n_coq += 1
                log_cnts.msg_idx += 1

                t0 = time.time()

                # important line -- getting prediction
                response = predict_server.predict(msg)

                t1 = time.time()
                if t1-t0 > log_cnts.max_t_predict:
                    log_cnts.max_t_predict = t1-t0
                    log_cnts.argmax_t_predict = log_cnts.msg_idx

                log_cnts.t_predict += (t1 - t0)
                log_cnts.n_predict += 1

                # important line -- sending prediction
                predict_server.response_history.record_response(response)
                prediction_requests.send(response)

            elif isinstance(msg, CheckAlignmentMessage):
                logger.info("checkAlignment request")
                response = predict_server.check_alignment(context)
                logger.info("sending checkAlignment response to coq")
                predict_server.response_history.record_response(response)
                prediction_requests.send(response)

            elif isinstance(msg, GlobalContextMessage):
                prediction_loop(predict_server, msg)

            else:
                raise Exception("Capnp protocol error")

def start_prediction_loop(predict_server: PredictServer, capnp_socket: socket.socket, record_file: Optional[BinaryIO]):
    prediction_loop(predict_server, capnp_message_generator(capnp_socket, record_file))

def start_prediction_loop_with_replay(predict_server: PredictServer, replay_file: Path, record_file: Optional[BinaryIO]):
    with open(replay_file, "rb") as f:
        prediction_loop(predict_server, capnp_message_generator_from_file(f, record=record_file))

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

    parser.add_argument('--log_level', type=str,
                        default='info',
                        help='debug | verbose | info | summary | warning | error | critical')

    parser.add_argument('--tf_log_level', type=str,
                        default='info',
                        help='debug | verbose | info | summary | warning | error | critical')

    parser.add_argument('--record',
                        type = str,
                        default = None,
                        help='Record all exchanged messages to the specified file, so that they can later be ' +
                        'replayed through "pytact-fake-coq" or --replay')

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

    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument('--update_no_definitions', '--update-no-definitions',
        action='store_const', dest='update', const=None, default=None,
        help='for new definitions (not learned during training) use default embeddings (default)'
    )
    update_group.add_argument('--update_new_definitions', '--update-new-definitions',
        action='store_const', dest='update', const='new',
        help='for new definitions (not learned during training) use embeddings calculated from the model'
    )
    update_group.add_argument('--update_all_definitions', '--update-all-definitions',
        action='store_const', dest='update', const='all',
        help='overwrite all definition embeddings with new embeddings calculated from the model'
    )

    parser.add_argument('--progress_bar',
                        default=False,
                        action='store_true',
                        help="show the progress bar of update definition clusters")

    parser.add_argument('--tf_eager',
                        default=False,
                        action='store_true',
                        help="with tf_eager=True activated network may initialize faster but run slower, use carefully if you need")

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

        if config.exclude_tactics is not None:
            with Path(config.exclude_tactics).open('r') as yaml_file:
                exclude_tactics = yaml.load(yaml_file, Loader=yaml.SafeLoader)
            logger.info(f'excluding tactics {exclude_tactics}')
        else:
            exclude_tactics = None

        logger.info("importing TFGNNPredict class...")
        from graph2tac.tfgnn.predict import TFGNNPredict
        model = TFGNNPredict(log_dir=Path(config.model).expanduser().absolute(),
                               debug_dir=config.debug_predict,
                               checkpoint_number=config.checkpoint_number,
                               exclude_tactics=exclude_tactics)
    elif config.arch == 'hmodel':
        logger.info("importing HPredict class..")
        from graph2tac.loader.hmodel import HPredict
        model = HPredict(checkpoint_dir=Path(config.model).expanduser().absolute(), debug_dir=config.debug_predict)
    else:
        raise Exception(f'the provided model architecture {config.arch} is not supported')

    logger.info(f"initializing predict network from {Path(config.model).expanduser().absolute()}")
    return model

def main() -> ResponseHistory:

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
    model = load_model(config, log_levels)
    response_history = ResponseHistory(recording_on=(config.replay_file is not None))  # only record response history if replay is on
    predict_server = PredictServer(model, config, log_cnts, response_history)

    if config.record is not None:
        record_context = open(config.record, 'wb')
    else:
        record_context = contextlib.nullcontext()

    with record_context as record_file:
        if config.replay_file is not None:
            start_prediction_loop_with_replay(predict_server, config.replay_file, record_file)
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

                    start_prediction_loop(predict_server, capnp_socket, record_file)
                    logger.info(f"coq client disconnected {remote_addr}")
            finally:
                logger.info(f'closing the server on port {config.port}')
                server_sock.close()
        else:
            logger.info("starting stdin server")
            capnp_socket = socket.socket(fileno=sys.stdin.fileno())
            start_prediction_loop(predict_server, capnp_socket, record_file)
    
    return response_history  # return for testing purposes

if __name__ == '__main__':
    main()
