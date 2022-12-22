from collections.abc import Iterable
from contextlib import contextmanager
import sys
import signal
import socket
from pathlib import Path
import numpy as np
import pickle
import tqdm

from py_data_server import AbstractDataServer, DataServer
from pytact import graph_api_capnp
from pytact.data_reader import online_definitions_initialize, online_data_predict, capnp_message_generator
from pytact.graph_api_capnp_cython import PredictionProtocol_Request_Reader
from graph2tac.loader.data_classes import *

def apply_temperature(confidences, temperature):
    """
    Apply the temperature,
    assuming the truncated tail of the probability distribution
    is clustered in one unseen element
    """
    res = np.array(confidences, np.float64)**(1/temperature)
    res /= res.sum() + (1-sum(confidences))**(1/temperature)

# TODO: add logging messages
# TODO: LoggingCounters

class PredictServer(AbstractDataServer):
    def __init__(self,
                 model,
                 config,
    ):
        max_subgraph_size = model.get_max_subgraph_size()
        bfs_option = True
        stop_at_definitions = True
        super().__init__(
            max_subgraph_size = max_subgraph_size,
            bfs_option = bfs_option,
            stop_at_definitions = stop_at_definitions,
        )
        self.model = model
        self.config = predict_config
        self.current_definitions = None # only with a coq context

        self._tactic_i_to_numargs = list(model.get_tactic_index_to_numargs())
        self._tactic_i_to_hash = list(model.get_tactic_index_to_hash())
        self._tactic_to_i = {
            h : i
            for i,h in enumerate(self._tactic_i_to_hash)
        }
        #self._tactic_i_to_bytes = [b'UNKNOWN']*len(self._tactic_i_to_hash)
        self._tactic_i_to_bytes = list(model.get_tactic_index_to_string())

        self._node_i_to_name = model.get_label_to_name()
        self._node_i_in_spine = model.get_label_in_spine()
        self._num_train_nodes = len(self._node_i_to_name)

        del self._def_node_to_i # not usable without a coq_context
        self._def_name_to_i = {
            name : i
            for i,name in enumerate(self._node_i_to_name)
            if i >= self._base_node_label_num and self._node_i_in_spine[i]
        }

    def _enter_coq_context(self, definitions, tactics):
        self.current_definitions = definitions

        # tactics

        self.current_allowed_tactics = []
        for t in tactics:
            i = self._tactic_to_i.get(t.ident, None)
            if i is not None:
                self.current_allowed_tactics.append(i)

        # definitions

        self._globarg_i_to_nodeid = []
        global_context = []
        self._def_node_to_i = dict()
        for d in definitions.definitions:
            i = self._def_name_to_i.get(d.name, None)
            if i is None:
                i = self._register_definition(d)
            else:
                self._def_node_to_i[d.node] = i
            self._globarg_i_to_nodeid.append(d.node.nodeid)
            global_context.append(i)

        global_context = np.array(global_context, dtype = np.int32)
        self.global_context = global_context
        self.model.initialize(global_context)

        # definition recalculation

        if update_all_definitions:
            def_clusters_for_update = list(definitions.clustered_definitions)
            logger.info(f"Prepared for update all {len(def_clusters_for_update)} definition clusters")
        elif update_new_definitions:
            def_clusters_for_update = [
                cluster for cluster in definitions.clustered_definitions
                if self._def_node_to_i[cluster[0].node] >= self._num_train_nodes
            ]
            logger.info(f"Prepared for update {len(def_clusters_for_update)} definition clusters containing unaligned definitions")
        else:
            def_clusters_for_update = []
            logger.info(f"No update of the definition clusters requested")

        if def_clusters_for_update and self.config.proggress_bar:
            def_clusters_for_update = tqdm.tqdm(def_clusters_for_update)

        for cluster in def_clusters_for_update:
            cluster_state = self.cluster_to_graph(cluster)
            self.model.compute_new_definitions([cluster_state])

        logger.info(f"Definition clusters updated.")

    def _exit_coq_context(self):
        self.current_definitions = None

        del self.current_allowed_tactics
        del self._globarg_i_to_nodeid
        del self._def_node_to_i
        del self._node_i_to_name[self._num_train_nodes:]
        del self._node_i_in_spine[self._num_train_nodes:]

    @contextmanager
    def coq_context(self, msg):
        with online_definitions_initialize(msg.graph, msg.representative) as definitions:
            self._enter_coq_context(definitions, msg.tactics)
            yield
            self._exit_coq_context()

    def _decode_action(self, action, confidence, proof_state):
        res = dict()
        tactic = action[0,0]
        arguments = []
        for arg_type, arg_index in action[1:]:
            if arg_type == 0: # local argument
                arguments.append({'term' : {
                    'depIndex' : 0,
                    'nodeIndex': proof_state.context[arg_index].nodeid,
                }})
            else:
                arguments.append({'term' : {
                    'depIndex' : 1,
                    'nodeIndex': int(self._globarg_i_to_nodeid[arg_index]),
                }})
                print(self._globarg_i_to_nodeid[arg_index])
                print(self.global_context[arg_index])
                print(self._node_i_to_name[self.global_context[arg_index]])
        res['tactic'] = {'ident' : int(self._tactic_i_to_hash[tactic])}
        print(res['tactic'], ':', self._tactic_i_to_bytes[tactic])
        res['confidence'] = confidence
        res['arguments'] = arguments
        return res

    def predict(self, msg):
        if self.current_definitions is None:
            raise Exception("Cannot predict outside 'with predict_server.coq_context()'")
        with online_data_predict(
                self.current_definitions,
                msg) as proof_state:

            root = proof_state.root
            graph, node_to_i = self._downward_closure(
                [root]
            )
            root_i = 0
            local_context = proof_state.context
            local_context_i = [node_to_i[n] for n in local_context]
            dynamic_global_context = np.arange(len(self.global_context), dtype=np.uint32)

            context = ProofstateContext(
                local_context=np.array(local_context_i, dtype = np.uint32),
                global_context=np.array(dynamic_global_context, dtype = np.uint32),
            )
            dummy_proofstate_info = ProofstateMetadata(b'dummy_proofstate_name', -1, True)

            actions, confidences = self.model.ranked_predictions(
                state=LoaderProofstate(
                    graph=graph,
                    root=root_i,
                    context=context,
                    metadata=dummy_proofstate_info
                ),
                tactic_expand_bound=self.config.tactic_expand_bound,
                total_expand_bound=self.config.total_expand_bound,
                allowed_model_tactics=self.current_allowed_tactics,
                available_global=None
            )
            confidences = apply_temperature(confidences, self.config.temperature)
            # use only top-k
            actions = actions[:self.config.search_expand_bound]
            confidences = confidences[:self.config.search_expand_bound]
            response = graph_api_capnp.PredictionProtocol.Response.new_message(
                prediction = [
                    self._decode_action(action, confidence, proof_state)
                    for action, confidence in zip(actions, confidences)
                ]
            )
            print(response)
            return response

    def check_alignment(msg):
        with online_definitions_initialize(
                    msg.graph,
                    msg.representative) as definitions:
            unaligned_tactics = [
                tactic.ident for tactic in msg.tactics
                if tactic.ident not in self._tactic_to_i
            ]
            unaligned_definitions = [
                d.node.nodeid for d in definitions.definitions
                if d.name not in self._def_name_to_i
            ]

            unaligned_tactics = sorted(unaligned_tactics)
            unaligned_definitions = sorted(unaligned_definitions)
            return {
                'unalignedTactics': unaligned_tactics,
                'unalignedDefinitions': unaligned_definitions,
            }

def prediction_loop(predict_server, capnp_socket, record_file):
    if predict_server.config.with_meter:
        capnp_socket = tqdm.tqdm(capnp_socket)
    message_generator = capnp_message_generator(capnp_socket, record_file)

    msg = next(message_generator, None)
    while msg is not None:
        if msg.is_predict:
            raise Exception('Predict message received without a preceding initialize message')
        elif msg.is_synchronize:
            response = graph_api_capnp.PredictionProtocol.Response.new_message(
                synchronized=msg.synchronize
            )
            response.write_packed(capnp_socket)
            msg = next(message_generator, None)
        elif msg.is_check_alignment:
            response = predict_server.check_alignment(msg.check_alignment)
            response.write_packed(capnp_socket)
            msg = next(message_generator, None)
        elif msg.is_initialize:
            with predict_server.coq_context(msg.initialize):
                response = graph_api_capnp.PredictionProtocol.Response.new_message(initialized=None)
                response.write_packed(capnp_socket)
                msg = next(message_generator, None)
                while msg is not None and msg.is_predict:
                    response = predict_server.predict(msg.predict)
                    response.write_packed(capnp_socket)
                    msg = next(message_generator, None)
        else:
            raise Exception("Capnp protocol error")

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='graph2tac Predict python tensorflow server')

    parser.add_argument('--tcp', action='store_true',
                        help='start python server on tcp/ip socket')

    parser.add_argument('--port', type=int,
                        default=33333,
                        help='run python server on this port')
    parser.add_argument('--host', type=str,
                        default='',
                        help='run python server on this local ip')

    parser.add_argument('--model', type=str,
                        default=None,
                        help='checkpoint directory of the model')

    parser.add_argument('--arch', type=str,
                        default='tf2',
                        help='the model architecture tfgnn or tf2 or hmodel (current default is tf2)')

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
                        'replayed through "pytact-fake-coq"')

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

    return parser.parse_args()

def load_model(config, log_levels):

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
    if config.arch == 'tf2':
        logger.info("importing tensorflow...")
        import tensorflow as tf
        tf.get_logger().setLevel(int(log_levels[config.tf_log_level]))
        tf.config.run_functions_eagerly(config.tf_eager)

        logger.info("importing TF2Predict class..")
        from graph2tac.tf2.predict import TF2Predict
        predict = TF2Predict(checkpoint_dir=Path(config.model).expanduser().absolute(), debug_dir=config.debug_predict)
    elif config.arch == 'tfgnn':
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
        predict = TFGNNPredict(log_dir=Path(config.model).expanduser().absolute(),
                               debug_dir=config.debug_predict,
                               checkpoint_number=config.checkpoint_number,
                               exclude_tactics=exclude_tactics)
    elif config.arch == 'hmodel':
        logger.info("importing HPredict class..")
        from graph2tac.loader.hmodel import HPredict
        predict = HPredict(checkpoint_dir=Path(config.model).expanduser().absolute(), debug_dir=config.debug_predict)
    else:
        Exception(f'the provided model architecture {config.arch} is not supported')

    logger.info(f"initializing predict network from {Path(config.model).expanduser().absolute()}")
    return model

def main():

    config = parse_args()

    if config.record is not None:
        os.makedirs(config.record, exist_ok=True)
        logger.warning(f"WARNING!!!! the directory {config.record} was provided to record messages for debugging purposes. The capnp bin messages will be recorded to {config.record}. Please do not provide --record if you do not want to record messages for later debugging and inspection purposes!")
    
    log_levels = {'debug':'10',
                  'verbose':'15',
                  'info':'20',
                  'summary':'25',
                  'warning':'30',
                  'error':'40',
                  'critical':'50'}


    os.environ['G2T_LOG_LEVEL'] = log_levels[config.log_level]
    logger.setLevel(int(log_levels[config.log_level]))

    # This process uuid is used in logging as an identifier for this process.
    # There is a small probability of a collision with another process, but it is unlikely.
    process_uuid = uuid.uuid1()
    logger.info(f"UUID: {process_uuid}")

    model = load_model(config, log_levels)
    predict_server = PredictServer(model, config)

    if config.record is not None:
        record_context = open(config.record, 'wb')
    else:
        record_context = contextlib.nullcontext()

    with record_context as record_file:
        if config.tcp:
            logger.info("starting stdin server")
            capnp_socket = socket.socket(fileno=sys.stdin.fileno())
            prediction_loop(predict_server, capnp_socket, record_file)
        else:
            logger.info(f"starting tcp/ip server on port {args.port}")
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
                logger.info(f"tcp/ip server is listening on {args.port}")
                while True:
                    capnp_socket, remote_addr = server_sock.accept()
                    logger.info(f"coq client connected {remote_addr}")

                    prediction_loop(predict_server, capnp_socket, record_file)
                    logger.info(f"coq client disconnected {remote_addr}")
            finally:
                print(f'closing the server on port {args.port}')
                server_sock.close()

if __name__ == '__main__':
    main()
