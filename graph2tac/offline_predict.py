from graph2tac.loader.data_classes import DatasetConfig, LoaderAction
from graph2tac.loader.data_server import AbstractDataServer, DataServer, IterableLen
from graph2tac.loader.predict_server import apply_temperature, load_model
from graph2tac.common import logger
import json
from pathlib import Path
import argparse
import os
import tqdm

class AlignedDataServer(DataServer):
    def __init__(self, data_dir : Path, graph_constants):

        eval_theories_fname = data_dir / "eval_theories.txt"
        if eval_theories_fname.is_file():
            with eval_theories_fname.open() as f:
                prefixes = [
                    line.strip()
                    for line in f
                ]
            split_method = "file_prefix"
            split = [prefixes]
            logger.info(f"Evaluating on {len(prefixes)} prefixes found in {eval_theories_fname}")
        else:
            split_method = "hash"
            split = {'proportions' : [0,1], 'random_seed' : 0}
            logger.info(f"No prefix file {eval_theories_fname} --> evaluating on the entire dataset")

        dataset_config = DatasetConfig(
            data_config = graph_constants.data_config,
            split_method = split_method,
            split = split,
            restrict_to_spine = False,
            exclude_none_arguments = False,
            exclude_not_faithful = False,
            required_tactic_occurrence = 1,
            shuffle_random_seed = 0,
        )

        self._ident_to_node_i = {
            ident : node_i
            for node_i, ident in enumerate(graph_constants.label_to_ident)
            if node_i >= graph_constants.base_node_label_num
        }
        self._train_node_i_to_name = graph_constants.label_to_names
        self._train_node_i_to_ident = graph_constants.label_to_ident
        self._train_node_i_in_spine = graph_constants.label_in_spine
        self._train_tactic_i_to_numargs = graph_constants.tactic_index_to_numargs
        self._train_tactic_i_to_string = graph_constants.tactic_index_to_string
        self._train_tactic_i_to_hash = graph_constants.tactic_index_to_hash
        self.num_train_nodes = len(graph_constants.label_to_ident)
        self.num_train_tactics = len(graph_constants.tactic_index_to_numargs)
        super().__init__(data_dir, dataset_config)

        logger.info("Checking alignment...")
        num_aligned_nodes = sum(self._node_i_aligned)
        num_unaligned_nodes = sum(not x for x in self._node_i_aligned)
        num_new_nodes = len(self._node_i_to_name) - len(self._train_node_i_to_name)
        num_aligned_tactics = sum(self._tactic_i_aligned)
        num_unaligned_tactics = sum(not x for x in self._tactic_i_aligned)
        num_new_tactics = len(self._tactic_i_to_string) - len(self._train_tactic_i_to_string)
        logger.info(f"Aligned nodes: {num_aligned_nodes}")
        logger.info(f"Unaligned nodes: {num_unaligned_nodes}")
        logger.info(f"New nodes: {num_new_nodes}")
        logger.info(f"Aligned tactics: {num_aligned_tactics}")
        logger.info(f"Unaligned tactics: {num_unaligned_tactics}")
        logger.info(f"New tactics: {num_new_tactics}")

    def _register_definition(self, d):
        node = d.node
        if node in self._node_to_node_i:
            return self._node_to_node_i[node]
        if len(self._node_i_to_name) < self.num_train_nodes:
            n = len(self._node_i_to_name)
            self._node_i_to_name.extend(self._train_node_i_to_name[n:])
            self._node_i_to_ident.extend(self._train_node_i_to_ident[n:])
            self._node_i_in_spine.extend(self._train_node_i_in_spine[n:])
            self._node_i_aligned = [True]*n + [False]*len(self._node_i_to_name[n:])
        i = self._ident_to_node_i.get(node.identity)
        if i is not None:
            self._node_to_node_i[node] = i
            self._node_i_aligned[i] = True
            return i
        new_i = len(self._node_i_to_name)
        self._node_to_node_i[node] = new_i
        self._node_i_to_name.append(d.name)
        self._node_i_to_ident.append(d.node.identity)
        self._node_i_in_spine.append(False)
        return new_i

    def _register_tactic(self, tactic_usage):
        if len(self._tactic_i_to_numargs) == 0:
            self._tactic_i_to_numargs.extend(self._train_tactic_i_to_numargs)
            self._tactic_i_to_string.extend(self._train_tactic_i_to_string)
            self._tactic_i_to_hash.extend(self._train_tactic_i_to_hash)
            self._tactic_i_count.extend([100]*len(self._tactic_i_to_numargs))
            self._tactic_i_aligned = [False]*len(self._tactic_i_to_numargs)
            self._tactic_to_i = {
                tactic : tactic_i
                for tactic_i, tactic in enumerate(self._train_tactic_i_to_hash)
            }
        tactic_i = AbstractDataServer._register_tactic(self, tactic_usage)
        if tactic_i is not None and tactic_i < self.num_train_tactics:
            self._tactic_i_aligned[tactic_i] = True

    def is_new_definition(self, cluster_i):
        cluster = self._def_clusters[cluster_i]
        node0_i = self._node_to_node_i[cluster[0].node]
        return node0_i >= self.num_train_nodes

    def new_definitions(self):
        ids = self.def_cluster_indices()
        ids = [
            i for i in ids
            if self.is_new_definition(i)
        ]
        return IterableLen(map(self.def_cluster_subgraph, ids), len(ids))

    def all_definitions(self):
        ids = self.def_cluster_indices()
        return IterableLen(map(self.def_cluster_subgraph, ids), len(ids))

    def prediction_to_dict(self, action, confidence):
        tactic_i = action[0,0]
        args = [[],[]]
        global_args = []
        for arg_type, arg_index in action[1:]:
            args[arg_type].append(arg_index)
            args[1-arg_type].append(arg_index)
        res = self.action_to_dict(LoaderAction(
            tactic_id = tactic_i,
            local_args = args[0],
            global_args = args[1],
        ))
        res['confidence'] = float(confidence)
        return res

    def action_to_dict(self, action : LoaderAction):
        arguments = [None]*len(action.local_args)
        for i,arg in enumerate(action.local_args):
            if arg < 0: continue
            arguments[i] = {
                'type' : 'local',
                'index' : int(arg)
            }
        for i,arg in enumerate(action.global_args):
            if arg < 0: continue
            arguments[i] = {
                'type' : 'global',
                'index' : int(arg),
                'name' : str(self._node_i_to_name[arg]),
                'hash' : self._node_i_to_ident[arg],
            }

        return {
            'tactic_index' : int(action.tactic_id),
            'tactic_string' : str(self._tactic_i_to_string[action.tactic_id]),
            'tactic_hash' : self._tactic_i_to_hash[action.tactic_id],
            'arguments' : arguments,
        }
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='graph2tac Run offline predictions')

    # required by load_model()
    parser.add_argument('--model', type=str, required=True,
                        help='checkpoint directory of the model')

    # required by load_model()
    parser.add_argument('--arch', type=str,
                        default='tfgnn',
                        choices=['tfgnn', 'hmodel'],
                        help='the model architecture tfgnn or hmodel (current default is tfgnn)')

    parser.add_argument('--log-level', '--log_level', type=str,
                        default='info',
                        help='debug | verbose | info | summary | warning | error | critical')

    # required by load_model()
    parser.add_argument('--tf-log-level', '--tf_log_level', type=str,
                        default='info',
                        help='debug | verbose | info | summary | warning | error | critical')

    # required by load_model()
    parser.add_argument('--tactic-expand-bound', '--tactic_expand_bound',
                        type=int,
                        default=8,
                        help="tactic_expand_bound for ranked argument search")

    # required by load_model()
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
                        help="show the progress bars")

    # required by load_model()
    parser.add_argument('--tf-eager', '--tf_eager',
                        default=False,
                        action='store_true',
                        help="with tf_eager=True activated network may initialize faster but run slower, use carefully if you need")

    # needed for apply_temperature()
    parser.add_argument('--temperature',
                        type=float,
                        default=1.0,
                        help="temperature to apply to the probability distributions returned by the model")

    # required by load_model()
    parser.add_argument('--debug-predict', '--debug_predict',
                        type=Path,
                        default=None,
                        help="set this flag to run Predict in debug mode")

    # required by load_model()
    parser.add_argument('--checkpoint-number', '--checkpoint_number',
                        type=int,
                        default=None,
                        help="choose the checkpoint number to use (defaults to latest available checkpoint)")

    # required by load_model()
    parser.add_argument('--exclude-tactics', '--exclude_tactics',
                        type=Path,
                        default=None,
                        help="a list of tactic names to exclude from predictions")

    # required by load_model()
    parser.add_argument('--cpu-thread-count', '--cpu_thread_count',
                        type=int,
                        default=0,
                        help="number of cpu threads to use tensorflow to use (automatic by default)")

    # dataset specification
    dataset_source = parser.add_mutually_exclusive_group(required=True)
    dataset_source.add_argument("--data-dir", metavar="DIRECTORY", type=Path,
                                help="Location of the evaluated capnp dataset")
    
    # output file
    output_file = parser.add_mutually_exclusive_group(required=True)
    output_file.add_argument("--output-file", metavar="OUTPUT_FILE", type=Path,
                                help="File for output in jsonlines format")

    return parser.parse_args()

def main():
    config = parse_args()
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

    # load model
    model = load_model(config, log_levels)

    # load data
    data_server = AlignedDataServer(
        data_dir = config.data_dir,
        graph_constants = model.graph_constants,
    )

    if config.update is not None:
        if config.update == 'new':
            definitions = data_server.new_definitions()
            defined_nodes = set(range(data_server.num_train_nodes))
        elif config.update == 'all':
            definitions = data_server.all_definitions()
            defined_nodes = set(range(model.graph_constants.base_node_label_num))
        else: raise Exception(f"invalid config.update: {config.update}")

        def sanity_check(cluster_graph): # debugging function
            new_defined_nodes = cluster_graph.graph.nodes[:cluster_graph.num_definitions]
            used_nodes = cluster_graph.graph.nodes[cluster_graph.num_definitions:]
            for n in set(used_nodes):
                assert n in defined_nodes, (
                    f"Definition clusters out of order. "
                    f"Attempting to compute definition embedding for node labels {new_defined_nodes} "
                    f"({cluster_graph.definition_names}) without first computing "
                    f"the definition embedding for node label {n} used in that definition."
                )
            for n in new_defined_nodes:
                defined_nodes.add(n)
                # # Turns out that a definition can appear multiple times
                # # (a variable in a section)
                # assert n not in defined_nodes, (
                #     f"Something is wrong with the definition clusters. "
                #     f"Attempting to compute definition embedding for node labels {new_defined_nodes} "
                #     f"({cluster_graph.definition_names}) "
                #     f"for which node label {n} has already been computed."
                # )


        logger.info(f"Updating definition clusters.")
        if config.progress_bar: definitions = tqdm.tqdm(definitions)
        for cluster_graph in definitions:
            model.compute_new_definitions([cluster_graph])
            sanity_check(cluster_graph)
        logger.info(f"Definition clusters updated.")

    eval_data = data_server.data_valid()
    if config.progress_bar: eval_data = tqdm.tqdm(eval_data)
    logger.info(f"Evaluating proofstates.")
    with open(config.output_file, 'w') as f:
        all_tactics = list(range(model.graph_constants.tactic_num))
        for proofstate, true_action, i in eval_data:
            actions, confidences = model.ranked_predictions(
                proofstate,
                available_global = None,
                allowed_model_tactics = all_tactics,
            )
            confidences = apply_temperature(confidences, config.temperature)

            # use only top-k
            actions = actions[:config.search_expand_bound]
            confidences = confidences[:config.search_expand_bound]

            datapoint = {
                'data_server_index' : i,
                'name' : str(proofstate.metadata.name),
                'step' : proofstate.metadata.step,
                'is_faithful' : proofstate.metadata.is_faithful,
                'true_action': data_server.action_to_dict(true_action),
                'predictions': [
                    data_server.prediction_to_dict(action, confidence)
                    for action, confidence in zip(actions, confidences)
                ]
            }
            print(json.dumps(datapoint), file = f)

if __name__ == "__main__":
     main()
