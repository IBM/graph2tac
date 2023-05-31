from typing import Optional, List, Tuple

import tqdm
import hashlib
import argparse
import pickle
import numpy as np
from pathlib import Path

from graph2tac.loader.data_classes import DataConfig, DatasetConfig, LoaderAction, LoaderProofstate, LoaderDefinition
from graph2tac.loader.data_server import DataServer, SplitDisabled

from graph2tac.predict import Predict, predict_api_debugging


def my_hash(x: bytes):
    m = hashlib.sha256()
    m.update(x)
    return m.hexdigest()


def hash_nparray(array: np.ndarray):
    x = memoryview(array).tobytes()
    return my_hash(x)


def action_encode(action: LoaderAction,
                  local_context,
                  global_context):
    res = []
    for local_arg, global_arg in zip(action.local_args, action.global_args):
        if local_arg >= 0: res.append((0, local_context[local_arg]))
        elif global_arg >= 0: res.append((1, global_context[global_arg]))
        else: res.append((len(local_context), local_arg))

    return action.tactic_id, res


def my_hash_of(state: LoaderProofstate, with_context: bool):
    graph = state.graph
    graph_tuple = (graph.nodes, graph.edges, graph.edge_labels, graph.edge_offsets)
    what_to_hash = (*graph_tuple, state.context.local_context) if with_context else graph_tuple
    state_hash = my_hash(''.join(hash_nparray(x) for x in what_to_hash).encode())
    return state_hash


def args_decode(args,
                inverse_local_context,
                inverse_global_label,
                ):
    res = []
    inverse_map = (inverse_local_context, inverse_global_label)
    for arg in args:
        value = (arg[0], inverse_map[arg[0]].get(arg[1], -1))
        if value[1] == -1:
            # print("args", args, "inverse_map", inverse_map)
            return None
        res.append(value)
    return res


class Train:
    def __init__(self, data_dir: Path, output_dir: Path, max_subgraph_size, with_context, shuffled, dry):

        self._data_server = DataServer(
            data_dir=data_dir,
            dataset_config = DatasetConfig(
                data_config = DataConfig(
                    max_subgraph_size=max_subgraph_size,
                    bfs_option = True,
                    stop_at_definitions = True,
                    symmetrization = "bidirectional",
                    add_self_edges = True,
                ),
                split_method = "disabled",
                split = None,
                restrict_to_spine = False,
                exclude_none_arguments = False,
                exclude_not_faithful = False,
                required_tactic_occurrence = 1,
                shuffle_random_seed = 0,
            )
        )
        self._data = {}
        self.output_dir = output_dir
        self._tactic_index_to_hash = self._data_server.graph_constants().tactic_index_to_hash
        self._max_subgraph_size = max_subgraph_size
        self._with_context = with_context
        self._shuffled = shuffled
        self._dry = dry

    def train(self):
        none_counter = 0
        node_counter = 0
        edge_counter = 0
        context_counter = 0
        for state, action, idx in tqdm.tqdm(self._data_server.data_train(shuffled=self._shuffled)):
            node_counter += len(state.graph.nodes)
            edge_counter += len(state.graph.edges)
            context_counter += len(state.context.global_context)

            if self._dry:
                continue

            state_hash = my_hash_of(state, self._with_context)

            train_action = self._data.get(state_hash, [])
            try:
                train_action.append(action_encode(action,
                                              local_context=state.context.local_context,
                                              global_context=state.context.global_context))
            except IndexError:
                none_counter += 1


            self._data[state_hash] = train_action
        print(f"skipped {none_counter} proofstates with Nones")
        print(f"nodes {node_counter}, edges {edge_counter}, context elements {context_counter}")

        saved_model = {'graph_constants': self._data_server.graph_constants(),
                       'data': self._data,
                       'with_context': self._with_context,
                       'max_subgraph_size': self._max_subgraph_size}

        self.output_dir.mkdir(exist_ok=True, parents=True)
        pickle.dump(saved_model, open(self.output_dir / "hmodel.sav", 'wb'))
        print(f"model saved in {self.output_dir / 'hmodel.sav'}")

        return saved_model # return all data for testing purposes


class HPredict(Predict):
    def __init__(self,
                 checkpoint_dir: Path,
                 tactic_expand_bound: int = 20,
                 search_expand_bound: int = 1000,
                 debug_dir: Optional[Path] = None
    ):
        loaded_model = pickle.load(open(checkpoint_dir/'hmodel.sav', 'rb'))

        # initialize self._graph_constants
        super().__init__(
            graph_constants=loaded_model['graph_constants'],
            tactic_expand_bound=tactic_expand_bound,
            search_expand_bound=search_expand_bound,
            debug_dir=debug_dir,
        )

        self._data = loaded_model['data']
        self._label_to_index = None
        self._max_subgraph_size = loaded_model['max_subgraph_size']
        self._with_context = loaded_model['with_context']
        self._label_to_idx = dict()

    @predict_api_debugging
    def allocate_definitions(self, new_node_label_num : int) -> None:
        pass

    @predict_api_debugging
    def compute_new_definitions(self, clusters: List[LoaderDefinition]) -> None:
        """
        a list of cluster states on which we run dummy runs
        """
        pass

    @predict_api_debugging
    def ranked_predictions(self,
                           state: LoaderProofstate,
                           allowed_model_tactics: List[int],
                           available_global: Optional[np.ndarray] = None,
                           annotation: str = "",
                           debug: bool = False,
                           ) -> Tuple[np.ndarray, List]:
        state_hash = my_hash_of(state, self._with_context)
        inverse_local_context = {
            node_idx : i
            for i,node_idx in enumerate(state.context.local_context)
        }
        inverse_global_context = {
            node_idx : i
            for i,node_idx in enumerate(state.context.global_context)
        }

        predictions = self._data.get(state_hash, [])
        if len(predictions) > 0:
            pass
        else:
            if debug:
                print("HMODEL | STATE NOT NOT FOUND")
        decoded_predictions = []
        for prediction in predictions:
            tactic_idx, args = prediction

            args_decoded = args_decode(args,
                                       inverse_local_context=inverse_local_context,
                                       inverse_global_label=inverse_global_context)
            if args_decoded is not None:
                args_decoded = np.array(args_decoded, dtype=np.uint32).reshape((len(args_decoded), 2))
            if debug:
                print("HMODEL ", annotation, self.graph_constants.tactic_index_to_string[tactic_idx], "index into context:", args_decoded, end="")
            if tactic_idx in allowed_model_tactics:
                if args_decoded is not None:
                    decoded_predictions.append(np.concatenate([np.array([(tactic_idx, tactic_idx)], dtype=np.uint32), args_decoded]))
                else:
                    if debug:
                        print(": arg not allowed")
            else:
                if debug:
                    print(": tactic not allowed")

        if debug:
            print("predictions", decoded_predictions)
        collapsed_pred = dict()
        collapsed_val = dict()
        for d_prediction in decoded_predictions:
            pred_hash = hash_nparray(d_prediction)
            collapsed_val[pred_hash] = collapsed_val.get(pred_hash,0) + 1 / len(decoded_predictions)
            collapsed_pred[pred_hash] = d_prediction


        result = []
        for pred_hash in collapsed_pred.keys():
            result.append((collapsed_pred[pred_hash], np.log(collapsed_val[pred_hash])))

        sorted_result = sorted(result, key = lambda x: -x[1])
        result_pred = []
        result_val = []
        for (pred, val) in sorted_result:
            result_pred.append(pred)
            result_val.append(val)

        return np.array(result_pred)[:self._search_expand_bound], result_val[:self._search_expand_bound]

def main_with_return_value():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=Path,
                        help='location of the training data')

    parser.add_argument('--output_dir', type=Path,
                        help='location to save results (default: working directory)',
                        default=Path("."))

    parser.add_argument('--max_subgraph_size', type=int,
                        help='max subgraph size limit (default: 1024)',
                        default=1024)

    parser.add_argument('--with_context',
                        action='store_true',
                        help='use state.context as a part of hashed state')

    parser.add_argument('--shuffled',
                        action='store_true',
                        help='shuffle the training dataset')

    parser.add_argument('--dry',
                        action='store_true',
                        help='pass through data without training model')


    args = parser.parse_args()

    trainer = Train(
        data_dir=args.data_dir.expanduser().absolute(),
        output_dir=args.output_dir,
        max_subgraph_size=args.max_subgraph_size,
        with_context=args.with_context,
        shuffled=args.shuffled,
        dry=args.dry
    )

    return trainer.train()


def main():
    main_with_return_value()


if __name__ == '__main__':
    main()
