from typing import Optional, List, Tuple

import tqdm
import hashlib
import argparse
import pickle
import numpy as np
from pathlib import Path

from graph2tac.loader.data_server import DataServer, LoaderProofstate, LoaderDefinition
from graph2tac.predict import Predict, predict_api_debugging


def my_hash(x: bytes):
    m = hashlib.sha256()
    m.update(x)
    return m.hexdigest()


def hash_nparray(array: np.array):
    x = memoryview(array).tobytes()
    return my_hash(x)


def action_encode(action,
                  local_context,
                  global_context):
    tactic_idx, args = action
    context = (local_context, global_context)
    res = [(arg[0], context[arg[0]][arg[1]]) for arg in args]

    return tactic_idx, res


def my_hash_of(state: LoaderProofstate, with_context: bool):
    graph, root, context, _ = state
    what_to_hash = (*graph, context) if with_context else graph
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
            #print("args", args, "inverse_map", inverse_map)
            return None
        res.append(value)
    return res


class Train:
    def __init__(self, data_dir: Path, max_subgraph_size, with_context):
        self._data_server = DataServer(data_dir=data_dir,
                                       split=(1,0,0),
                                       bfs_option=True,
                                       restrict_to_spine=False,
                                       max_subgraph_size=max_subgraph_size)
        self._data = {}
        self._global_context = self._data_server.graph_constants().global_context
        self._tactic_index_to_hash = self._data_server.graph_constants().tactic_index_to_hash
        self._max_subgraph_size = max_subgraph_size
        self._with_context = with_context

    def train(self):
        none_counter = 0
        for state, action, idx in tqdm.tqdm(self._data_server.data_train()):
            graph, root, context, _ = state
            local_context, _ = context
            state_hash = my_hash_of(state, self._with_context)

            train_action = self._data.get(state_hash, [])
            try:
                train_action.append(action_encode(action,
                                              local_context=local_context,
                                              global_context=self._global_context))
            except IndexError:
                none_counter += 1


            self._data[state_hash] = train_action
        print(f"skipped {none_counter} proofstates with Nones")

        saved_model = {'graph_constants': self._data_server.graph_constants(),
                       'data': self._data,
                       'with_context': self._with_context,
                       'max_subgraph_size': self._max_subgraph_size}

        pickle.dump(saved_model, open('hmodel.sav', 'wb'))
        print(f"model saved in hmodel.sav")



class HPredict(Predict):
    def __init__(self, checkpoint_dir: Path, debug_dir: Optional[Path] = None):
        loaded_model = pickle.load(open(checkpoint_dir/'hmodel.sav', 'rb'))

        # initialize self._graph_constants
        super().__init__(graph_constants=loaded_model['graph_constants'], debug_dir=debug_dir)

        self._data = loaded_model['data']
        self._label_to_index = None
        self._max_subgraph_size = loaded_model['max_subgraph_size']
        self._with_context = loaded_model['with_context']
        self._label_to_idx = dict()

    @predict_api_debugging
    def initialize(self, global_context: Optional[List[int]] = None) -> None:
        if global_context is not None:
            for idx, label in enumerate(global_context):
                self._label_to_idx[label] = idx

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
                           tactic_expand_bound: int = 20,
                           total_expand_bound: int = 1000000,
                           annotation: str = "",
                           debug: bool = False
                           ) -> Tuple[np.ndarray, List]:
        graph, root, context, _ = state
        state_hash = my_hash_of(state, self._with_context)
        inverse_local_context = dict()
        for (i, node_idx) in enumerate(context):
           inverse_local_context[node_idx] = i

        predictions = self._data.get(state_hash, [])
        # print("allowed_model_tactics", allowed_model_tactics)
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
                                       inverse_global_label=self._label_to_idx)
            if args_decoded is not None:
                args_decoded = np.array(args_decoded, dtype=np.uint32).reshape((len(args_decoded), 2))
            if debug:
                print("HMODEL ", annotation, self._graph_constants.tactic_index_to_string[tactic_idx].decode(), "index into context:", args_decoded, end="")
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
            result.append((collapsed_pred[pred_hash], collapsed_val[pred_hash]))

        sorted_result = sorted(result, key = lambda x: -x[1])
        result_pred = []
        result_val = []
        for (pred, val) in sorted_result:
            result_pred.append(pred)
            result_val.append(val)

        return np.array(result_pred), result_val

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='data_dir to the model')

    parser.add_argument('--max_subgraph_size', type=int,
                        help='max subgraph size limit',
                        default=1024)
    parser.add_argument('--with_context',
                        action='store_true',
                        help='use state.context as a part of hashed state')


    args = parser.parse_args()

    trainer = Train(Path(args.data_dir).expanduser().absolute(),
                    args.max_subgraph_size, args.with_context)

    trainer.train()


if __name__ == '__main__':
    main()
