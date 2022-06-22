import pickle


from pathlib import Path
from graph2tac.loader.data_server import DataServer
from graph2tac.loader.data_server import GraphConstants
from typing import Iterable

import argparse

import tqdm

import numpy as np

import hashlib



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

    return (tactic_idx, res)

def my_hash_of(state, with_context):
    graph, root, context = state
    if with_context:
        what_to_hash = tuple(*graph, context)
    else:
        what_to_hash = graph
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
                                       split=[1,0,0],
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
            graph, root, context = state
            state_hash = my_hash_of(state, self._with_context)

            train_action = self._data.get(state_hash, [])
            try:
                train_action.append(action_encode(action,
                                              local_context=context,
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



class Predict:
    def __init__(self, checkpoint_dir: Path):
        loaded_model = pickle.load(open(checkpoint_dir/'hmodel.sav', 'rb'))
        self._graph_constants: GraphConstants = loaded_model['graph_constants']
        self._data = loaded_model['data']
        self._label_to_index = None
        self._max_subgraph_size = loaded_model['max_subgraph_size']
        self._with_context = loaded_model['with_context']

    def get_tactic_index_to_numargs(self) -> Iterable:
        """
        Public API
        """
        return self._graph_constants.tactic_index_to_numargs

    def get_tactic_index_to_hash(self) -> Iterable:
        """
        Public API
        """
        return self._graph_constants.tactic_index_to_hash

    def get_node_label_to_name(self) -> Iterable:
        """
        Public API
        """
        return self._graph_constants.label_to_names


    def get_node_label_in_spine(self) -> Iterable:
        """
        Public API
        """
        return self._graph_constants.label_in_spine

    def get_max_subgraph_size(self):
        """
        Public API
        """
        return self._max_subgraph_size


    def initialize(self, global_context):
        self._label_to_idx = dict()
        for idx, label in enumerate(global_context):
            self._label_to_idx[label] = idx

    def compute_new_definitions(self, clusters: list[tuple[np.ndarray]]):
        """
        a list of cluster states on which we run dummy runs
        """
        pass



    def ranked_predictions(self, state: tuple, allowed_model_tactics: list, available_global=None, tactic_expand_bound=20, total_expand_bound=1000000, annotation="", debug=False):
        graph, root, context = state
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

        return result_pred, result_val

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
