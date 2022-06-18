import pickle


from pathlib import Path
from graph2tac.loader.data_server import DataServer
from graph2tac.loader.data_server import GraphConstants
from typing import Iterable

import argparse

import tqdm

import numpy as np

def hash_nparray(array: np.array):
    return hash(memoryview(array).tobytes())

def action_encode(action, global_context):
    tactic_idx, args = action
    res = [(1, global_context[arg[1]]) if arg[0] == 1 else (0, arg[1]) for arg in args]
    return (tactic_idx, res)

def args_decode(args, label_to_idx):
    res = []
    for arg in args:
        value = (1, label_to_idx.get(arg[1], -1)) if arg[0] == 1 else (0, arg[1])
        if value == (1,-1):
            return None
        res.append(value)
    return res


class Train:
    def __init__(self, data_dir: Path, max_subgraph_size):
        self._data_server = DataServer(data_dir=data_dir,
                                       split=[1,0,0],
                                       bfs_option=True,
                                       restrict_to_spine=False,
                                       max_subgraph_size=max_subgraph_size)
        self._data = {}
        self._global_context = self._data_server.graph_constants().global_context
        self._tactic_index_to_hash = self._data_server.graph_constants().tactic_index_to_hash

    def train(self):
        for state, action, idx in tqdm.tqdm(self._data_server.data_train()):
            graph, root, context = state
            state_hash = hash(tuple(hash_nparray(x) for x in (*graph, context)))
            train_action = self._data.get(state_hash, [])
            train_action.append(action_encode(action, self._global_context))
            self._data[state_hash] = train_action
        saved_model = {'graph_constants': self._data_server.graph_constants(),
                       'data': self._data}
        pickle.dump(saved_model, open('hmodel.sav', 'wb'))
        print(f"model saved in hmodel.sav")



class Predict:
    def __init__(self, checkpoint_dir: Path):
        loaded_model = pickle.load(open(checkpoint_dir/'hmodel.sav', 'rb'))
        self._graph_constants = loaded_model['graph_constants']
        self._data = loaded_model['data']
        self._label_to_index = None

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


    def initialize(self, global_context):
        self._label_to_idx = dict()
        for idx, label in enumerate(global_context):
            self._label_to_idx[label] = idx


    def ranked_predictions(self, state: tuple, allowed_model_tactics: list, available_global=None, tactic_expand_bound=20, total_expand_bound=1000000):
        graph, root, context = state
        state_hash = hash(tuple(hash_nparray(x) for x in (*graph, context)))
        predictions = self._data.get(state_hash, [])
        decoded_predictions = []
        for prediction in predictions:
            tactic_idx, args = prediction
            if tactic_idx in allowed_model_tactics:
                args_decoded = args_decode(args, self._label_to_idx)
                args_decoded = np.array(args_decoded, dtype=np.uint32).reshape((len(args_decoded), 2))
                if args_decoded is not None:
                    try:
                        decoded_predictions.append(np.concatenate([np.array([(tactic_idx, tactic_idx)], dtype=np.uint32), np.array(args_decoded, dtype=np.uint32)]))
                    except ValueError:
                        print("args decoded", args_decoded)
                        raise ValueError
        return decoded_predictions, np.ones(len(decoded_predictions))/len(decoded_predictions)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='data_dir to the model')

    parser.add_argument('--max_subgraph_size', type=int,
                        help='max subgraph size limit',
                        default=1024)


    args = parser.parse_args()

    trainer = Train(Path(args.data_dir).expanduser().absolute(), args.max_subgraph_size)

    trainer.train()


if __name__ == '__main__':
    main()
