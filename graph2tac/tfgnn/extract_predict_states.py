from collections import defaultdict
import gzip
import itertools
import json
from typing import Any, Iterable, Optional

import argparse
import pickle
from pathlib import Path
import numpy as np
import numpy.typing as npt
import tqdm

from graph2tac.loader.data_server import DataServer, LoaderProofstate, LoaderAction
from graph2tac.loader.predict_server import load_model, Predict

#from pympler.tracker import SummaryTracker
#tracker = SummaryTracker()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='graph2tac Data extractor python tensorflow server')

    parser.add_argument(
        "--data-dir", "--data_dir",
        type=Path,
        required=True,
        help="Location of the data"
    )

    parser.add_argument(
        "--output-dir", "--output_dir",
        type=Path,
        required=True,
        help="Location of the output"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many datapoints to extract (default is all of them)"
    )

    parser.add_argument(
        "--file_size",
        type=int,
        default=100000,
        help="How many datapoints pre file (default is 100,000)"
    )

    parser.add_argument('--model', type=Path, required=True,
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
    
    parser.add_argument('--max-tactic-args', '--max_tactic_args',
                        type=int,
                        default=255,
                        help="exclude any tactic with more than this many arguments (default: 255)")

    parser.add_argument('--hard-code-arg-pred-logit-temp', '--hard_code_arg_pred_logit_temp',
                        default=False,
                        action='store_true',
                        help="For debugging only. (Needed for compatibility with a particular previously trained model which had a bug.)")
    
    parser.add_argument('--knn-proofstep-limit', '--knn_proofstep_limit',
                        type=int,
                        default=0,
                        help="Number of recent proof states to use for k-NN tactic prediction (0 disables k-NN), defaults to 0")
    
    parser.add_argument('--knn-keys-ignore-tactic-head', '--knn_keys_ignore_tactic_head',
                        default=False,
                        action='store_true',
                        help="Use pre-tactic-head embeddings for key embeddings in k-NN tactic prediction")
    
    parser.add_argument('--knn-logit-normalize-mean', '--knn_logit_normalize_mean',
                        default=False,
                        action='store_true',
                        help="Normalize logits mean to 0 (independently for knn and trained tactics)")
    
    parser.add_argument('--knn-logit-normalize-max', '--knn_logit_normalize_max',
                        default=False,
                        action='store_true',
                        help="Use same max score for top predictions from each of knn and trained tactics")
    
    parser.add_argument('--knn-logit-normalize-prob', '--knn_logit_normalize_prob',
                        default=False,
                        action='store_true',
                        help="Normalize logits to be a log probability distribution (independently for knn and trained tactics)")
    
    parser.add_argument('--knn-logit-normalize-var', '--knn_logit_normalize_var',
                        default=False,
                        action='store_true',
                        help="Normalize knn logits to have same variance as trained tactic logits")
    
    parser.add_argument('--knn-logit-normalize-std', '--knn_logit_normalize_std',
                        type=float,
                        default=None,
                        help="Normalize knn logits to have specific standard deviation, defaults to None")
    
    parser.add_argument('--knn-logit-temp', '--knn_logit_temp',
                        type=float,
                        default=None,
                        help="Logit temperature for k-NN tactic prediction, defaults to None")
    
    parser.add_argument('--knn-only', '--knn_only',
                        default=False,
                        action='store_true',
                        help="Don't use learned tactic embeddings as keys for tactic prediction (`knn_proofstep_limit` must be positive)")
    
    parser.add_argument('--knn-duplicate-reduction', '--knn_duplicate_reduction',
                        type=str,
                        default="none",
                        help="How to combine logits if the same tactic is selected multiple times (options: 'none', 'mean', 'sum', 'max', 'softmax', 'frequency', 'order'), defaults to 'none'")
    
    parser.add_argument('--knn-use-learned-tactic-embeddings-for-arg-prediction', '--knn_use_learned_tactic_embeddings_for_arg_prediction',
                        default=False,
                        action='store_true',
                        help="Use a learned tactic embedding (if one exists) for argument prediction instead of the embedding from the k-NN proof state example")
    
    parser.add_argument('--knn-dist', '--knn_dist',
                        type=str,
                        default="inner_prod",
                        help="The distance to use in the knn (options: 'inner_prod', 'cosine', 'euclidean'), defaults to 'inner_prod'")
    
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

    parser.add_argument('--proofstep-profiler-logdir', '--proofstep_profiler_logdir',
                        type=Path, default=None,
                        help='Supply logdir to profile the proofstep processing')

    parser.add_argument('--proofstep-profiler-start', '--proofstep_profiler_start',
                        type=int, default=10,
                        help='Proofstep processing steps to start profiling (default: 10).')
    
    parser.add_argument('--proofstep-profiler-end', '--proofstep_profiler_end',
                        type=int, default=15,
                        help='Proofstep processing steps to stop profiling (exclusive) (default: 15).')
    
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


class DataExtractor:
    def __init__(self, data_server: DataServer, model: Predict, output_dir: Path):
        self.data_server = data_server
        self.model = model
        self.output_dir = output_dir

        self.all_cluster_subgraphs = self.data_server.def_cluster_subgraphs() 
        self.train_proofstates = self.data_server.data_train()
        self.valid_proofstates = self.data_server.data_valid()

        # each name could have many is
        self.name_to_is: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self.model.graph_constants.label_to_names):
            self.name_to_is[name].append(i)

        self.name_data: dict[int, dict[str, Any]] = {}
        self.visited_names: dict[int, list[int]] = {}

        # verify that the alignment is the same
        # TODO(jrute): Make it so that we don't have to run on the exact same data used for training
        assert len(self.data_server._node_i_to_name) == len(self.model.graph_constants.label_to_names), "Dataset (from data) and graph constants (from model) don't align"
        assert self.data_server._node_i_to_name == self.model.graph_constants.label_to_names, "Dataset (from data) and graph constants (from model) don't align"
        assert len(self.data_server._tactic_i_to_string) == len(self.model.graph_constants.tactic_index_to_string), "Dataset (from data) and graph constants (from model) don't align"
        assert self.data_server._tactic_i_to_string == self.model.graph_constants.tactic_index_to_string, "Dataset (from data) and graph constants (from model) don't align"

        #model.allocate_definitions(
        #    len(all_cluster_subgraphs),
        #    len(train_proofstates) + len(valid_proofstates),
        #)
        #for df in tqdm.tqdm(all_cluster_subgraphs):
        #    model.compute_new_definitions([df])
        #    #print("Names", df.definition_names)
    
    def name_to_name_id(
        self,
        name: bytes,
        global_cxt: list[int]
    ):
        """Get the name id from the name.

        Since names aren't always unique and the name id is not stored in the metadata,
        we have to infer the name id from the available possible ids and the global context.
        The name id should be the first id with this name greater than the node ids.

        :param name: The byte name of the current proofstate
        :param global_cxt: The ids in the global context elements
        """
        name_str = name.decode("utf8")
        indices = self.name_to_is[name_str]  # sorted list
        max_cxt_id = max(global_cxt)
        for index in indices:
            if index <= max_cxt_id:
                continue
            else:
                break
        assert index > max_cxt_id, f"No index for {name_str} available larger than the maximum node id {max_cxt_id}. Indices: {indices}"
        return index

    @staticmethod
    def sorted_list_to_list_of_ranges(l: list) -> list[list[int]]:
        # if l is sorted and no duplicates, then the difference between x and the index i is a group id
        groups = (list(group) for _, group in itertools.groupby(enumerate(l), key=lambda p: p[1] - p[0]))
        intervals = [[group[0][1], group[-1][1]+1] for group in groups]
        return intervals

    def process_name_data(
        self,
        proof_state: LoaderProofstate,
    ) -> int:
        #name_bytes = proof_state.metadata.name
        #cxt = [int(cxt_id) for cxt_id in proof_state.context.global_context]
        #name_id = self.name_to_name_id(name_bytes, cxt)
        name_id = proof_state.metadata.definition_index
        step_id = int(proof_state.metadata.step)
        
        if name_id not in self.visited_names:
            name_bytes = proof_state.metadata.name
            cxt = [int(cxt_id) for cxt_id in proof_state.context.global_context]
            cxt = sorted(cxt)
            self.name_data[name_id] = {
                "name_id": name_id,
                "name": name_bytes.decode("utf8"),
                "global_context_ranges": self.sorted_list_to_list_of_ranges(cxt)
            }
            
            self.visited_names[name_id] = [step_id]
        else:
            #cxt = sorted(cxt)
            #cxt = self.sorted_list_to_list_of_ranges(cxt)
            #if name_id in self.name_data:
            #    assert cxt == self.name_data[name_id]["global_context_ranges"], (name_bytes, cxt, self.name_data[name_id])
            if step_id in self.visited_names[name_id]:
                #name_bytes = proof_state.metadata.name
                #print("WARNING", f"Visited more than once: {name_bytes} (id {name_id}), step {step_id}", flush=True)
                pass
            else:
                self.visited_names[name_id].append(step_id)

        return name_id
    
    def compute_and_process_datapoint(
        self,
        i: int,
        split: str,
        proof_state: LoaderProofstate,
        action: LoaderAction,
        name_id: int,
    ) -> tuple[dict[str, Any], npt.NDArray[np.float32]]:
        #self.model.compute_new_proofstep(
        #    proof_state=proof_state,
        #    tactic_id=action.tactic_id
        #)
        emb = self.model._compute_proofstate_emb(proof_state).numpy()
        datapoint = {
            "id": int(i),
            "metadata_name_id": name_id,
            "metadata_step": int(proof_state.metadata.step),
            "tactic_id": int(action.tactic_id),
            "split": split,
        }
        # TODO(jrute): The tactic_inference_tack and _pop_tactic_embs is specific to the tfgnn model
        #emb = self.model.tactic_inference_task.proof_step_embeddings.get_value(self.model.tactic_inference_task.proof_step_embeddings.length-1) # type: ignore
        #self.model._pop_tactic_embs(0) # type: ignore
        
        return datapoint, emb 
    
    def save_metadata(self):
        with (self.output_dir / "context_names.jsonl").open("w") as f:
            for x in self.model.graph_constants.label_to_names:
                print(x, file=f)
        
        with (self.output_dir / "tactic_names.jsonl").open("w") as f:
            for x in self.model.graph_constants.tactic_index_to_string:
                print(x, file=f)
        
    def save_data(self, file_ix: int, name_data: dict[int, dict[str, Any]], data: list[dict[str, Any]], embeddings: npt.ArrayLike):
        data_jsonl_gz_file = self.output_dir / f"data/data{file_ix}.jsonl.gz"
        with gzip.open(data_jsonl_gz_file, "wb") as f:
            file_contents = "\n".join(json.dumps(x) for x in data)
            f.write(file_contents.encode())

        data_jsonl_gz_file = self.output_dir / f"name_data/name_data{file_ix}.jsonl.gz"
        with gzip.open(data_jsonl_gz_file, "wb") as f:
            file_contents = "\n".join(json.dumps(x) for x in name_data.values())
            f.write(file_contents.encode())
        
        np.save(self.output_dir / f"embeddings/embeddings{file_ix}.npy", np.array(embeddings))

    def compute_and_save_data(self, limit: Optional[int], file_size: int):
        self.save_metadata()
                
        data = []
        embeddings = []

        cnt = 0
        file_ix = 0
        for split, proofstates in [("train", self.train_proofstates), ("valid", self.valid_proofstates)]:
            for proof_state, action, i in tqdm.tqdm(proofstates):
                cnt += 1
                if limit is not None and cnt >= limit:
                    break

                # this updates self.name_data with the name, name_id, and context
                # the name_id is not unique to a proof state
                name_id = self.process_name_data(
                    proof_state=proof_state
                )

                datapoint, emb = self.compute_and_process_datapoint(
                    i=i,
                    split=split,
                    proof_state=proof_state,
                    action=action,
                    name_id=name_id,
                )
                data.append(datapoint)
                embeddings.append(emb)

                # to save memory, periodically save data
                if len(data) == file_size:
                    self.save_data(file_ix, self.name_data, data, embeddings)
                    file_ix += 1
                    # reset data
                    self.name_data = {}
                    data = []
                    embeddings = []
                    #tracker.print_diff()
                

        if data:
            self.save_data(file_ix, self.name_data, data, embeddings)


def main():
    args = parse_args()

    import tensorflow as tf
    import tensorflow_gnn as tfgnn
    print(f'Using TensorFlow v{tf.__version__} and TensorFlow-GNN v{tfgnn.__version__}')

    from graph2tac.tfgnn.predict import TFGNNPredict

    # load the model the same way as the predict server
    # it is very hacky since it requires a Namespace object set up the same as the predict server
    # TODO(jrute): Move this to be it's own thing with a clean interface
    model = load_model(
        config=args,
        log_levels={
            'debug':'10',
            'verbose':'15',
            'info':'20',
            'summary':'25',
            'warning':'30',
            'error':'40',
            'critical':'50',
        },
    )

    # load the data using the dataset config inside the model
    data_server = DataServer.from_yaml_config(
        data_dir=args.data_dir,
        yaml_filepath=args.model.expanduser().resolve() / "config/dataset.yaml"
    )

    data_extractor = DataExtractor(
        model=model,
        data_server=data_server,
        output_dir=args.output_dir
    )
    data_extractor.compute_and_save_data(
        limit=args.limit,
        file_size=args.file_size
    )
    

    #results = predict_evaluation(
    #    dataset=data_server,
    #    log_dir=Path(model),
    #    checkpoint_number=checkpoint_number,
    #    tactic_expand_bound=args.tactic_expand_bound,
    #    total_expand_bound=args.total_expand_bound,
    #    search_expand_bound=None,
    #    cache_file=None,
    #)

    #fig, ax = plt.subplots(figsize=args.plot_size)
    #for color_model in args.model:
    #    color, model = color_model.split(':', 1)
    #    per_proofstate = []
    #    per_proofstate_with_reconstruction = []
    #    for checkpoint_number in checkpoint_numbers:
    #        results = predict_evaluation(dataset=dataset,
    #                                     log_dir=Path(model),
    #                                     checkpoint_number=checkpoint_number,
    #                                     tactic_expand_bound=args.tactic_expand_bound,
    #                                     total_expand_bound=args.total_expand_bound,
    #                                     search_expand_bound=None,
    #                                     cache_file=None,
    #                                    )
    #        per_proofstate.append(results['per_proofstate'])
    #        per_proofstate_with_reconstruction.append(results['per_proofstate_with_reconstruction'])
    #    ax.plot(checkpoint_numbers, per_proofstate, color=color, label=f'{model} (without def. reconstruction)')
    #    ax.plot(checkpoint_numbers, per_proofstate_with_reconstruction, color=color, label=f'{model} (with def. reconstruction)', linestyle='--')
    #ax.set_title('tactic_expand_bound = {args.tactic_expand_bound}, total_expand_bound = {args.total_expand_bound} (per-proofstate)', fontsize=20)
    #ax.set_xlabel('epoch', fontsize=16)
    #ylabel = 'strict accuracy'
    #if args.exclude_not_faithful: ylabel = ylabel+" (only faithful)"
    #ax.set_ylabel(ylabel, fontsize=16)
    #ax.legend(fontsize=12)
    #plt.savefig(args.output_fname)

if __name__ == "__main__":
    main()
