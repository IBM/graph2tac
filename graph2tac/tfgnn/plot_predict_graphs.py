from typing import Optional

import argparse
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog='plot_predict_graphs.py',
                                     description='Makes supervised evaluation of the predict class, and plot graphs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Location of the data"
    )
    parser.add_argument('model', metavar='model', type=str, nargs='+',
                        help='color:path_to_the_model')
    parser.add_argument(
        "--output_fname",
        type=Path,
        default='./plot.png',
        help=("Filename for the generated plot")
    )
    parser.add_argument(
        '--checkpoint_q',
        default = 1.5,
        type=float,
        help="Coefficient of the geometric sequence of loaded checkpoints"
    )
    parser.add_argument(
        '--checkpoint_last',
        default = 985,
        type=int,
        help="The last checkpoint number to be loaded"
    )
    def floatpair(string):
        a,b = string.split(',')
        return float(a), float(b)
    parser.add_argument(
        '--plot_size',
        default = (16,10),
        type=floatpair,
        help="The size of the plot 'height,width'"
    )
    parser.add_argument(
        '--tactic_expand_bound',
        default = 1,
        type=int,
        help="Predict class argument"
    )
    parser.add_argument(
        '--total_expand_bound',
        default = 1,
        type=int,
        help="Predict class argument"
    )
    parser.add_argument(
        '--undirected', dest='symmetrization', action='store_const',
        const="UNDIRECTED", default="BIDIRECTIONAL",
        help='Dataset option, UNDIRECTED symmetrization'
    )
    parser.add_argument(
        '--no_self_edges', dest='add_self_edges', action='store_const',
        const=False, default=True,
        help='Dataset option'
    )
    parser.add_argument(
        '--max_subgraph_size', default = 1024,
        type = int,
        help='Dataset option'
    )
    parser.add_argument(
        '--exclude_none_arguments', dest='exclude_none_arguments', action='store_const',
        const=True, default=False,
        help='Dataset option'        
    )
    parser.add_argument(
        '--include_not_faithful', dest='exclude_not_faithful', action='store_const',
        const=False, default=True,
        help='Dataset option'
    )

    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import tensorflow as tf
    import tensorflow_gnn as tfgnn
    print(f'Using TensorFlow v{tf.__version__} and TensorFlow-GNN v{tfgnn.__version__}')

    from graph2tac.tfgnn.dataset import DataServerDataset, BIDIRECTIONAL, UNDIRECTED
    from graph2tac.tfgnn.predict import TFGNNPredict

    def predict_evaluation(dataset: DataServerDataset,
                           log_dir: Path,
                           checkpoint_number: Optional[int],
                           tactic_expand_bound: int,
                           total_expand_bound: int,
                           search_expand_bound: Optional[int],
                           cache_file: Optional[Path],
                           batch_size: int = 64):
        # read the cache file if it was specified
        if cache_file is not None:
            with cache_file.open('rb') as pickle_jar:
                cache = pickle.load(pickle_jar)
        else:
            cache = {}

        # only compute the evaluation if it is not in the cache
        fingerprint = (log_dir.stem, checkpoint_number, tactic_expand_bound, total_expand_bound, search_expand_bound)
        if fingerprint not in cache.keys():
            # get the evaluation data from the dataset
            all_proofstates, _ = dataset.proofstates(split=(1,0))
            all_cluster_subgraphs = dataset.data_server.def_cluster_subgraphs()

            # compute accuracies without definition reconstruction
            global_argument_predict = TFGNNPredict(log_dir=log_dir,
                                                   checkpoint_number=checkpoint_number)

            per_proofstate, per_lemma = global_argument_predict._evaluate(all_proofstates, 
                                                                          batch_size=batch_size, 
                                                                          tactic_expand_bound=tactic_expand_bound,
                                                                          total_expand_bound=total_expand_bound,
                                                                          search_expand_bound=search_expand_bound)

            # reconstruct all definitions
            global_argument_predict.compute_new_definitions(all_cluster_subgraphs)

            # compute accuracies with definition reconstruction
            per_proofstate_with_reconstruction, per_lemma_with_reconstruction = global_argument_predict._evaluate(all_proofstates, 
                                                                                                                  batch_size=batch_size, 
                                                                                                                  tactic_expand_bound=tactic_expand_bound, 
                                                                                                                  total_expand_bound=total_expand_bound,
                                                                                                                  search_expand_bound=search_expand_bound)

            # update the cache
            cache[fingerprint] = {
                'per_proofstate': per_proofstate, 
                'per_lemma': per_lemma,
                'per_proofstate_with_reconstruction': per_proofstate_with_reconstruction,
                'per_lemma_with_reconstruction': per_lemma_with_reconstruction
            }

            # write to disk if the cache file was specified
            if cache_file is not None:
                with cache_file.open('wb') as pickle_jar:
                    pickle.dump(cache, pickle_jar)
        return cache[fingerprint]

    if args.symmetrization == "BIDIRECTIONAL": symmetrization = BIDIRECTIONAL
    elif args.symmetrization == "UNDIRECTED": symmetrization = UNDIRECTED
    else: raise Exception()
    dataset = DataServerDataset(data_dir=args.data_dir,
                                symmetrization=symmetrization,
                                add_self_edges=args.add_self_edges,
                                max_subgraph_size=args.max_subgraph_size,
                                exclude_none_arguments=args.exclude_none_arguments,
                                exclude_not_faithful=args.exclude_not_faithful)

    q = args.checkpoint_q
    last = args.checkpoint_last
    i = 0
    checkpoint_numbers = [0]
    while int(q**i) < last:
        checkpoint_numbers.append(int(q**i))
        i += 1
    checkpoint_numbers.append(last)

    fig, ax = plt.subplots(figsize=args.plot_size)
    for color_model in args.model:
        color, model = color_model.split(':', 1)
        per_proofstate = []
        per_proofstate_with_reconstruction = []
        for checkpoint_number in checkpoint_numbers:
            results = predict_evaluation(dataset=dataset,
                                         log_dir=Path(model),
                                         checkpoint_number=checkpoint_number,
                                         tactic_expand_bound=args.tactic_expand_bound,
                                         total_expand_bound=args.total_expand_bound,
                                         search_expand_bound=None,
                                         cache_file=None,
                                        )
            per_proofstate.append(results['per_proofstate'])
            per_proofstate_with_reconstruction.append(results['per_proofstate_with_reconstruction'])
        ax.plot(checkpoint_numbers, per_proofstate, color=color, label=f'{model} (without def. reconstruction)')
        ax.plot(checkpoint_numbers, per_proofstate_with_reconstruction, color=color, label=f'{model} (with def. reconstruction)', linestyle='--')
    ax.set_title('tactic_expand_bound = {args.tactic_expand_bound}, total_expand_bound = {args.total_expand_bound} (per-proofstate)', fontsize=20)
    ax.set_xlabel('epoch', fontsize=16)
    ylabel = 'strict accuracy'
    if args.exclude_not_faithful: ylabel = ylabel+" (only faithful)"
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(fontsize=12)
    plt.savefig(args.output_fname)

if __name__ == "__main__":
    main()
