from pathlib import Path
from pytact.data_reader import data_reader
from collections import defaultdict
import yaml
from dataclasses import dataclass
import random
import sys
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class BasicStats:
    dataset_to_num_thms : Dict[str,int]
    dataset_to_num_proofstates : Dict[str,int]
    dataset_to_deps : Dict[str, List[str]]

def compute_basic_stats():
    dataset_path = Path("/home/olsak/datasets_coq/v15-opam-coq8.11-partial/dataset")
    dataset_to_num_thms = defaultdict(int)
    dataset_to_num_proofstates = defaultdict(int)
    dataset_to_deps = defaultdict(set)

    with data_reader(dataset_path) as data:
        for datafile in data.values():

            num_thms = 0
            num_proofstates = 0
            for d in datafile.definitions():
                if proof := d.proof:
                    num_thms += 1
                    for proof_step in proof:
                        num_proofstates += len(proof_step.outcomes)

            package_name =  datafile.filename.parts[0]
            dataset_to_num_thms[package_name] += num_thms
            dataset_to_num_proofstates[package_name] += num_proofstates
            print(f"package_name: += {num_thms} thms {num_proofstates} proof_states", file=f)

            dataset_to_deps[package_name].update(
                path.parts[0]
                for path in datafile.dependencies
            )

    with open("basic_stats.yaml", 'w') as f:
        yaml.dump({
            "dataset_to_num_thms": dict(dataset_to_num_thms),
            "dataset_to_num_proofstates": dict(dataset_to_num_proofstates),
            "dataset_to_deps": {
                name : sorted(deps)
                for name, deps in dataset_to_deps.items()
            }
        }, f)

def load_basic_stats():

    with open("basic_stats.yaml") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return BasicStats(**data)

def topo_sort(basic_stats, key = None, reverse = False):
    dataset_to_deps = {
        name : set(deps) - set([name])
        for name, deps in basic_stats.dataset_to_deps.items()
    }
    dataset_to_rdeps = {name : set() for name in dataset_to_deps.keys()}
    for d,deps in dataset_to_deps.items():
        for dep in deps:
            dataset_to_rdeps[dep].add(d)

    if reverse: dataset_to_deps, dataset_to_rdeps = dataset_to_rdeps, dataset_to_deps

    res = []
    available = [d for d,deps in dataset_to_deps.items() if not deps]
    while available:
        if key == None:
            x = random.choice(sorted(available))
        else:
            x = max(available, key = key)
        res.append(x)
        available.remove(x)
        del dataset_to_deps[x]
        for rdep in dataset_to_rdeps.pop(x):
            deps = dataset_to_deps[rdep]
            deps.remove(x)
            if not deps:
                available.append(rdep)

    assert not dataset_to_deps
    assert not dataset_to_rdeps

    return res

def print_percentages(sort_mode, show_plot = True, plot_fname = None, from_end = False, split_line = 0.9, hott_extra = False, f = sys.stdout):

    assert sort_mode in ("biggest_first", "smallest_first", "random")
    
    basic_stats = load_basic_stats()
    total_num_thms = sum(basic_stats.dataset_to_num_thms.values())
    total_num_proofstates = sum(basic_stats.dataset_to_num_proofstates.values())
    dataset_to_num_thms_rel = {
        name : num_thms / total_num_thms
        for name, num_thms in basic_stats.dataset_to_num_thms.items()
    }
    dataset_to_num_proofstates_rel = {
        name : num_proofstates / total_num_proofstates
        for name, num_proofstates in basic_stats.dataset_to_num_proofstates.items()
    }
    if sort_mode == "random": key = None
    else:
        if sort_mode == "biggest_first": sgn = 1
        else: sgn = -1
        key = lambda x : sgn*(dataset_to_num_thms_rel[x] + dataset_to_num_proofstates_rel[x])

    names = topo_sort(
        basic_stats,
        key = key,
        reverse = from_end,
    )
    if from_end: names.reverse()
    if hott_extra:
        hott_name = "coq-hott.8.11"
        names.remove(hott_name)
        print("================= Hott ================", file=f)
        print("   Theorems        Proofstates", file=f)
        name = hott_name
        num_thms = basic_stats.dataset_to_num_thms[name]
        num_pfs = basic_stats.dataset_to_num_proofstates[name]
        num_thms_rel = dataset_to_num_thms_rel[name]
        num_pfs_rel = dataset_to_num_proofstates_rel[name]            
        print(f"{num_thms:5} = {num_thms_rel:6.2%} {num_pfs:8} = {num_pfs_rel:6.2%} -- {name}", file=f)
        total_num_thms -= num_thms
        total_num_proofstates -= num_pfs
        dataset_to_num_thms_rel = {
            name : num_thms / total_num_thms
            for name, num_thms in basic_stats.dataset_to_num_thms.items()
        }
        dataset_to_num_proofstates_rel = {
            name : num_proofstates / total_num_proofstates
            for name, num_proofstates in basic_stats.dataset_to_num_proofstates.items()
        }

    cum_rel_thms = [0.0]
    cum_rel_pfs = [0.0]
    cum_thms = [0]
    cum_pfs = [0]

    num_training_files = None
    print("================= Training ================", file=f)
    print("   Theorems        Proofstates", file=f)
    for name in names:
        num_thms = basic_stats.dataset_to_num_thms[name]
        num_pfs = basic_stats.dataset_to_num_proofstates[name]
        num_thms_rel = dataset_to_num_thms_rel[name]
        num_pfs_rel = dataset_to_num_proofstates_rel[name]
        print(f"{num_thms:5} = {num_thms_rel:6.2%} {num_pfs:8} = {num_pfs_rel:6.2%} -- {name}", file=f)
        cum_rel_thms.append(cum_rel_thms[-1]+num_thms_rel)
        cum_rel_pfs.append(cum_rel_pfs[-1]+num_pfs_rel)
        cum_thms.append(cum_thms[-1]+num_thms)
        cum_pfs.append(cum_pfs[-1]+num_pfs)
        if num_training_files is None and sum(cum_rel_pfs[-2:] + cum_rel_thms[-2:]) / 4 > split_line:
            print("================= Evaluation ================", file=f)
            print("   Theorems        Proofstates", file=f)
            num_training_files = len(cum_rel_pfs) -1

    print("================= Statistics ================", file=f)
    print(f"number of testing directories: {len(names) - num_training_files} = {1-num_training_files/len(names):.2%}", file=f)
    num_testing_theorems = cum_thms[-1] - cum_thms[num_training_files]
    num_rel_testing_theorems = cum_rel_thms[-1] - cum_rel_thms[num_training_files]
    num_testing_proofstates = cum_pfs[-1] - cum_pfs[num_training_files]
    num_rel_testing_proofstates = cum_rel_pfs[-1] - cum_rel_pfs[num_training_files]
    print(f"number of testing theorems: {num_testing_theorems} = {num_rel_testing_theorems:.2%}", file=f)
    print(f"number of testing proofstates: {num_testing_proofstates} = {num_rel_testing_proofstates:.2%}", file=f)

    num_training_theorems = cum_thms[num_training_files]
    num_rel_training_theorems = cum_rel_thms[num_training_files]
    num_training_proofstates = cum_pfs[num_training_files]
    num_rel_training_proofstates = cum_rel_pfs[num_training_files]
    print(f"number of training directories: {num_training_files} = {num_training_files/len(names):.2%}", file=f)
    print(f"number of training theorems: {num_training_theorems} = {num_rel_training_theorems:.2%}", file=f)
    print(f"number of training proofstates: {num_training_proofstates} = {num_rel_training_proofstates:.2%}", file=f)

    x = np.arange(len(names)+1)

    if show_plot or plot_fname:

        # Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
        fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
        ax.plot(x, cum_rel_thms, label='relative cumulative theorems')
        ax.plot(x, cum_rel_pfs, label='relative cumulative proofstates')
        ax.plot(x, np.full(len(x), split_line), label=f"possible {split_line} cutoff")
        ax.set_xlabel('number of taken main directories')
        ax.set_ylabel('proportion')
        ax.set_title("Dataset Statistics")
        ax.legend()

        if plot_fname is not None: plt.savefig(plot_fname)
        if show_plot: plt.show()

from_end = True
seed = 0
split_name = f"candidate{seed}"
if from_end: split_name = split_name+"_fe"
random.seed(seed)
with open(split_name+".txt", 'w') as f:
    print_percentages(
        "random",
        from_end = True,
        show_plot = False,
        hott_extra = True,
        plot_fname = split_name+".png",
        f = f,
    )
