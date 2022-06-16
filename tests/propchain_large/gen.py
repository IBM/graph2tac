prelude_string = """
From Tactician Require Export Ltac1.Record.
Global Set Default Proof Mode "Tactician Ltac1".

Reserved Notation "x -> y" (at level 99, right associativity, y at level 200).
Notation "A -> B" := (forall (_ : A), B).
"""



import random
import os
def generate_random(n: int, k: int, thm_name: str):
    """
    n is the number of given variables (in example below n = 5);
    k is the index  of a variable to use 0 <= k < n

    in example k = 3

    output:

    Theorem <thm_name> (A0 A1 A2 A3 A4:  Prop) : (A0->A1->A2->A3->A4->A3). intros. apply H<k-1>. Qed.
    # if k-1 == -1 ==> ''

    """

    prefix = f"Theorem {thm_name}"
    variables = " ".join(f"A{i}" for i in range(n))
    proposition = "->".join(f"A{i}" for i in range(n)) + f"->A{k}"

    hyp = "H" if k==0 else f"H{k-1}"

    return f"{prefix} ({variables}: Prop) : ({proposition}). intros. apply {hyp}. Qed."


def get_fname(i: int,
                degree: int,
                prefix: str):

    def char_of_level(level: int) -> str:
        return chr(ord('a') + level)
    digits = []
    while (i != 0):
        d = i % degree;
        digits.append(d)
        i = i // degree
    digits.reverse()
    return prefix + '/' + '/'.join([f"{char_of_level(level)}{d}" for level, d in enumerate(digits)])


def make_dataset(n_vars: int,
                 total_n_files: int,
                 degree: int,
                 theorems_per_file: int,
                 seed: int=0,
                 prefix="dataset"):
    with open(os.path.join(prefix, "prelude_import.v"), 'w') as prelude_file:
        prelude_file.write(prelude_string)



    """
    generate chain theorems spread over multiple in deeper hierarchy,
    such as
    dataset/a0/a1/c0/d3
    dataset/a0/b2/..

    n_vars: number of variables
    n_th  : number of theorems
    seed  : random number generator seed
    degree: how many subdirector
    """

    random.seed(seed)

    for i in range(total_n_files):
        th_filename = get_fname(i + 1, degree, prefix)
        print(th_filename)
        th_dirname = os.path.dirname(th_filename)
        os.makedirs(th_dirname, exist_ok=True)
        with open(th_filename + '.v','w') as f:
            f.write("Require Import prelude_import.\n")
            print(f"In {th_filename}:")
            for j in range(theorems_per_file):
                th_str = generate_random(n_vars, random.randint(0, n_vars - 1), f"th{i*theorems_per_file+j}")
                print(th_str)
                print(th_str, file=f)


make_dataset(n_vars=5, total_n_files=100, theorems_per_file=2, degree=3, seed=0, prefix="theories")
