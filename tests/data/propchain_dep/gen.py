import random

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


def make_dataset(n_vars: int, n_th: int, seed: int):
    """
    n_vars: number of variables
    n_th  : number of theorems
    seed  : random number generator seed
    """

    random.seed(seed)

    for i in range(n_th):
        th_str = generate_random(n_vars, random.randint(0, n_vars - 1), f"th{i}")
        print(th_str)


make_dataset(5, 100, seed=0)
