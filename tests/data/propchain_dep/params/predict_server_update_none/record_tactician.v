(* Model used: Output of the test_training_tfgnn[propchain_dep-normal] test *)
Set Tactician Neural Executable "g2t-server --arch tfgnn --log_level=info --tf_log_level=critical --tactic_expand_bound=3 --total_expand_bound=10 --search_expand_bound=4 --model MODEL --record record_file.bin".
From Tactician Require Import Ltac1.

Tactician Neural Alignment.

Reserved Notation "x -> y" (at level 99, right associativity, y at level 200).
Notation "A -> B" := (forall (_ : A), B).

Theorem start (A: Prop) : (A->A).
  intros. apply H.
  Qed.
Theorem th0 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A3).
  intros.
  apply H2.
Qed.

Theorem th1_ (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A3).
  Debug Suggest.
  try (timeout 10 (debug synth)).
Abort.

Theorem th1 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A3).
  pose th0.
  apply p.
Qed.

Theorem th2_other_ (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2).
  Debug Suggest.
  try (timeout 10 (debug synth)).
Abort.

Theorem th2_other (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2).
  pose th1.
  pose th0.
  intros.
  apply H1.
Qed.

Theorem th2_ (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2).
  Debug Suggest.
  try (timeout 10 (debug synth)).
Abort.

Theorem th2 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2).
  pose th0.
  pose th1.
  intros.
  apply H1.
Qed.
