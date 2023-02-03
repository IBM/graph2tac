Require Import prelude_import.

Theorem start (A: Prop) : (A->A).
  intros. apply H.
  Qed.
Theorem th0 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A3).
  intros.
  apply H2.
Qed.

Theorem th1 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A3).
  pose th0.
  apply p.
Qed.

Theorem th2_other (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2).
  pose th1.
  pose th0.
  intros.
  apply H1.
Qed.

Theorem th2 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2).
  pose th0.
  pose th1.
  intros.
  apply H1.
Qed.
