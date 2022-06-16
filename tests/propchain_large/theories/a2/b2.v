Require Import prelude_import.
Theorem th14 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2). intros. apply H1. Qed.
Theorem th15 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A1). intros. apply H0. Qed.
