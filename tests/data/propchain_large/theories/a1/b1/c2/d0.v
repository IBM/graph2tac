Require Import prelude_import.
Theorem th82 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A1). intros. apply H0. Qed.
Theorem th83 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2). intros. apply H1. Qed.
