Require Import prelude_import.
Theorem th2 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A0). intros. apply H. Qed.
Theorem th3 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2). intros. apply H1. Qed.
