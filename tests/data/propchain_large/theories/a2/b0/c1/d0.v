Require Import prelude_import.
Theorem th112 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2). intros. apply H1. Qed.
Theorem th113 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A4). intros. apply H3. Qed.
