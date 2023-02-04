Require Import prelude_import.
Theorem th74 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A4). intros. apply H3. Qed.
Theorem th75 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A2). intros. apply H1. Qed.
