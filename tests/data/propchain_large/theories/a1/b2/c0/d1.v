Require Import prelude_import.
Theorem th90 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A3). intros. apply H2. Qed.
Theorem th91 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A0). intros. apply H. Qed.
