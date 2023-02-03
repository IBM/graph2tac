Require Import prelude_import.
Theorem th148 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A4). intros. apply H3. Qed.
Theorem th149 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A0). intros. apply H. Qed.
