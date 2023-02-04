Require Import prelude_import.
Theorem th10 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A4). intros. apply H3. Qed.
Theorem th11 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A1). intros. apply H0. Qed.
