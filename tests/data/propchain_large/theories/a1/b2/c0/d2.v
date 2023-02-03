Require Import prelude_import.
Theorem th92 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A0). intros. apply H. Qed.
Theorem th93 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A1). intros. apply H0. Qed.
