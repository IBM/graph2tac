Require Import prelude_import.
Theorem th142 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A0). intros. apply H. Qed.
Theorem th143 (A0 A1 A2 A3 A4: Prop) : (A0->A1->A2->A3->A4->A3). intros. apply H2. Qed.
