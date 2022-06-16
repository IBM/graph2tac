Require Import prelude_import.
Theorem theorem0: forall X: Prop, (X->X). intro. intro. apply H. Qed.
Theorem theorem1: forall A B: Prop, (A->(B->B)). intro. intro. intro. intro. apply H0. Qed.
Theorem theorem2: forall A B: Prop, (A->(B->A)). intro. intro. intro. intro. apply H. Qed.
Theorem theorem3: forall C D: Prop, (C->(D->D)).  intro.  intro.  intro.  apply theorem0. Qed.
