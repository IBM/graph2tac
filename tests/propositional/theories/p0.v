Require Export prelude_import.
Theorem theorem0: forall A: Prop, (A->A). intro. intro. apply H. Qed.
Theorem theorem1: forall A B: Prop, (A->(B->B)). intro. intro. intro. intro. apply H0. Qed.
Theorem theorem2: forall A B: Prop, (A->(B->A)). intro. intro. intro. intro. apply H. Qed.
