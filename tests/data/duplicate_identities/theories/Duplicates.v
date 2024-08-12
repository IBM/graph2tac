From Tactician Require Export Ltac1.Record.
From Tactician Require Export Ltac1.
(* Model used: Unclear. What is the purpose of this test? Should it be running against a trained model
   with this file as a dataset? Unknown right now. *)
Set Tactician Neural Executable "g2t-server --arch tfgnn --log_level=info --tf_log_level=critical --tactic_expand_bound=2 --total_expand_bound=10 --search_expand_bound=4 --model MODEL --record record_file.bin".
Global Set Default Proof Mode "Tactician Ltac1".

Reserved Notation "x -> y" (at level 99, right associativity, y at level 200).
Notation "A -> B" := (forall (_ : A), B).

Goal Prop -> Prop.
intro.
assumption.
Qed.

Section X.
Variable A : Prop.
Definition t : A -> A.
try (synth) || (intro; assumption).
Qed.

Tactician Neural Alignment.
End X.

Section X.
Variable A : Prop. (* Duplicate! *)
Definition u : A -> A.
try (synth) || (intro; assumption).
Qed.

Tactician Neural Alignment.
End X.

Module Type Y.
Parameter B : Prop.
End Y.

Module M1(MyY : Y).
Definition t : MyY.B -> MyY.B.
try (synth) || (intro; assumption).
Qed.

Tactician Neural Alignment.
End M1.

Definition somethinguseless := Prop.

Module M2(MyY : Y). (* Duplicate MyY.B *)
Definition t : MyY.B -> MyY.B.
try (synth) || (intro; assumption).
Qed.


End M2.