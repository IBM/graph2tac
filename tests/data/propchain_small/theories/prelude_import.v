From Tactician Require Export Ltac1.Record.
Global Set Default Proof Mode "Tactician Ltac1".

Reserved Notation "x -> y" (at level 99, right associativity, y at level 200).
Notation "A -> B" := (forall (_ : A), B).
