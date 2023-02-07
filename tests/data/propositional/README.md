This directory contain micro unit test involving several
coq modules (in theories/).

The module p0.v contains 3 independent propositional theorems.

The module p1.v contains 3 independent propositional theorems and 1
theorem whose proof depends on a theorem proved in the same module
p1.v.

The module p2.v contains 1 propositional theorems whose proof depends
on a theorem proved in the other module p0.v.

The module p2.v depends on p0.v using global argument.

The module p3.v depends on p0.v using a pose to a context of a global argument.

The module p4.v is a dummy independent module containing a single
theorem but testing theorem name collision.

Therefore, we have the following DAG of module dependencies:

p0.v {}
p1.v {}
p2.v {p0.v}
