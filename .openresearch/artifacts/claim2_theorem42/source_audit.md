# Source audit — Theorem 4.2

Source: arXiv:2606.17816v1, Theorem 4.2 at `#S4.Thmtheorem2`, using the
SwiGLU map in Equation 4. The theorem quantifies over every conservation law
and every hidden index, gives `nabla_B h=0` and its `A,C` PDE, and identifies
`||A[:,i]||^2-||C[i,:]||^2` as the characteristic invariants.

The current verifier directly certifies that each displayed norm difference
is conserved for arbitrary dimensions and differentiable losses. It does not
machine-formalize the paper's stronger completeness statement that there are
no other independent `C1` laws; that residual interpretation risk is stated
on the evaluator page and in `limitations.md`.

