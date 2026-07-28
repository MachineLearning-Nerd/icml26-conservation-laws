# Limitations and deviations

- This is a proof-by-counterexample to one universally quantified component,
  not a training-scale empirical study.
- It falsifies the normalized-sigmoid dense-MoE claim through the gating-row
  sum invariant. Other Theorem 4.7 invariants may still hold.
- The counterexample uses one data point, which is sufficient because the
  paper quantifies over every dataset. It is not presented as evidence about
  typical training behavior.
- No source-code release from the paper authors was available in the starting
  repository, so the gate was reconstructed directly from Equation/Theorem
  anchors.

