# Limitations and deviations

- The exact certificates establish every displayed invariant for arbitrary
  dimensions. They do not independently formalize the paper's stronger
  completeness classification of all `C1` conservation laws.
- The model-scale audit uses random inputs and squared loss, not PTB or
  WikiText-103 training. Its role is implementation and scale calibration;
  the universal evidence is algebraic.
- Runtime is uncertain, so the formal cumulative run uses HF `cpu-upgrade`.
  No GPU is used.

