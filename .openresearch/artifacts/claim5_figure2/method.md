# Four verification-oriented routes

1. **Discretization/non-circularity.** Derive the exact one-step Euler error
   for a quadratic invariant. Check slope two and a non-conserved slope-one
   control.
2. **Configuration/evidence audit.** Enumerate every named dataset, model,
   learning-rate sweep, seed count, metric, and runtime requirement. Audit the
   candidate for training code, datasets, and raw trajectories.
3. **Independent resource calibration.** Benchmark one paper-width projection
   on the allocated CPU and form a deliberately weak WikiText-only work lower
   bound. The bound omits backward propagation and almost all model work.
4. **Mandatory falsification route.** Apply the exact normalized-sigmoid
   counterexample to the discrete step. It yields slope one while a matched
   softmax control gives zero, but it is not a named paper trajectory and
   therefore cannot falsify the exact historical empirical observation.

The first three routes do not raise confidence above LOW. Route four is
therefore mandatory. The final verdict is BLOCKED, not PASS.
