# Current audit — Claim 5 / Figure 2

**Verdict: BLOCKED. Confidence: LOW.** No toy or proxy result is presented as
paper-scale validation.

The exact claim is the reported `O(tau^2 k)` conservation-error behavior for
Qwen-3-style and ViT models on WikiText-103, PTB, CIFAR-10, and ImageNet-1K,
using three rates and ten independent seeds per configuration. Appendix D used
one 80GB H100 and up to four hours per configuration. This campaign was
authorized for CPU only.

## Four completed routes

1. **Non-circular discretization audit.** A one-step quadratic invariant has
   exact `tau^2` drift after its linear term cancels. The observed slope is
   approximately 2; a non-conserved control has slope approximately 1. This
   proves that the historical tiny one-step slope is algebraic, not
   architecture- or dataset-level validation.
2. **Exact configuration audit.** All four datasets, models, schedules, rates,
   seeds, and the paper's block metric were enumerated. The released
   reproduction contains neither the paper training implementation nor raw
   named-dataset trajectories.
3. **Independent CPU calibration.** The formal run benchmarks one
   paper-width projection. WikiText-103 alone requires 450,000 optimizer
   steps across the stated rates/seeds and 5,529,600,000 token positions.
   The reported lower bound deliberately omits backward propagation and most
   of the model.
4. **Mandatory falsification attempt.** The exact normalized-sigmoid
   counterexample produces first-order, slope-one gate-row-sum drift, while a
   matched softmax control gives zero. It falsifies the conservation premise,
   but it is not one of the named dataset trajectories, so it does not
   contradict the exact historical empirical observation.

Download the [four-route raw record](../../evidence/claim5_figure2/formal_run_summary.json),
[claim contract](../../evidence/claim5_figure2/claim_contract.json), and
[fail-closed checker](../../evidence/claim5_figure2/verify.py). The
[full four-route source](../../conservation_repro/claim5_figure2.py) is
also visible.

## Unblocker

The authors' executable training code and raw trajectories, or authorization
and suitable accelerator capacity for the exact 120+-configuration campaign.
Neither is available under this CPU-only task.
