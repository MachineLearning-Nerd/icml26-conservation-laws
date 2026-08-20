# Claim evidence ledger

Each verdict below comes from committed verifier source, a machine-readable
contract or summary, a source anchor, and a negative control where the claim
allows one. A VERIFIED_SCOPED result supports the stated finite or structural
scope; it does not turn finite computation into a proof of a universal
quantifier. FALSIFIED_SCOPED identifies an explicit assumption-matched
contradiction. BLOCKED means the exact claim could not be answered without
missing assets or unauthorized compute.

| Claim | Verdict | Primary evidence | Production path |
| --- | --- | --- | --- |
| C1 | VERIFIED_SCOPED | .openresearch/artifacts/claim1_theorem41/raw/formal_run_summary.json | conservation_repro/claim1_theorem41.py::verify |
| C2 | VERIFIED_SCOPED | .openresearch/artifacts/claim2_theorem42/raw/model_scale_summary.json and structural_certificate.json | conservation_repro/claim2_swiglu.py::verify |
| C3 | VERIFIED_SCOPED | .openresearch/artifacts/claim3_attention/raw/formal_run_summary.json and structural_certificate.json | conservation_repro/claim3_attention.py::verify |
| C4 | FALSIFIED_SCOPED | .openresearch/artifacts/claim4_theorem47/raw/exact_certificate.json | conservation_repro/claim4_theorem47.py::verify |
| C5 | BLOCKED | .openresearch/artifacts/claim5_figure2/raw/formal_run_summary.json | conservation_repro/claim5_figure2.py::verify |

## C1 — Theorem 4.1, GELU and SiLU

The paper claims that every C1 conservation law for the stated GELU and SiLU
feedforward layers is constant under dataset-universal Euclidean gradient
flow. The audit reconstructs the arbitrary-width Taylor and pole argument:
at a generic parameter point, the coefficients of phi(B_ij t) and
t phi'(B_ij t) produce a rational numerator map whose distinct-pole ranks
force all candidate-law derivatives to vanish.

The committed verifier checks exact Fraction ranks for widths 1 through 8
with ranks 2, 4, 6, 8, 10, 12, 14, and 16, and rejects a repeated-gamma
rank-loss control. Ten deterministic sampled Jacobian checks provide an
independent finite full-span audit for both activations. The result is
VERIFIED_SCOPED: the derivation carries the universal argument, while the
finite checks calibrate its algebraic consequences.

## C2 — Theorem 4.2, SwiGLU

For each hidden index i, the inverse scaling A[:,i] -> exp(t) A[:,i] and
C[i,:] -> exp(-t) C[i,:] leaves the model unchanged. Differentiating the
symmetry gives conservation of ||A[:,i]||^2 - ||C[i,:]||^2. The verifier
also reconstructs the iff gradient condition and checks the structural
exponent sum.

The model-scale audit uses d=256, hidden width 1024, 12 repeated layer
instances, 10 seeds, float64 arithmetic, and an independent elementwise
non-law control. The maximum relative invariant-rate residual is
3.9075619819679394e-16; the minimum intended control rate is
0.001807331989681632. The result is VERIFIED_SCOPED.

## C3 — Theorems 4.3–4.4, MHA and RoPE

The audit derives the displayed invariants from exact functional symmetries:
Q -> Q E and K -> K E^-T preserves attention scores, V -> V F and
O -> O F^-T preserves the output contraction, and matched inverse scaling of
each RoPE Q/K block preserves the rotary score. The corresponding Q/K, V/O,
and RoPE-block norm differences are then checked under gradient flow.

The recorded campaign uses 12 layers, d=192, three 64-dimensional heads,
sequence length 32, and 10 deterministic seeds. Worst observed values are
2.2073201955743162e-16 relative residual, 1.051088885904111e-14 V/O
matrix-rate Frobenius norm, and 6.661338147750939e-16 RoPE block rate; the
non-conserved control is 0.3806736206839703. The result is
VERIFIED_SCOPED. The broader claim that no other C1 laws exist is not
machine-formalized here.

## C4 — Theorem 4.7, normalized-sigmoid MoE

The paper claims that every conservation law for normalized-sigmoid-gated
dense and sparse MoE coincides with the softmax case. The audit tests the
paper's stated consequence that the gate-row sum is conserved. It uses d=3,
two exact SwiGLU experts, one hidden unit, x=e1, target zero, squared loss,
and logits (0, ln 3). The normalized sigmoid gates are (2/5, 3/5).

Exact arithmetic gives dL/dc = -9/250 for a common logit shift and hence
d/dt(W11 + W21) = 9/250, while the matched softmax control gives zero. An
independent 80-digit checker agrees. The result is FALSIFIED_SCOPED for the
normalized-sigmoid claim. Dense and sparse softmax laws, and expert-level
SwiGLU invariants, remain separate positive results.

## C5 — Figure 2 scaling

The exact empirical claim concerns O(tau^2 k) conservation-error behavior
for Qwen-3-style and ViT training on WikiText-103, PTB, CIFAR-10, and
ImageNet-1K, with three rates and ten independent seeds per configuration.
The audit completes four routes:

1. A one-step quadratic invariant and a non-conserved control establish the
   expected tau-squared versus tau slopes without presenting it as the paper
   experiment.
2. The exact architecture, dataset, schedule, seed, and block-metric
   contract is enumerated; the training implementation and raw trajectories
   are absent.
3. CPU calibration shows that even the named WikiText lower bound is not a
   suitable substitute for the paper's H100 campaign.
4. The normalized-sigmoid counterexample and softmax control are checked as
   a mandatory falsification route, but they are not named-dataset
   trajectories.

Therefore C5 is BLOCKED, not verified or falsified. The unblockers are the
authors' executable training code and raw trajectories, or authorized
accelerator capacity for the exact campaign.
