# Current verification — Claim 4 / Theorem 4.7

## FALSIFIED: normalized-sigmoid gating does not conserve the softmax row-sum invariant

Paper source: [Theorem 4.7](https://ar5iv.labs.arxiv.org/html/2606.17816#S4.Thmtheorem7)
and [Theorem 4.5](https://ar5iv.labs.arxiv.org/html/2606.17816#S4.Thmtheorem5).

Theorem 4.7 says every conservation law of normalized-sigmoid-gated MoE
coincides with the softmax-gated case. Theorem 4.5 includes
`h(W)=sum_i W_i`. Because conservation is quantified over every dataset and
initialization, one assumption-satisfying point with `dh/dt != 0` falsifies
that exact statement.

### Explicit construction and exact result

Use `d=3`, two SwiGLU experts, one hidden unit, `x=e1`, target `y=0`, and
squared loss. Explicit expert parameters produce `E1(x)=0` and `E2(x)=e1`.
Set gate logits to `(0, ln 3)`. The normalized sigmoid values are `(1/2,3/4)`
and the normalized gates are `(2/5,3/5)`.

Exact rational differentiation gives:

| Quantity | Exact value |
| --- | ---: |
| `dL/dz1` | `-9/125` |
| `dL/dz2` | `9/250` |
| Common-shift derivative `dL/dc` | `-9/250` |
| Gradient-flow derivative `d/dt(W11+W21)` | **`9/250`** |
| Matched softmax control `dL/dc` | **`0`** |

The claimed invariant therefore changes at nonzero instantaneous rate.

### Dense and sparse softmax variants are also addressed

For Theorem 4.5, a common shift of every gate row adds the same scalar to all
logits, leaving dense softmax probabilities unchanged. For Theorem 4.6
(`k>1`), that shift also preserves the complete ordering, the Top-k set, and
the renormalized softmax probabilities inside the selected set. Every expert
retains its independent SwiGLU inverse-scaling symmetry. A control that shifts
only one selected expert changes the sparse output as intended. Download the
[dense/sparse structural certificate](../../evidence/claim4_theorem47/softmax_sparse_certificate.json).

### Independent checker and fail-closed control

An independent 80-digit central difference along the common-logit-shift
direction returned
`-0.0359999999999999999999999999999999999999999999999999772`,
absolute error `2.28e-53` from `-9/250`. The matched softmax derivative had
absolute magnitude `5e-56`.

The standalone verifier exits zero only when the exact counterexample,
independent checker, and softmax control all pass:

```bash
uv run --frozen --no-dev python -m conservation_repro.claim4_theorem47
```

The deliberately false theorem assertion is the negative control and exits 1:

```bash
uv run --frozen --no-dev python -m conservation_repro.claim4_theorem47 --assert-theorem
```

Download:
[raw exact certificate](../../evidence/claim4_theorem47/exact_certificate.json) ·
[independent checker output](../../evidence/claim4_theorem47/independent_checker_output.json) ·
[standalone verifier](../../evidence/claim4_theorem47/verify.py) ·
[claim contract](../../evidence/claim4_theorem47/claim_contract.json) ·
[full verifier source](../../source/conservation_repro/claim4_theorem47.py).

### Assumptions and scope

Squared loss is `C2` and has `V_ell=R^3`; `h` is linear and therefore `C1`;
the dynamics are Euclidean gradient flow; and the experts are exact instances
of Equation 4. The displayed dense and sparse softmax laws are structurally
verified. The counterexample falsifies the normalized-sigmoid dense-gating
portion of Theorem 4.7. It does not falsify softmax laws or expert-level
SwiGLU invariants.

The prior `verify` and `overview` pages are preserved as the
**Historical rejected baseline** and are superseded by this current verifier.
