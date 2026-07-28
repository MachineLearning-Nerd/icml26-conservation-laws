# Current verification — Claim 2 / Theorem 4.2

## VERIFIED: the stated SwiGLU norm-difference invariants

Paper source: [Theorem 4.2](https://ar5iv.labs.arxiv.org/html/2606.17816#S4.Thmtheorem2).

For every hidden index, the transformation
`A[:,i] -> exp(t)A[:,i]`, `C[i,:] -> exp(-t)C[i,:]` leaves the SwiGLU model
exactly unchanged: each term has exponent `+1+0-1=0`. Differentiating this
identity for an arbitrary differentiable dataset loss gives

`d/dt (||A[:,i]||^2-||C[i,:]||^2) = 0`

under Euclidean gradient flow, for arbitrary dimensions and parameter values.

### Independent model-scale audit

The fixed cumulative command ran on HF `cpu-upgrade` at the paper's CIFAR-10
ViT widths:

| Configuration | Result |
| --- | ---: |
| Layers / width / intermediate | `12 / 256 / 1024` |
| Independent seeds | `10` |
| Maximum absolute invariant-rate residual | `6.6613e-16` |
| Maximum relative residual | `3.9076e-16` |
| Minimum intended nonconserved-control rate | `1.8073e-3` |
| Formal run / Git | `03fc4630…` / `e441d0c…` |
| CPU / runtime | 64 logical CPUs visible, 1 effective thread; 1.293s verifier, 21s job |

Fixed command:

```bash
uv run --frozen --no-dev python -m conservation_repro.run
```

Download:
[raw summary](../../evidence/claim2_theorem42/model_scale_summary.json) ·
[structural certificate](../../evidence/claim2_theorem42/structural_certificate.json) ·
[fail-closed checker](../../evidence/claim2_theorem42/verify.py).

Limitation: this independently verifies the displayed invariants. It does not
machine-formalize the paper's stronger completeness classification over every
possible `C1` conservation law.

