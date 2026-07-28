# Current verification — Claim 2 / Theorem 4.2

## VERIFIED: the stated SwiGLU norm-difference invariants

Paper source: [Theorem 4.2](https://ar5iv.labs.arxiv.org/html/2606.17816#S4.Thmtheorem2).

For every hidden index, the transformation
`A[:,i] -> exp(t)A[:,i]`, `C[i,:] -> exp(-t)C[i,:]` leaves the SwiGLU model
exactly unchanged: each term has exponent `+1+0-1=0`. Differentiating this
identity for an arbitrary differentiable dataset loss gives

`d/dt (||A[:,i]||^2-||C[i,:]||^2) = 0`

under Euclidean gradient flow, for arbitrary dimensions and parameter values.

For completeness, orthogonality at `x=t e_j` is expanded in the
arbitrary-width family `SiLU(B_ij t), t SiLU'(B_ij t)`. The same exact
Taylor/rational-pole certificate used on Claim 1 proves this family is
linearly independent on a dense generic set. Its coefficients force
`grad_B h=0` and
`<grad_A[:,i] h,C[i,:]>+<A[:,i],grad_C[i,:] h>=0`; C1 continuity extends the
conditions globally. Substitution proves the converse, establishing the
theorem's iff characterization.

### Independent model-scale audit

The fixed cumulative command ran on HF `cpu-upgrade` at the paper's CIFAR-10
ViT widths:

| Configuration | Result |
| --- | ---: |
| Repeated layer audits / width / intermediate | `12 / 256 / 1024` |
| Independent seeds | `10` |
| Maximum absolute invariant-rate residual | `6.6613e-16` |
| Maximum relative residual | `3.9076e-16` |
| Minimum intended nonconserved-control rate | `1.8073e-3` |
| Formal run / Git | `03fc4630…` / `e441d0c…` |
| CPU / runtime | 64 logical CPUs visible; 1.293s verifier, 21s job |

Fixed command:

```bash
uv run --frozen --no-dev python -m conservation_repro.run
```

Download:
[raw summary](../../evidence/claim2_theorem42/model_scale_summary.json) ·
[structural certificate](../../evidence/claim2_theorem42/structural_certificate.json) ·
[fail-closed checker](../../evidence/claim2_theorem42/verify.py) ·
[full verifier source](../../conservation_repro/claim2_swiglu.py).

Deviation: Appendix D's CIFAR-10 ViT has six layers. The numerical audit
repeats the paper widths over 12 independently sampled layer instances; scale
is corroboration, while the arbitrary-width derivation above carries the
universal result.
