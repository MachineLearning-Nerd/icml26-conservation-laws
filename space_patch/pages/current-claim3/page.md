# Current verification — Claim 3 / Theorems 4.3–4.4

**Verdict: VERIFIED for every displayed MHA and RoPE invariant.** This
supersedes the standard-attention-only historical rejected baseline.

## Exact contracts

Standard MHA conserves `Q_i^T Q_i-K_i^T K_i` and
`V_i^T V_i-O_i^T O_i` per head. With RoPE, each two-coordinate rotary block
conserves the corresponding Q/K norm difference. Sources:
[Theorems 4.3–4.4](https://ar5iv.labs.arxiv.org/html/2606.17816#S4.Thmtheorem3).

The functional certificates are exact:

- `Q -> Q E`, `K -> K E^{-T}` leaves `Q K^T` unchanged.
- `V -> V F`, `O -> O F^{-T}` leaves `V O^T` unchanged.
- For every fixed RoPE rotation `R_(p-q)`, scaling a Q block by `exp(t)` and
  its matched K block by `exp(-t)` leaves `Q R_(p-q) K^T` unchanged.

Differentiating these symmetries and applying Euclidean gradient flow gives
the displayed conserved matrices.

## Formal run

HF `cpu-upgrade`, commit
`f9a9805c941e4e57e4116b444a50c06f73522778`, 21 s job wall time. The audit
used 12 layers, `d=192`, three 64-dimensional heads, sequence length 32, and
ten deterministic seeds.

| Quantity | Worst observed value |
| --- | ---: |
| RoPE Q/K block rate | `6.661338147750939e-16` |
| V/O matrix rate, Frobenius | `1.051088885904111e-14` |
| Relative invariant rate | `2.2073201955743162e-16` |
| Intended non-conserved control | `0.3806736206839703` |

Download the [raw summary](../../evidence/claim3_attention/formal_run_summary.json),
[structural certificate](../../evidence/claim3_attention/structural_certificate.json),
and [fail-closed checker](../../evidence/claim3_attention/verify.py). The
[full verifier source](../../conservation_repro/claim3_attention.py) is
visible here.

## Reproduce and limitation

```bash
uv run --frozen --no-dev python -m conservation_repro.run
```

The exact symmetries establish the displayed invariants at arbitrary
dimensions. This campaign did not separately machine-formalize the paper's
stronger completeness classification of every possible C1 law.
