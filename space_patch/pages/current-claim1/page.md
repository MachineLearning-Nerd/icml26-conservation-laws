# Current verification — Claim 1 / Theorem 4.1

**Verdict: VERIFIED.** This page supersedes the one-candidate drift check in
the historical rejected baseline.

## Exact claim and assumptions

For `f(x;A,B)=A GELU(Bx)` and its SiLU analogue, every C1 conservation law is
constant. “Conservation law” means constant along every Euclidean
gradient-flow trajectory for every dataset and initialization. The loss is C2
and its output-gradient span is the full output space. Source:
[Theorem 4.1](https://ar5iv.labs.arxiv.org/html/2606.17816#S4.Thmtheorem1),
with proofs in Appendix A.1–A.2.

## Why this is proof-level evidence

At `x=t e_j`, orthogonality reduces to a linear combination of
`phi(B_ij t)` and `t phi'(B_ij t)`. GELU and SiLU both have a linear term and
a nonzero coefficient at every positive even power. Equating those
coefficients gives

`sum_i (u_i + m w_i) gamma_i^m = 0` for every `m>=1`,
where `gamma_i=B_ij^2`.

Summing in `m` produces

`sum_i u_i gamma_i z/(1-gamma_i z) + w_i gamma_i z/(1-gamma_i z)^2 = 0`.

For distinct nonzero `gamma_i`, the double pole at `1/gamma_i` forces
`w_i=0`, then its simple pole forces `u_i=0`. Thus every partial derivative
of a C1 law vanishes on a dense generic set. Continuity extends this to all
parameters, and the Euclidean parameter space is connected, so the law is
constant.

The checker validates the rational numerator map with exact Fraction
arithmetic at widths 1–8. A negative control repeats one `gamma`; exact rank
drops as intended. An independent sampled-Jacobian audit checks the derived
full-span consequence for GELU and SiLU at dimensions 8×12, five seeds.

## Reproduce

Pinned Python 3.12 and NumPy 2.3.2 are in `pyproject.toml` and `uv.lock`.

```bash
uv run --frozen --no-dev python -m conservation_repro.run
```

The cumulative verifier exits nonzero on any failed certificate or control.
Download the [standalone verifier](../../evidence/claim1_theorem41/verify.py),
[claim contract](../../evidence/claim1_theorem41/claim_contract.json), and
[formal run summary](../../evidence/claim1_theorem41/formal_run_summary.json).
The [full verifier source](../../conservation_repro/claim1_theorem41.py)
and [cumulative entrypoint](../../conservation_repro/run.py) are also
evaluator-visible.

## Limitation

No interactive proof assistant was used. The universal step is the explicit
arbitrary-width pole-order argument above; the finite exact and numerical
checks are supporting audits, not extrapolated proof.
