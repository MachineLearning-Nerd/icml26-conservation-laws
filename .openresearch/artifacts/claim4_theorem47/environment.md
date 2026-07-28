# Command and environment

- Fixed inherited command:
  `uv run --frozen --no-dev python -m conservation_repro.run`
- Standalone verifier:
  `uv run --frozen --no-dev python -m conservation_repro.claim4_theorem47`
- Fail-closed control:
  `uv run --frozen --no-dev python -m conservation_repro.claim4_theorem47 --assert-theorem`
- Python: `>=3.12,<3.13`, locked by `uv.lock`.
- NumPy: `2.3.2` (used only by cumulative historical regression checks).
- Exact counterexample dependencies: Python standard library only.
- Determinism: exact arithmetic; no random seed is involved in the
  counterexample. Historical regressions retain seeds 0–5.
- Estimated compute: one core, under five minutes.
- Selected compute: local backend, one effective thread.
- Git SHA and formal run ID are populated in the experiment description after
  the formal run; the verifier prints visible CPU allocation and runtime.

