# Environment and reproduction contract

## Fixed command

~~~bash
uv sync --frozen
uv run --frozen --no-dev python -m conservation_repro.run
~~~

The same cumulative command is exposed in the evaluator-visible space_patch
source and is fail-closed on any recorded certificate or intended control
failure.

## Pinned environment

- Python: >=3.12,<3.13
- NumPy: 2.3.2
- Dependencies: pyproject.toml and uv.lock
- Backend for recorded formal jobs: CPU, Hugging Face cpu-upgrade
- Logical CPUs visible during recorded jobs: 64
- GPU: none
- Threading: the recorded formal jobs used an effective single-thread limit
  where stated in the raw summaries

The exact counterexample uses rational arithmetic plus an independent
high-precision decimal checker. C1–C3 use structural certificates and finite
float64 audits. C5 is deliberately not rerun with substitute models,
datasets, checkpoints, or unauthorized accelerator access.

## Recorded evidence inputs

- C1 run: dc3ad414-e440-4a58-96cb-c4ecde4de22b
- C2 run: 03fc4630-8a4f-45ae-a310-672283d1b97e
- C3 run: dd30dd24-c4b3-48d0-b39a-2b4987e67580
- C5 audit run: 0ec4b425-e079-4ad4-b221-05809ef01326
- Exact C4 result: .openresearch/artifacts/claim4_theorem47/raw/exact_certificate.json

Raw summaries, source anchors, limitations, and standalone checkers are
committed under .openresearch/artifacts/ and mirrored where appropriate under
space_patch/evidence/.
