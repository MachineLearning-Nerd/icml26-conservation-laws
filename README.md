# Conservation Laws for Modern Neural Architectures

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-conservation-laws/blob/main/notebooks/conservation_laws_tutorial.py)

Independent claim-by-claim reproduction audit for [*Conservation Laws for Modern Neural Architectures*](https://arxiv.org/abs/2606.17816), by Viet-Hoang Tran, Vinh Khanh Bui, Tan Lai Ngoc, Nam Nguyen, Tuan Dam, and Tan M. Nguyen. This repository is an independent reproduction and evidence audit, not the authors' official implementation.

The standardized dossier is available in [STATUS.md](STATUS.md),
[CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md), [SOURCE_AUDIT.md](SOURCE_AUDIT.md),
[ENVIRONMENT.md](ENVIRONMENT.md), [REPORT.md](REPORT.md),
[CITATION.cff](CITATION.cff), [AUTHOR_THANK_YOU.md](AUTHOR_THANK_YOU.md),
[claims.json](claims.json), [reproduction_verdicts.json](reproduction_verdicts.json),
and [verify_final.py](verify_final.py).

## Paper in one paragraph

The paper develops a unified framework for conservation laws under gradient flow in modern neural architectures. It studies feedforward networks with GELU, SiLU, and SwiGLU activations, multi-head attention with sinusoidal and rotary positional encodings, and mixture-of-experts models with different gating designs. The paper derives structural invariants and supports them with numerical experiments, while this repository separates theorem-level certificates, finite audits, exact counterexamples, and unavailable paper-scale experiments.

## Audit headline

The previous live judged score is **5/10**. The current audit records Claims 1, 2, and 3 as **VERIFIED within the stated audit scopes**, Claim 4 as **FALSIFIED within the exact counterexample scope**, and Claim 5 as **BLOCKED**. The conservative forecast is 7–9/10, with 9/10 the best-supported possible score; that forecast is not a judge result and no current score is claimed.

## Claim and evidence ledger

| Claim | Paper result | Audit status | How the result is produced |
| --- | --- | --- | --- |
| 1. GELU/SiLU conservation laws | Theorem 4.1 characterizes the constant conservation laws | **VERIFIED** | Reconstruct the arbitrary-width Taylor/pole argument, check exact Fraction ranks through width 8 and exact ranks 2–16, and use a repeated-pole rank-loss control. |
| 2. SwiGLU conservation laws | Theorem 4.2 gives the gradient characterization and norm-difference laws | **VERIFIED** | Prove the inverse scaling symmetry, reconstruct the iff gradient condition, then audit paper-width blocks (d=256, width 1024, 12 repeated instances, 10 seeds) with an independent checker. |
| 3. MHA and positional-encoding laws | Theorem 4.4 gives attention and sinusoidal/RoPE invariants | **VERIFIED** | Apply exact Q/K and V/O group actions, include the matched RoPE rotations, and compare invariant rates with non-conserved controls across 12 layers and 10 seeds. |
| 4. Dense, sparse, and normalized-sigmoid MoE laws | Theorem 4.7 claims the normalized-sigmoid laws coincide with softmax laws | **FALSIFIED** | At d=3 with two exact SwiGLU experts, input e1, logits (0, ln 3), target zero, and squared loss, exact arithmetic gives d/dt (W_1,1 + W_2,1) = 9/250; an 80-digit independent checker confirms the mismatch. Dense and sparse softmax laws are audited separately. |
| 5. Figure 2 scaling on four named datasets | The paper reports scaling behavior for the named experiments | **BLOCKED** | Complete four routes: source/asset availability, non-circularity, CPU workload calibration, and assumption-preserving falsification. The exact 120+-configuration H100 campaign, training implementation, and raw trajectories are unavailable under the authorized CPU-only contract. |

Claim 5 is deliberately **BLOCKED**, not failed or verified. A toy WikiText run or the exact Claim 4 counterexample cannot establish or contradict the paper's named-dataset trajectories.

## How each claim is produced

Each claim follows the same evidence path:

1. Freeze the theorem or empirical contract in .openresearch/artifacts/<claim>/claim_contract.json.
2. Implement the primary proof, symmetry, counterexample, or availability audit in conservation_repro/.
3. Run an independent checker and a deliberately non-conserved or assumption-breaking control.
4. Preserve raw summaries, source audits, limitations, exact commands, and fail-closed outputs.
5. Publish the cumulative result through reports/conservation-laws/, the space_patch/ evaluator mirror, and its release manifests.

The fixed local command is:

~~~
uv sync --frozen
uv run --frozen --no-dev python -m conservation_repro.run
~~~

The locked environment is Python 3.12 with NumPy 2.3.2. Short checks ran locally on CPU; uncertain or multi-core formal jobs used Hugging Face cpu-upgrade. No GPU was used for this reproduction.

## Repository contents

- conservation_repro/ — claim verifiers, exact certificates, symmetry checks, and independent controls.
- reports/conservation-laws/report.md — illustrated technical report with equations, evidence, and limitations.
- reports/conservation-laws/release-report.md — evaluator-facing score and release record.
- notebooks/conservation_laws_tutorial.py — self-contained tutorial notebook.
- .openresearch/artifacts/ — claim contracts, source audits, raw summaries, and checker outputs.
- space_patch/ — evaluator-visible static Space source, evidence pages, and manifests.
- branch-audit.md — mapping from former generated branch names to clean names.

## Branch map

main is the cumulative publication surface. Focused branches preserve the theorem, counterexample, availability, and release lineage; the complete migration mapping is in branch-audit.md.

| Clean branch | Purpose | Status |
| --- | --- | --- |
| [historical/judged-baseline](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/historical/judged-baseline) | Preserve the original 5/10 judged toy baseline and source provenance | Historical record |
| [audit/claim4-normalized-sigmoid](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/audit/claim4-normalized-sigmoid) | Add the exact Theorem 4.7 normalized-sigmoid counterexample | Claim 4 falsified |
| [audit/claim2-swiglu-symmetry](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/audit/claim2-swiglu-symmetry) | Reconstruct the SwiGLU theorem and model-scale audit | Claim 2 evidence |
| [audit/claim3-attention-rope](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/audit/claim3-attention-rope) | Add MHA and RoPE group-action certificates | Claim 3 evidence |
| [audit/claim1-gelu-silu](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/audit/claim1-gelu-silu) | Calibrate the arbitrary-width GELU/SiLU completeness theorem | Claim 1 evidence |
| [audit/claim5-figure2-availability](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/audit/claim5-figure2-availability) | Complete the four-route audit of the paper-scale Figure 2 claim | Claim 5 blocked |
| [release/evaluator-candidate](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/release/evaluator-candidate) | Assemble the cumulative evaluator-visible release | Release candidate |
| [release/space-root-package](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/release/space-root-package) | Make the fixed verifier runnable from the Space root | Published package |
| [main](https://github.com/MachineLearning-Nerd/icml26-conservation-laws/tree/main) | Current README, report, notebook, and evidence mirror | Current |

## Citation

~~~
@article{tran2026conservation,
  title         = {Conservation Laws for Modern Neural Architectures},
  author        = {Tran, Viet-Hoang and Bui, Vinh Khanh and Ngoc, Tan Lai and Nguyen, Nam and Dam, Tuan and Nguyen, Tan M.},
  journal       = {arXiv preprint arXiv:2606.17816},
  year          = {2026},
  doi           = {10.48550/arXiv.2606.17816},
  url           = {https://arxiv.org/abs/2606.17816}
}
~~~

Paper: [arXiv:2606.17816](https://arxiv.org/abs/2606.17816), published at ICML 2026. The evaluator-visible historical artifact is preserved in the [DineshAI/ay4Q69fAJL Space](https://huggingface.co/spaces/DineshAI/ay4Q69fAJL/commit/6219f9cc8d0132ab7407b2d31712ec4180c13c67).

## Thank you

Thank you to Viet-Hoang Tran, Vinh Khanh Bui, Tan Lai Ngoc, Nam Nguyen, Tuan Dam, and Tan M. Nguyen for developing the conservation-law framework and sharing the paper that made this independent audit possible. The reproduction distinguishes theorem evidence, exact counterexamples, and genuinely unavailable experiments so the scope of the authors' claims remains visible.

## Attribution and limitations

This repository is maintained by [MachineLearning-Nerd](https://github.com/MachineLearning-Nerd). It is not affiliated with the paper's authors. Symbolic certificates and finite numerical audits support the theorem claims but do not replace a formal proof assistant or the paper's full experimental campaign. Claim 5 requires the authors' executable training code, raw trajectories, or equivalent paper-scale accelerator access.
