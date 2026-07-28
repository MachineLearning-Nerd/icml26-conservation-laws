# Evaluator-blind pre-publication review

Review basis: only this downloaded candidate tree and the evaluator rubric.
No OpenResearch logs, dashboard state, unpublished branch knowledge, or
instructions about evidence locations were used to fill gaps.

## First pass and fix

The first blind traversal opened the canonical README and five current pages.
It found that source and lock files were reachable but nested under `source/`
and `environment/`; therefore the advertised root-level fixed command was not
directly executable from the Space checkout. Publication was stopped. A
packaging-only child mirrored the locked environment and cumulative source to
Space root and reran the complete formal regression successfully.

The findings below are the mandatory second blind traversal of the freshly
assembled fixed candidate.

## Files opened from the canonical entrypoint

1. `README.md`
2. `logbook.json`
3. `pages/index.md`
4. `pages/current-claim1/page.md`
5. `pages/current-claim2/page.md`
6. `pages/current-claim3/page.md`
7. `pages/current-claim4/page.md`
8. `pages/current-claim5/page.md`
9. Every raw JSON, checker, claim contract, certificate, and source file
   directly linked by those pages and by the README visibility matrix
10. `pyproject.toml` and `uv.lock`
11. Historical `pages/verify/page.md` and `pages/overview/page.md`

## Findings

- The current verifier is obvious from README and appears before historical
  pages in navigation.
- All five claim contracts, assumptions, source anchors, commands, pinned
  environment, raw values, source code, checkers, controls, limitations,
  run/commit provenance, CPU data, and verdicts are reachable.
- All local Markdown links in the traversed current pages resolve.
- All five evaluator checkers execute successfully and fail closed on their
  recorded conditions.
- The root command shown in every page matches the immutable OpenResearch
  command. Root source and lock files are present.
- Claim 5 is visibly BLOCKED, not passed or described as full scale.
- The exact two historical pages are byte-identical to the judged revision and
  are labeled “Historical rejected baseline” in current navigation.

Conclusions that could not be verified: none for the published verdicts.
Claim 5’s paper-scale empirical result itself cannot be verified; that missing
capability is the reason for its explicit BLOCKED verdict.

Review result: PASS. Visibility matrix has no missing cells.
