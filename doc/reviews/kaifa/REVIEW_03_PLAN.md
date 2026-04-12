# Kaifa Review #3 Revision Plan

Date: 2026-04-09
Review cycle: `kaifa-2026-r3`
Source tracker: `manuscript_kaifa_archive/REVISION_TRACKER.md`

## Goal

Turn synthetic review `#3` into an execution sequence that preserves as much of
the current manuscript as possible while addressing the most serious remaining
credibility problems.

## Recommendation

Do not treat review `#3` as another appendix-expansion pass. It is asking for a
strategic tightening of the paper.

Recommended path:

1. Keep the paper as an exploratory cross-sectional manuscript.
2. Keep SEM only if the manuscript stops pretending that the measurement model
   is stronger than it is.
3. Demote or simplify the weakest parts of the design rather than layering on
   more secondary material.
4. Move most remaining spatial/gap/risk content out of the main article.

This is the lowest-change path that still responds honestly to the review.

## Decision Gate

Before making line edits, choose the paper form.

### Recommended lane: Narrow exploratory SEM paper

Keep the current two-factor SEM as a descriptive measurement-and-association
framework, but:

- stop treating `Recovery Performance` as equally mature to `Recovery Timeliness`
- stop leaning on the reflective logic as if it were settled
- stop using the unrecovered `N = 169` subset as substantive robustness support
- stop letting the spatial/gap/risk sections compete with the SEM

### Escalation lane: Simpler observed-variable or composite-path paper

Switch if the next round of revisions makes it impossible to defend the
reflective SEM framing without overclaiming.

Trigger conditions:

- the team is not willing to demote or re-specify `Recovery Performance`
- state/local pooled comparability cannot be defended even with stratified
  robustness
- stronger timing controls cannot be recovered cleanly enough for the current
  SEM to remain credible

## Priority Order

### Phase 0: Lock The Manuscript Form

Priority: Critical

Tasks:

- Confirm that the paper will remain an exploratory SEM manuscript rather than
  pivoting immediately to composites/path analysis.
- If staying with SEM, explicitly state that the SEM is a heuristic latent
  structure, not a fully validated confirmatory measurement model.
- Decide now whether `Recovery Performance` will be:
  - demoted within the current SEM framing, or
  - replaced with a simpler observed/composite outcome in a later escalation.

Exit criteria:

- The revision team has chosen one lane.
- The manuscript’s core claim is defined as the workload-manageability /
  timeliness result, not the full current model stack.

### Phase 1: Tighten The Core Claim

Priority: Critical

Why first:

This review is telling us the paper’s strongest claim is narrower than the
current article still implies.

Tasks:

- Rewrite the abstract so the aggregated cross-sectional design is stated in the
  first sentence cluster.
- Reframe the introduction and discussion so the paper’s main contribution is
  workload manageability and timeliness, not a comprehensive latent theory of
  governmental capacity.
- Recast the negative `Administrative Resources -> Recovery Performance`
  coefficient as a potentially confounded association, not as a substantive
  story about resources harming performance.
- Make it explicit that `Recovery Performance` is the weaker and less settled of
  the two outcome constructs.
- Tighten the SVI interpretation so vulnerable populations are framed as the
  contexts in which structural and administrative barriers are harder to
  overcome.

Comments addressed:

- Major 5
- Major 7
- Minor 1
- Minor 7

Primary files:

- `manuscript_kaifa_archive/code/revise_full_sem_manuscript.py`
- `manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_full_revision.docx`

Exit criteria:

- The paper no longer treats performance and timeliness as equally robust.
- The negative resources coefficient is discussed cautiously.
- The abstract foregrounds the pooled cross-sectional design.

### Phase 2: Reduce Measurement Ambition

Priority: Critical

Why second:

The biggest conceptual attack in review `#3` is that the current reflective SEM
logic is doing more work than the indicators can support.

Tasks:

- Add explicit manuscript language that the factors are exploratory latent
  summaries, not settled confirmatory constructs.
- Remove the weak/negative performance indicator from the main model, or move
  the current full performance factor into a secondary/sensitivity role.
- Add a full appendix structural table with:
  - all modeled covariate paths
  - factor correlations
  - estimation details
  - the source of reported AIC/BIC values
- Verify the current AIC/BIC reporting directly from the model outputs and note
  the scale explicitly.
- Decide whether to keep the current `Recovery Performance` factor in the main
  text at all.

Comments addressed:

- Major 1
- Major 5
- Major 9
- Minor 5

Primary files:

- `src/capacity_sem/models/kaifa_recovered_analysis.py`
- `manuscript_kaifa_archive/code/revise_full_sem_manuscript.py`
- `data_work/diagnostics/kaifa_recovered_analysis/`

Exit criteria:

- The manuscript stops relying on a strong confirmatory-reading of the SEM.
- The appendix contains a complete reporting package.
- AIC/BIC reporting is documented and verified.

### Phase 3: Handle Timing And Pooled Comparability More Honestly

Priority: High

Why now:

These are the strongest design threats after the measurement issue.

Tasks:

- Add stronger exogenous timing controls if recoverable:
  - disaster year
  - first QPR date
  - quarters since start
  - grant allocation year
- If clean recovery is not feasible, narrow the timeliness claim further and
  emphasize that even the burden-timeliness result is maturity-sensitive.
- Add explicit state/local stratified robustness or subgroup summaries that show
  whether the main signal is pooled-only.
- Add clearer manuscript language that pooled state/local estimates are
  convenience comparisons, not strong institutional equivalence claims.
- If clustered uncertainty cannot be modeled, acknowledge the likely direction
  of bias in the standard errors.

Comments addressed:

- Major 2
- Major 4

Primary files:

- `src/capacity_sem/models/kaifa_recovered_analysis.py`
- `manuscript_kaifa_archive/code/revise_full_sem_manuscript.py`

Exit criteria:

- The paper either contains stronger timing controls or sharply narrowed timing
  claims.
- The paper either contains stratified robustness or much stronger pooled-model
  caveats.

### Phase 4: Clean Reproducibility And Data-Flow Weaknesses

Priority: High

Why before the next review:

Several review comments now converge on the need for a transparent analysis
pipeline, not just more prose.

Tasks:

- Add a raw-QPR -> jurisdiction-quarter -> 573-unit data-flow table.
- Include unresolved-match and manual-review counts where they can be recovered
  honestly.
- Remove the `N = 169` subset from any main substantive robustness claim unless
  its rule can be reconstructed exactly.
- If retained at all, describe `N = 169` purely as an unrecovered historical
  filtered subset.
- Clean up formula presentation so `avg_employment`, `avg_payroll`, and the
  workload ratios are stated only one way.
- Add frequency reporting for ratios greater than `1.0` and one simple
  truncation/winsorization or robust-estimation sensitivity.

Comments addressed:

- Major 9
- Minor 2
- Minor 3
- Minor 6

Primary files:

- `src/capacity_sem/models/kaifa_recovered_analysis.py`
- `manuscript_kaifa_archive/code/revise_full_sem_manuscript.py`
- `data_work/diagnostics/kaifa_recovered_analysis/`

Exit criteria:

- The paper has a clear data-flow table.
- The `N = 169` subset no longer undermines the article’s credibility.
- Ratio handling is transparent and tested.

### Phase 5: Shrink The Article To Its Strongest Form

Priority: High

Why late:

Scope should be cut only after the core evidence package is decided.

Tasks:

- Move nearly all spatial/gap/risk content to appendix or supplement.
- Keep only the minimum spatial material needed to orient the reader.
- Ensure all maps clearly distinguish state-agency units from county-linked
  local units.
- Fix figure and section numbering so appendix figures are never referenced as
  main-text figures.
- Tighten the literature review around planning, recovery governance,
  implementation capacity, intergovernmental relations, and administrative
  burden.

Comments addressed:

- Major 6
- Major 8
- Minor 4
- Minor 8

Primary files:

- `manuscript_kaifa_archive/code/revise_full_sem_manuscript.py`
- `manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_full_revision.docx`
- `manuscript_kaifa_archive/references.bib`

Exit criteria:

- The main article reads as an SEM-centered paper, not a bundle of side
  projects.
- The literature review has a clearer governance backbone.
- Visual labeling is internally consistent.

## Minimal-Change Package

If the goal is to respond to review `#3` with the least possible manuscript
rebuild, do this sequence:

1. Reframe the abstract, introduction, discussion, and conclusion around an
   exploratory pooled SEM with one main claim: workload manageability aligns
   more consistently with timeliness than raw resource intensity alone.
2. Demote `Recovery Performance` clearly and stop using it as an equal partner
   to timeliness.
3. Add a full appendix coefficient/fit table and an AIC/BIC verification note.
4. Remove the `N = 169` subset from anything that reads like formal robustness.
5. Add the data-flow table and ratio-above-`1.0` frequency/sensitivity table.
6. Move almost all spatial/gap/risk material to appendix or supplement.

That is the cleanest low-churn path.

## Stop Conditions

Pause and reconsider the manuscript form if any of these happen:

- the paper cannot defend the reflective SEM language even after demoting
  `Recovery Performance`
- stronger timing controls cannot be recovered and the team is unwilling to
  narrow the claims further
- pooled state/local robustness looks too unstable to support a shared model
- the manuscript still requires too many caveats to read as a coherent article

If any stop condition is triggered, move to the escalation lane:

- observed composites or simpler path model
- timeliness-centered paper
- spatial/gap/risk material as supplement only

## Immediate Next Actions

1. Write the abstract/core-claim reframing.
2. Decide the fate of `Recovery Performance` in the main model.
3. Add the full appendix coefficient table and AIC/BIC verification note.
4. Remove or fully demote the unrecovered `N = 169` subset.
5. Build the raw-QPR -> 573-unit data-flow table.
