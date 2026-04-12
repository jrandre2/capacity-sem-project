# Kaifa Revision Plan

**Manuscript**: `manuscript_kaifa_archive/`
**Review cycle**: `kaifa-2026-r1`
**Last updated**: 2026-04-09

## Goal

Convert the AI-review triage into an execution sequence that fixes the most serious credibility defects before any cosmetic editing or journal-specific polishing.

## Recommendation

Do not treat this as a line-edit pass. The manuscript currently has a credibility problem, not a wording problem.

Recommended strategy:

1. Stabilize the empirical argument.
2. Reframe the manuscript as cross-sectional and associational unless a redesign is completed.
3. Expand transparency around measurement, timing, data construction, and reproducibility.
4. Only then clean figures, captions, references, and style.

## Priority Order

### Phase 0: Quantitative Audit And Go/No-Go Decision

**Priority**: Critical
**Why first**: The current manuscript cannot be revised responsibly until the fitted-model facts are reconciled.

Tasks:

- Recompute and verify Table 1 and Table 2 directly from the analysis outputs.
- Confirm whether the standard and alternative SEM labels were swapped or the explanatory paragraph was written incorrectly.
- Export the full measurement-model outputs for the preferred and alternative specifications.
- Check whether the state and local models in the later results sections are based on the same definitions described in the main model-comparison section.
- Write a short audit note stating which model is actually preferred and why.

Comments addressed:

- Major 1
- Major 2
- Minor 6

Primary files:

- `manuscript_kaifa_archive/index.qmd`
- `manuscript_kaifa_archive/appendix-b-methods.qmd`
- `manuscript_kaifa_archive/code/`
- `manuscript_kaifa_archive/figures/`

Exit criteria:

- Table 1 and narrative agree.
- Preferred model choice is defensible.
- Full loadings and diagnostics are available for manuscript reporting.

### Phase 1: Design Reframing Or Redesign Decision

**Priority**: Critical
**Why second**: Several review findings are structural and cannot be solved by better wording alone.

Tasks:

- Remove all moderation language unless true moderation is added.
- Make the unit of analysis explicit and defend the cross-sectional aggregation.
- Correct the `2001` versus `2003` sample-window inconsistency.
- Add an explicit limitation section on right-censoring, maturity bias, cohort effects, and pooled state/county comparability.
- Decide between two paths:
  - `Salvage path`: keep the paper as an exploratory, cross-sectional SEM manuscript with narrower claims.
  - `Redesign path`: rebuild the empirical core around separate cohorts, multi-group/invariance testing, or a longitudinal/survival framework.

Comments addressed:

- Major 3
- Major 4
- Major 5
- Major 9

Primary files:

- `manuscript_kaifa_archive/index.qmd`
- `manuscript_kaifa_archive/appendix-a-data.qmd`
- `manuscript_kaifa_archive/appendix-b-methods.qmd`

Exit criteria:

- The manuscript no longer claims moderation without a moderation model.
- The framing is consistent with the actual design.
- The team has explicitly chosen `salvage` or `redesign`.

### Phase 2: Data Construction And Reproducibility Appendix

**Priority**: High
**Why now**: Even a narrowed manuscript remains weak without transparent source handling.

Tasks:

- Document QCEW extraction rules, suppression/missingness handling, and why NAICS `925110` is used as a proxy.
- Document how SVI vintages were aligned, or narrow the analysis to avoid invalid cross-vintage comparison.
- Replace or validate the Gigasheet-based geography workflow against official Census relationship or gazetteer files.
- Report geographic match shares by method, unresolved cases, and manual-audit results.
- Rewrite the data-availability statement to separate public source files from derived linkage artifacts.

Comments addressed:

- Major 6
- Major 7
- Major 8
- Major 11

Primary files:

- `manuscript_kaifa_archive/index.qmd`
- `manuscript_kaifa_archive/appendix-a-data.qmd`
- `manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-07.docx`

Exit criteria:

- A reader can understand exactly how the analytic dataset was constructed.
- Public versus restricted materials are clearly distinguished.

### Phase 3: Metric Definitions And Interpretation Cleanup

**Priority**: High
**Why after data appendix**: The gap and risk metrics need definitions that depend on the audited model and cleaned data description.

Tasks:

- Define the administrative gap measures with explicit formulas.
- Define the Recovery Governance Risk Index with weights, scaling, and threshold logic.
- Add a short sensitivity discussion for alternate weighting or scaling choices.
- Rewrite causal and determinative language across the abstract, results, discussion, and conclusion.
- Distinguish workload, capacity proxy, and recovery outcomes more carefully in the interpretation.

Comments addressed:

- Major 9
- Major 10

Primary files:

- `manuscript_kaifa_archive/index.qmd`
- `manuscript_kaifa_archive/appendix-b-methods.qmd`

Exit criteria:

- All composite metrics are reproducible from text alone.
- The manuscript is consistently framed as associative unless the design is upgraded.

### Phase 4: Figure, Terminology, And Reference Cleanup

**Priority**: Medium
**Why later**: These are real issues, but they should not consume attention before the design and measurement problems are settled.

Tasks:

- Fix Figure 3 panel labels.
- Fix the Figure 5 caption/discussion mismatch.
- Replace informal phrasing.
- Standardize `grant`, `program`, `activity`, `grantee`, and `jurisdiction`.
- Rename or explain the so-called partial dependence plots.
- Audit the bibliography for publication status, formatting, and gray-literature labeling.

Comments addressed:

- Major 12
- Minor 1
- Minor 2
- Minor 3
- Minor 4
- Minor 5

Primary files:

- `manuscript_kaifa_archive/index.qmd`
- `manuscript_kaifa_archive/references.bib`
- `manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-07.docx`

Exit criteria:

- No figure-label contradictions remain.
- Bibliography entries are accurate and transparently labeled.
- Terminology is stable throughout the paper.

### Phase 5: Final Verification And Submission Readiness

**Priority**: Medium

Tasks:

- Re-render the manuscript and confirm captions, tables, and appendices match the revised claims.
- Regenerate the Kaifa response letter from the tracker.
- Update the tracker statuses from `VALID - ACTION NEEDED` to final dispositions as changes are completed.
- Re-run `review_verify --manuscript kaifa`.
- Decide whether the manuscript is now:
  - a credible exploratory working paper
  - a submission-ready revised manuscript
  - an archived SEM lineage document that should not be pushed toward publication

Primary files:

- `manuscript_kaifa_archive/REVISION_TRACKER.md`
- `doc/reviews/kaifa/response_letter_*.md`
- `manuscript_kaifa_archive/_output/`

## Immediate Next Actions

If the team wants the fastest path forward, do these next:

1. Audit Table 1 and the underlying SEM outputs.
2. Remove the false moderation language.
3. Rewrite the abstract and conclusion to an associative framing.
4. Add a methods appendix table with full loadings and model diagnostics.
5. Draft the data-construction appendix for QCEW, SVI, and geography matching.

## Stop Conditions

Pause revision and reconsider the manuscript strategy if any of these happen:

- The preferred SEM model cannot be defended after the Table 1 audit.
- The QCEW or SVI constructions cannot be justified cleanly enough for publication.
- The cross-sectional timing design remains too distorted by censoring and cohort bias to support the central claims.

If any stop condition is triggered, the manuscript should be reframed as historical or exploratory, not pushed through as a polished submission draft.
