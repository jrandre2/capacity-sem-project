---
name: sem-manuscript-audit
description: Audit structural equation modeling manuscripts for model-reporting, measurement, design, timing, and interpretation defects. Use when Codex needs to review an SEM paper, compare tables against prose, verify that moderation or causal claims match the actual specification, check measurement-model completeness, evaluate aggregation or censoring choices, or convert audit findings into tracker-ready revisions.
---

# SEM Manuscript Audit

Use this skill when auditing `manuscript_quarto/` (primary SEM manuscript) or `manuscript_kaifa_archive/` (archived source draft).

## Audit Workflow

1. Build the artifact set.
- Read the abstract, methods, results, appendices, figure captions, tables, and references.
- Read code or outputs if they exist and the manuscript makes claims that can be checked against them.

2. Start with consistency failures.
- Compare model-comparison tables to the surrounding narrative.
- Compare sample sizes, year ranges, units, and fit statistics across sections.
- Confirm figure captions and panel references match the discussion.

3. Audit the measurement model.
- Check whether each latent has enough indicators.
- Look for full loadings, standard errors, residuals, latent correlations, reliability or validity evidence, and respecification logic.
- Flag fit claims that do not match the reported thresholds.

4. Audit the structural claims.
- Compare moderation, mediation, causality, invariance, or multi-group claims against the actual equations.
- Treat additive controls as controls, not moderators.

5. Audit design and timing.
- Check unit of analysis, aggregation, pooling of heterogeneous units, censoring, time-window consistency, and outlier rules.
- Flag cross-sectional claims that are dressed up as longitudinal inference.

6. Audit proxies and data sources.
- Check whether QCEW, SVI, matching crosswalks, or risk indices are adequately defined and defended.

7. Audit interpretation.
- Downgrade deterministic or causal language when the design is associative.
- Flag misleading labels such as `partial dependence plots` when the workflow is not actually PDP-based.
- Flag metacommentary that makes the paper read like a project memo instead of a standalone article.
- Treat phrases like `this revision`, `the revised manuscript`, `the current repo`, `the audit found`, or `the imported outputs` as manuscript defects unless the user explicitly wants a reflexive methods note.

8. Output findings in the right format.
- If the user asked for review, present findings ordered by severity with file references.
- If the user asked for revision support, convert findings into tracker entries or a revision plan.

## High-Risk Failure Modes

- Model-comparison paragraph contradicts its own table.
- Fit indices are poor but described as strong without caveat.
- Moderation or invariance is claimed but not estimated.
- Timing outcomes ignore right-censoring.
- State and local units are pooled without adequate justification.
- Latent constructs rely on proxies that are not defended.
- Data-availability statement hides which components are actually public.
- References overstate preprints or gray literature.

## References

- Read `references/audit-checklist.md` for the detailed issue list and evidence prompts.
