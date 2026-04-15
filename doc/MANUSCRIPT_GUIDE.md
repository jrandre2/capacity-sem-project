# Manuscript Guide

## Which Manuscript Is Active

- Primary research manuscript: `manuscript_quarto/` — *"A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery"*
- Archived survival draft: `manuscript_velocity/` (superseded)

The active Capacity-SEM paper is `manuscript_quarto/index.qmd`. After ten synthetic peer review cycles (R1–R10), it is structured around a six-item measurement-sensitivity audit protocol with a specification-curve dashboard as the deliverable.

## Current Writing Rule

The contribution is the **audit protocol**, not a substantive capacity-timeliness coefficient. The cross-sectional SEM and Cox survival serve as demonstrations of audit items, not as authoritative inferential claims.

Do not treat the following as current findings unless you are explicitly discussing superseded framings:
- The reference β = +0.266 as a substantive estimate (it is one row of the dashboard)
- "Capacity matters for timeliness" / "high capacity speeds recovery" headlines
- Pre-pivot framings that present the SEM as the primary inferential engine
- "Near zero" verdicts that privilege the measurement-appropriate slice (R10 walked this back)

Current framing should align with [PROJECT_STATUS.md](PROJECT_STATUS.md):
- Headline: capacity-timeliness coefficient is *not stably identified* under principled measurement perturbations
- Primary specification: local-only (N=543), β = +0.257
- Pooled supplementary (N=573): β = +0.266 with cluster-bootstrap CI [−0.129, +0.531] crosses zero
- Dashboard spans positive, near-zero, and negative estimates across Class Ia / Ib / II-C / II-O perturbations
- Cross-framework Cox is itself measurement-sensitive (null time-varying; positive baseline)
- Vulnerability *direction* is more dashboard-stable than capacity-timeliness, but specific theme-to-outcome assignments are vintage-sensitive

## File Map

| Path | Purpose |
|------|---------|
| `manuscript_quarto/index.qmd` | Main paper (cross-sectional SEM) |
| `manuscript_quarto/appendix-a-data.qmd` | Data appendix |
| `manuscript_quarto/appendix-b-methods.qmd` | Methods appendix |
| `manuscript_quarto/appendix-c-robustness.qmd` | Robustness appendix |
| `manuscript_quarto/REVISION_TRACKER.md` | Review and verification tracker |
| `manuscript_quarto/render_all.sh` | Multi-format render script |
| `manuscript_velocity/index.qmd` | Archived survival draft |

## Rendering

```bash
cd manuscript_quarto
./render_all.sh
```

Re-render without changing upstream analysis inputs:

```bash
cd manuscript_quarto
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh
```

## Review Commands

```bash
python src/pipeline.py review_status --manuscript quarto
python src/pipeline.py review_verify --manuscript quarto
python src/pipeline.py review_new --manuscript quarto --focus par_general
```

Current tracker status should be read from:
- `manuscript_quarto/REVISION_TRACKER.md`
- `doc/MANUSCRIPT_REVISION_CHECKLIST.md`

## Writing Guardrails

- Present the audit protocol as the contribution; the SEM and Cox results are demonstrations.
- Present the dashboard as the deliverable; do not privilege any single slice (positive ε-offset, near-zero non-suppressed, negative bridge) as the "true" effect.
- Use stability text labels (Stable / Attenuated / Reversed); never emoji flags.
- Use the class taxonomy (Ia / Ib / II-C / II-O / III) consistently across §4.1 prose, tables, and figure.
- Avoid metacommentary ("this study," "advances the literature," "novel contribution").
- Avoid internal "SEM vs survival" victory framing; cross-framework divergence is itself an audit-item finding (Item 4), not an invalidation of one method.
- `manuscript_quarto/` is the primary manuscript; `manuscript_velocity/` and `manuscript_kaifa_archive/` are archived.
