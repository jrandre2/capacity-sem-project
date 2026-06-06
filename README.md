# A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in CDBG-DR Disaster Recovery

Does government administrative capacity determine how quickly jurisdictions spend disaster-recovery
dollars? That question drives a large body of CDBG-DR research — but the answer turns out to depend
heavily on *how capacity is measured*. This project demonstrates that the capacity–timeliness
coefficient is **not stably identified**: depending on which operationalization, sample scope, or
analytical framework is applied to the same underlying data, the estimated coefficient spans
strongly positive, near-zero, and negative values. Rather than delivering a verdict, this work
offers a **six-item measurement-sensitivity audit protocol** whose output is a specification-curve
dashboard — a tool practitioners and reviewers can use to evaluate whether any capacity finding is
robust enough to support benchmarking or policy anchoring.

**Target journal**: Public Administration Review (PAR)
**Manuscript status**: Ten synthetic review cycles completed (R1–R10); ready for editorial submission.

---

## Manuscript

| File | Description |
|------|-------------|
| [`manuscript.pdf`](manuscript.pdf) | Latest rendered manuscript (PDF) |
| [`manuscript.docx`](manuscript.docx) | Latest rendered manuscript (Word) |
| [`manuscript_quarto/index.qmd`](manuscript_quarto/index.qmd) | Source of truth (Quarto) |
| [`manuscript_quarto/`](manuscript_quarto/) | Full source: appendices, bibliography, render scripts |

Rendered outputs at repo root are copied from `manuscript_quarto/_output/` on each render pass.
The Quarto source is the authoritative version.


## Key Findings

The specification-curve dashboard spans three qualitatively distinct zones:

- **Positive and significant** (β ≈ +0.26 to +0.30): primary local-only SEM (N=543), pooled SEM
  (N=573), residualized burden, and several measurement-preserving variants
- **Near zero or attenuated**: fixed-horizon reconstructed-panel outcomes (q=8/12/16), time-varying
  Cox model (HR ≈ 1.0), non-suppressed-QCEW subsample (β=+0.132, n.s.)
- **Negative and reversed** (β ≈ −0.24 to −0.60): mature-jurisdiction-only sample, raw portfolio
  counts, within-SEM financial-ratio operationalization

The within-Cox divergence is particularly telling: a single-spell baseline Cox on the *same data*
yields HR=1.46–2.58 (p<0.01), the opposite sign from the time-varying Cox. Same data, different
specification, opposite answer.

The pooled-SEM 95% CI under state-clustered bootstrap (1,000 iterations, 35 clusters) is
[−0.129, +0.531] — it crosses zero. Current CDBG-DR capacity estimates are not robust enough for
policy anchoring or benchmarking.

---

## Current Status

- **Primary manuscript**: `manuscript_quarto/index.qmd`
- **Title**: "A Measurement-Sensitivity Audit Protocol for Administrative-Capacity Studies in
  CDBG-DR Disaster Recovery"
- **Contribution**: Six-item measurement-sensitivity audit protocol; deliverable is a
  specification-curve dashboard
- **Data**: 543 local administering jurisdictions (primary); N=573 with state agencies
  (supplementary); 151 grantee-disaster pairs for cross-framework survival analysis
- **Archived drafts**: `manuscript_velocity/` (survival-only draft, superseded) and
  `manuscript_kaifa_archive/` (original SEM draft)

For the full analytical state, next steps, and trusted-findings table see
[doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md). For the review cycle history see
[manuscript_quarto/REVISION_TRACKER.md](manuscript_quarto/REVISION_TRACKER.md).

---

## Repository Map

```
manuscript_quarto/          Primary manuscript source (Quarto / PAR target)
  index.qmd                 Main text
  appendix-a-data.qmd       Data appendix
  appendix-b-methods.qmd    Methods appendix
  appendix-c-robustness.qmd Robustness / sensitivity appendix
  REVISION_TRACKER.md       Review cycle log (R1–R10)
manuscript.pdf / .docx      Latest rendered outputs (root copies)
manuscript_velocity/        Archived survival-only draft (superseded)
manuscript_kaifa_archive/   Archived original SEM draft (source material)

scripts/                    Analysis scripts (sensitivity runs, figures)
src/                        Pipeline library and analysis modules
  pipeline.py               Main CLI entry point
  stages/                   Pipeline stages
  capacity_sem/             Core analysis modules

figures/                    Analysis figures (numbered: fig_01 … fig_13 + extras)
outputs/                    Model outputs, tables, reports
data_raw/                   Source datasets (read-only; large files not committed)
  svi_historical/           CDC/ATSDR SVI vintages 2000–2022
data_work/                  Derived data (.parquet files)

doc/                        All deeper documentation (see below)
tests/                      Regression tests
```

**Data note**: Raw CDBG-DR DRGR/QPR data are large restricted files not committed to this
repository. The committed files in `data_raw/` are the HUD QPR extract used for this study
(`qpr_data.csv`). SVI historicals in `data_raw/svi_historical/` were downloaded 2026-04-14.
Derived `.parquet` files in `data_work/` are generated by the pipeline and are committed for
reproducibility but are not primary data.

---

## Reproducibility

The analysis pipeline requires Python 3.10+ and Quarto. See [doc/PIPELINE.md](doc/PIPELINE.md)
for the full workflow, stage-by-stage commands, and expected outputs.

Quick start (Python environment):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py run_all
```

To render the manuscript: `cd manuscript_quarto && ./render_all.sh`

Quarto is an external CLI; use a system install or the vendored wrapper in `tools/bin/quarto`.
The environment variable `CAPACITY_SEM_SKIP_PIPELINE=1` skips the pipeline re-run during render.

---

## Documentation

| File | Contents |
|------|----------|
| [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md) | Current analytical state, trusted findings table, next steps |
| [doc/PIPELINE.md](doc/PIPELINE.md) | Full pipeline commands, stage reference, review management |
| [doc/METHODOLOGY.md](doc/METHODOLOGY.md) | SEM and survival analysis methods |
| [doc/DATA_DICTIONARY.md](doc/DATA_DICTIONARY.md) | Variable definitions |
| [doc/ETL_STANDARDIZATION.md](doc/ETL_STANDARDIZATION.md) | Fixed-denominator standardization |
| [doc/ANALYSIS_JOURNEY.md](doc/ANALYSIS_JOURNEY.md) | Methodological history and pivots |
| [doc/MANUSCRIPT_GUIDE.md](doc/MANUSCRIPT_GUIDE.md) | Manuscript locations and writing conventions |
| [doc/SYNTHETIC_REVIEW_PROCESS.md](doc/SYNTHETIC_REVIEW_PROCESS.md) | Synthetic review workflow |
| [doc/CHANGELOG.md](doc/CHANGELOG.md) | Cycle-by-cycle revision log |
| [manuscript_quarto/REVISION_TRACKER.md](manuscript_quarto/REVISION_TRACKER.md) | R1–R10 review history and final manuscript metrics |

Historical reports and archived analyses remain in the repo for provenance. Consult
[doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md) before citing any file predating the current revision.
