# Manuscript Writing Guide

**Target Journal**: Public Administration Review (PAR)

---

## Formatting Requirements

| Requirement | Specification |
|-------------|---------------|
| Word limit | 8,000 words (including abstract, endnotes, references) |
| Abstract | 150 words maximum |
| Font | 12-point Times New Roman |
| Spacing | Double-spaced |
| Margins | 1-inch on all sides |
| Citation style | Chicago Manual of Style 16th ed., Author-Date |
| Reference names | Full first names required |
| Review type | Blind review (no author identification) |
| Special section | "Evidence for Practice" (3-5 bullet points) |

Reference: https://publicadministrationreview.wordpress.com/submission-process/

---

## Writing Style Rules

### Avoid Metacommentary

Present findings directly without self-referential framing.

| Avoid | Use Instead |
|-------|-------------|
| "This study examines..." | "Government administrative capacity affects..." |
| "This approach advances the literature..." | [Simply present the analysis] |
| "This study provides the first application..." | [Let the contribution speak for itself] |
| "Our findings suggest..." | "Higher disbursement ratios predict..." |
| "We demonstrate that..." | "The results indicate..." |

### Avoid Internal Project References

The manuscript must be standalone, not a response to prior internal work.

| Avoid | Reason |
|-------|--------|
| "Prior latent variable approaches may overcomplicate..." | References archived internal SEM manuscript |
| "Why survival analysis succeeds where SEM fails" | Internal methodology comparison |
| "Unlike traditional SEM approaches..." | Implies response to internal work |
| "More parsimonious than complex measurement models" | Comparative framing against internal work |

### Legitimate External References

Discussing published research is appropriate and expected:

- **GAO/HUD reports**: @gao2019, @hud2020, @hudoig2021
- **Academic literature**: @gerber2022, @peacock2022, @smith2021
- **Methodological references**: @kline2016, @bollen1989

External citations support the literature review and theoretical framework. The prohibition is against referencing prior iterations of this specific project.

### Writing Checklist

Before submitting, verify:

- [ ] No "this study" or "our study" self-references
- [ ] No comparisons to "prior SEM approaches" (internal work)
- [ ] No "advances the literature" or "first application" claims
- [ ] No references to "latent constructs" as inferior alternatives
- [ ] No "why survival analysis succeeds" framing
- [ ] Abstract presents findings directly, not methodology comparisons
- [ ] Evidence for Practice section included
- [ ] Word count under 8,000

---

## File Locations

### Manuscript Files

| File | Purpose |
|------|---------|
| `manuscript_quarto/index.qmd` | Main manuscript |
| `manuscript_quarto/appendix-a-data.qmd` | Appendix A: Data sources and construction |
| `manuscript_quarto/appendix-b-methods.qmd` | Appendix B: Technical methodology |
| `manuscript_quarto/appendix-c-robustness.qmd` | Appendix C: Robustness checks |
| `manuscript_quarto/references.bib` | BibTeX references |
| `manuscript_quarto/csl/chicago-author-date.csl` | Citation style |

### Output

| File | Purpose |
|------|---------|
| `manuscript_quarto/_output/index.html` | Preview format |
| `manuscript_quarto/_output/index.pdf` | Submission format |
| `manuscript_quarto/_output/index.docx` | PAR submission format (primary) |

---

## Rendering the Manuscript

### Standard Render

```bash
cd manuscript_quarto
./render_all.sh
```

### Skip Pipeline (Re-render Only)

```bash
cd manuscript_quarto
CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh
```

### Demo Mode (No Real Data)

```bash
cd manuscript_quarto
CAPACITY_SEM_DEMO=1 ./render_all.sh
```

---

## Archived Manuscript

The original Kaifa SEM manuscript is archived in `manuscript_kaifa_archive/` for reference only.

**Do not copy content that**:
- References internal methodology comparisons
- Uses "this study advances" framing
- Compares survival analysis to "prior SEM approaches"

The archived manuscript used different methodology (SEM with latent variables) that had known issues with right-censoring and sample size. See `manuscript_kaifa_archive/README.md` for details.

---

## Appendix Guidelines

### Appendix A: Data

- Data sources and construction
- Variable definitions
- Sample selection criteria
- No methodology comparisons

### Appendix B: Methods

- Technical specification of survival models
- Cox PH and AFT model details
- Estimation procedures
- No "why this is better than SEM" framing

### Appendix C: Robustness

- Sensitivity analyses are appropriate here
- SEM model comparisons belong in C.2 as technical sensitivity checks
- Present as "alternative specifications" not "failed approaches"
- Keep technical, avoid value judgments

---

## Quick Reference: DO and DO NOT

### DO NOT

- Use "this study" self-references
- Compare to internal prior work (Kaifa SEM manuscript)
- Use metacommentary ("advances the literature", "first application")
- Reference "latent constructs" or "complex measures" comparatively
- Include "why survival analysis succeeds where SEM fails" framing
- Write abstracts that discuss methodology choices rather than findings

### DO

- Present findings directly without self-referential framing
- Reference legitimate external literature appropriately
- Let the methodology speak for itself
- Keep robustness comparisons in appendices
- Follow PAR formatting requirements exactly
- Include Evidence for Practice section
