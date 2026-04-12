# Revision Tracker: Predictors Manuscript

## Current Status

**Phase**: Initial Draft Complete
**Last Updated**: December 27, 2025

## Approach Summary

This manuscript presents the **robust null finding** on velocity as the central contribution, with exploratory analysis documenting why alternative predictors fail validation.

**Key Message**: Spending velocity does not predict CDBG-DR program completion once reverse causality is properly addressed.

## Analysis Checklist

- [x] Run predictor discovery analysis (`run_predictor_discovery.py`)
- [x] Review LASSO-selected features
- [x] Review RSF importance rankings
- [x] Identify concordant predictors
- [x] Build final Cox model
- [x] Validate temporal stability of Housing_Pct - **FAILED**
- [x] Retract "velocity paradox" claim (methodological artifact)
- [x] Complete manuscript text with rigorous findings
- [ ] Internal review
- [ ] Synthetic peer review

## Key Findings

### Primary Finding: Robust NULL on Velocity

| Specification | HR | p-value | Status |
|---------------|----|----|--------|
| Time-varying, lag=1 | ~1.00 | >0.95 | **ROBUST NULL** |
| Across thresholds 20-100% | ~1.00 | >0.05 | **ROBUST NULL** |
| 50+ operationalizations | ~1.00 | >0.05 | **ROBUST NULL** |

### Retracted Claims

| Initial Claim | Status | Reason |
|---------------|--------|--------|
| "Velocity paradox" (higher velocity → slower completion) | **RETRACTED** | Reverse causality artifact |
| Housing_Pct predicts completion (HR=15.01) | **EXCLUDED** | Failed temporal stability (78.7% show >30pp change) |

## Validation Protocol

For any candidate predictor to support causal claims, it must pass:

| Criterion | Threshold | Housing_Pct Result |
|-----------|-----------|-------------------|
| Within-program std | < 0.10 | 0.26 **FAIL** |
| First-to-last drift | < 0.15 | 0.32 **FAIL** |
| Programs meeting both | > 50% | 19.7% **FAIL** |

## Manuscript Structure

1. **Introduction**: Frame the velocity hypothesis
2. **Methodological Challenge**: Why static analysis fails (reverse causality)
3. **Methods**: Time-varying Cox with lagged covariates
4. **Results**:
   - Primary: Velocity NULL (HR ≈ 1.00)
   - Secondary: Housing_Pct fails validation
5. **Discussion**: The null finding IS the finding
6. **Evidence for Practice**: Policy implications

## Review History

| Cycle | Date | Focus | Status |
|-------|------|-------|--------|
| 1 | Dec 27, 2025 | Initial draft | Complete |
| 2 | TBD | Internal review | Pending |

## Open Questions (Resolved)

1. ~~Does employment data predict completion?~~ → No significant signal
2. ~~Do external capacity measures outperform velocity?~~ → No, none validated
3. ~~What is the concordance index?~~ → 0.662 (final Cox), 0.880 (RSF)

## Files

| File | Description |
|------|-------------|
| `index.qmd` | Main manuscript |
| `references.bib` | Bibliography |
| `render_all.sh` | Rendering script |
| `_quarto.yml` | Quarto configuration |
