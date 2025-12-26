# Kaifa's Original Manuscript (Archived)

**Archived:** December 26, 2024

This directory contains the original manuscript based on Kaifa's SEM methodology.

## Key Claims in Original Manuscript

- **Structural path:** β = 71.024, p = 0.01
- **Sample:** N=36-40 grantees
- **Methodology:** Cross-sectional SEM with latent capacity construct

## Methodological Issues Identified

See `doc/ANALYSIS_COMPARISON_REPORT.md` for detailed comparison.

1. **Right-censoring:** 73.7% of observations lack valid Duration at 95% threshold
2. **Mathematical circularity:** Timeliness = 1/Duration as capacity indicator creates artificial coupling with Duration outcome
3. **Grantee-level aggregation:** Averaging across disasters reduces variance and may inflate effects

## Why This Was Archived

Alternative analysis using survival methods (Cox Proportional Hazards, AFT) with proper censoring treatment (N=152) showed:
- **Cox:** HR = 4.37, p = 0.006 (significant)
- **AFT:** TR = 0.16, p = 0.0001 (significant)

However, standard SEM approaches without the Timeliness/Duration circularity showed no significant effects (p > 0.13).

The new manuscript uses survival analysis as the primary analytical approach.
