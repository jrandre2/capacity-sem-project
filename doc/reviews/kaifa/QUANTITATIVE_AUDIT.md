# Kaifa Quantitative Audit

**Date**: 2026-04-09  
**Review cycle**: `kaifa-2026-r1`  
**Decision**: no-go for confirmatory SEM salvage; go for exploratory archival redraft

## Scope

This audit reconciles the Kaifa Quarto manuscript, the April 7, 2026 Word
draft, and the archived diagnostic tables in:

- `manuscript_kaifa_archive/data/`
- `data_work/diagnostics/`
- `src/stages/s03_manuscript_replication.py`

## Findings

### 1. The model-comparison narrative in the Word draft is not reliable

The Word manuscript's Table 1 narrative does not match the reported fit
statistics. The archive does not support a single cleanly preferred model:

- `full` has the strongest conventional CFI/TLI among archived models but
  poor RMSEA and overlapping timing indicators.
- `exp_optimal_v1` has the lowest AIC/BIC but weak incremental fit.
- `improved_3x3` produces inadmissible CFI/TLI values above 1 in a very
  small sample.

### 2. The large positive legacy state coefficient is reproducible only in the flawed legacy design

The archived replication can reproduce a large positive state-level
capacity-to-outcome path:

- archived replication: `beta = 113.652`, `p < 0.001`, `N = 36`
- later Word draft claim: `beta = 71.024`, `p = 0.01`

That result depends on:

- right-censoring incomplete programs by setting duration equal to current
  observation time;
- defining `Timeliness_censored = 1 / Duration_censored`;
- placing `Duration_censored` on the outcome factor while placing its
  inverse on the capacity factor.

This is not a defensible confirmatory measurement design.

### 3. Non-circular SEM specifications do not support the same substantive claim

Archived baseline and alternative models are uniformly weak:

- `exp_optimal_v1`, all grantees: `beta = 0.320`, `p = 0.958`, `N = 40`
- `exp_optimal_v1`, state grantees: `beta = 10.031`, `p = 0.834`, `N = 15`
- `exp_optimal_v1`, state grantee-disaster: `beta = 0.362`, `p = 0.970`, `N = 23`
- duration-free and milestone alternatives remain non-significant across the
  archived tables

### 4. State-local comparisons are too underpowered for the current manuscript's claims

The archived subset comparison has:

- state subset: `N = 23`
- local subset: `N = 17`

Those sample sizes are too small for the strong state-versus-local claims in
the Word draft.

### 5. The current repository does not fully reconstruct the Word draft's external-data extensions

The later Word draft references QCEW, SVI, and geography matching, but the
current reproducible Quarto archive does not contain a complete audited
reconstruction of those linkage artifacts. They should not be interpreted as
active empirical inputs in the redrafted manuscript until that provenance
work is done.

## Go / No-Go Decision

### No-go for confirmatory SEM salvage

The archive cannot support a publication-ready confirmatory SEM manuscript
that keeps the later Word draft's state/local effect claims.

### Go for exploratory archival redraft

The archive can support a transparent, narrower manuscript that:

1. presents the descriptive throughput patterns;
2. reports the archived model-comparison results honestly;
3. treats the large legacy state result as a design-dependent replication
   artifact;
4. frames all SEM results as exploratory associations;
5. documents the provenance boundary for later QCEW, SVI, and geography
   variables.

## Implementation Outcome

The Quarto manuscript was rewritten to follow the exploratory archival
redraft path. Unsupported confirmatory claims were removed rather than
patched with unverifiable prose.
