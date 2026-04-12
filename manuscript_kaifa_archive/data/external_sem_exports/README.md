# External SEM Exports

This directory vendors later Kaifa SEM export bundles that were originally
found outside the repository and imported on 2026-04-09 for provenance and
manuscript audit work.

## Imported bundles

### `baseline_sem/`

Primary later-manuscript SEM export bundle.

- Source: `/Users/jesseandrews/Downloads/baseline_sem`
- Apparent role: main expanded cross-sectional SEM with QCEW-, SVI-, and
  recovery-performance variables
- Most useful files:
  - `sem_model_comparison.csv`
  - `sem_2factor_parameter_estimates.csv`
  - `sem_2factor_fit_statistics.csv`
  - `sem_2factor_ready_dataset.csv`
- Notes:
  - `N = 573`
  - reproduces the one-factor vs two-factor model-comparison issue flagged in
    the Kaifa review
  - `*_factor_loadings.csv` files are empty, but loadings are present in the
    `*_parameter_estimates.csv` files

### `baseline_sem_admin_4/`

Smaller robustness or alternate-sample version of the same expanded SEM
family.

- Source: `/Users/jesseandrews/Downloads/baseline_sem_admin_4`
- Apparent role: reduced or filtered sensitivity sample
- Notes:
  - `N = 169`
  - same variable schema as `baseline_sem/`
  - weaker fit than the larger bundle, so treat as appendix-only robustness

## Not imported here

The following external folders were reviewed but not imported into the Kaifa
archive because they are either duplicates or low-value for the current
manuscript:

- OneDrive snapshot folders duplicating other SEM exports
- Harvey-specific growth-model reports and outputs with very thin effective
  panel counts

Those materials may still be useful for historical exploration, but they are
not the primary evidence base for the current Kaifa manuscript edits.

## Temporary reconstruction workflow

These imported bundles now support a repository-local reconstruction stage:

```bash
python src/pipeline.py run_kaifa_external_replication --bundle baseline_sem
```

That stage is deliberately labeled as a `PROVISIONAL_EXTERNAL_RECONSTRUCTION`.
It regenerates comparison tables and parameter outputs from the imported ready
datasets, but it should not be confused with Kaifa Lu's original analysis
code. Generated files are written with an `external_reconstruction_*` prefix
under `data_work/diagnostics/kaifa_external_replication/`.
