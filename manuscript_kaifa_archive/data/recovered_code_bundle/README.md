# Recovered Kaifa Code Bundle

This directory stores the recovered Kaifa analysis ZIP and the extracted files
used to update the manuscript analysis workspace.

## Contents

- [`sem_output_and_partial_dependence_plots.zip`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/sem_output_and_partial_dependence_plots.zip)
- [`extracted/SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/SEM_model_latest_all_funds_latent_all_year_v2_final.ipynb)
- [`extracted/sem_partial_dependence_outputs_4_v2_final.ipynb`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/sem_partial_dependence_outputs_4_v2_final.ipynb)
- [`extracted/all_state_local_fund_latent_var_4_v2.csv`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/data/recovered_code_bundle/extracted/all_state_local_fund_latent_var_4_v2.csv)

The ZIP also contained a Word manuscript, which is preserved separately at
[`source_docs/SEM_Manuscript_2026-04-09_code_bundle.docx`](/Volumes/T9/Projects/capacity-sem-project/manuscript_kaifa_archive/source_docs/SEM_Manuscript_2026-04-09_code_bundle.docx).

## Status

The core SEM notebook is now the primary recovered source-code reference for
the later Kaifa manuscript. Run it through the repository port with:

```bash
python src/pipeline.py run_kaifa_recovered_analysis
```

Important caveat:

- the saved notebook outputs reference a `573`-row SEM run
- the recovered CSV currently bundled here has `577` rows

The current best reconstruction of the missing `573`-row sample excludes:

- `Collier County, FL`
- `KY`
- `Leon County, FL`
- `Nash County, NC`

The repository therefore treats this as a recovered-code bundle with an
input-version discrepancy, not as a perfectly closed raw reproduction.

For a fuller audit, see
[`doc/reviews/kaifa/CODE_BUNDLE_AUDIT.md`](/Volumes/T9/Projects/capacity-sem-project/doc/reviews/kaifa/CODE_BUNDLE_AUDIT.md).
