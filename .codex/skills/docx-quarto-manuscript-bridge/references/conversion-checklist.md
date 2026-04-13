# DOCX Quarto Conversion Checklist

Use this checklist when converting or aligning manuscripts between DOCX and Quarto.

## Front Matter

- Title matches the supplied manuscript
- Author line matches the supplied manuscript
- Affiliations preserved
- Abstract label matches manuscript convention
- Keywords preserved if present

## Structure

- Numbered section heads preserved or intentionally changed
- Appendix boundaries preserved
- Data-availability statement retained
- References section retained
- Manuscript reads as a standalone article rather than a revision memo
- No metacommentary such as `this revision`, `the current repository`, or `the audit found`
- Limitations are phrased as study/design limitations, not process-history notes

## Tables And Figures

- Table count checked before and after conversion
- Figure references still resolve
- Captions are present and readable
- No table is split or clipped in the rendered PDF

## Citations And References

- `.bib` includes all cited works
- No broken citation keys
- Reference heading style matches the intended manuscript convention
- Preprints and gray literature labeled transparently if used

## Quarto And DOCX Bridge

- If Quarto `reference-doc` is insufficient, use `python-docx` postprocessing
- If the supplied DOCX uses direct formatting, compare visible conventions instead of style names
- Re-render after each meaningful formatting adjustment

## Verification

- Inspect the actual `.docx`, not just source `.qmd`
- Spot-check rendered PDF pages for front matter and tables
- Update review trackers only after render verification
