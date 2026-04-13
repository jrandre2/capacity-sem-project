---
name: docx-quarto-manuscript-bridge
description: Convert or align manuscript content between DOCX and Quarto in this repository. Use when Codex needs to turn a provided Word manuscript into Quarto section structure, preserve front matter and references while rewriting in `.qmd`, or reformat a Quarto-rendered DOCX so it matches a supplied Word manuscript's layout and conventions.
---

# DOCX Quarto Manuscript Bridge

Use this skill for manuscript conversion and formatting-bridge work in this repository, especially under `manuscript_quarto/` (primary) and `manuscript_kaifa_archive/` (archived source).

## Workflow

1. Identify the direction of travel.
- `DOCX -> Quarto`: extract manuscript structure and convert the Word document into `index.qmd` plus appendices.
- `Quarto -> DOCX style sync`: render Quarto first, then reapply the supplied Word manuscript's formatting conventions to the rendered DOCX.

2. Inspect the source manuscript before editing.
- Use `python-docx` to inspect paragraph order, front matter, heading patterns, table count, and reference formatting.
- Do not assume the supplied Word manuscript is a true reusable template. Many manuscript DOCX files use direct `Normal` formatting rather than stable Word styles.

3. For `DOCX -> Quarto`, preserve structure before prose cleanup.
- Capture the title, author line, affiliations, abstract, keywords, numbered sections, tables, appendices, and data-availability statement.
- Keep the manuscript's section numbering and major headings stable unless the user explicitly wants a redesign.
- Move core manuscript content into `index.qmd` and split long methods/data/robustness sections into appendix `.qmd` files only when that materially improves maintainability.

4. For `DOCX -> Quarto`, preserve references and manuscript conventions.
- Carry over the full bibliography and check that every in-text citation has a matching `.bib` entry.
- Keep journal-facing labels such as `Abstract:`, `Keywords:`, and `Data Availability Statement` if those are part of the supplied manuscript style.
- When Quarto cannot express a Word-only convention cleanly, document the gap instead of silently dropping it.

5. Preserve standalone manuscript voice.
- The manuscript must read as a self-contained journal article, not as a revision memo, repository audit, or project status report.
- Do not write phrases such as `this revision`, `the revised manuscript`, `the current repository`, `the audit found`, `the imported bundle`, `the workflow in this repo`, or similar metacommentary into the manuscript body.
- Put process history, provenance disputes, audit notes, and revision rationale in trackers, memos, appendices for reproducibility, or companion notes, not in the article narrative itself.
- If a limitation must remain because evidence is incomplete, state it as a manuscript-facing limitation of the study, data, or measurement design, not as commentary about what Codex or the repo did.

6. For `Quarto -> DOCX style sync`, render first and then post-process.
- Run the manuscript render script or Quarto directly.
- If the repository already has a manuscript-specific postprocessor, use it instead of inventing a new one.
- In `manuscript_kaifa_archive/`, use `code/postprocess_word_format.py` after rendering to map the Quarto DOCX back toward the supplied Word manuscript format.

7. Treat formatting as data, not as vibes.
- Compare the supplied DOCX and the rendered DOCX on front matter, author lines, affiliations, abstract label, keywords, heading style, table captions, reference heading, and bibliography layout.
- If the source DOCX uses direct formatting, replicate the visible conventions with `python-docx` rather than relying only on Quarto `reference-doc`.

8. Verify the output artifact, not just the source files.
- For DOCX outputs, inspect the actual rendered `.docx` with `python-docx`.
- For PDF outputs, render pages and spot-check front matter, table-heavy pages, and appendix pages.
- Close review checklists only after the rendered artifact matches the intended manuscript conventions.

## Repo-Specific Conventions

- Treat Quarto as the content source of truth once the manuscript is under repository control.
- Treat supplied manuscript DOCX files as formatting and provenance references unless the user explicitly wants Word-first editing.
- Keep manuscript-specific formatting helpers near the manuscript they serve.
- For Kaifa work, the current formatting bridge is:
  `manuscript_kaifa_archive/code/postprocess_word_format.py`

## References

- Read `references/conversion-checklist.md` before converting a manuscript between DOCX and Quarto.
