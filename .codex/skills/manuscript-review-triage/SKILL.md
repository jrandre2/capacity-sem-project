---
name: manuscript-review-triage
description: Triage manuscript reviews, journal decision letters, pasted reviewer comments, and AI-generated critiques into the repository's unified review workflow. Use when Codex needs to split review text into discrete comments, classify each point as VALID - ACTION NEEDED / ALREADY ADDRESSED / BEYOND SCOPE / INVALID, update REVISION_TRACKER.md, label the review truthfully as synthetic or actual, generate a prioritized revision plan, or refresh review_response and review_verify artifacts.
---

# Manuscript Review Triage

Use this skill for review intake and tracker maintenance in this repository, especially for `manuscript_quarto/` (primary manuscript targeting PAR).

## Workflow

1. Identify the manuscript key and current tracker state.
- Run `python src/pipeline.py review_status --manuscript <key>` before editing.
- Use `quarto` for the primary SEM manuscript (targeting PAR), `velocity` for the archived null-results draft, and `kaifa` for the archived source SEM draft.

2. Gather the minimum source set.
- Read the current `REVISION_TRACKER.md`.
- Read the review artifact itself: pasted text, DOCX, PDF, email, or generated critique.
- Read enough of the manuscript to test whether each critique is valid.

3. Normalize the review into discrete comments.
- Split compound paragraphs into separate actionable items.
- Keep conceptual, methodological, data, and design issues under `## Major Comments`.
- Keep caption, terminology, reference, style, and small reporting issues under `## Minor Comments`.

4. Classify each comment with evidence.
- `VALID - ACTION NEEDED`: real defect or omission that needs revision.
- `ALREADY ADDRESSED`: only when the manuscript already resolves the issue and you can point to the exact section.
- `BEYOND SCOPE`: valid point deferred for explicit reasons.
- `INVALID`: reviewer misunderstanding or claim contradicted by the manuscript or code.
- Use `Validity Assessment` to separate the status from the strength of the critique. `PARTIALLY VALID` is appropriate when the manuscript gestures at the issue but does not resolve it.

5. Update the tracker in the parseable format.
- Use the exact section pattern in `references/tracker-format.md`.
- Keep `source_type` semantically correct.
  - Use `synthetic` for AI-generated, prompted, or manually generated internal reviews.
  - Use `actual` for real external or journal reviews.
- Preserve unified review numbering already in the tracker unless the user explicitly asks to reset it.

6. Turn triage into execution.
- When the user asks for next steps, create `doc/reviews/<manuscript>/REVISION_PLAN.md`.
- Order work by credibility risk, not by ease.
- Put design defects, model contradictions, and data provenance ahead of stylistic cleanup.
- When drafting manuscript text from the plan, keep revision logic out of the manuscript itself.
- Use trackers, memos, and response letters for process narration; use article-facing prose in the manuscript.

7. Refresh derived review artifacts.
- Run `python src/pipeline.py review_response --manuscript <key>` after meaningful tracker updates.
- Run `python src/pipeline.py review_verify --manuscript <key>` to update checklist progress.

## Repo Conventions

- Do not claim a review is actual if it was AI-generated.
- Do not mark a point `ALREADY ADDRESSED` without a file or section citation.
- Prefer Quarto source files for edits; treat DOCX files as source artifacts unless the user explicitly wants Word-first editing.
- For Kaifa work, keep the manuscript archived and clearly separate from the active `manuscript_quarto/` manuscript.
- Preserve parseability for `review_response` and `review_verify`.
- Do not let tracker language leak into the manuscript body. A completed revision should read like a submission-ready paper, not like a changelog.

## References

- Read `references/tracker-format.md` before editing a tracker by hand.
