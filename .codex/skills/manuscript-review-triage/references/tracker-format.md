# Tracker Format

Use this exact structure so `python src/pipeline.py review_response` and `review_verify` can parse the tracker.

## Major Comment Pattern

```md
### Comment 1: Short title

**Status**: VALID - ACTION NEEDED

**Reviewer's Concern**:
> Quoted or paraphrased concern

**Validity Assessment**: VALID

[Evidence-based assessment]

**Response**:

[Concrete revision path]

**Files Modified**:
- path/to/file
```

Accepted heading variants:

- `### Comment 1: ...`
- `### R1 Major 1: ...`

## Minor Comment Pattern

```md
### Minor 1: Short title

**Status**: VALID - ACTION NEEDED
**Concern**: Short concern
**Response**: Concrete fix
```

## Status Rules

- Use `VALID - ACTION NEEDED` for unresolved real defects.
- Use `ALREADY ADDRESSED` only with a file or section citation.
- Use `BEYOND SCOPE` only with an explicit reason.
- Use `INVALID` only when the manuscript or code clearly contradicts the critique.

## Checklist Rules

- Update the verification checklist after triage and after each major revision wave.
- Treat the revision plan as a concrete path item in the checklist when one exists.
- Keep process language in the tracker. Do not copy tracker phrasing like `validity assessment`, `revision path`, or `this revision` into the manuscript itself.
