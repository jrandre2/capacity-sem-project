# Synthetic Review Index

**Related**: [SYNTHETIC_REVIEW_PROCESS.md](../SYNTHETIC_REVIEW_PROCESS.md) | [MANUSCRIPT_REVISION_CHECKLIST.md](../MANUSCRIPT_REVISION_CHECKLIST.md)
**Project**: Capacity-SEM Manuscript for Public Administration Review
**Last Updated**: 2025-12-26

---

## Overview

This directory tracks all synthetic peer review cycles for the Capacity-SEM manuscript. Each review cycle generates a detailed `REVISION_TRACKER.md` document that is archived here upon completion.

**Purpose**: Stress-test methodology, identify weaknesses, and strengthen the manuscript before submission to PAR.

---

## Review Cycle Summary

| Review | Date | Focus | LLM | Major | Minor | Addressed | Beyond Scope | Status |
|--------|------|-------|-----|-------|-------|-----------|--------------|--------|
| *Awaiting first review* | - | - | - | - | - | - | - | - |

**Legend**:
- **Focus**: par_general (comprehensive), methods (methodology), policy (practitioner relevance), clarity (writing)
- **LLM**: Claude Opus 4.5, Claude Sonnet 3.5, GPT-4, etc.
- **Status**: Active, Completed, Archived

---

## Archived Reviews

All completed review cycles are stored in the `archive/` subdirectory with the naming convention:

```
archive/review_NN_YYYY-MM-DD_FOCUS.md
```

**Example**: `archive/review_01_2025-12-26_par_general.md`

### How to Archive a Review

```bash
# Automatically archives the current REVISION_TRACKER.md
python src/pipeline.py review_archive
```

This command:
1. Copies `manuscript_quarto/REVISION_TRACKER.md` to `archive/review_NN_YYYY-MM-DD_FOCUS.md`
2. Updates this README.md with the review summary
3. Resets `REVISION_TRACKER.md` for the next cycle
4. Updates `MANUSCRIPT_REVISION_CHECKLIST.md`

---

## Review Cycle Workflow

### 1. Generate New Review

```bash
python src/pipeline.py review_new --focus par_general
```

Creates a new `manuscript_quarto/REVISION_TRACKER.md` with:
- Review number (auto-incremented)
- Embedded PAR-specific review prompt
- Template for tracking responses
- Summary statistics table

### 2. Obtain Review from LLM

Provide the LLM with:
- Manuscript text (from `manuscript_quarto/_output/`)
- Relevant tables/figures
- The embedded review prompt

Paste the LLM's review into `REVISION_TRACKER.md`.

### 3. Triage Comments

Classify each comment:
- **VALID - ACTION NEEDED**: Implement fix
- **ALREADY ADDRESSED**: Document where
- **BEYOND SCOPE**: Explain why deferred
- **INVALID**: Clarify misunderstanding

### 4. Implement Changes

For valid concerns:
- Modify code/analysis as needed
- Update manuscript text
- Re-render manuscript
- Document changes in tracker

### 5. Verify and Archive

```bash
# Verify all changes work
python src/pipeline.py review_verify

# Archive when complete
python src/pipeline.py review_archive
```

---

## Recurring Themes

As multiple review cycles are completed, track recurring concerns here:

| Theme | Reviews Raised | Status | Notes |
|-------|----------------|--------|-------|
| *None yet* | - | - | *Track recurring issues across reviews* |

**Purpose**: If multiple independent reviews raise the same concern, it likely requires attention.

---

## Key Lessons Learned

### From Review #N (FOCUS)
*Add key insights from each review cycle*

**Example**:
- Endogeneity concerns were raised but already partially addressed in limitations
- Policy recommendations needed more specificity (addressed by adding dollar amounts and timelines)
- Evidence for Practice section strengthened with concrete examples

---

## Statistics Across All Reviews

| Metric | Count |
|--------|-------|
| Total review cycles | 0 |
| Total major comments | 0 |
| Total minor comments | 0 |
| Comments addressed | 0 |
| Comments beyond scope | 0 |
| Comments invalid | 0 |

*Auto-updated by `review_report` command*

---

## Focus Area Coverage

Track which focus areas have been reviewed:

| Focus | Reviewed? | Date | Review # |
|-------|-----------|------|----------|
| PAR General (comprehensive) | ❌ | - | - |
| Methods (methodology) | ❌ | - | - |
| Policy (practitioner relevance) | ❌ | - | - |
| Clarity (writing/presentation) | ❌ | - | - |

**Recommendation**: Before submission, complete at least:
1. One **par_general** comprehensive review
2. One **methods** deep methodological review
3. One **policy** practitioner relevance check

---

## Comparison to Actual Peer Review (Post-Submission)

*After PAR provides actual peer reviews, compare them to synthetic reviews here*

| Actual Reviewer Concern | Anticipated in Synthetic Review? | Review # |
|-------------------------|----------------------------------|----------|
| *Complete after actual reviews received* | - | - |

**Purpose**: Learn which concerns were successfully anticipated and which were missed.

---

## Related Documentation

- [SYNTHETIC_REVIEW_PROCESS.md](../SYNTHETIC_REVIEW_PROCESS.md) - Detailed methodology guide
- [MANUSCRIPT_REVISION_CHECKLIST.md](../MANUSCRIPT_REVISION_CHECKLIST.md) - High-level revision tracking
- [METHODOLOGY.md](../METHODOLOGY.md) - Survival analysis and SEM methodology
- [MANUSCRIPT_GUIDE.md](../MANUSCRIPT_GUIDE.md) - PAR formatting and writing rules

---

## Commands Quick Reference

```bash
# Start new review cycle
python src/pipeline.py review_new --focus par_general
python src/pipeline.py review_new --focus methods
python src/pipeline.py review_new --focus policy
python src/pipeline.py review_new --focus clarity

# Check current review status
python src/pipeline.py review_status

# Run verification checklist
python src/pipeline.py review_verify

# Archive completed review
python src/pipeline.py review_archive

# Generate summary report
python src/pipeline.py review_report
```

---

*Last updated: 2025-12-26*
*Status: Ready for first review cycle*
