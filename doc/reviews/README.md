# Synthetic Review Index

**Related**: [SYNTHETIC_REVIEW_PROCESS.md](../SYNTHETIC_REVIEW_PROCESS.md) | [MANUSCRIPT_REVISION_CHECKLIST.md](../MANUSCRIPT_REVISION_CHECKLIST.md)
**Project**: Capacity-SEM Analysis
**Last Updated**: 2025-12-27

---

## Overview

This directory tracks synthetic peer review cycles for all manuscript approaches in the Capacity-SEM project. Each manuscript has its own subdirectory with review history and archives.

**Purpose**: Stress-test methodology, identify weaknesses, and strengthen manuscripts before submission.

---

## Manuscript Approaches

| Manuscript | Directory | Status | Target Journal | Key Method |
|------------|-----------|--------|----------------|------------|
| **Velocity** | `velocity/` | **Active** | PAR | Contingent Capacity Framework |
| ~~Capacity-SEM~~ | `archive/` | Archived | - | SEM with latent constructs |

### Active: Velocity Manuscript

**Location**: `manuscript_velocity/`
**Approach**: Contingent Capacity Framework - velocity effects depend on time constraints and substitutability
**Key Finding**: Median HR=4.60 across 16 contexts, ranging from null (housing) to extreme (wildfire)

See [velocity/README.md](velocity/README.md) for review details.

### Archived: Capacity-SEM Manuscript

**Location**: `manuscript_kaifa_archive/`
**Approach**: SEM with latent capacity constructs
**Status**: Archived due to methodological limitations (see `doc/ANALYSIS_COMPARISON_REPORT.md`)

---

## Review Cycle Summary (All Manuscripts)

| Manuscript | Review | Date | Focus | Major | Minor | Addressed | Status |
|------------|--------|------|-------|-------|-------|-----------|--------|
| velocity | *Awaiting first review* | - | - | - | - | - | - |

---

## Directory Structure

```
doc/reviews/
├── README.md                    # This file
├── velocity/                    # Velocity manuscript reviews
│   ├── README.md               # Velocity-specific index
│   └── archive/                # Completed velocity reviews
└── archive/                    # Legacy/archived manuscript reviews
```

---

## Commands Quick Reference

```bash
# Velocity manuscript (current)
python src/pipeline.py review_new --manuscript velocity --focus par_general
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_archive --manuscript velocity

# List all manuscripts and reviews
python src/pipeline.py review_report
```

---

## Focus Areas

| Focus | Description | When to Use |
|-------|-------------|-------------|
| `par_general` | Comprehensive PAR review | First review, pre-submission |
| `methods` | Deep methodological critique | After major analysis changes |
| `policy` | Practitioner relevance | Before finalizing recommendations |
| `clarity` | Writing and presentation | Final polish |

---

## Review Workflow

### 1. Start New Review
```bash
python src/pipeline.py review_new --manuscript velocity --focus par_general
```

### 2. Obtain LLM Review
- Provide rendered manuscript (`manuscript_velocity/_output/index.pdf`)
- Use prompts from `velocity/README.md`
- Paste review into `manuscript_velocity/REVISION_TRACKER.md`

### 3. Triage Comments
Classify each as:
- **VALID - ACTION NEEDED**: Implement fix
- **ALREADY ADDRESSED**: Document where
- **BEYOND SCOPE**: Explain deferral
- **INVALID**: Clarify misunderstanding

### 4. Implement & Verify
```bash
# Make changes, then verify
python src/pipeline.py review_verify --manuscript velocity
```

### 5. Archive
```bash
python src/pipeline.py review_archive --manuscript velocity
```

---

## Recurring Themes

Track concerns raised across multiple reviews:

| Theme | Reviews Raised | Status | Notes |
|-------|----------------|--------|-------|
| *None yet* | - | - | *Track recurring issues across reviews* |

---

## Statistics Across All Manuscripts

| Metric | Velocity | Total |
|--------|----------|-------|
| Review cycles | 0 | 0 |
| Major comments | 0 | 0 |
| Minor comments | 0 | 0 |
| Comments addressed | 0 | 0 |

*Updated by `review_report` command*

---

## Adding a New Manuscript Approach

When creating a new analytical approach:

1. Create manuscript directory: `manuscript_{name}/`
2. Create review subdirectory: `doc/reviews/{name}/`
3. Create `doc/reviews/{name}/README.md` with approach-specific prompts
4. Create `manuscript_{name}/REVISION_TRACKER.md`
5. Update this README with new manuscript entry
6. Register in pipeline.py `MANUSCRIPTS` dict

---

## Related Documentation

- [SYNTHETIC_REVIEW_PROCESS.md](../SYNTHETIC_REVIEW_PROCESS.md) - Detailed methodology
- [MANUSCRIPT_REVISION_CHECKLIST.md](../MANUSCRIPT_REVISION_CHECKLIST.md) - High-level tracking
- [velocity/README.md](velocity/README.md) - Velocity manuscript reviews
- [../METHODOLOGY.md](../METHODOLOGY.md) - Analysis methodology
