#!/usr/bin/env python3
"""
Compatibility wrapper for review management.

The project previously maintained a separate local review workflow in this
module. The authoritative implementation now lives in
`centaur.stages.s07_reviews`, which supports synthetic reviews, actual journal
reviews, diffs, response-letter generation, and DOCX feedback import.

Keep this module as a thin shim so older imports continue to work without
preserving a second, divergent review system.
"""
from __future__ import annotations

from typing import Optional

from centaur.stages import s07_reviews

DEFAULT_MANUSCRIPT = "velocity"
FOCUS_PROMPTS = s07_reviews.FOCUS_PROMPTS
MANUSCRIPTS = s07_reviews.MANUSCRIPTS


def get_manuscript_paths(manuscript: str = DEFAULT_MANUSCRIPT):
    """Return manuscript paths from the unified CENTAUR review engine."""
    return s07_reviews.get_manuscript_paths(manuscript)


def status(manuscript: str = DEFAULT_MANUSCRIPT):
    """Display current review-cycle status."""
    return s07_reviews.status(manuscript=manuscript)


def new_cycle(
    focus: str = "par_general",
    manuscript: str = DEFAULT_MANUSCRIPT,
    source_type: str = "synthetic",
    journal: Optional[str] = None,
    submission_round: Optional[str] = None,
    decision: Optional[str] = None,
    reviewer_ids: Optional[list[str]] = None,
):
    """Initialize a new review cycle through the unified CENTAUR engine."""
    return s07_reviews.new_cycle(
        manuscript=manuscript,
        focus=focus,
        source_type=source_type,
        journal=journal,
        submission_round=submission_round,
        decision=decision,
        reviewer_ids=reviewer_ids,
    )


def ingest_docx_review(
    manuscript: str = DEFAULT_MANUSCRIPT,
    docx_path: Optional[str] = None,
    journal: Optional[str] = None,
    submission_round: Optional[str] = None,
    decision: Optional[str] = None,
    reviewer_ids: Optional[list[str]] = None,
    dry_run: bool = False,
    force: bool = False,
):
    """Import DOCX comments or tracked changes into the active tracker."""
    return s07_reviews.ingest_docx_review(
        manuscript=manuscript,
        docx_path=docx_path,
        journal=journal,
        submission_round=submission_round,
        decision=decision,
        reviewer_ids=reviewer_ids,
        dry_run=dry_run,
        force=force,
    )


def archive(
    manuscript: str = DEFAULT_MANUSCRIPT,
    create_tag: bool = True,
    tag_name: Optional[str] = None,
):
    """Archive the active review cycle."""
    return s07_reviews.archive(
        manuscript=manuscript,
        create_tag=create_tag,
        tag_name=tag_name,
    )


def verify(manuscript: str = DEFAULT_MANUSCRIPT):
    """Run the verification checklist for the active review cycle."""
    return s07_reviews.verify(manuscript=manuscript)


def report():
    """Generate a summary report across manuscripts."""
    return s07_reviews.report()


def diff(
    manuscript: str = DEFAULT_MANUSCRIPT,
    from_cycle: Optional[int] = None,
    to_cycle: Optional[int] = None,
    from_commit: Optional[str] = None,
    format: str = "markdown",
):
    """Generate manuscript diffs for review work."""
    return s07_reviews.diff(
        manuscript=manuscript,
        from_cycle=from_cycle,
        to_cycle=to_cycle,
        from_commit=from_commit,
        format=format,
    )


def generate_response_letter(
    manuscript: str = DEFAULT_MANUSCRIPT,
    format: str = "markdown",
    include_diffs: bool = False,
):
    """Generate a response letter from the active tracker."""
    return s07_reviews.generate_response_letter(
        manuscript=manuscript,
        format=format,
        include_diffs=include_diffs,
    )


def main(
    action: str = "status",
    focus: str = "par_general",
    manuscript: str = DEFAULT_MANUSCRIPT,
):
    """Backward-compatible CLI dispatcher for older direct imports."""
    if action == "status":
        return status(manuscript=manuscript)
    if action == "new":
        return new_cycle(focus=focus, manuscript=manuscript)
    if action == "archive":
        return archive(manuscript=manuscript)
    if action == "verify":
        return verify(manuscript=manuscript)
    if action == "report":
        return report()
    raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
