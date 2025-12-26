#!/usr/bin/env python3
"""
Review Management for Capacity-SEM Manuscript

Purpose: Manage synthetic peer review cycles for PAR submission.

Commands
--------
status : Display current review cycle status
new    : Initialize a new review cycle with focus-specific template
archive: Archive current cycle and reset for new one
verify : Run verification checklist (including PAR compliance)
report : Generate summary report of all review cycles

Usage
-----
    python src/pipeline.py review_status
    python src/pipeline.py review_new --focus par_general
    python src/pipeline.py review_archive
    python src/pipeline.py review_verify
    python src/pipeline.py review_report
"""
from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DOC_DIR = PROJECT_ROOT / 'doc'
REVIEWS_DIR = DOC_DIR / 'reviews'
ARCHIVE_DIR = REVIEWS_DIR / 'archive'
MANUSCRIPT_DIR = PROJECT_ROOT / 'manuscript_quarto'
TRACKER_FILE = MANUSCRIPT_DIR / 'REVISION_TRACKER.md'
CHECKLIST_FILE = DOC_DIR / 'MANUSCRIPT_REVISION_CHECKLIST.md'
REVIEWS_INDEX = REVIEWS_DIR / 'README.md'

# PAR-specific focus prompts
FOCUS_PROMPTS = {
    'par_general': '''Act as a peer reviewer for **Public Administration Review (PAR)**, the top journal in public administration. PAR emphasizes:
1. Relevance to practitioners and public managers
2. Evidence-based policy recommendations
3. Methodological rigor appropriate to the research question
4. Clear, accessible writing for multidisciplinary audience

**Manuscript Context**:
- Title: Government Capacity and Disaster Recovery: A Survival Analysis of CDBG-DR Fund Management
- Method: Survival analysis (Cox PH, AFT) of 152 grantee-disaster pairs
- Key Finding: Disbursement capacity predicts completion timing (HR=4.37, p=0.006)
- Contribution: First application of survival analysis to disaster recovery administration

**Review the manuscript for**:

**1. Practitioner Relevance & Evidence for Practice**
- Does the Evidence for Practice section provide actionable insights?
- Are policy recommendations specific and implementable?
- Would HUD program managers find this useful?

**2. Methodological Appropriateness**
- Is survival analysis justified for this research question?
- Are Cox PH and AFT models appropriate?
- Is the 95% completion threshold defensible?
- Are capacity indicators (disbursement/expenditure ratios) valid proxies?

**3. Threats to Validity**
- Endogeneity: Do faster programs cause higher ratios, or vice versa?
- Selection bias: Are incomplete programs systematically different?
- Measurement error: How reliable are QPR self-reports?
- External validity: Does CDBG-DR generalize to other programs?

**4. Statistical Rigor**
- Is N=152 adequate for survival analysis?
- Are proportional hazards assumptions tested?
- Are standard errors robust?
- Are effect sizes practically meaningful?

**5. Robustness & Heterogeneity**
- Are findings robust to specification changes?
- Do effects vary by disaster type (hurricanes vs wildfires)?
- Do effects vary by government type (state vs local)?
- Are outliers handled appropriately?

**6. Literature & Theory**
- Does the administrative capacity framework add value?
- Is the survival analysis novelty overstated?
- Are prior disaster recovery studies properly cited?
- Is the contribution to public administration clear?

**7. Presentation & Clarity**
- Is the abstract clear and compelling?
- Are figures/tables well-designed?
- Is the discussion balanced (not overclaiming)?
- Are limitations honestly acknowledged?
- Is the writing accessible to non-specialists?

**8. PAR-Specific Issues**
- Word count under 8,000? (currently ~7,565)
- Evidence for Practice section strong?
- Policy recommendations actionable?
- No "this study" metacommentary? (verified clean)

**Format your review as**:

## MAJOR COMMENTS
1. [Issue title]
   - Concern: [description]
   - Severity: [CRITICAL | HIGH | MODERATE]
   - Recommendation: [specific action]

## MINOR COMMENTS
1. [Issue]
2. [Issue]

## OVERALL ASSESSMENT
- Recommendation: [ACCEPT | MINOR REVISIONS | MAJOR REVISIONS | REJECT]
- Strengths: [3-5 key strengths]
- Weaknesses: [3-5 key weaknesses]
- Confidence: [HIGH | MEDIUM | LOW] (in your assessment)

Be critical but fair. PAR rejects ~90% of submissions, so high standards apply.''',

    'methods': '''Act as a methodologist reviewing a manuscript for Public Administration Review.

Focus exclusively on the **survival analysis methodology** used to analyze CDBG-DR disaster recovery completion timing.

**Evaluate**:

**1. Model Specification**
- Are Cox Proportional Hazards and AFT Lognormal models appropriate?
- Are covariates correctly specified?
- Are proportional hazards assumptions tested and met?
- Is the baseline hazard appropriately modeled?

**2. Censoring Handling**
- Is right-censoring (73.7% of sample) appropriately handled?
- Is the 95% completion threshold justified?
- Are informative censoring concerns addressed?

**3. Sample Size Adequacy**
- Is N=152 (40 events) adequate for the model complexity?
- Are confidence intervals appropriately wide given sample size?
- Should simpler models be considered?

**4. Capacity Indicator Validity**
- Are disbursement/expenditure ratios valid capacity proxies?
- Are cumulative mean ratios appropriate (vs. other aggregations)?
- Is measurement error discussed?

**5. Endogeneity**
- Is reverse causality adequately addressed?
- Are instrumental variables or sensitivity analyses needed?
- Are lagged capacity measures explored?

**6. Robustness**
- Are alternative specifications tested?
- Are outliers handled appropriately?
- Are results stable across subgroups?

**7. Diagnostics**
- Are residual diagnostics shown?
- Are influential observations identified?
- Are model fit statistics reported?

Format your response with numbered major and minor methodological concerns.''',

    'policy': '''Act as a public administration practitioner reviewing a manuscript for PAR.

You are a HUD program manager or state/local disaster recovery coordinator. Evaluate this manuscript for **practical relevance and actionability**.

**Evaluate**:

**1. Evidence for Practice Section**
- Are the bullet points specific and actionable?
- Would practitioners understand how to apply these insights?
- Are recommendations feasible given real-world constraints?

**2. Policy Recommendations**
- Are recommendations concrete (not generic)?
- Do they specify WHO should do WHAT?
- Are timelines and resource requirements realistic?
- Are political/institutional barriers acknowledged?

**3. Generalizability**
- Do findings apply beyond CDBG-DR to other disaster programs?
- Would this help with FEMA, EDA, or state programs?
- Are context-specific limitations clear?

**4. Practitioner Accessibility**
- Is the writing clear for non-technical audiences?
- Are key findings summarized effectively?
- Are technical details appropriately placed in appendices?

**5. Real-World Examples**
- Are specific disasters and jurisdictions used effectively?
- Do examples resonate with practitioner experiences?
- Are success stories and failures both represented?

**6. Implementation Barriers**
- Are political, legal, and resource constraints discussed?
- Are change management issues addressed?
- Are capacity-building recommendations included?

Format your response as if writing a memo to the author about improving practitioner relevance.''',

    'clarity': '''Act as a writing and communication expert reviewing a manuscript for PAR.

Focus on **presentation quality, clarity, and accessibility** for PAR's multidisciplinary audience.

**Evaluate**:

**1. Abstract**
- Is it under 150 words?
- Does it convey the research question, method, finding, and implication?
- Would a non-specialist understand the contribution?

**2. Introduction**
- Is the research question clear within the first page?
- Is the motivation compelling?
- Is the structure signposted?

**3. Literature Review**
- Is the theoretical framework accessible?
- Are connections between concepts clear?
- Is jargon minimized or explained?

**4. Methods**
- Can a non-specialist understand why survival analysis is appropriate?
- Are technical details appropriately placed?
- Are key assumptions explained in plain language?

**5. Results**
- Are tables and figures self-explanatory?
- Are effect sizes interpreted substantively (not just statistically)?
- Are key findings highlighted effectively?

**6. Discussion**
- Is the interpretation balanced?
- Are limitations acknowledged honestly?
- Are claims supported by evidence?

**7. Conclusions**
- Are policy implications clear?
- Is the broader significance articulated?
- Does it avoid overclaiming?

**8. Overall Flow**
- Are transitions smooth between sections?
- Is the narrative coherent?
- Are repetitions minimized?

**9. PAR Style**
- No "this study" self-references?
- Chicago Author-Date citations correct?
- Direct presentation of findings?

Format your response with specific line-level and structural suggestions.'''
}


def status():
    """Display current review cycle status from REVISION_TRACKER.md."""
    print("Review Status")
    print("=" * 50)

    if not TRACKER_FILE.exists():
        print("\nNo active review cycle.")
        print(f"Start one with: python src/pipeline.py review_new --focus <name>")
        print(f"Available focus areas: {', '.join(FOCUS_PROMPTS.keys())}")
        return

    content = TRACKER_FILE.read_text()

    # Parse summary statistics
    print("\nCurrent Tracker:", TRACKER_FILE)

    # Extract review number and focus
    review_match = re.search(r'\*\*Review\*\*:\s*#?(\w+)', content)
    focus_match = re.search(r'\*\*Focus\*\*:\s*([\w_]+)', content)

    if review_match:
        print(f"Review: #{review_match.group(1)}")
    if focus_match:
        print(f"Focus: {focus_match.group(1)}")

    # Extract summary table
    summary_match = re.search(
        r'\|\s*Category\s*\|.*?\n\|[-\s|]+\n((?:\|.*?\n)+)',
        content,
        re.MULTILINE
    )

    if summary_match:
        print("\nSummary:")
        rows = summary_match.group(1).strip().split('\n')
        for row in rows:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            if len(cells) >= 5:
                print(f"  {cells[0]}: {cells[1]} total, {cells[2]} addressed, "
                      f"{cells[3]} beyond scope, {cells[4]} pending")

    # Count verification items
    checked = len(re.findall(r'- \[x\]', content, re.IGNORECASE))
    unchecked = len(re.findall(r'- \[ \]', content))
    total = checked + unchecked

    if total > 0:
        print(f"\nVerification: {checked}/{total} items complete")

    print("\n" + "=" * 50)


def new_cycle(focus: str = 'par_general'):
    """Initialize a new review cycle with focus-specific template."""
    if focus not in FOCUS_PROMPTS:
        print(f"ERROR: Unknown focus '{focus}'")
        print(f"Available: {', '.join(FOCUS_PROMPTS.keys())}")
        return

    print(f"Initializing new review cycle (focus: {focus})")
    print("=" * 50)

    # Determine review number
    review_num = 1
    if ARCHIVE_DIR.exists():
        existing = list(ARCHIVE_DIR.glob('review_*.md'))
        if existing:
            nums = [int(re.search(r'review_(\d+)', f.name).group(1))
                    for f in existing if re.search(r'review_(\d+)', f.name)]
            if nums:
                review_num = max(nums) + 1

    # Check if there's an active review
    if TRACKER_FILE.exists():
        content = TRACKER_FILE.read_text()
        if 'PENDING' in content.upper() or re.search(r'\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[1-9]', content):
            print("\nWARNING: Active review has pending items.")
            print("Archive current review first with: python src/pipeline.py review_archive")
            return

    # Create new tracker from template
    today = datetime.now().strftime('%Y-%m-%d')
    template = f'''# Revision Tracker: Response to Synthetic Review

**Document**: Capacity-SEM Manuscript for Public Administration Review
**Review**: #{review_num}
**Focus**: {focus}
**Last Updated**: {today}

---

## Summary Statistics

| Category | Total | Addressed | Beyond Scope | Pending |
|----------|-------|-----------|--------------|---------|
| Major Comments | 0 | 0 | 0 | 0 |
| Minor Comments | 0 | 0 | 0 | 0 |

---

## Prompt Used

```
{FOCUS_PROMPTS[focus]}
```

---

## Major Comments

### Comment 1: [Title]

**Status**: [VALID - ACTION NEEDED | ALREADY ADDRESSED | BEYOND SCOPE | INVALID]

**Reviewer's Concern**:
> [Paste reviewer comment here]

**Validity Assessment**: [VALID | PARTIALLY VALID | INVALID]

[Explain why this is/isn't a legitimate concern]

**Response**:

[Describe what you did to address this, OR why it was not addressed]

**Files Modified**:
- [List specific files and line numbers]

---

## Minor Comments

### Minor 1: [Title]

**Status**: [STATUS]

**Concern**: [Brief description]

**Response**: [What was done]

---

## Verification Checklist

- [ ] All VALID - ACTION NEEDED items addressed
- [ ] All code runs without errors (`python src/pipeline.py run_all`)
- [ ] Manuscript text updated
- [ ] Tables/figures reflect changes
- [ ] Quarto renders without errors (`CAPACITY_SEM_SKIP_PIPELINE=1 ./render_all.sh`)
- [ ] Word count still under 8,000 (`wc -w manuscript_quarto/*.qmd`)
- [ ] Changes committed to git
- [ ] MANUSCRIPT_REVISION_CHECKLIST.md updated

---

## PAR Compliance Check

- [ ] Word count ≤ 8,000 (total)
- [ ] Abstract ≤ 150 words
- [ ] Evidence for Practice section present and actionable
- [ ] No "this study" self-references
- [ ] Chicago Author-Date citations

---

*Review generated: {today}*
*Last updated: {today}*
'''

    TRACKER_FILE.write_text(template)
    print(f"\nCreated: {TRACKER_FILE}")
    print(f"\nReview #{review_num} initialized with '{focus}' focus.")
    print("\nNext steps:")
    print("1. Generate a synthetic review using the prompt above")
    print("2. Paste reviewer comments into the tracker")
    print("3. Triage each comment with a status classification")
    print("4. Implement changes and update tracker")
    print("5. Run: python src/pipeline.py review_verify")


def archive():
    """Archive current review cycle and reset for new one."""
    print("Archiving Review Cycle")
    print("=" * 50)

    if not TRACKER_FILE.exists():
        print("No active review to archive.")
        return

    # Ensure archive directory exists
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine archive filename
    content = TRACKER_FILE.read_text()
    review_match = re.search(r'\*\*Review\*\*:\s*#?(\d+)', content)
    focus_match = re.search(r'\*\*Focus\*\*:\s*([\w_]+)', content)

    if review_match:
        review_num = int(review_match.group(1))
    else:
        # Find next available number
        existing = list(ARCHIVE_DIR.glob('review_*.md'))
        nums = [int(re.search(r'review_(\d+)', f.name).group(1))
                for f in existing if re.search(r'review_(\d+)', f.name)]
        review_num = max(nums) + 1 if nums else 1

    focus = focus_match.group(1) if focus_match else 'unknown'
    today = datetime.now().strftime('%Y-%m-%d')
    archive_file = ARCHIVE_DIR / f'review_{review_num:02d}_{today}_{focus}.md'

    # Copy to archive
    shutil.copy(TRACKER_FILE, archive_file)
    print(f"Archived to: {archive_file}")

    # Reset tracker
    TRACKER_FILE.unlink()
    print(f"Removed: {TRACKER_FILE}")

    print(f"\nReview #{review_num} archived successfully.")
    print("Start new review with: python src/pipeline.py review_new --focus <name>")


def verify():
    """Run verification checklist for current review cycle."""
    print("Verification Checklist")
    print("=" * 50)

    if not TRACKER_FILE.exists():
        print("No active review to verify.")
        return

    content = TRACKER_FILE.read_text()

    # Find all checklist items
    checked = re.findall(r'- \[x\]\s+(.+)', content, re.IGNORECASE)
    unchecked = re.findall(r'- \[ \]\s+(.+)', content)

    print(f"\nCompleted ({len(checked)}):")
    for item in checked:
        print(f"  ✓ {item}")

    print(f"\nPending ({len(unchecked)}):")
    for item in unchecked:
        print(f"  ☐ {item}")

    total = len(checked) + len(unchecked)
    if total > 0:
        pct = (len(checked) / total) * 100
        print(f"\nProgress: {len(checked)}/{total} ({pct:.0f}%)")

        if len(unchecked) == 0:
            print("\n✓ All verification items complete!")
            print("Ready to archive: python src/pipeline.py review_archive")
        else:
            print(f"\n{len(unchecked)} items remaining before archive.")

    # PAR-specific checks
    print("\n" + "=" * 50)
    print("PAR Compliance Checks")
    print("=" * 50)

    # Word count check
    try:
        qmd_files = list(MANUSCRIPT_DIR.glob('*.qmd'))
        word_count_result = subprocess.run(
            ['wc', '-w'] + [str(f) for f in qmd_files],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if word_count_result.returncode == 0:
            lines = word_count_result.stdout.strip().split('\n')
            if len(lines) > 1:
                total_line = lines[-1]
                total_words = int(total_line.strip().split()[0])
                print(f"\nWord count: {total_words:,} / 8,000")
                if total_words <= 8000:
                    print(f"  ✓ Under limit ({8000 - total_words:,} words remaining)")
                else:
                    print(f"  ✗ OVER LIMIT by {total_words - 8000:,} words!")
    except Exception as e:
        print(f"\nWord count check failed: {e}")

    # Check for "this study"
    try:
        index_qmd = MANUSCRIPT_DIR / 'index.qmd'
        if index_qmd.exists():
            text = index_qmd.read_text().lower()
            this_study_count = text.count('this study')
            if this_study_count == 0:
                print("  ✓ No 'this study' self-references found")
            else:
                print(f"  ✗ Found {this_study_count} instances of 'this study'")
    except Exception as e:
        print(f"\n'This study' check failed: {e}")


def report():
    """Generate summary report of all review cycles."""
    print("Review Cycles Report")
    print("=" * 50)

    # Check for archived reviews
    if not ARCHIVE_DIR.exists():
        archived = []
    else:
        archived = sorted(ARCHIVE_DIR.glob('review_*.md'))

    # Check for active review
    active = TRACKER_FILE.exists()

    total_cycles = len(archived) + (1 if active else 0)
    print(f"\nTotal review cycles: {total_cycles}")
    print(f"  Archived: {len(archived)}")
    print(f"  Active: {'Yes' if active else 'No'}")

    if archived:
        print("\nArchived Reviews:")
        print("-" * 60)
        for f in archived:
            content = f.read_text()
            focus_match = re.search(r'\*\*Focus\*\*:\s*([\w_]+)', content)
            date_match = re.search(r'\*Review generated:\s*(\d{4}-\d{2}-\d{2})', content)

            focus = focus_match.group(1) if focus_match else 'unknown'
            date = date_match.group(1) if date_match else 'unknown'

            # Count comments
            major = len(re.findall(r'### Comment \d+:', content))
            minor = len(re.findall(r'### Minor \d+:', content))

            print(f"  {f.name}")
            print(f"    Focus: {focus}, Date: {date}")
            print(f"    Comments: {major} major, {minor} minor")

    if active:
        print("\nActive Review:")
        print("-" * 60)
        status()


def main(action: str = 'status', focus: str = 'par_general'):
    """Main entry point for review management."""
    if action == 'status':
        status()
    elif action == 'new':
        new_cycle(focus)
    elif action == 'archive':
        archive()
    elif action == 'verify':
        verify()
    elif action == 'report':
        report()
    else:
        print(f"Unknown action: {action}")
        print("Available: status, new, archive, verify, report")


if __name__ == '__main__':
    main()
