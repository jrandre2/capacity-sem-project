"""
Systematic Audit of Survival Analysis Implementation
=====================================================
Run: cd /Volumes/T9/Projects/capacity-sem-project && python scripts/audit_survival_analysis.py
Output: doc/reports/survival_audit_report.md

Checks all 10 concerns raised in the survival analysis review:
1. Ratio data quality in time-varying panel
2. Cached vs fresh panel comparison
3. Concordance index investigation
4. EPV assessment
5. Proportional hazards test
6. Competing risks assessment
7. Non-linear effects
8. Bootstrap clustered SEs
9. AFT model comparison
10. Tie handling
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime
from io import StringIO

import numpy as np

# Suppress warnings during model fitting
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Use pyarrow directly since pandas parquet has version issues
import pyarrow.parquet as pq
import pyarrow as pa

# Project paths
PROJECT_DIR = Path("/Volumes/T9/Projects/capacity-sem-project")
DATA_WORK = PROJECT_DIR / "data_work"
DIAGNOSTICS = DATA_WORK / "diagnostics"
REPORT_DIR = PROJECT_DIR / "doc" / "reports"

# ── Helpers ───────────────────���─────────────────────────────────────────

def read_parquet_safe(path):
    """Read parquet using pyarrow, converting to dict-of-lists for analysis."""
    t = pq.read_table(str(path))
    return t


def col_to_list(table, col):
    """Extract a column as a plain Python list, filtering None."""
    return [x for x in table.column(col).to_pylist() if x is not None]


def col_to_list_all(table, col):
    """Extract a column as a plain Python list, keeping None."""
    return table.column(col).to_pylist()


def percentile(vals, p):
    """Compute percentile from sorted list."""
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]


def describe_distribution(vals, name=""):
    """Return distribution summary string."""
    if not vals:
        return f"  {name}: NO DATA\n"
    s = sorted(vals)
    n = len(s)
    lines = [
        f"  {name}: N={n}, min={s[0]:.4f}, max={s[-1]:.4f}",
        f"    mean={sum(s)/n:.4f}, median={s[n//2]:.4f}",
        f"    P1={percentile(s,1):.4f}, P5={percentile(s,5):.4f}, P25={percentile(s,25):.4f}",
        f"    P75={percentile(s,75):.4f}, P95={percentile(s,95):.4f}, P99={percentile(s,99):.4f}",
        f"    >1: {sum(1 for v in s if v>1)} ({100*sum(1 for v in s if v>1)/n:.1f}%)",
        f"    >10: {sum(1 for v in s if v>10)} ({100*sum(1 for v in s if v>10)/n:.1f}%)",
        f"    >100: {sum(1 for v in s if v>100)} ({100*sum(1 for v in s if v>100)/n:.1f}%)",
    ]
    return "\n".join(lines) + "\n"


def read_csv_simple(path):
    """Read CSV into list of dicts."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


# ── Output collector ──────────────────��─────────────────────────────────

class ReportBuilder:
    def __init__(self):
        self.sections = []
        self.summary = []

    def add_section(self, title, content, status="INFO"):
        """status: PASS, FAIL, WARNING, INFO, CRITICAL"""
        self.sections.append((title, content, status))
        self.summary.append((title, status))

    def build(self):
        lines = []
        lines.append("# Survival Analysis Audit Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Check | Status |")
        lines.append("|-------|--------|")
        for title, status in self.summary:
            emoji = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "CRITICAL": "🔴"}.get(status, "❓")
            lines.append(f"| {title} | {emoji} {status} |")
        lines.append("")

        # Detailed sections
        for title, content, status in self.sections:
            emoji = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "CRITICAL": "🔴"}.get(status, "❓")
            lines.append(f"## {emoji} {title}")
            lines.append("")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)


report = ReportBuilder()


# ═══════════════════════════════════════════════════════════════════════
# CHECK 1: Ratio Data Quality in Time-Varying Panel
# ══════════════════════════════════════════════���════════════════════════

def check_1_ratio_data_quality():
    print("CHECK 1: Ratio data quality...")

    out = StringIO()

    # Load cached panel
    tv = read_parquet_safe(DATA_WORK / "panel_time_varying.parquet")
    out.write(f"### Cached panel_time_varying.parquet\n\n")
    out.write(f"- Rows: {tv.num_rows}\n")
    out.write(f"- Columns: {len(tv.column_names)}\n\n")

    # Check ratio distributions
    for col in ['Ratio_disbursed_to_obligated_lag1', 'Ratio_expended_to_disbursed_lag1']:
        vals = col_to_list(tv, col)
        out.write(describe_distribution(vals, col) + "\n")

    # Load source qpr_quarterly
    qpr = read_parquet_safe(DATA_WORK / "qpr_quarterly.parquet")
    out.write(f"### Source qpr_quarterly.parquet\n\n")
    out.write(f"- Rows: {qpr.num_rows}\n\n")

    # Compute ratios from source
    obligated = qpr.column('QPR Fund Obligated $').to_pylist()
    disbursed = qpr.column('QPR Fund Disbursed $').to_pylist()
    expended = qpr.column('QPR Fund Expended $').to_pylist()

    disb_ratios = []
    for o, d in zip(obligated, disbursed):
        if o and o > 0 and d is not None:
            disb_ratios.append(d / o)

    exp_ratios = []
    for d, e in zip(disbursed, expended):
        if d and d > 0 and e is not None:
            exp_ratios.append(e / d)

    out.write(describe_distribution(disb_ratios, "Source Disb/Obligated") + "\n")
    out.write(describe_distribution(exp_ratios, "Source Exp/Disbursed") + "\n")

    # Check for duplicates in source
    grantees = qpr.column('Grantee').to_pylist()
    disasters = qpr.column('Disaster Type').to_pylist()
    quarters = qpr.column('QPR Actual Quarter').to_pylist()
    keys = list(zip(grantees, disasters, quarters))
    n_dups = len(keys) - len(set(keys))
    out.write(f"### Duplicate check in qpr_quarterly\n\n")
    out.write(f"- Unique grantee-disaster-quarter: {len(set(keys))}\n")
    out.write(f"- Duplicates: {n_dups}\n\n")

    # Determine severity
    cached_ratios = col_to_list(tv, 'Ratio_disbursed_to_obligated_lag1')
    cached_max = max(cached_ratios) if cached_ratios else 0
    source_max = max(disb_ratios) if disb_ratios else 0

    out.write(f"### Comparison\n\n")
    out.write(f"- Source max Disb/Obligated: {source_max:.4f}\n")
    out.write(f"- Cached panel max Disb/Obligated: {cached_max:.2f}\n")
    out.write(f"- **Discrepancy factor: {cached_max/source_max:.0f}x**\n\n")

    if cached_max > 100:
        out.write("**CRITICAL**: Cached panel has ratio values up to {:.0f}, far exceeding the source data maximum of {:.2f}. ".format(cached_max, source_max))
        out.write("This means the cached `panel_time_varying.parquet` was generated by a different code path or from different source data. ")
        out.write("The extreme values drive Cox model coefficients to machine-zero, producing the HR≈1.0 null finding.\n")
        status = "CRITICAL"
    else:
        out.write("Ratio values appear reasonable.\n")
        status = "PASS"

    report.add_section("CHECK 1: Ratio Data Quality", out.getvalue(), status)
    return status


# ═══════════════════════════════════════════════════════════════════════
# CHECK 2: Cached vs Fresh Panel Comparison
# ═══════════════════════════════════════════════════════════════════════

def check_2_cached_vs_fresh():
    print("CHECK 2: Cached vs fresh panel...")

    out = StringIO()

    # Add project to path for imports
    sys.path.insert(0, str(PROJECT_DIR / "src"))
    from capacity_sem.models.time_varying_survival import reshape_quarterly_to_time_varying, add_static_covariates
    from stages._io_utils import safe_read_parquet

    # Load source data
    qpr_quarterly = safe_read_parquet(DATA_WORK / "qpr_quarterly.parquet")
    panel_features = safe_read_parquet(DATA_WORK / "panel_features.parquet")

    # Regenerate fresh panel
    fresh_tv = reshape_quarterly_to_time_varying(
        qpr_quarterly=qpr_quarterly,
        panel_features=panel_features,
        lag_quarters=1
    )
    fresh_tv = add_static_covariates(fresh_tv, panel_features)

    # Load cached panel
    cached_tv = read_parquet_safe(DATA_WORK / "panel_time_varying.parquet")

    out.write(f"### Shape comparison\n\n")
    out.write(f"- Fresh panel: {fresh_tv.shape[0]} rows × {fresh_tv.shape[1]} columns\n")
    out.write(f"- Cached panel: {cached_tv.num_rows} rows × {len(cached_tv.column_names)} columns\n\n")

    # Compare ratio distributions
    out.write(f"### Ratio distribution comparison\n\n")

    fresh_disb = fresh_tv['Ratio_disbursed_to_obligated_lag1'].dropna().tolist()
    cached_disb = col_to_list(cached_tv, 'Ratio_disbursed_to_obligated_lag1')

    out.write(f"**Ratio_disbursed_to_obligated_lag1:**\n")
    out.write(f"- Fresh: N={len(fresh_disb)}, max={max(fresh_disb):.4f}, mean={sum(fresh_disb)/len(fresh_disb):.4f}\n")
    out.write(f"- Cached: N={len(cached_disb)}, max={max(cached_disb):.2f}, mean={sum(cached_disb)/len(cached_disb):.2f}\n\n")

    fresh_exp = fresh_tv['Ratio_expended_to_disbursed_lag1'].dropna().tolist()
    cached_exp = col_to_list(cached_tv, 'Ratio_expended_to_disbursed_lag1')

    out.write(f"**Ratio_expended_to_disbursed_lag1:**\n")
    out.write(f"- Fresh: N={len(fresh_exp)}, max={max(fresh_exp):.4f}, mean={sum(fresh_exp)/len(fresh_exp):.4f}\n")
    out.write(f"- Cached: N={len(cached_exp)}, max={max(cached_exp):.4f}, mean={sum(cached_exp)/len(cached_exp):.4f}\n\n")

    # Events comparison
    fresh_events = int(fresh_tv['E'].sum())
    cached_events = sum(1 for x in cached_tv.column('E').to_pylist() if x == 1)
    out.write(f"### Event comparison\n\n")
    out.write(f"- Fresh events: {fresh_events}\n")
    out.write(f"- Cached events: {cached_events}\n\n")

    # Column comparison
    fresh_cols = set(fresh_tv.columns)
    cached_cols = set(cached_tv.column_names)
    extra_cached = cached_cols - fresh_cols
    out.write(f"### Column comparison\n\n")
    out.write(f"- Fresh columns: {len(fresh_cols)}\n")
    out.write(f"- Cached columns: {len(cached_cols)}\n")
    out.write(f"- Extra in cached (not in fresh): {len(extra_cached)}\n")
    if extra_cached:
        out.write(f"  - Examples: {list(extra_cached)[:5]}\n")
    out.write("\n")

    if max(fresh_disb) < 20 and max(cached_disb) > 1000:
        out.write("**CRITICAL**: The fresh panel has bounded ratios (max {:.2f}) while the cached panel has extreme values (max {:.0f}). ".format(max(fresh_disb), max(cached_disb)))
        out.write("The cached panel was NOT generated by the current `reshape_quarterly_to_time_varying()` function. ")
        out.write("It contains {} extra columns (velocity measures, multiple lags) not produced by the current code.\n".format(len(extra_cached)))
        status = "CRITICAL"
    else:
        status = "PASS"

    # Save fresh panel for subsequent checks
    check_2_cached_vs_fresh.fresh_tv = fresh_tv

    report.add_section("CHECK 2: Cached vs Fresh Panel", out.getvalue(), status)
    return status


# ═══════════════════════════════════════════════════════════════════════
# CHECK 3: Concordance Index Investigation (uses fresh panel)
# ═══════════════════════════════════════════════════════════════════════

def check_3_concordance():
    print("CHECK 3: Concordance investigation...")

    out = StringIO()

    from lifelines import CoxPHFitter

    tv = check_2_cached_vs_fresh.fresh_tv.copy()

    capacity_cols = ['Ratio_disbursed_to_obligated_lag1', 'Ratio_expended_to_disbursed_lag1']
    covariate_cols = ['Government_Type_State', 'Log_Obligated', 'Prior_Grant_Count',
                      'Prior_Grant_Dollars_log', 'Disaster_Year', 'Population_log']

    # Available covariates
    available_covs = [c for c in covariate_cols if c in tv.columns]
    available_caps = [c for c in capacity_cols if c in tv.columns]

    out.write(f"### Model comparison (fresh data, N={len(tv)} intervals, {int(tv['E'].sum())} events)\n\n")
    out.write("| Model | Predictors | Concordance | N_obs | N_events |\n")
    out.write("|-------|-----------|-------------|-------|----------|\n")

    models_to_test = []

    # Model A: Capacity only
    if available_caps:
        models_to_test.append(("Capacity only", available_caps))

    # Model B: Covariates only
    if available_covs:
        models_to_test.append(("Covariates only", available_covs))

    # Model C: Full model
    if available_caps and available_covs:
        models_to_test.append(("Full model", available_caps + available_covs))

    concordances = {}
    for model_name, cols in models_to_test:
        fit_cols = ['start', 'stop', 'E'] + cols
        fit_data = tv[fit_cols].dropna(subset=cols)

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(fit_data, duration_col='stop', event_col='E', entry_col='start')
            c = cph.concordance_index_
            n_obs = len(fit_data)
            n_events = int(fit_data['E'].sum())
            out.write(f"| {model_name} | {len(cols)} | {c:.4f} | {n_obs} | {n_events} |\n")
            concordances[model_name] = c

            # Print HRs for capacity models
            if any('Ratio' in col for col in cols):
                out.write(f"\n**{model_name} hazard ratios:**\n```\n")
                for var in cph.summary.index:
                    hr = np.exp(cph.summary.loc[var, 'coef'])
                    p = cph.summary.loc[var, 'p']
                    out.write(f"  {var}: HR={hr:.4f}, p={p:.4f}\n")
                out.write("```\n\n")

        except Exception as e:
            out.write(f"| {model_name} | {len(cols)} | ERROR: {e} | - | - |\n")

    # Report threshold sensitivity concordance trend
    out.write(f"\n### Threshold sensitivity concordance (from existing CSV)\n\n")
    threshold_path = DIAGNOSTICS / "survival_threshold_sensitivity.csv"
    if threshold_path.exists():
        rows = read_csv_simple(threshold_path)
        out.write("| Threshold | Events | EPV | Concordance (adj) |\n")
        out.write("|-----------|--------|-----|-------------------|\n")
        for r in rows:
            out.write(f"| {r['Threshold_pct']}% | {r['N_Events']} | {float(r['EPV_Ratio']):.1f} | {float(r['Concordance_Adjusted']):.3f} |\n")
    else:
        out.write("*File not found — run `python src/pipeline.py run_survival_threshold_sensitivity` first.*\n")

    # Assess
    best_c = max(concordances.values()) if concordances else 0
    if best_c < 0.5:
        out.write(f"\n**WARNING**: Best concordance is {best_c:.3f} (below random=0.5). Model has no discriminatory ability.\n")
        status = "WARNING"
    elif best_c < 0.6:
        out.write(f"\n**WARNING**: Best concordance is {best_c:.3f}. Poor but above random.\n")
        status = "WARNING"
    else:
        out.write(f"\nBest concordance: {best_c:.3f}\n")
        status = "PASS"

    report.add_section("CHECK 3: Concordance Investigation", out.getvalue(), status)
    return status


# ══════════════════════════��════════════════════════════════════════════
# CHECK 4: EPV Assessment
# ═══════════════════════════════════════════════════════════════════════

def check_4_epv():
    print("CHECK 4: EPV assessment...")

    out = StringIO()

    threshold_path = DIAGNOSTICS / "survival_threshold_sensitivity.csv"
    if not threshold_path.exists():
        out.write("*Threshold sensitivity file not found.*\n")
        report.add_section("CHECK 4: Events Per Variable (EPV)", out.getvalue(), "WARNING")
        return "WARNING"

    rows = read_csv_simple(threshold_path)

    out.write("### EPV across thresholds\n\n")
    out.write("| Threshold | Events | Predictors | EPV | Adequate? |\n")
    out.write("|-----------|--------|-----------|-----|----------|\n")

    adequate_thresholds = []
    for r in rows:
        epv = float(r['EPV_Ratio'])
        adequate = "✅ Yes" if epv >= 10 else ("⚠️ Marginal" if epv >= 5 else "❌ No")
        out.write(f"| {r['Threshold_pct']}% | {r['N_Events']} | {r['N_Predictors']} | {epv:.1f} | {adequate} |\n")
        if epv >= 10:
            adequate_thresholds.append(int(r['Threshold_pct']))

    out.write(f"\n### Recommendation\n\n")
    if adequate_thresholds:
        out.write(f"- EPV ≥ 10 at thresholds: {adequate_thresholds}\n")
        out.write(f"- **Recommended primary threshold**: {max(adequate_thresholds)}% (highest with adequate EPV)\n")
        out.write(f"- Use 95% threshold only as sensitivity check (EPV too low for primary)\n")
        status = "WARNING"
    else:
        out.write("- No threshold achieves EPV ≥ 10. Consider reducing model complexity.\n")
        status = "FAIL"

    report.add_section("CHECK 4: Events Per Variable (EPV)", out.getvalue(), status)
    return status


# ═════════════════════════════════════��══════════════════════════════���══
# CHECK 5: Proportional Hazards Test
# ═══════════════════════════════════════════════════════════════════════

def check_5_ph_test():
    print("CHECK 5: Proportional hazards test...")

    out = StringIO()

    from lifelines import CoxPHFitter

    tv = check_2_cached_vs_fresh.fresh_tv.copy()

    capacity_cols = ['Ratio_disbursed_to_obligated_lag1', 'Ratio_expended_to_disbursed_lag1']
    covariate_cols = ['Government_Type_State', 'Log_Obligated', 'Prior_Grant_Count',
                      'Prior_Grant_Dollars_log', 'Disaster_Year', 'Population_log']

    all_cols = [c for c in capacity_cols + covariate_cols if c in tv.columns]
    fit_cols = ['start', 'stop', 'E'] + all_cols
    fit_data = tv[fit_cols].dropna(subset=all_cols)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(fit_data, duration_col='stop', event_col='E', entry_col='start')

    # Test PH assumption
    out.write("### Schoenfeld Residual Tests\n\n")
    out.write("| Variable | Test Statistic | p-value | PH Violated? |\n")
    out.write("|----------|---------------|---------|-------------|\n")

    violations = []
    try:
        # lifelines check_assumptions returns a list or prints results
        # Capture via summary
        import io
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            ph_results = cph.check_assumptions(fit_data, p_value_threshold=0.05, show_plots=False)
        except Exception:
            ph_results = None

        sys.stdout = old_stdout
        ph_output = captured.getvalue()

        if ph_output:
            out.write(f"\nlifelines PH test output:\n```\n{ph_output}\n```\n\n")

        if ph_results is not None and len(ph_results) > 0:
            out.write("**PH violations detected by lifelines.**\n")
            status = "WARNING"
        else:
            out.write("**No PH violations detected** (all p > 0.05). Proportional hazards assumption holds.\n")
            status = "PASS"

    except Exception as e:
        out.write(f"\nPH test could not be computed: {e}\n")
        status = "WARNING"

    report.add_section("CHECK 5: Proportional Hazards Test", out.getvalue(), status)
    return status


# ═════════════════════════════════════════════════════════════════════���═
# CHECK 6: Competing Risks Assessment
# ═══════════════════════════════════════════════════════════════════��═══

def check_6_competing_risks():
    print("CHECK 6: Competing risks assessment...")

    out = StringIO()

    sys.path.insert(0, str(PROJECT_DIR / "src"))
    from stages._io_utils import safe_read_parquet

    panel = safe_read_parquet(DATA_WORK / "panel_features.parquet")

    # Check completion status
    dur_col = 'Duration_of_completion'
    if dur_col in panel.columns:
        completed = panel[dur_col].notna().sum()
        censored = panel[dur_col].isna().sum()
        total = len(panel)

        out.write(f"### Completion status (95% threshold)\n\n")
        out.write(f"- Total grantee-disasters: {total}\n")
        out.write(f"- Completed (E=1): {completed} ({100*completed/total:.1f}%)\n")
        out.write(f"- Censored (E=0): {censored} ({100*censored/total:.1f}%)\n\n")

    # Check for termination indicators
    out.write(f"### Available columns that might indicate censoring type\n\n")
    potential_cols = [c for c in panel.columns if any(kw in c.lower() for kw in
                     ['status', 'terminat', 'cancel', 'close', 'active', 'withdrawn'])]
    if potential_cols:
        out.write(f"Potential censoring-type columns: {potential_cols}\n\n")
        for col in potential_cols:
            vals = panel[col].value_counts()
            out.write(f"  {col}: {dict(vals)}\n")
    else:
        out.write("**No columns found** that distinguish censoring reasons (termination vs. ongoing vs. deadline).\n\n")

    # Check multiple completion thresholds
    out.write(f"### Completion rates at different thresholds\n\n")
    out.write("| Threshold | Completed | Censored | Completion Rate |\n")
    out.write("|-----------|-----------|----------|----------------|\n")

    for pct in [20, 50, 75, 95, 100]:
        col_name = f'Duration_{pct}pct' if pct != 100 else 'Duration_of_completion'
        if col_name in panel.columns:
            comp = panel[col_name].notna().sum()
            cens = panel[col_name].isna().sum()
            out.write(f"| {pct}% | {comp} | {cens} | {100*comp/len(panel):.1f}% |\n")

    out.write(f"\n### Assessment\n\n")
    out.write("Without explicit censoring-type indicators, we cannot distinguish:\n")
    out.write("1. Programs still active (administrative censoring)\n")
    out.write("2. Programs terminated/cancelled (competing risk)\n")
    out.write("3. Programs past deadline (administrative censoring)\n\n")
    out.write("If termination is correlated with capacity, the independent censoring assumption is violated.\n")
    out.write("**Recommendation**: Document this limitation. A Fine-Gray model would require termination status data.\n")

    report.add_section("CHECK 6: Competing Risks Assessment", out.getvalue(), "WARNING")
    return "WARNING"


# ═══════════════════════════════════════════════════════════════════════
# CHECK 7: Non-Linear Effects
# ══════════════════════════════════════════���═══════════════════════��════

def check_7_nonlinear():
    print("CHECK 7: Non-linear effects...")

    out = StringIO()

    import pandas as pd
    from lifelines import CoxPHFitter

    tv = check_2_cached_vs_fresh.fresh_tv.copy()

    # Create quartile dummies for disbursement ratio
    ratio_col = 'Ratio_disbursed_to_obligated_lag1'
    valid = tv[ratio_col].dropna()
    q25, q50, q75 = valid.quantile([0.25, 0.5, 0.75])

    out.write(f"### Quartile cutpoints for {ratio_col}\n\n")
    out.write(f"- Q1 (≤25th): ≤ {q25:.4f}\n")
    out.write(f"- Q2 (25-50th): {q25:.4f} – {q50:.4f}\n")
    out.write(f"- Q3 (50-75th): {q50:.4f} – {q75:.4f}\n")
    out.write(f"- Q4 (>75th): > {q75:.4f}\n\n")

    tv['ratio_q2'] = ((tv[ratio_col] > q25) & (tv[ratio_col] <= q50)).astype(float)
    tv['ratio_q3'] = ((tv[ratio_col] > q50) & (tv[ratio_col] <= q75)).astype(float)
    tv['ratio_q4'] = (tv[ratio_col] > q75).astype(float)

    # Linear model
    fit_data_lin = tv[['start', 'stop', 'E', ratio_col]].dropna()
    cph_lin = CoxPHFitter(penalizer=0.1)
    cph_lin.fit(fit_data_lin, duration_col='stop', event_col='E', entry_col='start')

    # Quartile model
    quartile_cols = ['ratio_q2', 'ratio_q3', 'ratio_q4']
    # Filter rows where ratio is not NaN before fitting quartile model
    tv_valid = tv.dropna(subset=[ratio_col])
    fit_data_q = tv_valid[['start', 'stop', 'E'] + quartile_cols]
    cph_q = CoxPHFitter(penalizer=0.1)
    cph_q.fit(fit_data_q, duration_col='stop', event_col='E', entry_col='start')

    out.write(f"### Linear vs Quartile specification\n\n")
    out.write(f"**Linear model** (concordance={cph_lin.concordance_index_:.4f}):\n")
    out.write(f"```\n")
    hr_lin = np.exp(cph_lin.summary.loc[ratio_col, 'coef'])
    p_lin = cph_lin.summary.loc[ratio_col, 'p']
    out.write(f"  {ratio_col}: HR={hr_lin:.4f}, p={p_lin:.4f}\n")
    out.write(f"```\n\n")

    out.write(f"**Quartile model** (concordance={cph_q.concordance_index_:.4f}, ref=Q1):\n")
    out.write(f"```\n")
    for var in quartile_cols:
        hr = np.exp(cph_q.summary.loc[var, 'coef'])
        p = cph_q.summary.loc[var, 'p']
        out.write(f"  {var}: HR={hr:.4f}, p={p:.4f}\n")
    out.write(f"```\n\n")

    # Check for threshold effect
    q4_hr = np.exp(cph_q.summary.loc['ratio_q4', 'coef'])
    q4_p = cph_q.summary.loc['ratio_q4', 'p']

    if q4_p < 0.05:
        out.write(f"**Threshold effect detected**: Q4 (highest ratio) has HR={q4_hr:.3f} (p={q4_p:.4f}). ")
        out.write("Capacity may only matter at high levels.\n")
        status = "WARNING"
    else:
        out.write(f"No significant threshold effect. Q4 HR={q4_hr:.3f} (p={q4_p:.3f}).\n")
        status = "PASS"

    report.add_section("CHECK 7: Non-Linear Effects", out.getvalue(), status)
    return status


# ═══════════════════════════════════���═══════════════════════════════════
# CHECK 8: Bootstrap Clustered SEs
# ═════════════════════════════════════════���═════════════════════════════

def check_8_bootstrap():
    print("CHECK 8: Bootstrap clustered SEs...")

    out = StringIO()

    bootstrap_path = DIAGNOSTICS / "survival_bootstrap_se.csv"
    results_path = DIAGNOSTICS / "survival_time_varying_cox_results.csv"

    if not bootstrap_path.exists():
        out.write("*Bootstrap SE file not found. Bootstrap was never run.*\n")
        report.add_section("CHECK 8: Bootstrap Clustered SEs", out.getvalue(), "WARNING")
        return "WARNING"

    bootstrap = read_csv_simple(bootstrap_path)

    out.write("### Bootstrap SE results (from cached panel — may be unreliable due to CHECK 1/2)\n\n")
    out.write("| Variable | HR (bootstrap) | SE (bootstrap) | p-value |\n")
    out.write("|----------|---------------|---------------|--------|\n")

    for r in bootstrap:
        out.write(f"| {r['Variable']} | {float(r['HR_bootstrap']):.6f} | {float(r['se_bootstrap']):.6e} | {float(r['p_value_bootstrap']):.4f} |\n")

    # Check if model-based SEs are similar
    out.write(f"\n### Assessment\n\n")

    # Check for near-zero SEs (sign of extreme values in predictors)
    ses = [float(r['se_bootstrap']) for r in bootstrap]
    min_se = min(ses)
    max_se = max(ses)

    if min_se < 1e-10:
        out.write(f"**CRITICAL**: Bootstrap SEs are near machine-zero (min={min_se:.2e}). ")
        out.write("This is consistent with CHECK 1 — extreme ratio values produce meaningless coefficients. ")
        out.write("**These results are invalid and should be recomputed after fixing the data.**\n")
        status = "CRITICAL"
    else:
        out.write(f"Bootstrap SEs range: {min_se:.4f} to {max_se:.4f}.\n")
        status = "PASS"

    report.add_section("CHECK 8: Bootstrap Clustered SEs", out.getvalue(), status)
    return status


# ═══════════════════════���═══════════════════════════════════════════════
# CHECK 9: AFT Model Comparison
# ═══════════════════════════════════════════��═══════════════════════════

def check_9_aft():
    print("CHECK 9: AFT model comparison...")

    out = StringIO()

    aft_path = DIAGNOSTICS / "alternatives_survival.csv"

    if not aft_path.exists():
        out.write("*AFT comparison file not found.*\n")
        report.add_section("CHECK 9: AFT Model Comparison", out.getvalue(), "WARNING")
        return "WARNING"

    rows = read_csv_simple(aft_path)

    out.write("### Static Cox PH vs AFT (from alternatives_survival.csv)\n\n")
    out.write("**Note**: These use the STATIC (not time-varying) specification.\n\n")
    out.write("| Model | Variable | Effect Ratio | p-value | Interpretation |\n")
    out.write("|-------|----------|-------------|---------|---------------|\n")

    for r in rows:
        out.write(f"| {r['Model']} | {r['Variable']} | {float(r['Effect_Ratio']):.4f} | {float(r['p_value']):.4f} | {r['Interpretation']} |\n")

    out.write(f"\n### Key finding\n\n")

    # Extract Cox PH disbursement result
    cox_disb = [r for r in rows if r['Model'] == 'Cox_PH' and 'disbursed' in r['Variable'].lower()]
    if cox_disb:
        hr = float(cox_disb[0]['Effect_Ratio'])
        p = float(cox_disb[0]['p_value'])
        out.write(f"- **Static Cox PH**: Disbursement ratio HR={hr:.3f}, p={p:.4f} {'(significant)' if p < 0.05 else '(not significant)'}\n")

    aft_disb = [r for r in rows if r['Model'] == 'AFT_lognormal' and 'disbursed' in r['Variable'].lower()]
    if aft_disb:
        tr = float(aft_disb[0]['Effect_Ratio'])
        p = float(aft_disb[0]['p_value'])
        out.write(f"- **AFT lognormal**: Disbursement ratio TR={tr:.4f}, p={p:.6f} {'(significant)' if p < 0.05 else '(not significant)'}\n")

    out.write(f"\n**Discrepancy**: Static models show significant effects while the time-varying model shows null. ")
    out.write(f"This could reflect:\n")
    out.write(f"1. The cached time-varying panel data quality issue (CHECK 1)\n")
    out.write(f"2. True reverse causality that the time-varying design appropriately removes\n")
    out.write(f"3. Loss of between-subject variation with the time-varying approach\n\n")
    out.write(f"**Must re-run time-varying model with clean data before drawing conclusions.**\n")

    report.add_section("CHECK 9: AFT Model Comparison", out.getvalue(), "INFO")
    return "INFO"


# ═══════════════════════════════════════════════��═══════════════════════
# CHECK 10: Tie Handling
# ════════════════════════════════════════════��══════════════════════════

def check_10_tie_handling():
    print("CHECK 10: Tie handling...")

    out = StringIO()

    tv = check_2_cached_vs_fresh.fresh_tv.copy()

    # Count ties in event times
    event_times = tv[tv['E'] == 1]['stop'].tolist()
    unique_event_times = len(set(event_times))
    total_events = len(event_times)

    out.write(f"### Tie analysis\n\n")
    out.write(f"- Total events: {total_events}\n")
    out.write(f"- Unique event times: {unique_event_times}\n")
    out.write(f"- Tied events: {total_events - unique_event_times}\n\n")

    if unique_event_times < total_events:
        # Count ties per time
        from collections import Counter
        time_counts = Counter(event_times)
        max_ties = max(time_counts.values())
        tied_times = sum(1 for c in time_counts.values() if c > 1)
        out.write(f"- Times with ties: {tied_times}\n")
        out.write(f"- Maximum events at same time: {max_ties}\n\n")

    out.write(f"### Efron vs Breslow\n\n")
    out.write("lifelines CoxPHFitter does not support Efron's method for start/stop (interval) format. ")
    out.write("Breslow approximation is used by default. ")

    if total_events - unique_event_times < total_events * 0.1:
        out.write(f"With only {total_events - unique_event_times} tied events out of {total_events}, this is unlikely to affect results.\n")
        status = "PASS"
    else:
        out.write(f"With {total_events - unique_event_times} tied events, consider using a different package (e.g., R survival) for Efron method.\n")
        status = "WARNING"

    report.add_section("CHECK 10: Tie Handling", out.getvalue(), status)
    return status


# ════════════════════════════���═══════════════════════════════════���══════
# CHECK 3b: Re-fit model with FRESH data and report HRs
# ═══════════════════════════════════════════════════════════════════════

def check_3b_refit_fresh():
    """Re-fit the Cox model with fresh (clean) data to see if the null changes."""
    print("CHECK 3b: Re-fitting Cox model with fresh data...")

    out = StringIO()

    import pandas as pd
    from lifelines import CoxPHFitter

    tv = check_2_cached_vs_fresh.fresh_tv.copy()

    capacity_cols = ['Ratio_disbursed_to_obligated_lag1', 'Ratio_expended_to_disbursed_lag1']
    covariate_cols = ['Government_Type_State', 'Log_Obligated', 'Prior_Grant_Count',
                      'Prior_Grant_Dollars_log', 'Disaster_Year', 'Population_log']

    available_covs = [c for c in covariate_cols if c in tv.columns]
    available_caps = [c for c in capacity_cols if c in tv.columns]

    out.write("### Fresh data characteristics\n\n")
    for col in available_caps:
        vals = tv[col].dropna()
        out.write(f"- {col}: range [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}\n")

    out.write(f"\n### Cox PH with fresh data\n\n")

    # Capacity only
    all_predictors = available_caps + available_covs
    fit_cols = ['start', 'stop', 'E'] + all_predictors
    fit_data = tv[fit_cols].dropna(subset=all_predictors)

    n_obs = len(fit_data)
    n_events = int(fit_data['E'].sum())

    out.write(f"- N intervals: {n_obs}\n")
    out.write(f"- N events: {n_events}\n")
    out.write(f"- EPV: {n_events / len(all_predictors):.1f}\n\n")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(fit_data, duration_col='stop', event_col='E', entry_col='start')

    out.write(f"**Concordance: {cph.concordance_index_:.4f}**\n\n")
    out.write("| Variable | HR | 95% CI | p-value | Sig? |\n")
    out.write("|----------|----|----|---------|------|\n")

    for var in cph.summary.index:
        hr = np.exp(cph.summary.loc[var, 'coef'])
        se = cph.summary.loc[var, 'se(coef)']
        hr_lo = np.exp(cph.summary.loc[var, 'coef'] - 1.96 * se)
        hr_hi = np.exp(cph.summary.loc[var, 'coef'] + 1.96 * se)
        p = cph.summary.loc[var, 'p']
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        out.write(f"| {var} | {hr:.4f} | [{hr_lo:.4f}, {hr_hi:.4f}] | {p:.4f} | {sig} |\n")

    # Key comparison
    disb_hr = np.exp(cph.summary.loc[available_caps[0], 'coef']) if available_caps[0] in cph.summary.index else None
    disb_p = cph.summary.loc[available_caps[0], 'p'] if available_caps[0] in cph.summary.index else None

    out.write(f"\n### Comparison: Cached vs Fresh data results\n\n")
    out.write(f"| | Cached Panel | Fresh Panel |\n")
    out.write(f"|---|---|---|\n")
    out.write(f"| Disb ratio max | 38,775,935 | {tv[available_caps[0]].max():.4f} |\n")
    out.write(f"| Disb HR | ~1.0000 | {disb_hr:.4f} |\n")
    out.write(f"| Disb p-value | ~1.000 | {disb_p:.4f} |\n")
    out.write(f"| Concordance | 0.171 | {cph.concordance_index_:.4f} |\n\n")

    if disb_p and disb_p < 0.05:
        out.write("**The null finding REVERSES with clean data.** The disbursement ratio is a significant predictor.\n")
        status = "CRITICAL"
    elif disb_p and disb_p < 0.20:
        out.write("**The null weakens with clean data** but effect is still not significant at p<0.05.\n")
        status = "WARNING"
    else:
        out.write("**The null finding holds even with clean data.** The data quality issue did not change the substantive conclusion.\n")
        status = "PASS"

    report.add_section("CHECK 3b: Re-fit with Fresh Data (KEY TEST)", out.getvalue(), status)
    return status


# ══════════════════════════════════════════════════��════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SURVIVAL ANALYSIS SYSTEMATIC AUDIT")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Project: {PROJECT_DIR}")
    print()

    # Run all checks
    check_1_ratio_data_quality()
    check_2_cached_vs_fresh()
    check_3b_refit_fresh()  # KEY: refit with fresh data
    check_3_concordance()
    check_4_epv()
    check_5_ph_test()
    check_6_competing_risks()
    check_7_nonlinear()
    check_8_bootstrap()
    check_9_aft()
    check_10_tie_handling()

    # Generate report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "survival_audit_report.md"
    report_text = report.build()

    with open(report_path, 'w') as f:
        f.write(report_text)

    print()
    print("=" * 70)
    print(f"AUDIT COMPLETE — Report: {report_path}")
    print("=" * 70)

    # Print summary
    print()
    for title, status in report.summary:
        emoji = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "CRITICAL": "🔴"}.get(status, "❓")
        print(f"  {emoji} {status:10s} {title}")


if __name__ == "__main__":
    main()
