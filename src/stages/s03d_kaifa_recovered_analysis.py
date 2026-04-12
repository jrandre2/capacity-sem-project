"""
Stage 03d - Recovered Kaifa notebook analysis port.

This stage runs the repository-local port of the recovered Kaifa notebook
bundle imported under:

    manuscript_kaifa_archive/data/recovered_code_bundle/

It preserves an explicit distinction between:
1. the saved notebook outputs from the recovered bundle, which reference a
   573-row SEM run, and
2. the CSV currently bundled in that ZIP, which reruns at 577 rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from capacity_sem.models.kaifa_recovered_analysis import (
    KAIFA_RECOVERED_LABEL,
    KAIFA_RECOVERED_WARNING,
    run_recovered_analysis,
)


def _print_fit_summary(reference_fit_comparison):
    """Print a compact fit comparison against the saved reference bundle."""
    if reference_fit_comparison is None:
        print("  No reference fit comparison available.")
        return

    for statistic in ["CFI", "TLI", "RMSEA", "AIC", "BIC"]:
        row = reference_fit_comparison.loc[
            reference_fit_comparison["Statistic"] == statistic
        ]
        if row.empty:
            continue
        record = row.iloc[0]
        print(
            f"  {statistic}: "
            f"reference one-factor {record['OneFactor']:.6f} -> rerun {record['OneFactor_rerun']:.6f} "
            f"(delta {record['Delta_OneFactor']:+.6f}); "
            f"reference two-factor {record['TwoFactor']:.6f} -> rerun {record['TwoFactor_rerun']:.6f} "
            f"(delta {record['Delta_TwoFactor']:+.6f})"
        )


def main(output_dir: str | None = None) -> None:
    """Run the recovered Kaifa notebook analysis port."""
    print("=" * 72)
    print("KAIFA RECOVERED NOTEBOOK ANALYSIS")
    print("=" * 72)
    print(f"Label: {KAIFA_RECOVERED_LABEL}")
    print(KAIFA_RECOVERED_WARNING)

    result = run_recovered_analysis(output_dir=output_dir, write_outputs=True)
    notebook_metadata = result["notebook_metadata"]
    partial_audit = result["partial_audit"]

    print("\n" + "-" * 72)
    print("Row-count audit")
    print("-" * 72)
    print(f"  Raw CSV rows rerun: {result['raw_df'].shape[0]}")
    print(f"  SEM-ready rows rerun: {result['sem_data'].shape[0]}")
    print(f"  Saved notebook rows: {notebook_metadata.get('saved_dataset_rows')}")
    print(
        "  Saved notebook state/local counts: "
        f"{notebook_metadata.get('saved_state_count')} state / "
        f"{notebook_metadata.get('saved_local_count')} local"
    )
    if result["excluded_grantees"]:
        print(
            "  Identified extra grantees in recovered CSV: "
            + ", ".join(result["excluded_grantees"])
        )

    alignment = result["candidate_alignment_summary"]
    if alignment is not None:
        exact = alignment.loc[alignment["exact_match"], "column"].tolist()
        drift = alignment.loc[~alignment["exact_match"], "column"].tolist()
        print(f"  Candidate 573 subset exact-match columns: {len(exact)}")
        print("  Candidate 573 subset drifting columns: " + ", ".join(drift))

    if result["maturity_summary"] is not None:
        print("\n  Portfolio maturity proxy summary:")
        for _, row in result["maturity_summary"].iterrows():
            print(
                f"    {row['maturity_band']}: n={int(row['jurisdictions'])}, "
                f"state={int(row['state_cases'])}, local={int(row['local_cases'])}, "
                f"median programs={row['median_programs']:.1f}, "
                f"median duration={row['median_duration_months']:.1f} months"
            )

    print("\n" + "-" * 72)
    print("Reference vs rerun fit")
    print("-" * 72)
    _print_fit_summary(result["reference_fit_comparison"])

    if result["candidate_reference_fit_comparison"] is not None:
        print("\n" + "-" * 72)
        print("Reference vs candidate 573 fit")
        print("-" * 72)
        for statistic in ["CFI", "TLI", "RMSEA", "AIC", "BIC"]:
            row = result["candidate_reference_fit_comparison"].loc[
                result["candidate_reference_fit_comparison"]["Statistic"] == statistic
            ]
            if row.empty:
                continue
            record = row.iloc[0]
            print(
                f"  {statistic}: "
                f"reference one-factor {record['OneFactor']:.6f} -> candidate {record['OneFactor_candidate_573']:.6f} "
                f"(delta {record['Delta_OneFactor_candidate_573']:+.6f}); "
                f"reference two-factor {record['TwoFactor']:.6f} -> candidate {record['TwoFactor_candidate_573']:.6f} "
                f"(delta {record['Delta_TwoFactor_candidate_573']:+.6f})"
            )

    print("\n" + "-" * 72)
    print("Partial notebook audit")
    print("-" * 72)
    print(
        "  Uses avg_employment-only denominator: "
        f"{partial_audit['uses_avg_employment_only_denominator']}"
    )
    print(
        "  Missing shapefiles: "
        + ", ".join(partial_audit["missing_shapefiles"])
        if partial_audit["missing_shapefiles"]
        else "  Missing shapefiles: none"
    )

    if result["sensitivity_summary"] is not None:
        print("\n" + "-" * 72)
        print("Light-touch sensitivity summary")
        print("-" * 72)
        for _, row in result["sensitivity_summary"].iterrows():
            print(
                f"  {row['specification']}: CFI={row['CFI']:.3f}, RMSEA={row['RMSEA']:.3f}, "
                f"burden->timeliness beta={row['burden_to_timeliness_beta']:.3f}, "
                f"burden->performance beta={row['burden_to_performance_beta']:.3f}"
            )

    if result["official_geography_summary"] is not None:
        print("\n" + "-" * 72)
        print("Official geography audit")
        print("-" * 72)
        for _, row in result["official_geography_summary"].iterrows():
            print(f"  {row['mapping_step']}: {row['count_or_implication']}")

    if result["proxy_validation_summary"] is not None:
        print("\n" + "-" * 72)
        print("Proxy validation summary")
        print("-" * 72)
        for _, row in result["proxy_validation_summary"].head(5).iterrows():
            print(
                f"  {row['check']}: estimate={row['estimate']:.3f} "
                f"({row['sample']})"
            )

    if result["subset_forensics_summary"] is not None:
        print("\n" + "-" * 72)
        print("N=169 subset forensics")
        print("-" * 72)
        for _, row in result["subset_forensics_summary"].iterrows():
            print(f"  {row['finding']}: {row['value']}")

    if result["data_flow_summary"] is not None:
        print("\n" + "-" * 72)
        print("Recoverable data flow")
        print("-" * 72)
        for _, row in result["data_flow_summary"].iterrows():
            print(f"  {row['stage']}: {row['count']}")

    if result["ratio_artifact_summary"] is not None:
        print("\n" + "-" * 72)
        print("Ratio artifact summary")
        print("-" * 72)
        for _, row in result["ratio_artifact_summary"].head(4).iterrows():
            print(f"  {row['element']}: {row['value']}")
    print(f"\n  Wrote recovered analysis outputs to: {result['output_dir']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for direct stage execution."""
    parser = argparse.ArgumentParser(
        description="Run the recovered Kaifa notebook analysis port."
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data_work/diagnostics/kaifa_recovered_analysis/",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir)
