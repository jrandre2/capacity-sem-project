"""
Stage 03c - Provisional external reconstruction of Kaifa's later SEM workflow.

IMPORTANT
---------
This stage is not Kaifa Lu's original analysis code. It rebuilds the later
SEM tables from imported external exports that were added to:

    manuscript_kaifa_archive/data/external_sem_exports/

Use it only as a temporary reconstruction layer until the original
notebook/scripts are recovered.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from capacity_sem.models.sem_external_replication import (
    EXTERNAL_SEM_BUNDLES,
    KAIFA_EXTERNAL_LABEL,
    KAIFA_EXTERNAL_WARNING,
    run_external_replication,
)


def _print_fit_summary(fit_comparison):
    """Print a compact comparison of exported versus reconstructed fit."""
    for statistic in ["CFI", "TLI", "RMSEA", "AIC", "BIC"]:
        row = fit_comparison.loc[fit_comparison["Statistic"] == statistic]
        if row.empty:
            continue
        record = row.iloc[0]
        print(
            f"  {statistic}: "
            f"one-factor {record['Exported_OneFactor']:.6f} -> {record['Refit_OneFactor']:.6f} "
            f"(delta {record['Delta_OneFactor']:+.6f}); "
            f"two-factor {record['Exported_TwoFactor']:.6f} -> {record['Refit_TwoFactor']:.6f} "
            f"(delta {record['Delta_TwoFactor']:+.6f})"
        )


def main(bundle: str = "baseline_sem", output_dir: str | None = None) -> None:
    """Run the provisional external reconstruction for one or all bundles."""
    bundles = sorted(EXTERNAL_SEM_BUNDLES) if bundle == "all" else [bundle]

    print("=" * 72)
    print("KAIFA EXTERNAL SEM RECONSTRUCTION")
    print("=" * 72)
    print(f"Label: {KAIFA_EXTERNAL_LABEL}")
    print(KAIFA_EXTERNAL_WARNING)

    for bundle_name in bundles:
        bundle_output_dir = None
        if output_dir:
            bundle_output_dir = str(Path(output_dir) / bundle_name) if bundle == "all" else output_dir

        print("\n" + "-" * 72)
        print(f"Bundle: {bundle_name}")
        print("-" * 72)
        result = run_external_replication(
            bundle_name=bundle_name,
            output_dir=bundle_output_dir,
            write_outputs=True,
        )
        dataset = result["dataset"]
        fit_comparison = result["fit_comparison"]
        print(f"  Dataset shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
        _print_fit_summary(fit_comparison)
        print(f"  Wrote external reconstruction outputs to: {result['output_dir']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for direct stage execution."""
    parser = argparse.ArgumentParser(
        description="Run the provisional external Kaifa SEM reconstruction."
    )
    parser.add_argument(
        "--bundle",
        default="baseline_sem",
        choices=[*sorted(EXTERNAL_SEM_BUNDLES), "all"],
        help="Imported external SEM bundle to reconstruct (default: baseline_sem)",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to data_work/diagnostics/kaifa_external_replication/<bundle>/",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(bundle=args.bundle, output_dir=args.output_dir)
