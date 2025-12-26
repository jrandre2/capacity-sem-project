#!/usr/bin/env python3
"""
Capacity-SEM Analysis Pipeline

A command-line interface for running structural equation model analysis
of government capacity effects on disaster recovery outcomes.

Usage:
    python src/pipeline.py <command> [options]

Commands:
    ingest_data       Load QPR and external data
    build_panel       Construct analysis panel
    compute_features  Calculate indicators and features
    run_estimation    Fit SEM models
    run_robustness    Run robustness checks
    run_survival      Run time-varying survival analysis
    run_survival_threshold_sensitivity
                      Run survival analysis across all completion thresholds (20%-100%)
    make_figures      Generate publication figures
    run_all           Run complete pipeline

    review_status     Display current review cycle status
    review_new        Initialize new review cycle
    review_archive    Archive current review cycle
    review_verify     Run verification checklist
    review_report     Generate summary report of all reviews

Examples:
    python src/pipeline.py ingest_data
    python src/pipeline.py run_estimation --model exp_optimal_v1 --subset state
    python src/pipeline.py make_figures --style publication
    python src/pipeline.py review_new --focus par_general
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def cmd_ingest_data(args):
    """Load QPR and external data."""
    from stages import s00_ingest
    s00_ingest.main(demo=args.demo)


def cmd_build_panel(args):
    """Construct analysis panel."""
    from stages import s01_link
    s01_link.main()


def cmd_compute_features(args):
    """Calculate indicators and features."""
    from stages import s02_features
    s02_features.main()


def cmd_run_estimation(args):
    """Fit SEM models."""
    from stages import s03_estimation
    s03_estimation.main(model=args.model, subset=args.subset)


def cmd_run_robustness(args):
    """Run robustness checks."""
    from stages import s04_robustness
    s04_robustness.main(models=args.models)


def cmd_run_survival(args):
    """Run time-varying survival analysis."""
    from stages import s03b_survival_estimation
    s03b_survival_estimation.main()


def cmd_run_survival_threshold_sensitivity(args):
    """Run survival analysis across all completion thresholds."""
    from stages import s03b_survival_estimation
    results_df = s03b_survival_estimation.run_threshold_sensitivity_analysis()

    # Print summary
    print("\n" + "="*80)
    print("THRESHOLD SENSITIVITY SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))

    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find thresholds with EPV >= 10
    adequate_power = results_df[results_df['EPV_Ratio'] >= 10.0]
    if len(adequate_power) > 0:
        print(f"\nThresholds with adequate power (EPV ≥ 10):")
        for _, row in adequate_power.iterrows():
            print(f"  {row['Threshold_pct']}%: {row['N_Events']} events, EPV={row['EPV_Ratio']:.1f}")
    else:
        print("\nNo thresholds achieve EPV ≥ 10 (guideline for stable estimates)")

    # Find significant effects
    significant = results_df[results_df['Disb_p_Adjusted'] < 0.05]
    if len(significant) > 0:
        print(f"\nThresholds with significant disbursement effect (p < 0.05):")
        for _, row in significant.iterrows():
            print(f"  {row['Threshold_pct']}%: HR={row['Disb_HR_Adjusted']:.3f}, p={row['Disb_p_Adjusted']:.3f}")
    else:
        print("\nNo thresholds show significant disbursement effects at p < 0.05")


def cmd_run_alternatives(args):
    """Run alternative modeling approaches."""
    from stages import s06_alternatives

    # Handle convenience flags
    if args.survival_only:
        methods = ['survival']
    elif args.sem_only:
        methods = ['threshold', 'duration_free', 'milestone']
    elif args.methods and 'all' in args.methods:
        methods = None
    else:
        methods = args.methods

    s06_alternatives.main(methods=methods, subset=args.subset)


def cmd_make_figures(args):
    """Generate publication figures."""
    from stages import s05_figures
    s05_figures.main(style=args.style)


def cmd_run_all(args):
    """Run complete pipeline."""
    print("=" * 60)
    print("Running Complete Pipeline")
    print("=" * 60)

    # Stage 0: Ingest
    print("\n" + "=" * 60)
    from stages import s00_ingest
    s00_ingest.main(demo=args.demo)

    # Stage 1: Panel
    print("\n" + "=" * 60)
    from stages import s01_link
    s01_link.main()

    # Stage 2: Features
    print("\n" + "=" * 60)
    from stages import s02_features
    s02_features.main()

    # Stage 3: Estimation
    print("\n" + "=" * 60)
    from stages import s03_estimation
    s03_estimation.main(model=args.model, subset=args.subset)

    # Stage 4: Robustness
    if not args.skip_robustness:
        print("\n" + "=" * 60)
        from stages import s04_robustness
        s04_robustness.main()

    # Stage 5: Figures
    print("\n" + "=" * 60)
    from stages import s05_figures
    s05_figures.main()

    print("\n" + "=" * 60)
    print("✓ Pipeline complete!")
    print("=" * 60)


def cmd_review_status(args):
    """Display current review cycle status."""
    import review_management
    review_management.status()


def cmd_review_new(args):
    """Initialize a new review cycle."""
    import review_management
    review_management.new_cycle(focus=args.focus)


def cmd_review_archive(args):
    """Archive current review cycle."""
    import review_management
    review_management.archive()


def cmd_review_verify(args):
    """Run verification checklist."""
    import review_management
    review_management.verify()


def cmd_review_report(args):
    """Generate summary report of all review cycles."""
    import review_management
    review_management.report()


def cmd_list_models(args):
    """List available SEM model specifications."""
    from capacity_sem.models.sem_specifications import list_available_models

    models = list_available_models()

    print("Available SEM Model Specifications")
    print("=" * 60)

    for name, description in models.items():
        print(f"\n{name}:")
        # Print first 2 lines of description
        lines = description.strip().split('\n')[:2]
        for line in lines:
            print(f"  {line}")

    print(f"\n\nTotal: {len(models)} models")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capacity-SEM Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        metavar="<command>"
    )

    # ingest_data
    p_ingest = subparsers.add_parser(
        "ingest_data",
        help="Load QPR and external data"
    )
    p_ingest.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Use demo/synthetic data"
    )
    p_ingest.set_defaults(func=cmd_ingest_data)

    # build_panel
    p_panel = subparsers.add_parser(
        "build_panel",
        help="Construct analysis panel"
    )
    p_panel.set_defaults(func=cmd_build_panel)

    # compute_features
    p_features = subparsers.add_parser(
        "compute_features",
        help="Calculate indicators and features"
    )
    p_features.set_defaults(func=cmd_compute_features)

    # run_estimation
    p_estimate = subparsers.add_parser(
        "run_estimation",
        help="Fit SEM models"
    )
    p_estimate.add_argument(
        "--model", "-m",
        default="exp_optimal_v1",
        help="Model specification (default: exp_optimal_v1)"
    )
    p_estimate.add_argument(
        "--subset", "-s",
        default="all",
        choices=["all", "state", "local"],
        help="Government subset (default: all)"
    )
    p_estimate.set_defaults(func=cmd_run_estimation)

    # run_robustness
    p_robust = subparsers.add_parser(
        "run_robustness",
        help="Run robustness checks"
    )
    p_robust.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        help="Model specifications to check"
    )
    p_robust.set_defaults(func=cmd_run_robustness)

    # run_survival
    p_survival = subparsers.add_parser(
        "run_survival",
        help="Run time-varying survival analysis"
    )
    p_survival.set_defaults(func=cmd_run_survival)

    # run_survival_threshold_sensitivity
    p_survival_threshold = subparsers.add_parser(
        "run_survival_threshold_sensitivity",
        help="Run survival analysis across all completion thresholds (20%-100%)"
    )
    p_survival_threshold.set_defaults(func=cmd_run_survival_threshold_sensitivity)

    # run_alternatives
    p_alternatives = subparsers.add_parser(
        "run_alternatives",
        help="Run alternative modeling approaches (survival, threshold, duration-free, milestone)"
    )
    p_alternatives.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        choices=['survival', 'threshold', 'duration_free', 'milestone', 'all'],
        help="Methods to run (default: all)"
    )
    p_alternatives.add_argument(
        "--subset", "-s",
        default="all",
        choices=["all", "state", "local"],
        help="Government subset"
    )
    p_alternatives.add_argument(
        "--survival-only",
        action="store_true",
        help="Run only survival analysis"
    )
    p_alternatives.add_argument(
        "--sem-only",
        action="store_true",
        help="Run only SEM alternatives (no survival)"
    )
    p_alternatives.set_defaults(func=cmd_run_alternatives)

    # make_figures
    p_figures = subparsers.add_parser(
        "make_figures",
        help="Generate publication figures"
    )
    p_figures.add_argument(
        "--style", "-s",
        default="publication",
        choices=["publication", "presentation"],
        help="Figure style (default: publication)"
    )
    p_figures.set_defaults(func=cmd_make_figures)

    # run_all
    p_all = subparsers.add_parser(
        "run_all",
        help="Run complete pipeline"
    )
    p_all.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Use demo/synthetic data"
    )
    p_all.add_argument(
        "--model", "-m",
        default="exp_optimal_v1",
        help="Model specification for main estimation"
    )
    p_all.add_argument(
        "--subset", "-s",
        default="all",
        choices=["all", "state", "local"],
        help="Government subset"
    )
    p_all.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness checks"
    )
    p_all.set_defaults(func=cmd_run_all)

    # review_status
    p_review_status = subparsers.add_parser(
        "review_status",
        help="Display current review cycle status"
    )
    p_review_status.set_defaults(func=cmd_review_status)

    # review_new
    p_review_new = subparsers.add_parser(
        "review_new",
        help="Initialize a new review cycle"
    )
    p_review_new.add_argument(
        "--focus", "-f",
        default="par_general",
        choices=["par_general", "methods", "policy", "clarity"],
        help="Review focus area (default: par_general)"
    )
    p_review_new.set_defaults(func=cmd_review_new)

    # review_archive
    p_review_archive = subparsers.add_parser(
        "review_archive",
        help="Archive current review cycle"
    )
    p_review_archive.set_defaults(func=cmd_review_archive)

    # review_verify
    p_review_verify = subparsers.add_parser(
        "review_verify",
        help="Run verification checklist"
    )
    p_review_verify.set_defaults(func=cmd_review_verify)

    # review_report
    p_review_report = subparsers.add_parser(
        "review_report",
        help="Generate summary report of all review cycles"
    )
    p_review_report.set_defaults(func=cmd_review_report)

    # list_models
    p_models = subparsers.add_parser(
        "list_models",
        help="List available SEM model specifications"
    )
    p_models.set_defaults(func=cmd_list_models)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
