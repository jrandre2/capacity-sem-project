#!/usr/bin/env python3
"""
Capacity-SEM Analysis Pipeline

A command-line interface for running the active standardized survival-analysis
workflow and the legacy SEM pipeline for replication/comparison.

Usage:
    python src/pipeline.py <command> [options]

Commands:
    ingest_data       Load QPR and external data
    standardize_data  Standardize QPR data with fixed denominators (NEW)
    build_panel       Construct analysis panel
    build_features_std Build features from standardized data (NEW)
    aggregate_program_types
                      Aggregate standardized program types (NEW)
    compute_features  Calculate indicators and features
    run_estimation    Fit SEM models
    run_robustness    Run robustness checks
    run_survival      Run time-varying survival analysis
    run_survival_threshold_sensitivity
                      Run survival analysis across all completion thresholds (20%-100%)
    run_kaifa_external_replication
                      Rebuild later Kaifa SEM tables from imported external exports
    run_kaifa_recovered_analysis
                      Run recovered Kaifa notebook analysis on the imported ZIP bundle
    make_figures      Generate publication figures
    capacity_summary  Generate corrected capacity summary table/figure
    run_all           Run active standardized survival pipeline
    run_all_legacy    Run legacy SEM pipeline

    review_status     Display current review cycle status
    review_new        Initialize new review cycle
    review_archive    Archive current review cycle
    review_verify     Run verification checklist
    review_report     Generate summary report of all reviews
    review_diff       Generate manuscript diffs for review cycles
    review_response   Generate a response letter from the active tracker
    review_ingest_docx
                      Import DOCX comments/track changes into the review tracker
    centaur           Run vendored CENTAUR framework tools

Examples:
    python src/pipeline.py run_all
    python src/pipeline.py run_survival
    python src/pipeline.py run_kaifa_external_replication --bundle baseline_sem
    python src/pipeline.py run_kaifa_recovered_analysis
    python src/pipeline.py run_all_legacy --model exp_optimal_v1 --subset state
    python src/pipeline.py review_new --focus par_general
    python src/pipeline.py review_ingest_docx --manuscript kaifa
    python src/pipeline.py centaur analyze_project --path /path/to/project
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


def cmd_standardize_data(args):
    """Standardize QPR data with fixed denominators."""
    from stages import s00b_standardize
    s00b_standardize.standardize_qpr_data()


def cmd_build_panel(args):
    """Construct analysis panel."""
    from stages import s01_link
    s01_link.main()


def cmd_build_features_std(args):
    """Build features from standardized data."""
    from stages import s01b_features
    s01b_features.build_standardized_features()


def cmd_aggregate_program_types(args):
    """Aggregate program type features to grantee-disaster level."""
    from stages import s01c_program_types
    s01c_program_types.run_aggregate_program_types()


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

    s06_alternatives.main(
        methods=methods,
        subset=args.subset,
        capacity_sets=args.capacity_sets
    )


def cmd_make_figures(args):
    """Generate publication figures."""
    from stages import s05_figures
    s05_figures.main(style=args.style)


def cmd_run_kaifa_external_replication(args):
    """Run the provisional external Kaifa SEM reconstruction."""
    from stages import s03c_kaifa_external_replication
    s03c_kaifa_external_replication.main(
        bundle=args.bundle,
        output_dir=args.output_dir,
    )


def cmd_run_kaifa_recovered_analysis(args):
    """Run the recovered Kaifa notebook analysis port."""
    from stages import s03d_kaifa_recovered_analysis

    s03d_kaifa_recovered_analysis.main(output_dir=args.output_dir)


def cmd_capacity_summary(args):
    """Generate corrected capacity summary table/figure."""
    from stages import s07_capacity_summary
    s07_capacity_summary.main()


def cmd_run_all(args):
    """Run active standardized survival-analysis workflow."""
    print("=" * 60)
    print("Running Active Standardized Pipeline")
    print("=" * 60)

    # Stage 0: Ingest
    print("\n" + "=" * 60)
    from stages import s00_ingest
    s00_ingest.main(demo=args.demo)

    # Stage 0b: Standardize
    print("\n" + "=" * 60)
    from stages import s00b_standardize
    s00b_standardize.standardize_qpr_data()

    # Stage 1: Panel
    print("\n" + "=" * 60)
    from stages import s01_link
    s01_link.main()

    # Stage 1b: Standardized features
    print("\n" + "=" * 60)
    from stages import s01b_features
    s01b_features.build_standardized_features()

    # Stage 1c: Program types
    print("\n" + "=" * 60)
    from stages import s01c_program_types
    s01c_program_types.run_aggregate_program_types()

    # Stage 3b: Survival
    print("\n" + "=" * 60)
    from stages import s03b_survival_estimation
    s03b_survival_estimation.main()

    print("\n" + "=" * 60)
    print("✓ Active standardized pipeline complete!")
    print("=" * 60)


def cmd_run_all_legacy(args):
    """Run legacy SEM-oriented end-to-end pipeline."""
    print("=" * 60)
    print("Running Legacy SEM Pipeline")
    print("=" * 60)

    # Stage 0: Ingest
    print("\n" + "=" * 60)
    from stages import s00_ingest
    s00_ingest.main(demo=args.demo)

    # Stage 1: Panel
    print("\n" + "=" * 60)
    from stages import s01_link
    s01_link.main()

    # Stage 2: Legacy features
    print("\n" + "=" * 60)
    from stages import s02_features
    s02_features.main()

    # Stage 3: SEM estimation
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
    print("✓ Legacy SEM pipeline complete!")
    print("=" * 60)


def cmd_review_status(args):
    """Display current review cycle status."""
    from centaur.stages import s07_reviews
    s07_reviews.status(manuscript=args.manuscript)


def cmd_review_new(args):
    """Initialize a new review cycle."""
    from centaur.stages import s07_reviews
    source_type = "actual" if args.actual else "synthetic"
    s07_reviews.new_cycle(
        manuscript=args.manuscript,
        focus=args.focus,
        source_type=source_type,
        journal=args.journal,
        submission_round=args.submission_round,
        decision=args.decision,
        reviewer_ids=args.reviewers,
    )


def cmd_review_archive(args):
    """Archive current review cycle."""
    from centaur.stages import s07_reviews
    s07_reviews.archive(
        manuscript=args.manuscript,
        create_tag=not args.no_tag,
        tag_name=args.tag,
    )


def cmd_review_verify(args):
    """Run verification checklist."""
    from centaur.stages import s07_reviews
    s07_reviews.verify(manuscript=args.manuscript)


def cmd_review_report(args):
    """Generate summary report of all review cycles."""
    from centaur.stages import s07_reviews
    s07_reviews.report()


def cmd_review_diff(args):
    """Generate a manuscript diff for the active or archived review cycle."""
    from centaur.stages import s07_reviews
    s07_reviews.diff(
        manuscript=args.manuscript,
        from_cycle=args.from_cycle,
        to_cycle=args.to_cycle,
        from_commit=args.from_commit,
        format=args.format,
    )


def cmd_review_response(args):
    """Generate a response letter from the active review tracker."""
    from centaur.stages import s07_reviews
    s07_reviews.generate_response_letter(
        manuscript=args.manuscript,
        format=args.format,
        include_diffs=args.include_diffs,
    )


def cmd_review_ingest_docx(args):
    """Import DOCX comments/track changes into the unified review tracker."""
    from centaur.stages import s07_reviews
    s07_reviews.ingest_docx_review(
        manuscript=args.manuscript,
        docx_path=args.input,
        journal=args.journal,
        submission_round=args.submission_round,
        decision=args.decision,
        reviewer_ids=args.reviewers,
        dry_run=args.dry_run,
        force=args.force,
    )


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

    # standardize_data
    p_standardize = subparsers.add_parser(
        "standardize_data",
        help="Standardize QPR data with fixed denominators"
    )
    p_standardize.set_defaults(func=cmd_standardize_data)

    # build_panel
    p_panel = subparsers.add_parser(
        "build_panel",
        help="Construct analysis panel"
    )
    p_panel.set_defaults(func=cmd_build_panel)

    # build_features_std
    p_features_std = subparsers.add_parser(
        "build_features_std",
        help="Build features from standardized data"
    )
    p_features_std.set_defaults(func=cmd_build_features_std)

    # aggregate_program_types
    p_program_types = subparsers.add_parser(
        "aggregate_program_types",
        help="Aggregate program type features to grantee-disaster level"
    )
    p_program_types.set_defaults(func=cmd_aggregate_program_types)

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
        help="Run survival analysis across all completion thresholds (20 to 100 percent)"
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
    p_alternatives.add_argument(
        "--capacity-sets",
        nargs="+",
        default=None,
        help="Capacity sets for survival analysis (names from config, or 'all')"
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

    # run_kaifa_external_replication
    p_kaifa_external = subparsers.add_parser(
        "run_kaifa_external_replication",
        help="Rebuild later Kaifa SEM tables from imported external exports"
    )
    p_kaifa_external.add_argument(
        "--bundle",
        default="baseline_sem",
        choices=["baseline_sem", "baseline_sem_admin_4", "all"],
        help="Imported external SEM bundle to reconstruct (default: baseline_sem)"
    )
    p_kaifa_external.add_argument(
        "--output-dir",
        help="Optional output directory for generated external reconstruction files"
    )
    p_kaifa_external.set_defaults(func=cmd_run_kaifa_external_replication)

    # run_kaifa_recovered_analysis
    p_kaifa_recovered = subparsers.add_parser(
        "run_kaifa_recovered_analysis",
        help="Run recovered Kaifa notebook analysis on the imported ZIP bundle"
    )
    p_kaifa_recovered.add_argument(
        "--output-dir",
        help="Optional output directory for generated recovered-analysis files"
    )
    p_kaifa_recovered.set_defaults(func=cmd_run_kaifa_recovered_analysis)

    # capacity_summary
    p_capacity_summary = subparsers.add_parser(
        "capacity_summary",
        help="Generate corrected capacity summary table/figure"
    )
    p_capacity_summary.set_defaults(func=cmd_capacity_summary)

    # run_all
    p_all = subparsers.add_parser(
        "run_all",
        help="Run active standardized survival pipeline"
    )
    p_all.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Use demo/synthetic data"
    )
    p_all.set_defaults(func=cmd_run_all)

    # run_all_legacy
    p_all_legacy = subparsers.add_parser(
        "run_all_legacy",
        help="Run legacy SEM pipeline"
    )
    p_all_legacy.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Use demo/synthetic data"
    )
    p_all_legacy.add_argument(
        "--model", "-m",
        default="exp_optimal_v1",
        help="Model specification for main estimation"
    )
    p_all_legacy.add_argument(
        "--subset", "-s",
        default="all",
        choices=["all", "state", "local"],
        help="Government subset"
    )
    p_all_legacy.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip robustness checks"
    )
    p_all_legacy.set_defaults(func=cmd_run_all_legacy)

    # review_status
    p_review_status = subparsers.add_parser(
        "review_status",
        help="Display current review cycle status"
    )
    p_review_status.add_argument(
        "--manuscript", "-m",
        default="velocity",
        help="Manuscript to check (default: velocity)"
    )
    p_review_status.set_defaults(func=cmd_review_status)

    # review_new
    p_review_new = subparsers.add_parser(
        "review_new",
        help="Initialize a new review cycle"
    )
    p_review_new.add_argument(
        "--manuscript", "-m",
        default="velocity",
        help="Manuscript for review (default: velocity)"
    )
    p_review_new.add_argument(
        "--focus", "-f",
        default="par_general",
        choices=["economics", "engineering", "social_sciences", "general", "par_general", "methods", "policy", "clarity"],
        help="Review focus area (default: par_general)"
    )
    p_review_new.add_argument(
        "--actual",
        action="store_true",
        help="Create an actual review instead of a synthetic review"
    )
    p_review_new.add_argument(
        "--journal", "-j",
        help="Journal name for actual reviews"
    )
    p_review_new.add_argument(
        "--round",
        dest="submission_round",
        help="Submission round, e.g. initial or R&R1"
    )
    p_review_new.add_argument(
        "--decision",
        choices=["major_revision", "minor_revision", "reject", "accept"],
        help="Editor decision for actual reviews"
    )
    p_review_new.add_argument(
        "--reviewers",
        nargs="+",
        help="Reviewer identifiers, e.g. R1 R2"
    )
    p_review_new.set_defaults(func=cmd_review_new)

    # review_archive
    p_review_archive = subparsers.add_parser(
        "review_archive",
        help="Archive current review cycle"
    )
    p_review_archive.add_argument(
        "--manuscript", "-m",
        default="velocity",
        help="Manuscript to archive (default: velocity)"
    )
    p_review_archive.add_argument(
        "--no-tag",
        action="store_true",
        help="Skip git tag creation when archiving"
    )
    p_review_archive.add_argument(
        "--tag",
        help="Custom git tag name"
    )
    p_review_archive.set_defaults(func=cmd_review_archive)

    # review_verify
    p_review_verify = subparsers.add_parser(
        "review_verify",
        help="Run verification checklist"
    )
    p_review_verify.add_argument(
        "--manuscript", "-m",
        default="velocity",
        help="Manuscript to verify (default: velocity)"
    )
    p_review_verify.set_defaults(func=cmd_review_verify)

    # review_report
    p_review_report = subparsers.add_parser(
        "review_report",
        help="Generate summary report of all review cycles"
    )
    p_review_report.set_defaults(func=cmd_review_report)

    # review_diff
    p_review_diff = subparsers.add_parser(
        "review_diff",
        help="Generate manuscript diffs for review cycles"
    )
    p_review_diff.add_argument(
        "--manuscript", "-m",
        default="velocity",
        help="Manuscript to diff (default: velocity)"
    )
    p_review_diff.add_argument(
        "--from-cycle",
        type=int,
        help="Archived review cycle number to diff from"
    )
    p_review_diff.add_argument(
        "--to-cycle",
        type=int,
        help="Reserved for future cycle-to-cycle comparisons"
    )
    p_review_diff.add_argument(
        "--commit",
        dest="from_commit",
        help="Git commit SHA to compare from"
    )
    p_review_diff.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "unified"],
        help="Diff output format"
    )
    p_review_diff.set_defaults(func=cmd_review_diff)

    # review_response
    p_review_response = subparsers.add_parser(
        "review_response",
        help="Generate a response letter from the active tracker"
    )
    p_review_response.add_argument(
        "--manuscript", "-m",
        default="velocity",
        help="Manuscript to use (default: velocity)"
    )
    p_review_response.add_argument(
        "--format",
        default="markdown",
        choices=["markdown"],
        help="Response output format"
    )
    p_review_response.add_argument(
        "--include-diffs",
        action="store_true",
        help="Reserved for future inline diff embedding"
    )
    p_review_response.set_defaults(func=cmd_review_response)

    # review_ingest_docx
    p_review_ingest_docx = subparsers.add_parser(
        "review_ingest_docx",
        help="Import DOCX comments/track changes into the review tracker"
    )
    p_review_ingest_docx.add_argument(
        "--manuscript", "-m",
        default="velocity",
        help="Manuscript key (default: velocity)"
    )
    p_review_ingest_docx.add_argument(
        "--input",
        help="Path to the source DOCX. If omitted, uses the manuscript's configured source_docx"
    )
    p_review_ingest_docx.add_argument(
        "--journal", "-j",
        help="Journal name for the imported review"
    )
    p_review_ingest_docx.add_argument(
        "--round",
        dest="submission_round",
        help="Submission round, e.g. initial or R&R1"
    )
    p_review_ingest_docx.add_argument(
        "--decision",
        choices=["major_revision", "minor_revision", "reject", "accept"],
        help="Editor decision"
    )
    p_review_ingest_docx.add_argument(
        "--reviewers",
        nargs="+",
        help="Explicit reviewer identifiers, e.g. R1 R2"
    )
    p_review_ingest_docx.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the imported tracker without writing it"
    )
    p_review_ingest_docx.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing active tracker"
    )
    p_review_ingest_docx.set_defaults(func=cmd_review_ingest_docx)

    # list_models
    p_models = subparsers.add_parser(
        "list_models",
        help="List available SEM model specifications"
    )
    p_models.set_defaults(func=cmd_list_models)

    from centaur.cli import register_centaur_parser
    register_centaur_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
