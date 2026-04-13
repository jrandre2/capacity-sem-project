"""
CLI helpers for the vendored CENTAUR framework.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .agents import analyze_project, generate_migration_plan, map_project
from .analysis.factory import get_engine_info, list_engines
from .config import DEFAULT_MANUSCRIPT, MANUSCRIPTS, ensure_directories


def register_centaur_parser(subparsers) -> None:
    """Register the `centaur` command group on the host CLI."""
    review_default = "quarto" if "quarto" in MANUSCRIPTS else DEFAULT_MANUSCRIPT

    p_centaur = subparsers.add_parser(
        "centaur",
        help="Run vendored CENTAUR framework tools",
    )
    centaur_subparsers = p_centaur.add_subparsers(
        dest="centaur_command",
        title="CENTAUR Tools",
        metavar="<tool>",
    )
    p_centaur.set_defaults(func=run_centaur, centaur_parser=p_centaur)

    p_analyze = centaur_subparsers.add_parser("analyze_project", help="Analyze an external project structure")
    p_analyze.add_argument("--path", required=True, help="Path to the project to analyze")
    p_analyze.add_argument("--output", help="Optional JSON output path")
    p_analyze.set_defaults(func=run_centaur)

    p_map = centaur_subparsers.add_parser("map_project", help="Map a project structure to the CENTAUR template")
    p_map.add_argument("--path", required=True, help="Path to the project to map")
    p_map.add_argument("--output", help="Optional JSON output path")
    p_map.set_defaults(func=run_centaur)

    p_plan = centaur_subparsers.add_parser("plan_migration", help="Generate a migration plan without executing it")
    p_plan.add_argument("--path", required=True, help="Source project path")
    p_plan.add_argument("--target", required=True, help="Target project path")
    p_plan.add_argument("--output", help="Optional markdown output path")
    p_plan.set_defaults(func=run_centaur)

    p_engines = centaur_subparsers.add_parser("engines", help="Inspect vendored CENTAUR analysis engines")
    p_engines.add_argument("action", choices=["list", "check"], help="List registered engines or validate installations")
    p_engines.set_defaults(func=run_centaur)

    p_ingest = centaur_subparsers.add_parser("ingest_data", help="Run CENTAUR Stage 00 ingestion")
    p_ingest.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    p_ingest.set_defaults(func=run_centaur)

    p_link = centaur_subparsers.add_parser("link_records", help="Run CENTAUR Stage 01 record linkage")
    p_link.add_argument("--sources", nargs="+", default=None, help="Additional parquet sources in data_work/centaur")
    p_link.add_argument("--keys", nargs="+", default=None, help="Key columns for exact matching")
    p_link.set_defaults(func=run_centaur)

    p_panel = centaur_subparsers.add_parser("build_panel", help="Run CENTAUR Stage 02 panel construction")
    p_panel.add_argument("--balance", action="store_true", help="Create a balanced panel")
    p_panel.add_argument("--fill-method", default="drop", choices=["drop", "ffill", "zero"], help="Method for balancing")
    p_panel.add_argument("--no-event-study", action="store_true", help="Skip event-study variables")
    p_panel.set_defaults(func=run_centaur)

    p_est = centaur_subparsers.add_parser("run_estimation", help="Run CENTAUR Stage 03 estimation")
    p_est.add_argument("--specification", "-s", default="baseline", help="Specification to run")
    p_est.add_argument("--sample", default="full", help="Sample restriction")
    p_est.add_argument("--all", action="store_true", help="Run all specifications")
    p_est.add_argument("--no-cache", action="store_true", help="Disable caching")
    p_est.add_argument("--sequential", action="store_true", help="Disable parallel execution")
    p_est.add_argument("--workers", "-w", type=int, default=None, help="Number of workers")
    p_est.add_argument("--engine", "-e", default=None, help="Analysis engine override")
    p_est.set_defaults(func=run_centaur)

    p_rob = centaur_subparsers.add_parser("estimate_robustness", help="Run CENTAUR Stage 04 robustness checks")
    p_rob.add_argument("--no-placebos", action="store_true", help="Skip placebo tests")
    p_rob.add_argument("--no-samples", action="store_true", help="Skip sample restrictions")
    p_rob.add_argument("--no-specs", action="store_true", help="Skip alternative specifications")
    p_rob.add_argument("--no-cache", action="store_true", help="Disable caching")
    p_rob.add_argument("--sequential", action="store_true", help="Disable parallel execution")
    p_rob.add_argument("--workers", "-w", type=int, default=None, help="Number of workers")
    p_rob.set_defaults(func=run_centaur)

    p_fig = centaur_subparsers.add_parser("make_figures", help="Run CENTAUR Stage 05 figure generation")
    p_fig.add_argument("--figures", nargs="+", default=None, help="Specific figures to generate")
    p_fig.add_argument("--no-cache", action="store_true", help="Disable caching")
    p_fig.add_argument("--sequential", action="store_true", help="Disable parallel execution")
    p_fig.add_argument("--workers", "-w", type=int, default=None, help="Number of workers")
    p_fig.set_defaults(func=run_centaur)

    p_validate = centaur_subparsers.add_parser("validate_submission", help="Run CENTAUR Stage 06 manuscript validation")
    p_validate.add_argument("--journal", "-j", default="jeem", help="Target journal")
    p_validate.add_argument("--report", action="store_true", help="Write markdown validation report")
    p_validate.set_defaults(func=run_centaur)

    p_review_status = centaur_subparsers.add_parser("review_status", help="Show CENTAUR review status")
    p_review_status.add_argument("--manuscript", "-m", default=review_default, help="Manuscript key")
    p_review_status.set_defaults(func=run_centaur)

    p_review_new = centaur_subparsers.add_parser("review_new", help="Start a CENTAUR review cycle")
    p_review_new.add_argument("--manuscript", "-m", default=review_default, help="Manuscript key")
    p_review_new.add_argument(
        "--focus",
        "-f",
        default="general",
        choices=["economics", "engineering", "social_sciences", "general", "par_general", "methods", "policy", "clarity"],
        help="Focus area for synthetic reviews",
    )
    p_review_new.add_argument("--actual", action="store_true", help="Create an actual review instead of a synthetic review")
    p_review_new.add_argument("--journal", "-j", help="Journal name for actual reviews")
    p_review_new.add_argument("--round", dest="submission_round", help="Submission round, e.g. R&R1")
    p_review_new.add_argument("--decision", choices=["major_revision", "minor_revision", "reject", "accept"], help="Editor decision")
    p_review_new.add_argument("--reviewers", nargs="+", help="Reviewer identifiers, e.g. R1 R2")
    p_review_new.set_defaults(func=run_centaur)

    p_review_archive = centaur_subparsers.add_parser("review_archive", help="Archive the current CENTAUR review cycle")
    p_review_archive.add_argument("--manuscript", "-m", default=review_default, help="Manuscript key")
    p_review_archive.add_argument("--no-tag", action="store_true", help="Skip git tag creation")
    p_review_archive.add_argument("--tag", help="Custom git tag name")
    p_review_archive.set_defaults(func=run_centaur)

    p_review_verify = centaur_subparsers.add_parser("review_verify", help="Run CENTAUR review verification")
    p_review_verify.add_argument("--manuscript", "-m", default=review_default, help="Manuscript key")
    p_review_verify.set_defaults(func=run_centaur)

    centaur_subparsers.add_parser("review_report", help="Generate a summary of CENTAUR reviews").set_defaults(func=run_centaur)

    p_review_diff = centaur_subparsers.add_parser("review_diff", help="Generate a manuscript diff for the active or archived review cycle")
    p_review_diff.add_argument("--manuscript", "-m", default=review_default, help="Manuscript key")
    p_review_diff.add_argument("--from-cycle", type=int, help="Archived review cycle number to diff from")
    p_review_diff.add_argument("--to-cycle", type=int, help="Reserved for future cycle-to-cycle comparisons")
    p_review_diff.add_argument("--commit", dest="from_commit", help="Git commit SHA to compare from")
    p_review_diff.add_argument("--format", default="markdown", choices=["markdown", "unified"], help="Diff output format")
    p_review_diff.set_defaults(func=run_centaur)

    p_review_response = centaur_subparsers.add_parser("review_response", help="Generate a response letter from the current tracker")
    p_review_response.add_argument("--manuscript", "-m", default=review_default, help="Manuscript key")
    p_review_response.add_argument("--format", default="markdown", choices=["markdown"], help="Response output format")
    p_review_response.add_argument("--include-diffs", action="store_true", help="Reserved for future inline diff embedding")
    p_review_response.set_defaults(func=run_centaur)

    p_review_ingest_docx = centaur_subparsers.add_parser("review_ingest_docx", help="Import DOCX comments/track changes into a review tracker")
    p_review_ingest_docx.add_argument("--manuscript", "-m", default=review_default, help="Manuscript key")
    p_review_ingest_docx.add_argument("--input", help="Path to the source DOCX. If omitted, uses the manuscript's configured source_docx")
    p_review_ingest_docx.add_argument("--journal", "-j", help="Journal name for the imported review")
    p_review_ingest_docx.add_argument("--round", dest="submission_round", help="Submission round, e.g. R&R1")
    p_review_ingest_docx.add_argument("--decision", choices=["major_revision", "minor_revision", "reject", "accept"], help="Editor decision")
    p_review_ingest_docx.add_argument("--reviewers", nargs="+", help="Explicit reviewer identifiers, e.g. R1 R2")
    p_review_ingest_docx.add_argument("--dry-run", action="store_true", help="Print the imported tracker without writing it")
    p_review_ingest_docx.add_argument("--force", action="store_true", help="Overwrite an existing active tracker")
    p_review_ingest_docx.set_defaults(func=run_centaur)

    centaur_subparsers.add_parser("journal_list", help="List available CENTAUR journal profiles").set_defaults(func=run_centaur)

    p_journal_validate = centaur_subparsers.add_parser("journal_validate", help="Validate a journal configuration")
    p_journal_validate.add_argument("--config", required=True, help="Journal config name without .yml")
    p_journal_validate.set_defaults(func=run_centaur)

    p_journal_compare = centaur_subparsers.add_parser("journal_compare", help="Compare the manuscript scaffold to a journal profile")
    p_journal_compare.add_argument("--journal", "-j", required=True, help="Journal config name without .yml")
    p_journal_compare.add_argument("--manuscript", help="Override manuscript directory")
    p_journal_compare.set_defaults(func=run_centaur)

    p_journal_parse = centaur_subparsers.add_parser("journal_parse", help="Parse journal guidelines into a YAML config")
    p_journal_parse.add_argument("--input", help="Path to local guidelines text")
    p_journal_parse.add_argument("--url", help="URL to fetch and parse")
    p_journal_parse.add_argument("--output", required=True, help="Output config filename")
    p_journal_parse.add_argument("--journal", help="Journal display name")
    p_journal_parse.add_argument("--template", default="template_comprehensive", help="Template name")
    p_journal_parse.add_argument("--save-raw", action="store_true", help="Save fetched raw guidelines")
    p_journal_parse.add_argument("--raw-dir", help="Override raw guidelines directory")
    p_journal_parse.add_argument("--overwrite", action="store_true", help="Overwrite existing raw files")
    p_journal_parse.set_defaults(func=run_centaur)

    p_journal_fetch = centaur_subparsers.add_parser("journal_fetch", help="Fetch journal guidelines from a URL")
    p_journal_fetch.add_argument("--url", required=True, help="Guidelines URL")
    p_journal_fetch.add_argument("--output", help="Output filename")
    p_journal_fetch.add_argument("--journal", help="Journal display name")
    p_journal_fetch.add_argument("--raw-dir", help="Override raw guidelines directory")
    p_journal_fetch.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p_journal_fetch.add_argument("--text", action="store_true", help="Save extracted plain text")
    p_journal_fetch.set_defaults(func=run_centaur)

    p_draft_results = centaur_subparsers.add_parser("draft_results", help="Draft a results section from a diagnostic table")
    p_draft_results.add_argument("--table", required=True, help="Diagnostic table name without .csv")
    p_draft_results.add_argument("--section", default="main", help="Section label for the draft file")
    p_draft_results.add_argument("--manuscript", "-m", default=DEFAULT_MANUSCRIPT, help="Manuscript key")
    p_draft_results.add_argument("--dry-run", action="store_true", help="Print the prompt instead of calling the provider")
    p_draft_results.add_argument("--provider", help="Provider override: anthropic or openai")
    p_draft_results.set_defaults(func=run_centaur)

    p_draft_captions = centaur_subparsers.add_parser("draft_captions", help="Draft figure captions")
    p_draft_captions.add_argument("--figure", required=True, help="Glob pattern for figure files")
    p_draft_captions.add_argument("--manuscript", "-m", default=DEFAULT_MANUSCRIPT, help="Manuscript key")
    p_draft_captions.add_argument("--dry-run", action="store_true", help="Print the prompt instead of calling the provider")
    p_draft_captions.add_argument("--provider", help="Provider override: anthropic or openai")
    p_draft_captions.set_defaults(func=run_centaur)

    p_draft_abstract = centaur_subparsers.add_parser("draft_abstract", help="Draft an abstract from the manuscript scaffold")
    p_draft_abstract.add_argument("--manuscript", "-m", default=DEFAULT_MANUSCRIPT, help="Manuscript key")
    p_draft_abstract.add_argument("--max-words", type=int, default=200, help="Target abstract length")
    p_draft_abstract.add_argument("--dry-run", action="store_true", help="Print the prompt instead of calling the provider")
    p_draft_abstract.add_argument("--provider", help="Provider override: anthropic or openai")
    p_draft_abstract.set_defaults(func=run_centaur)

    p_gui = centaur_subparsers.add_parser("gui", help="Launch the vendored CENTAUR FastAPI dashboard")
    p_gui.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    p_gui.add_argument("--port", "-p", type=int, default=8000, help="Port to bind to")
    p_gui.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    p_gui.set_defaults(func=run_centaur)

    centaur_subparsers.add_parser("list_stages", help="List available vendored CENTAUR stages").set_defaults(func=run_centaur)

    p_run_stage = centaur_subparsers.add_parser("run_stage", help="Run a vendored CENTAUR stage by module name")
    p_run_stage.add_argument("stage_name", help="Stage module name, e.g. s03_estimation")
    p_run_stage.set_defaults(func=run_centaur)


def run_centaur(args) -> None:
    """Execute a vendored CENTAUR CLI command."""
    ensure_directories()

    if args.centaur_command is None:
        args.centaur_parser.print_help()
        return

    if args.centaur_command == "analyze_project":
        analysis = analyze_project(args.path)
        print(analysis.summary())
        if args.output:
            _write_text(Path(args.output), analysis.to_json())
            print(f"\nJSON saved to: {args.output}")
        return

    if args.centaur_command == "map_project":
        analysis = analyze_project(args.path)
        mapping = map_project(analysis)
        print(mapping.summary())
        if args.output:
            _write_text(Path(args.output), mapping.to_json())
            print(f"\nJSON saved to: {args.output}")
        return

    if args.centaur_command == "plan_migration":
        analysis = analyze_project(args.path)
        mapping = map_project(analysis)
        plan = generate_migration_plan(analysis, mapping, args.target)
        print(plan.to_markdown())
        if args.output:
            _write_text(Path(args.output), plan.to_markdown())
            print(f"\nPlan saved to: {args.output}")
        return

    if args.centaur_command == "engines":
        _handle_engines(args.action)
        return

    if args.centaur_command == "ingest_data":
        from .stages import s00_ingest

        s00_ingest.main(use_demo=args.demo)
        return

    if args.centaur_command == "link_records":
        from .stages import s01_link

        s01_link.main(additional_sources=args.sources, key_columns=args.keys)
        return

    if args.centaur_command == "build_panel":
        from .stages import s02_panel

        s02_panel.main(
            balance=args.balance,
            fill_method=args.fill_method,
            create_event_study=not args.no_event_study,
        )
        return

    if args.centaur_command == "run_estimation":
        from .stages import s03_estimation

        s03_estimation.main(
            specification=args.specification,
            sample=args.sample,
            run_all=args.all,
            use_cache=not args.no_cache,
            parallel=not args.sequential,
            n_workers=args.workers,
            engine=args.engine,
        )
        return

    if args.centaur_command == "estimate_robustness":
        from .stages import s04_robustness

        s04_robustness.main(
            run_placebos=not args.no_placebos,
            run_samples=not args.no_samples,
            run_specs=not args.no_specs,
            use_cache=not args.no_cache,
            parallel=not args.sequential,
            n_workers=args.workers,
        )
        return

    if args.centaur_command == "make_figures":
        from .stages import s05_figures

        s05_figures.main(
            figures=args.figures,
            use_cache=not args.no_cache,
            parallel=not args.sequential,
            n_workers=args.workers,
        )
        return

    if args.centaur_command == "validate_submission":
        from .stages import s06_manuscript

        s06_manuscript.validate(journal=args.journal, report=args.report)
        return

    if args.centaur_command == "review_status":
        from .stages import s07_reviews

        s07_reviews.status(manuscript=args.manuscript)
        return

    if args.centaur_command == "review_new":
        from .stages import s07_reviews

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
        return

    if args.centaur_command == "review_archive":
        from .stages import s07_reviews

        s07_reviews.archive(
            manuscript=args.manuscript,
            create_tag=not args.no_tag,
            tag_name=args.tag,
        )
        return

    if args.centaur_command == "review_verify":
        from .stages import s07_reviews

        s07_reviews.verify(manuscript=args.manuscript)
        return

    if args.centaur_command == "review_report":
        from .stages import s07_reviews

        s07_reviews.report()
        return

    if args.centaur_command == "review_diff":
        from .stages import s07_reviews

        s07_reviews.diff(
            manuscript=args.manuscript,
            from_cycle=args.from_cycle,
            to_cycle=args.to_cycle,
            from_commit=args.from_commit,
            format=args.format,
        )
        return

    if args.centaur_command == "review_response":
        from .stages import s07_reviews

        s07_reviews.generate_response_letter(
            manuscript=args.manuscript,
            format=args.format,
            include_diffs=args.include_diffs,
        )
        return

    if args.centaur_command == "review_ingest_docx":
        from .stages import s07_reviews

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
        return

    if args.centaur_command == "journal_list":
        from .stages import s08_journal_parser

        s08_journal_parser.list_configs()
        return

    if args.centaur_command == "journal_validate":
        from .stages import s08_journal_parser

        s08_journal_parser.validate_config(args.config)
        return

    if args.centaur_command == "journal_compare":
        from .stages import s08_journal_parser

        s08_journal_parser.compare_manuscript(args.journal, args.manuscript)
        return

    if args.centaur_command == "journal_parse":
        from .stages import s08_journal_parser

        s08_journal_parser.parse_guidelines(
            input_file=args.input,
            output_file=args.output,
            journal_name=args.journal,
            url=args.url,
            template_name=args.template,
            save_raw=args.save_raw,
            raw_dir=args.raw_dir,
            overwrite=args.overwrite,
        )
        return

    if args.centaur_command == "journal_fetch":
        from .stages import s08_journal_parser

        s08_journal_parser.fetch_guidelines_cli(
            url=args.url,
            output=args.output,
            journal_name=args.journal,
            raw_dir=args.raw_dir,
            overwrite=args.overwrite,
            save_text=args.text,
        )
        return

    if args.centaur_command == "draft_results":
        from .stages import s09_writing

        s09_writing.draft_results(
            table_name=args.table,
            section=args.section,
            manuscript=args.manuscript,
            dry_run=args.dry_run,
            provider=args.provider,
        )
        return

    if args.centaur_command == "draft_captions":
        from .stages import s09_writing

        s09_writing.draft_captions(
            figure_pattern=args.figure,
            manuscript=args.manuscript,
            dry_run=args.dry_run,
            provider=args.provider,
        )
        return

    if args.centaur_command == "draft_abstract":
        from .stages import s09_writing

        s09_writing.draft_abstract(
            manuscript=args.manuscript,
            max_words=args.max_words,
            dry_run=args.dry_run,
            provider=args.provider,
        )
        return

    if args.centaur_command == "gui":
        try:
            from .gui.app import run_server
        except ImportError as exc:
            raise RuntimeError(
                "FastAPI GUI dependencies are missing. Install FastAPI and uvicorn to use `centaur gui`."
            ) from exc

        run_server(host=args.host, port=args.port, reload=not args.no_reload)
        return

    if args.centaur_command == "list_stages":
        _list_stages()
        return

    if args.centaur_command == "run_stage":
        _run_stage(args.stage_name)
        return

    raise ValueError(f"Unknown CENTAUR command: {args.centaur_command}")


def _handle_engines(action: str) -> None:
    if action == "list":
        print("=" * 50)
        print("Vendored CENTAUR Analysis Engines")
        print("=" * 50)
        for name, available in list_engines().items():
            status = "available" if available else "not found"
            print(f"  {name}: {status}")
        print()
        print("Use `python src/pipeline.py centaur engines check` for details.")
        return

    print("=" * 50)
    print("Vendored CENTAUR Engine Validation")
    print("=" * 50)
    for name in list_engines():
        info = get_engine_info(name)
        status = "OK" if info["available"] else "FAILED"
        print(f"\n{name}:")
        print(f"  Status:  {status}")
        print(f"  Version: {info['version']}")
        print(f"  Details: {info['message']}")


def _list_stages() -> None:
    from .gui.services.pipeline_service import get_pipeline_service

    stages = get_pipeline_service().discover_stages()
    print("Vendored CENTAUR Stages")
    print("=" * 50)
    for stage in stages:
        version = f" ({stage.version})" if stage.version else ""
        print(f"{stage.name}{version}: {stage.description}")


def _run_stage(stage_name: str) -> None:
    if stage_name == "s00_ingest":
        from .stages import s00_ingest

        s00_ingest.main()
        return
    if stage_name == "s01_link":
        from .stages import s01_link

        s01_link.main()
        return
    if stage_name == "s02_panel":
        from .stages import s02_panel

        s02_panel.main()
        return
    if stage_name == "s03_estimation":
        from .stages import s03_estimation

        s03_estimation.main()
        return
    if stage_name == "s04_robustness":
        from .stages import s04_robustness

        s04_robustness.main()
        return
    if stage_name == "s05_figures":
        from .stages import s05_figures

        s05_figures.main()
        return
    if stage_name == "s06_manuscript":
        from .stages import s06_manuscript

        s06_manuscript.validate()
        return
    if stage_name == "s07_reviews":
        from .stages import s07_reviews

        s07_reviews.status()
        return
    if stage_name == "s08_journal_parser":
        from .stages import s08_journal_parser

        s08_journal_parser.list_configs()
        return
    if stage_name == "s09_writing":
        raise ValueError("s09_writing requires a specific drafting command. Use centaur draft_results, draft_captions, or draft_abstract.")

    raise ValueError(f"Unknown stage: {stage_name}")


def _write_text(path: Path, content: str) -> None:
    """Write text output, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
