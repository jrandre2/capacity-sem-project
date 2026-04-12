# Review Index

**Last Updated**: 2026-04-09

## Active Coverage

| Manuscript | Directory | Status |
|------------|-----------|--------|
| Velocity manuscript | `manuscript_velocity/` | Active rewrite |
| Kaifa archive | `manuscript_kaifa_archive/` | DOCX review workspace |

Current CLI summary:

```bash
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_status --manuscript kaifa
python src/pipeline.py review_report
```

As of 2026-04-09:
- active reviews: 2
- archived reviews: 0
- velocity tracker verification: 4 / 13 items complete
- kaifa tracker verification: 0 / 6 items complete

## Commands

```bash
python src/pipeline.py review_new --manuscript velocity --focus par_general
python src/pipeline.py review_status --manuscript velocity
python src/pipeline.py review_diff --manuscript velocity
python src/pipeline.py review_response --manuscript velocity
python src/pipeline.py review_verify --manuscript velocity
python src/pipeline.py review_archive --manuscript velocity
python src/pipeline.py review_ingest_docx --manuscript kaifa
python src/pipeline.py review_report
```

## Working Rule

Review documentation should reflect the corrected post-bug state of the project. If a review prompt or comment still assumes strong positive velocity effects are the main result, it should be updated before reuse.
