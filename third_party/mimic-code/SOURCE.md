# mimic-code DuckDB concepts (vendored)

- Repository: https://github.com/MIT-LCP/mimic-code
- Commit: `303d26c623dcc9c49cc0f204468d4acc2f063797` (commit date 2026-09-01)
- Path: `mimic-iv/concepts_duckdb/`, stored verbatim in `concepts_duckdb/` (DuckDB dialect, generated upstream with sqlglot)
- License: MIT, see `LICENSE` in this folder.

Used by:
- `ucd/concepts.py`: the planner can use these as `concept.<name>` tables (KDIGO stages, SOFA, Sepsis-3, ...),
  built inside the gate on first use (`ucd/datasets.yaml`, `concepts.sql_dir`).
- `eval/mimiccode/build_refs.py`: the evaluator's reference cohorts, in the order given by `concepts_duckdb/duckdb.sql`.
