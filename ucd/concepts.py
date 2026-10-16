"""Tested SQL building blocks ("concepts") the planner can use as tables: concept.<name> (0.9.0).

A dataset's registry entry may name a folder of concept SQL files (`concepts:` in ucd/datasets.yaml), each one
`CREATE TABLE <derived>.<name> AS <SELECT ...>` over the dataset's own tables (for MIMIC-IV: the public mimic-code
DuckDB concepts - KDIGO stages, SOFA, Sepsis-3, suspicion of infection, ICU stay details, lab panels, vasopressors).

When planner SQL refers to concept.<name>, the gate builds that concept on first use - its dependencies first - inside
the gate, caches it as parquet under .kcd_cache/concepts/<dataset>/ (row-level, git-ignored, local) and exposes it as
the view concept.<name>. Every later query and run on this machine reuses the cache. The planner never sees rows: a
concept is a table like any other, reachable only through the gate's aggregate API and the runner.

  python -m ucd.concepts list  [dataset]            names, groups and columns (schema only)
  python -m ucd.concepts build [dataset] [names]    build now instead of on first use (e.g. once on a server)
  python -m ucd.concepts doc   [dataset]            write ucd/docs/<dataset>_concepts.md for the planner prompt
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".kcd_cache" / "concepts"
SCHEMA = "concept"
USE = re.compile(rf"\b{SCHEMA}\.([A-Za-z_][A-Za-z0-9_]*)", re.I)


def _conf(dataset: str) -> dict | None:
    from ucd import datasets
    return datasets.get(dataset).get("concepts")


@lru_cache(maxsize=8)
def index(dataset: str) -> dict[str, Path]:
    """concept name -> SQL file ({} when the dataset has no concept library)."""
    conf = _conf(dataset)
    if not conf:
        return {}
    d = (ROOT / conf["sql_dir"]).resolve()
    return {p.stem: p for p in sorted(d.rglob("*.sql")) if p.stem != "duckdb"} if d.is_dir() else {}


def body(dataset: str, name: str) -> str:
    """The concept's SELECT, rewritten to the gate's schema names (e.g. mimiciv_hosp.x -> hosp.x, the derived
    schema -> concept.x)."""
    conf = _conf(dataset) or {}
    sql = index(dataset)[name].read_text(encoding="utf-8")
    m = re.search(r"CREATE\s+TABLE\s+\S+\s+AS\s*(.*)\Z", sql, re.I | re.S)
    sql = (m.group(1) if m else sql).strip().rstrip(";")
    for src, dst in (conf.get("schemas") or {}).items():
        sql = re.sub(rf"\b{re.escape(src)}\.", f"{dst}.", sql)
    return sql


def dependencies(dataset: str, name: str) -> list[str]:
    return [d for d in dict.fromkeys(m.lower() for m in USE.findall(body(dataset, name))) if d != name]


def referenced(sql: str) -> list[str]:
    """Concept names that a query uses (outside string literals)."""
    code = re.sub(r"'(?:[^']|'')*'", "''", sql or "")
    return list(dict.fromkeys(m.lower() for m in USE.findall(code)))


def cache_file(dataset: str, name: str) -> Path:
    return CACHE / dataset / f"{name}.parquet"


MAX_NEW_BUILDS = 8                                     # concepts a query may name that are not built yet (R3, E-57)


def fingerprint(dataset: str, name: str, _seen: frozenset = frozenset()) -> str:
    """Hash of the concept's SQL and of its dependencies' SQL: a cached parquet built from other SQL is stale (R2)."""
    deps = [d for d in dependencies(dataset, name) if d in index(dataset) and d not in _seen]
    parts = [body(dataset, name)] + [fingerprint(dataset, d, _seen | {name}) for d in deps]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _fresh(dataset: str, name: str) -> bool:
    f = cache_file(dataset, name)
    sha = f.with_suffix(".sha")
    return f.exists() and sha.exists() and sha.read_text(encoding="utf-8").strip() == fingerprint(dataset, name)


def declare(con, dataset: str, names: list[str]) -> None:
    """Make concept.<name> bindable without computing anything: a fresh cache becomes a view over its parquet, a
    missing one a view over its SQL (DESCRIBE binds it in milliseconds). Used by the SQL pre-check (R3)."""
    idx = index(dataset)
    unknown = [n for n in names if n not in idx]
    if unknown:
        raise ValueError(f"unknown concept(s) {unknown}; available: {', '.join(sorted(idx)) or 'none for ' + dataset}")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    seen: set[str] = set()

    def visit(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        if _fresh(dataset, n):
            con.execute(f"CREATE OR REPLACE VIEW {SCHEMA}.{n} AS SELECT * FROM read_parquet('{cache_file(dataset, n).as_posix()}')")
            return
        for d in dependencies(dataset, n):
            if d in idx:
                visit(d)
        con.execute(f"CREATE OR REPLACE VIEW {SCHEMA}.{n} AS {body(dataset, n)}")

    for n in names:
        visit(n)


def register_cached(con, dataset: str) -> None:
    """Views for every concept already cached on this machine (cheap: no computation)."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    for f in sorted((CACHE / dataset).glob("*.parquet")):
        if f.stem in index(dataset) and _fresh(dataset, f.stem):
            con.execute(f"CREATE OR REPLACE VIEW {SCHEMA}.{f.stem} AS SELECT * FROM read_parquet('{f.as_posix()}')")


def ensure(con, dataset: str, names: list[str], log=None, max_new: int | None = MAX_NEW_BUILDS) -> list[str]:
    """Make concept.<name> available for each name (dependencies first): a cached parquet becomes a view; a missing
    one is computed once, written to the cache, then viewed. Returns the names that were computed now. Raises
    ValueError for an unknown name."""
    idx, built, seen = index(dataset), [], set()
    unknown = [n for n in names if n not in idx]
    if unknown:
        raise ValueError(f"unknown concept(s) {unknown}; available: {', '.join(sorted(idx)) or 'none for ' + dataset}")
    todo = [n for n in dict.fromkeys(names) if not _fresh(dataset, n)]
    if max_new is not None and len(todo) > max_new:          # None: no cap (the build command)
        raise ValueError(f"the query names {len(todo)} concepts that are not built yet (at most {max_new}); use fewer, "
                         "or build them once with: python -m ucd.concepts build")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    def visit(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        f = cache_file(dataset, n)
        if not _fresh(dataset, n):
            for d in dependencies(dataset, n):           # dependencies do not count against the cap (sepsis3: 21)
                if d in idx:
                    visit(d)
            t0 = time.time()
            f.parent.mkdir(parents=True, exist_ok=True)
            part = f.with_name(f"{n}.{os.getpid()}.{time.time_ns()}.part")      # unique per process (R1)
            con.execute(f"COPY ({body(dataset, n)}) TO '{part.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
            try:
                part.replace(f)
            except OSError:                            # another process finished first and holds the file
                part.unlink(missing_ok=True)
                if not f.exists():
                    raise
            f.with_suffix(".sha").write_text(fingerprint(dataset, n), encoding="utf-8")
            built.append(n)
            if log:
                log(f"concept {n} built in {time.time() - t0:.0f}s")
        con.execute(f"CREATE OR REPLACE VIEW {SCHEMA}.{n} AS SELECT * FROM read_parquet('{f.as_posix()}')")

    for n in names:
        visit(n)
    return built


def closure(dataset: str, names: list[str]) -> list[str]:
    """The named concepts and all their dependencies, dependencies first."""
    idx, out = index(dataset), []

    def visit(n: str) -> None:
        if n in out or n not in idx:
            return
        for d in dependencies(dataset, n):
            visit(d)
        out.append(n)
    for n in names:
        visit(n)
    return out


def inline(sql: str, dataset: str) -> str:
    """`sql` with every concept.<name> it uses (and their dependencies) inlined as WITH steps named concept__<name>:
    one query over the dataset's raw tables only, for an environment without the concept library or its cache
    (E-59). The original query is returned unchanged when it uses no concept."""
    from ucd import sqlcte
    names = closure(dataset, referenced(sql))
    if not names:
        return sql
    ren = lambda text: USE.sub(lambda m: f"{SCHEMA}__{m.group(1).lower()}", text)   # noqa: E731
    steps = [(f"{SCHEMA}__{n}", ren(body(dataset, n))) for n in names]
    if re.match(r"(?is)\s*WITH\s+RECURSIVE\b", sql):
        raise ValueError("a WITH RECURSIVE query cannot be inlined")
    parts = sqlcte.split(ren(sql))
    own, final = parts if parts else ([], ren(sql).strip())
    return sqlcte.join(steps + own, final)


def columns(dataset: str) -> dict[str, list[str]]:
    """Schema only: each concept's output columns, found by creating every concept as a view (no computation)."""
    import duckdb

    from ucd.gate import Gate
    g = Gate(dataset)
    con: duckdb.DuckDBPyConnection = g.con
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    out, todo = {}, list(index(dataset))
    for _ in range(len(todo) + 1):                   # views in dependency order (retry until nothing changes)
        left = []
        for n in todo:
            try:
                con.execute(f"CREATE OR REPLACE VIEW {SCHEMA}.{n} AS {body(dataset, n)}")
                out[n] = [r[0] for r in con.execute(f"DESCRIBE {SCHEMA}.{n}").fetchall()]
            except Exception:
                left.append(n)
        if not left or len(left) == len(todo):
            break
        todo = left
    return out


def doc(dataset: str) -> Path:
    """ucd/docs/<dataset>_concepts.md: the concept list the planner sees (group, name, columns; schema only)."""
    idx, cols = index(dataset), columns(dataset)
    lines = [f"# Tested concepts for {dataset} (use them as tables: {SCHEMA}.<name>)", "",
             "Built on first use from public, tested SQL; reuse them instead of re-deriving the same definitions.", ""]
    for group in sorted({p.parent.name for p in idx.values()}):
        lines.append(f"## {group}")
        for n, p in sorted(idx.items()):
            if p.parent.name == group:
                lines.append(f"- `{SCHEMA}.{n}`: {', '.join(cols.get(n, ['(columns unavailable)']))}")
        lines.append("")
    out = ROOT / "ucd" / "docs" / f"{dataset}_concepts.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def prompt_text(dataset: str) -> str:
    """The concept list for the planner prompt ('' when the dataset has none)."""
    f = ROOT / "ucd" / "docs" / f"{dataset}_concepts.md"
    return f.read_text(encoding="utf-8") if f.exists() and index(dataset) else ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ucd.concepts", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("cmd", choices=["list", "build", "doc"])
    p.add_argument("dataset", nargs="?", default=None)
    p.add_argument("names", nargs="*")
    a = p.parse_args(argv)
    from ucd import datasets
    ds = a.dataset or datasets.default()
    if a.cmd == "list":
        for n, c in sorted(columns(ds).items()):
            print(f"{n:<34} {', '.join(c)[:150]}")
    elif a.cmd == "doc":
        print(doc(ds))
    else:
        from ucd.gate import Gate
        con, built, failed = Gate(ds).con, [], []
        for n in a.names or list(index(ds)):            # every concept, no per-query cap (E-54); one failure
            try:                                        # does not stop the rest
                built += ensure(con, ds, [n], log=print, max_new=None)
            except Exception as e:                      # noqa: BLE001
                failed.append(n)
                print(f"concept {n} FAILED: {str(e).splitlines()[0][:300]}")
        print(f"{len(built)} concept(s) built, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}; "
              f"cache: {CACHE / ds}")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
