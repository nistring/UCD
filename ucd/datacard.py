"""Data Card: what the planner may know about a dataset without seeing rows.

Contents: tables, columns, types, row counts, per-column non-null fraction (aggregates), documented caveats
(docs/<dataset>_notes.md, docs/dataset_facts.yaml), and dictionary lookup (d_items / d_labitems / d_icd_* are
metadata, not patient data). Computed once per dataset and cached.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ucd.gate import CACHE, Gate

DOCS = Path(__file__).resolve().parent / "docs"
FACTS = Path(__file__).resolve().parent / "docs" / "dataset_facts.yaml"


def _parquet_dir(dataset: str) -> Path:
    from ucd.datasets import parquet_dir
    return parquet_dir(dataset).resolve()


def build(dataset: str, refresh: bool = False) -> dict[str, Any]:
    import hashlib
    tag = hashlib.sha1(str(_parquet_dir(dataset)).encode()).hexdigest()[:8]
    cf = CACHE / f"datacard_{dataset}_{tag}.json"   # keyed by source folder
    if cf.exists() and not refresh:
        return json.loads(cf.read_text(encoding="utf-8"))
    g = Gate(dataset)
    schema = g.schema()
    tables: dict[str, Any] = {}
    for module, ts in schema.items():
        for t, cols in ts.items():
            name = f"{module}.{t}"
            exprs = ", ".join(f'count("{c}")' for c, _ in cols)
            # one pass per table: rows and non-null counts together
            row = g.con.execute(f"SELECT count(*), {exprs} FROM {name}").fetchone()
            n = int(row[0])
            tables[name] = {
                "rows": n,
                "columns": [{"name": c, "type": ty,
                             "non_null": round(row[i + 1] / n, 3) if n else None}
                            for i, (c, ty) in enumerate(cols)],
            }
    card = {"dataset": dataset, "tables": tables}
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return card


def render(dataset: str, notes: bool = True) -> str:
    """The Data Card text. notes=False (ablation) leaves out the dataset notes and documented facts: schema only."""
    card = build(dataset)
    facts = (yaml.safe_load(FACTS.read_text(encoding="utf-8")) or {}).get(dataset, {}) if notes else {}
    nf = DOCS / f"{dataset}_notes.md"
    lines = [f"# Data Card: {facts.get('label', dataset)}", ""]
    if notes and nf.exists():
        lines += [nf.read_text(encoding="utf-8").strip(), ""]
    from ucd import datasets
    units = datasets.get(dataset).get("units") or {}
    if units:
        lines += ["## Units of analysis (values of unit_of_analysis, fine -> coarse)"]
        lines += [f"- `{lvl}`: one row per `{u['id']}` (`{u['table']}`)" for lvl, u in units.items()] + [""]
    if facts:
        lines += ["## Documented dataset facts", "```yaml", yaml.safe_dump(facts, allow_unicode=True, sort_keys=False).strip(),
                  "```", ""]
    lines += ["## Tables (rows; column: type [non-null fraction] - description)"]
    for name, t in card["tables"].items():
        lines.append(f"\n### {name}  ({t['rows']:,} rows)")
        for c in t["columns"]:
            nn = "" if c["non_null"] is None else f" [{c['non_null']:.2f}]"
            lines.append(f"- {c['name']}: {c['type']}{nn}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- lookup
def lookups(dataset: str, g: Gate | None = None) -> dict[str, tuple[str, list[str] | None, list[str] | None]]:
    """Dictionary tables of a dataset: from ucd/datasets.yaml, else every table named d_* (all text columns)."""
    from ucd import datasets
    conf = datasets.get(dataset).get("lookups") or {}
    if conf:
        return {k: (v[0], list(v[1]), list(v[2])) for k, v in conf.items()}
    g = g or Gate(dataset, memory="1GB", threads=2)
    return {t: (f"{m}.{t}", None, None) for m, ts in g.tables.items() for t in ts if t.startswith("d_")}


def lookup(dataset: str, term: str, kind: str = "all", limit: int = 40) -> str:
    """Case-insensitive search of dictionary tables. Returns markdown rows (metadata only)."""
    g = Gate(dataset, memory="1GB", threads=2)
    out = []
    known = lookups(dataset, g)
    if kind != "all" and kind not in known:
        return f"unknown kind {kind!r}; available: {', '.join(known)}, all"
    kinds = known if kind == "all" else {kind: known[kind]}
    pat = term.replace("'", "''")
    for k, (table, cols, search) in kinds.items():
        module, t = table.split(".")
        if t not in g.tables.get(module, []):
            out.append(f"## {k}: table {table} not available (no {module}/{t}.parquet in {_parquet_dir(dataset)}; "
                       f"run python -m ucd.build_parquet {dataset} or set the dataset's parquet folder)")
            continue
        if cols is None:                                   # unconfigured dictionary: show all, search text columns
            sch = g.schema()[module][t]
            cols = [c for c, _ in sch]
            search = [c for c, ty in sch if "VARCHAR" in ty.upper()] or cols
        where = " OR ".join(f"CAST({c} AS VARCHAR) ILIKE '%{pat}%'" for c in search)
        df = g.con.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE {where} LIMIT {limit + 1}").df()
        out.append(f"## {k} ({table}): {min(len(df), limit)} match(es)" + (" [truncated]" if len(df) > limit else ""))
        if len(df):
            out.append(df.head(limit).to_markdown(index=False))
    return "\n\n".join(out)


def schema_summary(dataset: str, max_chars: int = 9000) -> str:
    """Compact schema (table, rows, column names) for prompts that need to know what is measurable, e.g. intake."""
    card = build(dataset)
    lines = [f"{name} ({t['rows']:,} rows): " + ", ".join(c["name"] for c in t["columns"])
             for name, t in card["tables"].items()]
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[:max_chars] + "\n..."
