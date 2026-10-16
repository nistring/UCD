"""Dataset registry (ucd/datasets.yaml): the only place that names a concrete dataset."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).resolve().parent / "datasets.yaml"
LEVELS = ["icu_stay", "admission", "patient"]       # unit_of_analysis values, fine -> coarse


@lru_cache(maxsize=1)
def registry() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}


def names() -> list[str]:
    return list(registry())


def default() -> str:
    return os.environ.get("UCD_DATASET") or names()[0]


def get(name: str) -> dict[str, Any]:
    reg = registry()
    if name not in reg:
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(reg)} (ucd/datasets.yaml)")
    return {"name": name, "label": name, "description": "", "units": {}, "lookups": {}, **reg[name]}


def parquet_dir(name: str) -> Path:
    d = get(name)
    env = d.get("parquet_env") or f"UCD_{name.upper()}_DIR"
    return Path(os.path.expanduser(os.environ.get(env, str(ROOT / ".kcd_cache" / "parquet" / name))))


def source_dir(name: str) -> Path:
    """Raw tables <module>/<table>.csv[.gz]: env UCD_<NAME>_SOURCE, else the first existing `source` path."""
    d = get(name)
    env = os.environ.get(f"UCD_{name.upper()}_SOURCE")
    cands = [env] if env else ([d["source"]] if isinstance(d.get("source"), str) else list(d.get("source") or []))
    for c in cands:
        p = Path(os.path.expanduser(c))
        p = p if p.is_absolute() else (ROOT / p)
        if p.is_dir():
            return p.resolve()
    raise SystemExit(f"raw tables of {name!r} not found (looked in {cands}); set UCD_{name.upper()}_SOURCE")


def missing_tables(name: str) -> tuple[Path, list[str]]:
    """(parquet folder, registry tables (units and lookups) that have no parquet file there). Every table is missing
    when the folder is wrong or the build has not run."""
    d, ds = parquet_dir(name), get(name)
    need = [u["table"] for u in (ds.get("units") or {}).values()] + [v[0] for v in (ds.get("lookups") or {}).values()]
    return d, [t for t in dict.fromkeys(need) if not (d / t.split(".")[0] / f"{t.split('.')[1]}.parquet").is_file()]


def require_data(name: str) -> None:
    """Stop before any LLM call when the dataset's parquet is missing or incomplete (otherwise lookups say 'table
    not available', every design fails with I = 1.0, and the gateway credit is spent on nothing)."""
    d, miss = missing_tables(name)
    if miss:
        env = get(name).get("parquet_env")
        raise SystemExit(f"{name}: no parquet for {', '.join(miss)} in {d}.\n"
                         f"  Build it:  python -m ucd.build_parquet {name}   (raw files from {source_dir_hint(name)})\n"
                         + (f"  or point {env} to an existing parquet folder." if env else ""))


def source_dir_hint(name: str) -> str:
    try:
        return str(source_dir(name))
    except (Exception, SystemExit):                # no raw folder either: source_dir exits with its own message
        return "the 'source' paths in ucd/datasets.yaml"


def describe(name: str) -> str:
    d = get(name)
    return f"{d['label']}: {' '.join(str(d.get('description') or '').split())}"
