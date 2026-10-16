"""The only path from the planner to patient data.

* ``aggregate(source_sql, group_by, metrics)`` - the planner's only query: the gate builds
                       SELECT group, count(*) AS n, metrics FROM (source) GROUP BY group HAVING n >= K_MIN itself;
                       identifier / datetime columns are rejected; whitelisted metrics; at most MAX_ROWS rows;
                       per-run budget; every query is logged and cached by SQL hash (results are aggregates only).
* ``materialize(sql)`` - runs the design's analysis SQL into a DataFrame that stays inside the runner.

Views: <module>.<table> for every <parquet dir>/<module>/<table>.parquet of the dataset (ucd/datasets.yaml;
       MIMIC-IV -> hosp.*, icu.*). Parquet comes from `python -m ucd.build_parquet <dataset>`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / ".kcd_cache" / "parquet"
CACHE = ROOT / ".kcd_cache" / "ucd_cache"
GLOBAL_LOG = ROOT / ".kcd_cache" / "ucd_gate.log"

K_MIN = 10          # small-cell suppression threshold
MAX_ROWS = 50
BUDGET = 60         # explore queries per run per dataset

from ucd import datasets as _ds

DATASETS = _ds.names()     # registered datasets (ucd/datasets.yaml)

ID_COLUMNS = {"subject_id", "hadm_id", "stay_id", "unit_id", "subject_key", "transfer_id", "emar_id",
              "pharmacy_id", "poe_id", "orderid", "linkorderid", "caregiver_id", "order_provider_id",
              "note_id", "specimen_id", "labevent_id", "micro_specimen_id", "provider_id", "icustay_id"}
try:                       # every registry unit id is an identifier too (e.g. OMOP person_id, visit_occurrence_id)
    ID_COLUMNS |= {str(u["id"]).lower() for d in DATASETS for u in (_ds.get(d).get("units") or {}).values()}
except Exception:
    pass
ID_VALUED_MIN = 20         # a group column with more distinct values than this is checked for identifier values


def is_id_name(name: str) -> bool:
    """A column name that identifies a patient, an encounter or a record: the known ids, every registry unit id, and
    any *_id column except a *concept_id code (OMOP measurement_id, visit_detail_id ...; E-68)."""
    n = name.lower()
    return n in ID_COLUMNS or (n.endswith("_id") and not n.endswith("concept_id"))
FORBIDDEN = re.compile(
    r"\b(copy|attach|detach|install|load|pragma|create|insert|update|delete|drop|alter|export|import|set|call|"
    r"read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|parquet_scan|glob|httpfs|getenv|system)\b"
    r"|https?://|[a-z]:[\\/]|'/|\"/",
    re.IGNORECASE)


class GateError(ValueError):
    """Query refused by the gate (message is safe to show to the planner)."""


def _strip_sql(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return sql.strip().rstrip(";").strip()


PATHLIKE = re.compile(r"https?://|[a-z]:[\\/]|^/|\.(parquet|csv|gz|xlsx|json|db)\b", re.IGNORECASE)


def check_select(sql: str) -> str:
    s = _strip_sql(sql)
    if not re.match(r"^(select|with)\b", s, re.I):
        raise GateError("only a single SELECT / WITH statement is allowed")
    # string literals may contain ';' or words like 'set'; check keywords on SQL text outside literals only,
    # but refuse any literal that looks like a file path or URL
    literals = re.findall(r"'((?:[^']|'')*)'", s)
    code = re.sub(r"'(?:[^']|'')*'", "''", s)
    if ";" in code:
        raise GateError("only one statement is allowed (no ';')")
    m = FORBIDDEN.search(code)
    if m:
        raise GateError("statement uses a forbidden keyword or file/network access: " + m.group(0))
    for lit in literals:
        if PATHLIKE.search(lit):
            raise GateError("string literals that look like file paths or URLs are not allowed")
    return s


IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BANNED_AGG = re.compile(r"\b(min|max|arg_min|arg_max|argmin|argmax|first|last|any_value|list|array_agg|string_agg|"
                        r"group_concat|histogram|mode|approx_top_k|select|from|quantile_disc)\b", re.IGNORECASE)


def _check_metric(m: str) -> None:
    """Whitelist aggregate expressions; refuse anything that can return an individual record's value."""
    if ";" in m or BANNED_AGG.search(m):
        raise GateError(f"metric not allowed (no min/max/list/string aggregation or subqueries): {m!r}")
    if not re.search(r"\b(count|sum|avg|mean|median|quantile_cont|stddev|stddev_samp|var_samp|corr|regr_slope)\s*\(",
                     m, re.IGNORECASE):
        raise GateError(f"each metric must be an aggregate (count/sum/avg/median/quantile_cont/stddev/...): {m!r}")
    m_q = re.findall(r"quantile_cont\s*\([^,]+,\s*([0-9.]+)", m, re.IGNORECASE)
    for q in m_q:
        if not 0.05 <= float(q) <= 0.95:
            raise GateError("quantile_cont probabilities must be within 0.05-0.95 (extremes approximate single records)")


def cores() -> int:
    """CPU cores this process may use (respects CPU affinity / container limits where the OS reports them)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def ram_bytes() -> int:
    """Physical memory in bytes (0 if unknown)."""
    try:
        if os.name == "nt":
            import ctypes

            class _M(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "ext")]
            m = _M(); m.len = ctypes.sizeof(_M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.total)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return 0


def limits(memory: Optional[str] = None, threads: Optional[int] = None) -> tuple[Optional[str], int]:
    """DuckDB limits: UCD_MEMORY / UCD_THREADS win, then the caller's value, then every available core and DuckDB's
    own memory default (80 % of RAM). Until 0.5.0 the default was 2 GB / 4 threads, sized for parallel planners on a
    16 GB laptop; a single CLI run should use the machine (e.g. 120 cores on a KHDP pod)."""
    return os.environ.get("UCD_MEMORY") or memory, int(os.environ.get("UCD_THREADS") or threads or cores())


_SHARED: dict[tuple[str, str], "Gate"] = {}         # Gate.shared(): one gate per dataset and run folder


class Gate:
    def __init__(self, dataset: str, run_dir: Optional[Path] = None, memory: Optional[str] = None,
                 threads: Optional[int] = None):
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; available: {', '.join(DATASETS)}")
        self.dataset = dataset
        self.run_dir = Path(run_dir) if run_dir else None
        self.con = duckdb.connect()
        tmp = ROOT / ".kcd_cache" / "duckdb_tmp" / f"{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        memory, threads = limits(memory, threads)
        self.con.execute((f"SET memory_limit='{memory}'; " if memory else "") + f"SET threads={threads}; "
                         f"SET temp_directory='{tmp.as_posix()}'; SET preserve_insertion_order=false; SET enable_progress_bar=false;")
        self.tables: dict[str, list[str]] = {}
        self._steps: dict[str, str] = {}               # cached WITH steps: key -> temp table (see _with_cached_steps)
        self._prev_keys: set[str] = set()
        self.last_reused: list[str] = []
        self._make_views()
        from ucd import concepts                       # tested building blocks already cached on this machine
        if concepts.index(dataset):
            concepts.register_cached(self.con, dataset)

    # ------------------------------------------------------------------ views
    def _make_views(self) -> None:
        base = _ds.parquet_dir(self.dataset)
        layout = {m.name: sorted(p.stem for p in m.glob("*.parquet")) for m in sorted(base.glob("*")) if m.is_dir()}
        for module, tables in layout.items():
            self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {module}")
            for t in tables:
                f = base / module / f"{t}.parquet"
                self.con.execute(f"CREATE VIEW {module}.{t} AS SELECT * FROM read_parquet('{f.as_posix()}')")
            if tables:
                self.tables[module] = tables

    # ------------------------------------------------------------------ logging / budget
    def _log(self, kind: str, sql: str, info: dict[str, Any]) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": self.dataset, "kind": kind,
               "run_dir": str(self.run_dir) if self.run_dir else None, "sql": sql, **info}
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        GLOBAL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(GLOBAL_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
        if self.run_dir:
            with open(self.run_dir / "gate.log", "a", encoding="utf-8") as fh:
                fh.write(line)

    def _budget_file(self) -> Optional[Path]:
        return self.run_dir / f"budget_{self.dataset}.json" if self.run_dir else None

    def budget_used(self) -> int:
        f = self._budget_file()
        return json.loads(f.read_text())["used"] if f and f.exists() else 0

    def _spend(self) -> int:
        f = self._budget_file()
        used = self.budget_used() + 1
        if f:
            f.write_text(json.dumps({"used": used}))
        return used

    @staticmethod
    def sanitize(df: pd.DataFrame) -> dict[str, Any]:
        cols = [str(c) for c in df.columns]
        low = [c.lower() for c in cols]
        bad = [c for c in cols if is_id_name(c)]
        if bad:
            raise GateError(f"identifier columns are not allowed in results: {bad}")
        dt = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c]) or
              str(df[c].dtype).startswith(("datetime", "timedelta")) or
              (df[c].dtype == object and df[c].map(lambda v: hasattr(v, "isoformat")).any())]
        if dt:
            raise GateError(f"date/time columns are not allowed in results (aggregate them, e.g. hours): {dt}")
        if "n" not in low:
            raise GateError("internal: aggregate result without its count column n")
        ncol = cols[low.index("n")]
        n = pd.to_numeric(df[ncol], errors="coerce")
        keep = n >= K_MIN
        truncated = len(df) > MAX_ROWS
        df = df[keep].head(MAX_ROWS)
        rows = []
        for rec in df.to_dict(orient="records"):
            r = {}
            for k, v in rec.items():
                if isinstance(v, (float, np.floating)):
                    v = None if np.isnan(v) else float(f"{v:.4g}")
                elif isinstance(v, np.integer):
                    v = int(v)
                elif v is not None and not isinstance(v, (int, str, bool)):
                    v = str(v)
                r[str(k)] = v
            rows.append(r)
        return {"columns": cols, "rows": rows, "suppressed_rows": int((~keep).sum()),
                "truncated": bool(truncated)}

    def _unit_tables(self) -> list[tuple[str, str]]:
        """(id column, table) of every unit level in the registry: patient, admission, stay ..."""
        try:
            return [(u["id"], u["table"]) for u in (_ds.get(self.dataset).get("units") or {}).values()]
        except Exception:
            return []

    def _id_valued(self, src: str, col: str) -> str:
        """The unit id column whose values make up most of `col`'s distinct values ('' if none): one scan, the values
        compared as text so that a cast does not hide them."""
        units = self._unit_tables()
        if not units:
            return ""
        hits = ", ".join(f'count(*) FILTER (WHERE v IN (SELECT CAST("{c}" AS VARCHAR) FROM {t}))' for c, t in units)
        try:
            row = self.con.execute(f'SELECT count(*), {hits} FROM (SELECT DISTINCT CAST("{col}" AS VARCHAR) AS v '
                                   f"FROM ({src}) AS q) AS d").fetchone()
        except duckdb.Error:                         # a unit table that is not loaded: the other checks still apply
            return ""
        return next((c for (c, _), h in zip(units, row[1:]) if h > 0.5 * max(row[0], 1)), "")

    # ------------------------------------------------------------------ structured explore (planner-facing, safe)
    def aggregate(self, source_sql: str, group_by: list[str], metrics: list[str]) -> dict[str, Any]:
        """SELECT <group>, count(*) AS n, <metrics> FROM (<source>) GROUP BY <group> HAVING count(*) >= K_MIN.
        n is computed by the gate (cannot be forged); metrics must use whitelisted aggregate functions (no min/max/
        list/string aggregation that returns individual values); group columns must be plain column names that are
        not identifiers, not datetimes and not near-unique."""
        src = check_select(source_sql)
        for g in group_by:
            if not IDENT.match(g):
                raise GateError(f"group_by entries must be plain column names: {g!r}")
            if is_id_name(g):
                raise GateError(f"cannot group by an identifier column: {g}")
        for m in metrics:
            _check_metric(m)
        if self.run_dir and self.budget_used() >= BUDGET:
            raise GateError(f"explore budget of {BUDGET} queries for {self.dataset} is used up")
        try:
            self._concepts(src)
            src = self._with_cached_steps(src)         # steps computed here are reused by the next run, and back
            desc = self.con.execute(f"DESCRIBE SELECT * FROM ({src}) AS q").fetchall()
        except duckdb.Error as e:
            raise GateError(f"SQL error: {str(e)[:500]}") from None
        types = {r[0]: str(r[1]).upper() for r in desc}
        for g in group_by:
            if g not in types:
                raise GateError(f"group_by column {g!r} is not in the source; columns: {list(types)[:40]}")
            if any(t in types[g] for t in ("TIMESTAMP", "DATE", "TIME", "INTERVAL")):
                raise GateError(f"cannot group by a date/time column ({g}); derive hours or a category first")
        if group_by:
            cols = ", ".join(f'count(DISTINCT "{g}")' for g in group_by)
            row = self.con.execute(f"SELECT count(*), {cols} FROM ({src}) AS q").fetchone()
            total = max(int(row[0]), 1)
            for g, d in zip(group_by, row[1:]):
                if d > 200 and d / total > 0.2:
                    raise GateError(f"group_by column {g} is near-unique ({d} distinct values in {total} rows): "
                                    "looks like an identifier or raw value; bin it first")
                ids = self._id_valued(src, g) if d > ID_VALUED_MIN else ""
                if ids:                                # an id under another name (E-68): renaming must not leak it
                    raise GateError(f"group_by column {g} holds identifiers (most of its values are {ids} values); "
                                    "group by a category or a binned value")
        gsel = ", ".join(f'"{g}"' for g in group_by)
        msel = "".join(f", {m}" for m in metrics)
        sql = (f"SELECT {gsel + ', ' if gsel else ''}count(*) AS n{msel} FROM ({src}) AS q "
               + (f"GROUP BY {gsel} " if gsel else "") + f"HAVING count(*) >= {K_MIN} ORDER BY n DESC LIMIT {MAX_ROWS + 1}")
        used = self._spend()
        key = hashlib.sha1(f"{self.dataset}|AGG|{sql}".encode()).hexdigest()
        cf = CACHE / self.dataset / f"{key}.json"
        if cf.exists():
            out = json.loads(cf.read_text(encoding="utf-8"))
        else:
            try:
                df = self.con.execute(sql).df()
            except duckdb.Error as e:
                self._log("aggregate_error", sql, {"error": str(e)[:300]})
                raise GateError(f"SQL error: {str(e)[:500]}") from None
            out = self.sanitize(df)
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
        self._log("aggregate", sql, {"rows": len(out["rows"])})
        out["budget_left"] = BUDGET - used
        return out

    # ------------------------------------------------------------------ materialize (runner-only)
    def _concepts(self, sql: str, build: bool = True) -> None:
        """Every concept.<name> the query uses: built once per machine (build=True; rows stay inside the gate), or
        only declared as a view for binding (build=False, the SQL pre-check)."""
        from ucd import concepts, diag
        names = concepts.referenced(sql)
        if not names:
            return
        try:
            if build:
                for n in concepts.ensure(self.con, self.dataset, names):
                    self._log("concept_build", f"concept.{n}", {})
            else:
                concepts.declare(self.con, self.dataset, names)
        except ValueError as e:
            raise GateError(str(e)) from None
        except (duckdb.Error, OSError) as e:          # never show a raw build error: it may quote values (R6)
            diag.error("gate.concept_build", e)
            raise GateError(f"a concept failed to build ({', '.join(names)}); see the diagnostics log") from None

    def _with_cached_steps(self, sql: str) -> str:
        """Run each WITH step once per gate session and reuse it while its text and inputs are unchanged (0.7.5): a
        revised design recomputes only the steps it changed, plus the steps that read from them, plus the final
        SELECT. Each step is a temporary table inside this DuckDB session; its rows never leave the gate. Returns
        the query rewritten to read the cached steps; `self.last_reused` names the steps that were reused."""
        from ucd import sqlcte
        self.last_reused = []
        parts = sqlcte.split(sql)
        if parts is None:
            return sql
        steps, final = parts
        names = [sqlcte.bare(h) for h, _ in steps]
        keys: dict[str, str] = {}
        aliases: list[tuple[str, str]] = []
        for (header, body), name in zip(steps, names):
            earlier = names[:len(aliases)]
            deps = sqlcte.references(body, earlier)
            key = hashlib.sha1("|".join([header, " ".join(body.split())] + [keys[d] for d in deps]).encode()).hexdigest()[:16]
            keys[name] = key
            table = f"ucd_step_{key}"
            if key in self._steps:
                self.last_reused.append(name)
            else:
                self.con.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS " + sqlcte.join(aliases, body))
                self._steps[key] = table
            aliases.append((header, f"SELECT * FROM {table}"))
        live = set(keys.values())                      # keep this query's steps and the previous query's
        for k in [k for k in self._steps if k not in live | self._prev_keys]:
            self.con.execute(f"DROP TABLE IF EXISTS {self._steps.pop(k)}")
        self._prev_keys = live
        return sqlcte.join(aliases, final)

    @classmethod
    def shared(cls, dataset: str, run_dir: Optional[Path] = None) -> "Gate":
        """One gate per dataset and run folder in this process, so cached WITH steps survive between design
        iterations of the same run (the loop runs every iteration in one process)."""
        key = (dataset, str(Path(run_dir).resolve()) if run_dir else "")
        if key not in _SHARED:
            _SHARED[key] = cls(dataset, run_dir=run_dir)
        return _SHARED[key]

    def materialize(self, sql: str) -> pd.DataFrame:
        s = check_select(sql)
        t0 = time.time()
        try:
            self._concepts(s)
            df = self.con.execute(self._with_cached_steps(s)).df()
        except duckdb.Error as e:
            self._log("materialize_error", s, {"error": str(e)[:300]})
            from ucd import diag
            diag.error("gate.materialize", e, run=self.run_dir.name if self.run_dir else None)
            raise GateError(f"analysis_sql failed: {str(e)[:800]}{self.column_hint(str(e), s)}{self.clause_hint(str(e))}") from None
        self._log("materialize", s, {"rows": len(df), "seconds": round(time.time() - t0, 1),
                                     "steps_reused": len(self.last_reused)})
        return df

    def bind_check(self, sql: str, needed: Optional[list[str]] = None) -> str:
        """'' when the query binds and returns every needed column; else what is wrong, with a column hint (0.9.0).
        DuckDB binds the whole query (every table and column, concepts as views over their SQL) with DESCRIBE without
        reading rows or building concepts, so it costs milliseconds even on chartevents: an SQL slip is found before
        a design attempt or a critic call is spent."""
        try:
            s = check_select(sql)
            self._concepts(s, build=False)             # bind only: nothing is computed here
            cols = [r[0] for r in self.con.execute(f"DESCRIBE {s}").fetchall()]
        except GateError as e:
            return str(e)
        except duckdb.Error as e:
            return f"{str(e)[:600]}{self.column_hint(str(e), sql)}{self.clause_hint(str(e))}"
        missing = [c for c in (needed or []) if c not in cols]
        if missing:
            return (f"the query does not return the column(s) the spec needs: {missing}; it returns {cols[:40]}. "
                    "Add them to the final SELECT (or remove them from the spec).")
        return ""

    def _step_columns(self, alias: str, sql: str) -> str:
        """For an alias that stands for a WITH step: the columns that step really returns (schema only; E-55)."""
        from ucd import sqlcte
        try:
            parts = sqlcte.split(sql)
            if not parts:
                return ""
            steps, _ = parts
            names = [sqlcte.bare(h).lower() for h, _ in steps]
            m = re.search(rf"\b(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?{re.escape(alias)}\b", sql, flags=re.I)
            step = (m.group(1) if m else alias).lower()
            if step not in names:
                return ""
            k = names.index(step)
            cols = [r[0] for r in self.con.execute(f"DESCRIBE {sqlcte.join(steps[:k + 1], 'SELECT * FROM ' + step)}").fetchall()]
            return f" step '{step}' returns: {', '.join(cols[:40])};"
        except Exception:                              # a hint must never hide the original error
            return ""

    @staticmethod
    def clause_hint(error: str) -> str:
        """For 'WHERE clause cannot contain window functions / aggregates' (E-62): DuckDB lets WHERE name a SELECT
        alias and uses its expression, so the Postgres habit `row_number() OVER (...) AS rn ... WHERE rn = 1` fails."""
        if "WHERE clause cannot contain" not in error:
            return ""
        return ("\nHINT: WHERE cannot filter a window function or an aggregate, not even through its SELECT alias "
                "(`... AS rn ... WHERE rn = 1`). Use `QUALIFY rn = 1` for a window result, HAVING after GROUP BY for "
                "an aggregate, or filter in the next step.")

    def _relation_columns(self, name: str) -> str:
        """' (columns: a, b, ...)' for a named table or concept (schema only; E-64): concepts are not in the raw
        schema, so without this the hint for `FROM concept.chemistry che` named no column `che` really has."""
        try:
            cols = [r[0] for r in self.con.execute(f"DESCRIBE {name}").fetchall()]
        except Exception:                              # a hint must never hide the original error
            return ""
        return f" (columns: {', '.join(cols[:40])})"

    def column_hint(self, error: str, sql: str) -> str:
        """For 'column not found' errors: which tables really have that column, and which table the alias stands
        for (schema only, no data). Small models otherwise guess again (e.g. a.intime on hosp.admissions; E-35)."""
        m = re.search(r'column named "([^"]+)"|Referenced column "([^"]+)" not found', error)
        if not m:
            return ""
        col = (m.group(1) or m.group(2)).lower()
        try:
            schema = self.schema()
        except Exception:                              # a hint must never hide the original error
            return ""
        have = [f"{mod}.{t}" for mod, ts in schema.items() for t, cols in ts.items()
                if col in {c.lower() for c, _ in cols}]
        where = ""
        alias = re.search(r'(?:Table|Values list) "([^"]+)" does not have', error)
        if alias:
            src = re.search(rf"\b(?:FROM|JOIN)\s+(\w+\.\w+)\s+(?:AS\s+)?{re.escape(alias.group(1))}\b", sql, flags=re.I)
            where = (f" '{alias.group(1)}' is {src.group(1)}{self._relation_columns(src.group(1))};" if src else
                     f" '{alias.group(1)}' is a subquery or CTE: add the column to its SELECT list;"
                     f"{self._step_columns(alias.group(1), sql)}")
        return (f"\nHINT:{where} column '{col}' exists in: {', '.join(have[:8])}." if have else
                f"\nHINT:{where} no table has a column '{col}'; check the Data Card's column lists.")

    def schema(self) -> dict[str, dict[str, list[tuple[str, str]]]]:
        out: dict[str, dict[str, list[tuple[str, str]]]] = {}
        for module, tables in self.tables.items():
            for t in tables:
                d = self.con.execute(f"DESCRIBE {module}.{t}").fetchall()
                out.setdefault(module, {})[t] = [(r[0], r[1]) for r in d]
        return out
