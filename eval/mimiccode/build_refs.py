"""Build hidden external reference cohorts from the official MIT-LCP mimic-code
MIMIC-IV concepts (DuckDB dialect, see SOURCE.md).

Steps
  1. Register MIMIC-IV parquet files as views  mimiciv_hosp.<t> / mimiciv_icu.<t>.
  2. Run the needed mimic-code concepts (transitive closure of TARGETS) in the
     order of concepts_duckdb/duckdb.sql, materialising each as
     mimiciv_derived.<concept>, writing it to derived/<concept>.parquet and then
     replacing the in-memory table by a view over that parquet (keeps RAM low,
     makes reruns incremental).
  3. Build reference cohorts in ref/ (t1_aki, t2_sofa, t4_ne, t7_meld).

Usage:  python eval/mimiccode/build_refs.py [--force] [--wait] [--only c1,c2]
          --force  recompute concepts even if derived/<c>.parquet exists
          --wait   poll until missing source tables are converted (sleep <= 5 min)
Only aggregate statistics are printed.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARQ = ROOT / ".kcd_cache" / "parquet" / "mimic"
TMP = ROOT / ".kcd_cache" / "duckdb_tmp"
SQLDIR = ROOT / "third_party" / "mimic-code" / "concepts_duckdb"
DERIVED = HERE / "derived"
REF = HERE / "ref"

TARGETS = [
    "age", "icustay_detail",
    "kdigo_creatinine", "kdigo_uo", "kdigo_stages",
    "first_day_sofa", "sofa",
    "suspicion_of_infection", "sepsis3",
    "vasoactive_agent", "norepinephrine",
    "meld",
]

# Minimal dialect fixes applied to the official DuckDB SQL: (concept, pattern, replacement).
# Filled in only where the upstream DuckDB SQL fails on duckdb 1.x.
DIALECT_FIXES: list[tuple[str, str, str]] = []

REF_RE = re.compile(r"mimiciv_(derived|hosp|icu)\.([a-z_0-9]+)")


def log(*a):
    print(*a, flush=True)


def concept_order() -> dict[str, Path]:
    order: dict[str, Path] = {}
    for line in (SQLDIR / "duckdb.sql").read_text().splitlines():
        if line.startswith(".read "):
            rel = line.split(None, 1)[1].strip()
            order[Path(rel).stem] = SQLDIR / rel
    return order


def deps_of(path: Path) -> tuple[set[str], set[str]]:
    name = path.stem
    derived, raw = set(), set()
    for schema, tbl in REF_RE.findall(path.read_text()):
        if schema == "derived":
            if tbl != name:
                derived.add(tbl)
        else:
            raw.add(f"{schema}/{tbl}")
    return derived, raw


def closure(order, targets):
    need, stack = set(), list(targets)
    while stack:
        c = stack.pop()
        if c in need:
            continue
        need.add(c)
        stack.extend(deps_of(order[c])[0])
    return [c for c in order if c in need]


def raw_ready(t: str) -> bool:
    p = PARQ / f"{t}.parquet"
    # ucd/build_parquet.py writes <table>.part and renames to <table>.parquet when done
    return p.exists() and not p.with_suffix(".part").exists()


def connect():
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=8")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    for mod in ("hosp", "icu"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS mimiciv_{mod}")
    con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived")
    return con


def register_raw(con):
    n = 0
    for mod in ("hosp", "icu"):
        for p in sorted((PARQ / mod).glob("*.parquet")):
            if not raw_ready(f"{mod}/{p.stem}"):
                continue
            con.execute(f"CREATE OR REPLACE VIEW mimiciv_{mod}.{p.stem} AS "
                        f"SELECT * FROM read_parquet('{p.as_posix()}')")
            n += 1
    return n


def as_view(con, c):
    con.execute(f"DROP TABLE IF EXISTS mimiciv_derived.{c}")
    con.execute(f"CREATE OR REPLACE VIEW mimiciv_derived.{c} AS "
                f"SELECT * FROM read_parquet('{(DERIVED / (c + '.parquet')).as_posix()}')")


def run_concept(con, c, path):
    sql = path.read_text()
    for cc, pat, rep in DIALECT_FIXES:
        if cc == c:
            sql = re.sub(pat, rep, sql)
    t0 = time.time()
    con.execute(f"DROP VIEW IF EXISTS mimiciv_derived.{c}")
    con.execute(sql)
    out = DERIVED / f"{c}.parquet"
    tmp = DERIVED / f"{c}.parquet.tmp"
    con.execute(f"COPY mimiciv_derived.{c} TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out)
    n = con.execute(f"SELECT count(*) FROM mimiciv_derived.{c}").fetchone()[0]
    as_view(con, c)
    log(f"  [ok] {c}: {n:,} rows in {time.time() - t0:.0f}s")


def build_concepts(con, force=False, wait=False, only=None):
    DERIVED.mkdir(parents=True, exist_ok=True)
    order = concept_order()
    todo = closure(order, only or TARGETS)
    log(f"concepts needed ({len(todo)}): {', '.join(todo)}")
    done, failed = set(), {}
    pending = list(todo)
    while pending:
        progressed, still = False, []
        for c in pending:
            dderiv, draw = deps_of(order[c])
            if any(d in failed for d in dderiv):
                failed[c] = "dependency failed"
                progressed = True
                continue
            if not dderiv <= done or not all(raw_ready(t) for t in draw):
                still.append(c)
                continue
            out = DERIVED / f"{c}.parquet"
            try:
                if out.exists() and not force:
                    as_view(con, c)
                    log(f"  [cached] {c}")
                else:
                    run_concept(con, c, order[c])
                done.add(c)
            except Exception as e:  # noqa: BLE001
                failed[c] = f"{type(e).__name__}: {str(e)[:300]}"
                log(f"  [FAIL] {c}: {failed[c]}")
            progressed = True
        pending = still
        if pending and not progressed:
            missing = sorted({t for c in pending for t in deps_of(order[c])[1] if not raw_ready(t)})
            if not wait:
                log(f"  [skip] waiting on source tables {missing}: {', '.join(pending)}")
                break
            log(f"  ... waiting for {missing} ({time.strftime('%H:%M:%S')})")
            time.sleep(300)
            register_raw(con)
    return done, failed


# ---------------------------------------------------------------- references

ADULT_STAYS = """
  SELECT d.subject_id, d.hadm_id, d.stay_id, d.icu_intime AS intime, d.admission_age,
         d.hospital_expire_flag, d.first_icu_stay AS first_icu_stay_in_hadm,
         ROW_NUMBER() OVER (PARTITION BY d.subject_id ORDER BY d.icu_intime, d.stay_id) = 1
           AS first_icu_stay_patient
  FROM mimiciv_derived.icustay_detail d
  WHERE d.admission_age >= 18
"""


def build_refs(con, done):
    REF.mkdir(parents=True, exist_ok=True)
    built = {}

    def write(name, sql):
        p = REF / f"{name}.parquet"
        con.execute(f"COPY ({sql}) TO '{p.as_posix()}' (FORMAT PARQUET)")
        built[name] = p
        log(f"  [ref] {name}")

    if {"icustay_detail", "kdigo_stages"} <= done:
        write("t1_aki", f"""
          WITH a AS ({ADULT_STAYS}),
          k AS (
            SELECT a.stay_id, MAX(ks.aki_stage) AS max_kdigo_stage_7d,
                   MAX(ks.aki_stage_creat) AS max_stage_creat_7d,
                   MAX(ks.aki_stage_uo) AS max_stage_uo_7d,
                   COUNT(ks.charttime) AS n_kdigo_rows_7d
            FROM a JOIN mimiciv_derived.kdigo_stages ks
              ON ks.stay_id = a.stay_id
             AND ks.charttime <= a.intime + INTERVAL 7 DAY
            GROUP BY a.stay_id)
          SELECT a.stay_id, a.subject_id, a.hadm_id, a.admission_age,
                 k.max_kdigo_stage_7d, k.max_stage_creat_7d, k.max_stage_uo_7d,
                 COALESCE(k.n_kdigo_rows_7d, 0) AS n_kdigo_rows_7d,
                 a.hospital_expire_flag, a.first_icu_stay_patient, a.first_icu_stay_in_hadm
          FROM a LEFT JOIN k USING (stay_id)""")

    if {"icustay_detail", "suspicion_of_infection", "first_day_sofa"} <= done:
        write("t2_sofa", f"""
          WITH a AS ({ADULT_STAYS}),
          soi AS (
            SELECT a.stay_id, MIN(s.suspected_infection_time) AS suspected_infection_time
            FROM a JOIN mimiciv_derived.suspicion_of_infection s
              ON s.stay_id = a.stay_id AND s.suspected_infection = 1
             AND s.suspected_infection_time BETWEEN a.intime - INTERVAL 48 HOUR
                                               AND a.intime + INTERVAL 24 HOUR
            GROUP BY a.stay_id)
          SELECT a.stay_id, a.subject_id, a.hadm_id, a.admission_age,
                 soi.suspected_infection_time, f.sofa AS first_day_sofa,
                 f.respiration, f.coagulation, f.liver, f.cardiovascular, f.cns, f.renal,
                 a.hospital_expire_flag, a.first_icu_stay_patient, a.first_icu_stay_in_hadm
          FROM a JOIN soi USING (stay_id)
          LEFT JOIN mimiciv_derived.first_day_sofa f USING (stay_id)""")

    if {"icustay_detail", "sepsis3", "norepinephrine"} <= done:
        write("t4_ne", f"""
          WITH a AS ({ADULT_STAYS}),
          ne AS (SELECT stay_id, MIN(starttime) AS ne_start
                 FROM mimiciv_derived.norepinephrine GROUP BY stay_id)
          SELECT s.stay_id, a.subject_id, a.hadm_id, a.admission_age, a.intime,
                 s.suspected_infection_time, s.sofa_time, s.sofa_score,
                 ne.ne_start,
                 DATE_DIFF('second', a.intime, ne.ne_start) / 3600.0 AS hours_from_icu_intime_to_ne,
                 DATE_DIFF('second', LEAST(s.suspected_infection_time, s.sofa_time), ne.ne_start) / 3600.0
                   AS hours_from_sepsis_onset_to_ne,
                 a.hospital_expire_flag, a.first_icu_stay_patient
          FROM mimiciv_derived.sepsis3 s
          JOIN a USING (stay_id)
          JOIN ne USING (stay_id)
          WHERE s.sepsis3""")

    if {"icustay_detail", "meld"} <= done:
        write("t7_meld", f"""
          WITH a AS ({ADULT_STAYS})
          SELECT m.stay_id, a.subject_id, a.hadm_id, a.admission_age,
                 m.meld, m.meld_initial, m.rrt, m.creatinine_max, m.bilirubin_total_max,
                 m.inr_max, m.sodium_min,
                 a.hospital_expire_flag, a.first_icu_stay_patient
          FROM mimiciv_derived.meld m JOIN a USING (stay_id)
          WHERE m.meld IS NOT NULL""")
    return built


def summarise(con, built):
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    for name, p in built.items():
        src = f"read_parquet('{p.as_posix()}')"
        n, mort, nfirst = q(f"SELECT count(*), avg(hospital_expire_flag), "
                            f"sum(first_icu_stay_patient::INT) FROM {src}")[0]
        log(f"\n== {name}: {n:,} stays (first-stay-per-patient {nfirst:,}); "
            f"hospital mortality {mort:.3%}")
        if name == "t1_aki":
            nk, anyaki = q(f"SELECT count(max_kdigo_stage_7d), avg((max_kdigo_stage_7d>0)::INT) "
                           f"FROM {src}")[0]
            log(f"   stays with KDIGO data: {nk:,}; any-stage AKI among them: {anyaki:.2%}")
            for st, c in q(f"SELECT max_kdigo_stage_7d, count(*) FROM {src} GROUP BY 1 ORDER BY 1"):
                log(f"   stage {st}: {c:,} ({c / n:.1%})")
            for st, m in q(f"SELECT max_kdigo_stage_7d, avg(hospital_expire_flag) FROM {src} "
                           f"WHERE max_kdigo_stage_7d IS NOT NULL GROUP BY 1 ORDER BY 1"):
                log(f"   mortality stage {st}: {m:.1%}")
            nf, af = q(f"SELECT count(max_kdigo_stage_7d), avg((max_kdigo_stage_7d>0)::INT) "
                       f"FROM {src} WHERE first_icu_stay_patient")[0]
            log(f"   first stay/patient with KDIGO data: {nf:,}; any-stage AKI {af:.2%}")
        if name == "t2_sofa":
            try:
                from sklearn.metrics import roc_auc_score
                df = con.execute(f"SELECT first_day_sofa, hospital_expire_flag FROM {src} "
                                 f"WHERE first_day_sofa IS NOT NULL").df()
                auc = roc_auc_score(df.hospital_expire_flag, df.first_day_sofa)
                dff = con.execute(f"SELECT first_day_sofa, hospital_expire_flag FROM {src} "
                                  f"WHERE first_day_sofa IS NOT NULL AND first_icu_stay_patient").df()
                auc1 = roc_auc_score(dff.hospital_expire_flag, dff.first_day_sofa)
                log(f"   first-day SOFA AUROC for hospital mortality: {auc:.4f} (n={len(df):,}); "
                    f"first stay/patient {auc1:.4f} (n={len(dff):,})")
            except ImportError:
                log("   sklearn not available; AUROC skipped")
            med, q1, q3 = q(f"SELECT median(first_day_sofa), quantile_cont(first_day_sofa,0.25), "
                            f"quantile_cont(first_day_sofa,0.75) FROM {src}")[0]
            log(f"   first-day SOFA median {med} (IQR {q1}-{q3})")
        if name == "t4_ne":
            med, q1, q3, neg = q(f"SELECT median(hours_from_icu_intime_to_ne), "
                                 f"quantile_cont(hours_from_icu_intime_to_ne,0.25), "
                                 f"quantile_cont(hours_from_icu_intime_to_ne,0.75), "
                                 f"avg((hours_from_icu_intime_to_ne<0)::INT) FROM {src}")[0]
            log(f"   hours ICU intime->NE median {med:.1f} (IQR {q1:.1f}-{q3:.1f}); before intime {neg:.1%}")
        if name == "t7_meld":
            med, q1, q3 = q(f"SELECT median(meld), quantile_cont(meld,0.25), "
                            f"quantile_cont(meld,0.75) FROM {src}")[0]
            log(f"   MELD median {med:.1f} (IQR {q1:.1f}-{q3:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated target concepts")
    args = ap.parse_args()
    con = connect()
    log(f"registered {register_raw(con)} source tables")
    only = args.only.split(",") if args.only else None
    done, failed = build_concepts(con, force=args.force, wait=args.wait, only=only)
    built = build_refs(con, done)
    summarise(con, built)
    if "meld" not in done:
        log("\nNOTE: meld not built -> t7_meld skipped")
    if failed:
        log("\nFAILED concepts:")
        for c, e in failed.items():
            log(f"  {c}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
