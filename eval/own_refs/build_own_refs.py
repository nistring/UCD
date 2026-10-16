"""Evaluator-side, hand-written reference cohorts for t3 (dysnatremia), t5 (ICU readmission), t6 (transfusion).

These are OUR OWN references (no published mimic-code cohort exists for these topics). They implement the
protocols in ucd/topics.yaml directly on MIMIC-IV parquet, following papers.yaml design_checklist where the
protocol is silent. Writes eval/mimiccode/ref/{t3_dysnatremia,t5_readmission,t6_transfusion}.parquet
and eval/own_refs/own_refs_summary.json. Prints aggregates only.

    python eval/own_refs/build_own_refs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

ROOT = Path(__file__).resolve().parents[2]
P = (ROOT / ".kcd_cache" / "parquet" / "mimic").as_posix()
REF = ROOT / "eval" / "mimiccode" / "ref"
SOFA = ROOT / "eval" / "mimiccode" / "derived" / "first_day_sofa.parquet"
OUT = Path(__file__).resolve().parent / "own_refs_summary.json"

SURG_SERVICES = ("CSURG", "NSURG", "ORTHO", "PSURG", "SURG", "TRAUM", "TSURG", "VSURG", "GU", "GYN", "ENT")
# care units that MIMIC-IV icustays treats as ICU (icustays.first_careunit, excluding rare ward artefacts)
ICU_UNITS = ("Medical Intensive Care Unit (MICU)", "Medical/Surgical Intensive Care Unit (MICU/SICU)",
             "Cardiac Vascular Intensive Care Unit (CVICU)", "Surgical Intensive Care Unit (SICU)",
             "Coronary Care Unit (CCU)", "Trauma SICU (TSICU)", "Neuro Surgical Intensive Care Unit (Neuro SICU)",
             "Intensive Care Unit (ICU)", "Neuro Intermediate", "Neuro Stepdown")
NOT_WARD = ICU_UNITS + ("Emergency Department", "Emergency Department Observation", "Discharge Lounge",
                        "UNKNOWN", "Unknown")


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    (ROOT / ".kcd_cache" / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    c.execute("SET memory_limit='3GB'; SET threads=4; "
              f"SET temp_directory='{(ROOT / '.kcd_cache' / 'duckdb_tmp').as_posix()}'")
    for s, t in [("hosp", "admissions"), ("hosp", "patients"), ("hosp", "transfers"), ("hosp", "services"),
                 ("hosp", "labevents"), ("icu", "icustays"), ("icu", "inputevents")]:
        c.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{P}/{s}/{t}.parquet')")
    surg = ",".join(f"'{x}'" for x in SURG_SERVICES)
    # base: every ICU stay with age, sex, hospital outcome, surgical flag, order within patient / hadm
    c.execute(f"""
    CREATE TEMP TABLE base AS
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los, i.first_careunit,
           p.anchor_age + (year(i.intime) - p.anchor_year) AS age, (p.gender = 'M')::INT AS male,
           a.admission_type, a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag, a.discharge_location,
           (EXISTS (SELECT 1 FROM services s WHERE s.hadm_id = i.hadm_id AND s.curr_service IN ({surg})
                    AND s.transfertime <= i.intime + INTERVAL 24 HOUR))::INT AS surgical,
           row_number() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS pt_seq,
           row_number() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS hadm_seq
    FROM icustays i JOIN patients p USING (subject_id) JOIN admissions a USING (hadm_id)""")
    return c


def logit_or(df: pd.DataFrame, y: str, cats: list[str], ref: str, cat_col: str, covs: list[str]) -> dict:
    X = pd.get_dummies(df[cat_col]).astype(float).drop(columns=[ref])[[k for k in cats if k != ref]]
    X = pd.concat([X, df[covs].astype(float)], axis=1)
    m = sm.Logit(df[y].astype(float), sm.add_constant(X)).fit(disp=0)
    ci = m.conf_int()
    return {k: [round(float(np.exp(m.params[k])), 3), round(float(np.exp(ci.loc[k, 0])), 3),
                round(float(np.exp(ci.loc[k, 1])), 3)] for k in cats if k != ref}


# ------------------------------------------------------------------------------------------------ t3
NA_CATS = ["severe_hypo", "mild_hypo", "borderline_hypo", "normal", "borderline_hyper", "mild_hyper", "severe_hyper"]


def na_cat(v: float) -> str:
    if v < 125: return "severe_hypo"
    if v < 130: return "mild_hypo"
    if v < 135: return "borderline_hypo"
    if v <= 145: return "normal"
    if v <= 150: return "borderline_hyper"
    if v <= 155: return "mild_hyper"
    return "severe_hyper"


def t3(c) -> dict:
    df = c.execute("""
    WITH coh AS (SELECT * FROM base WHERE pt_seq = 1 AND age >= 18),
    na AS (SELECT b.stay_id, l.valuenum, row_number() OVER (PARTITION BY b.stay_id ORDER BY l.charttime) rn
           FROM coh b JOIN labevents l ON l.hadm_id = b.hadm_id
           WHERE l.itemid = 50983 AND l.valuenum IS NOT NULL
             AND l.charttime >= b.intime AND l.charttime <= b.intime + INTERVAL 24 HOUR)
    SELECT coh.stay_id, coh.age, coh.male, coh.surgical, coh.admission_type, coh.hospital_expire_flag,
           na.valuenum AS na_first_24h
    FROM coh LEFT JOIN na ON na.stay_id = coh.stay_id AND na.rn = 1""").df()
    n_first = len(df)
    df = df[df.na_first_24h.notna() & df.hospital_expire_flag.notna()].copy()
    df["na_cat"] = df.na_first_24h.map(na_cat)
    df["first_icu_stay_patient"] = True
    df["hospital_expire_flag"] = df.hospital_expire_flag.astype(int)
    df["emergency"] = df.admission_type.isin(["EW EMER.", "DIRECT EMER.", "URGENT"]).astype(int)
    covs = ["age", "male", "surgical", "emergency"]
    if SOFA.exists():
        s = pd.read_parquet(SOFA, columns=["stay_id", "sofa"])
        df = df.merge(s, on="stay_id", how="left")
        df["sofa"] = df["sofa"].fillna(0)        # mimic-code SOFA treats missing components as 0
        covs.append("sofa")
    out = {"n_first_adult_icu_stays": n_first, "n_cohort": len(df),
           "mortality": round(float(df.hospital_expire_flag.mean()), 4),
           "prevalence": {k: round(float((df.na_cat == k).mean()), 4) for k in NA_CATS},
           "mortality_by_cat": {k: round(float(df.loc[df.na_cat == k, "hospital_expire_flag"].mean()), 4) for k in NA_CATS},
           "covariates": covs,
           "aOR_vs_normal": logit_or(df, "hospital_expire_flag", NA_CATS, "normal", "na_cat", covs)}
    df[["stay_id", "first_icu_stay_patient", "na_first_24h", "na_cat", "hospital_expire_flag"]].to_parquet(
        REF / "t3_dysnatremia.parquet", index=False)
    return out


# ------------------------------------------------------------------------------------------------ t5
def t5(c) -> dict:
    df = c.execute("""
    WITH idx AS (SELECT * FROM base WHERE hadm_seq = 1 AND age >= 18),
    nxt AS (  -- first transfer event starting at/after ICU discharge in the same hadm
      SELECT x.stay_id, t.eventtype, t.careunit,
             row_number() OVER (PARTITION BY x.stay_id ORDER BY t.intime, t.transfer_id) rn
      FROM idx x JOIN transfers t ON t.hadm_id = x.hadm_id
      WHERE t.intime >= x.outtime)
    SELECT x.stay_id, x.subject_id, x.hadm_id, x.intime, x.outtime, x.hospital_expire_flag,
           x.deathtime, n.eventtype AS next_event, n.careunit AS next_unit,
           EXISTS (SELECT 1 FROM base b WHERE b.hadm_id = x.hadm_id AND b.intime > x.outtime) AS readmit
    FROM idx x LEFT JOIN nxt n ON n.stay_id = x.stay_id AND n.rn = 1""").df()
    n_idx = len(df)
    died_icu = df.deathtime.notna() & (df.deathtime <= df.outtime + pd.Timedelta(hours=1))
    to_ward = df.next_event.isin(["transfer", "admit"]) & df.next_unit.notna() & ~df.next_unit.isin(NOT_WARD)
    flow = {"n_first_icu_stay_of_hadm_adult": n_idx, "excluded_died_in_icu": int(died_icu.sum()),
            "excluded_not_to_ward": int((~died_icu & ~to_ward).sum())}
    df = df[~died_icu & to_ward].copy()
    df = df.sort_values("intime")
    df["first_icu_stay_patient"] = ~df.subject_id.duplicated()
    df["readmit"] = df.readmit.astype(int)
    df["hospital_expire_flag"] = df.hospital_expire_flag.astype(int)

    def rate(d):
        p, n = d.readmit.mean(), len(d)
        se = np.sqrt(p * (1 - p) / n)
        return {"n": n, "readmit_rate": round(float(p), 4), "ci95": [round(float(p - 1.96 * se), 4), round(float(p + 1.96 * se), 4)],
                "mortality_readmitted": round(float(d.loc[d.readmit == 1, "hospital_expire_flag"].mean()), 4),
                "mortality_not_readmitted": round(float(d.loc[d.readmit == 0, "hospital_expire_flag"].mean()), 4)}
    out = {"flow": flow, "all_hospitalizations": rate(df),
           "first_hospitalization_per_patient": rate(df[df.first_icu_stay_patient])}
    df[["stay_id", "first_icu_stay_patient", "hospital_expire_flag", "readmit"]].to_parquet(
        REF / "t5_readmission.parquet", index=False)
    return out


# ------------------------------------------------------------------------------------------------ t6
def t6(c) -> dict:
    df = c.execute("""
    WITH coh AS (SELECT * FROM base WHERE pt_seq = 1 AND age >= 18
                 AND NOT (los < 1 AND surgical = 1 AND admission_type IN ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION'))),
    rbc AS (SELECT b.stay_id, min(e.starttime) AS t_rbc
            FROM coh b JOIN inputevents e ON e.stay_id = b.stay_id
            WHERE e.itemid = 225168 AND e.amount > 0 AND e.starttime >= b.intime AND e.starttime <= b.outtime
            GROUP BY 1),
    hb AS (SELECT b.stay_id, l.charttime, l.valuenum
           FROM coh b JOIN labevents l ON l.hadm_id = b.hadm_id
           WHERE l.itemid IN (51222, 50811) AND l.valuenum > 0 AND l.valuenum < 25
             AND l.charttime >= b.intime - INTERVAL 6 HOUR AND l.charttime <= b.outtime + INTERVAL 24 HOUR),
    adm_hb AS (SELECT hb.stay_id, arg_min(hb.valuenum, hb.charttime) AS adm_hb FROM hb JOIN coh b USING (stay_id)
               WHERE hb.charttime <= b.intime + INTERVAL 24 HOUR GROUP BY 1),
    trig AS (SELECT r.stay_id, min(hb.valuenum) AS trigger_hb FROM rbc r JOIN hb USING (stay_id)
             WHERE CAST(hb.charttime AS DATE) = CAST(r.t_rbc AS DATE) GROUP BY 1)
    SELECT b.stay_id, b.age, b.male, b.hospital_expire_flag, b.intime, b.dischtime, b.deathtime,
           r.t_rbc, t.trigger_hb, a.adm_hb
    FROM coh b LEFT JOIN rbc r USING (stay_id) LEFT JOIN trig t USING (stay_id) LEFT JOIN adm_hb a USING (stay_id)
    """).df()
    df["first_icu_stay_patient"] = True
    df["rbc_any"] = df.t_rbc.notna()
    df["first_rbc_hours"] = (df.t_rbc - df.intime).dt.total_seconds() / 3600
    df["hospital_expire_flag"] = df.hospital_expire_flag.astype(int)
    tr = df[df.rbc_any]
    out = {"n_cohort": len(df), "prop_transfused": round(float(df.rbc_any.mean()), 4),
           "trigger_hb": {"n": int(tr.trigger_hb.notna().sum()), "mean": round(float(tr.trigger_hb.mean()), 2),
                          "sd": round(float(tr.trigger_hb.std()), 2), "median": round(float(tr.trigger_hb.median()), 2),
                          "iqr": [round(float(q), 2) for q in tr.trigger_hb.quantile([.25, .75])]},
           "first_rbc_hours_median": round(float(tr.first_rbc_hours.median()), 1),
           "mortality_transfused": round(float(tr.hospital_expire_flag.mean()), 4),
           "mortality_not_transfused": round(float(df.loc[~df.rbc_any, "hospital_expire_flag"].mean()), 4)}
    # time-dependent Cox: origin ICU intime, end = hospital discharge (death if hospital_expire_flag), split at first RBC
    d = df[df.adm_hb.notna()].copy()
    end = d.deathtime.where(d.hospital_expire_flag.eq(1) & d.deathtime.notna(), d.dischtime)
    d["T"] = ((end - d.intime).dt.total_seconds() / 3600).clip(lower=0.01)
    d["tr"] = d.first_rbc_hours.clip(upper=d["T"] - 0.001)   # transfusion recorded at/after end: count as exposed at end
    rows = []
    a = d[~d.rbc_any | (d.tr > 0)]
    pre = a.assign(start=0.0, stop=np.where(a.rbc_any, a.tr, a["T"]), rbc=0,
                   event=np.where(a.rbc_any, 0, a.hospital_expire_flag))
    rows.append(pre)
    b = d[d.rbc_any]
    rows.append(b.assign(start=b.tr.clip(lower=0), stop=b["T"], rbc=1, event=b.hospital_expire_flag))
    cp = pd.concat(rows, ignore_index=True)
    cp = cp[cp.stop > cp.start]
    X = cp[["rbc", "age", "male", "adm_hb"]].astype(float)
    m = PHReg(cp.stop.values, X, status=cp.event.values, entry=cp.start.values, ties="breslow").fit()
    i = list(X.columns).index("rbc")
    ci = m.conf_int()[i]
    out["cox_td"] = {"n_patients": int(d.stay_id.nunique()), "n_intervals": len(cp), "events": int(cp.event.sum()),
                     "covariates": ["age", "male", "adm_hb"],
                     "HR_rbc": [round(float(np.exp(m.params[i])), 3), round(float(np.exp(ci[0])), 3), round(float(np.exp(ci[1])), 3)]}
    df[["stay_id", "first_icu_stay_patient", "rbc_any", "first_rbc_hours", "trigger_hb", "hospital_expire_flag"]].to_parquet(
        REF / "t6_transfusion.parquet", index=False)
    return out


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    c = con()
    summary = {"note": "Own hand-written evaluator references (not published code). See README.md."}
    for name, f in [("t3_dysnatremia", t3), ("t5_readmission", t5), ("t6_transfusion", t6)]:
        summary[name] = f(c)
        print(name, json.dumps(summary[name], indent=1, default=str))
    OUT.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
