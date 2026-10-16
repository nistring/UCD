"""External error E per iteration. Computed after the runs and NEVER shown to the planner.

E_paper : agreement of the agent's MIMIC numbers with the published paper (eval/papers.yaml)
          per comparison: ratio metrics (OR, rate) -> min(1, |ln(agent/paper)| / ln 2); AUROC -> min(1, |diff| / 0.10)
E_code  : agreement with published MIMIC-IV code (mimic-code reference cohorts, eval/mimiccode/ref)
          on the same data: 1 - Jaccard(unit sets), outcome-rate log ratio, exposure/score agreement
E_qual  : design-checklist agreement, scored by a separate evaluator (eval/qual_scores.json)

  python -m eval.evaluate            -> eval_errors.csv (+ .json)
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from ucd.critic import rescore

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "ucd"
EXT = ROOT / "eval"
STAYS = EXT / "stays"
REF = EXT / "mimiccode" / "ref"
PAPERS = yaml.safe_load((EXT / "papers.yaml").read_text(encoding="utf-8"))


def _f(x: Any) -> Optional[float]:
    if isinstance(x, str):
        return None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def ratio_err(a: Optional[float], p: Optional[float]) -> Optional[float]:
    if a is None or p is None or a <= 0 or p <= 0:
        return None
    return min(1.0, abs(math.log(a / p)) / math.log(2))


def auc_err(a: Optional[float], p: Optional[float]) -> Optional[float]:
    if a is None or p is None:
        return None
    return min(1.0, abs(a - p) / 0.10)


# ---------------------------------------------------------------------------------------------- card extractors
def _terms(card: dict) -> list[dict]:
    m = card.get("model") or {}
    return m.get("terms") or (m.get("logistic") or {}).get("terms") or []


def _level(term: str) -> str:
    return term.split("=", 1)[1].strip().lower() if "=" in term else ""


def _num_in(s: str) -> Optional[float]:
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def term_or(card: dict, pred) -> Optional[float]:
    for t in _terms(card):
        if pred(t["term"]):
            return _f(t.get("or"))
    return None


def outcome_rate(card: dict) -> Optional[float]:
    return _f((card.get("outcome") or {}).get("rate"))


def first_score_auc(card: dict, pat: str) -> Optional[float]:
    for s in (card.get("model") or {}).get("scores", []):
        if re.search(pat, s["score"], re.I):
            return _f(s.get("auroc"))
    return None


def exposure_levels(card: dict) -> dict[str, float]:
    """level -> share, from the categorical exposure's table1 entry (first categorical with levels)."""
    for col, d in (card.get("table1") or {}).items():
        lv = d.get("levels")
        if lv:
            tot = sum(_f(v["n"]) or 0 for v in lv.values())
            if tot:
                return {k: (_f(v["n"]) or 0) / tot for k, v in lv.items()}
    return {}


def _stage_pred(k: int):
    return lambda t: "=" in t and (_num_in(_level(t)) == k) and not re.search(r"no|none", _level(t))


def paper_comparisons(topic: str, card: dict) -> list[dict]:
    P = PAPERS[topic]
    kr = {r["name"]: r for r in P.get("key_results", [])}
    C: list[tuple[str, Optional[float], Optional[float], str]] = []   # name, agent, paper, kind
    if topic == "t1_aki":
        lv = exposure_levels(card)
        none = [v for k, v in lv.items() if _num_in(k) == 0 or re.search(r"no|none", k.lower())]
        C.append(("AKI incidence", (1 - none[0]) if none else None, 0.573, "ratio"))
        for k, nm in ((1, "stage 1"), (2, "stage 2"), (3, "stage 3")):
            C.append((f"aOR {nm}", term_or(card, _stage_pred(k)),
                      kr[f"Adjusted OR hospital death, {nm} vs no AKI"]["value"], "ratio"))
    elif topic == "t2_sofa":
        C.append(("AUROC SOFA", first_score_auc(card, "sofa") or _f((card.get("model") or {}).get("primary", {}).get("auroc")), 0.753, "auc"))
        C.append(("AUROC qSOFA", first_score_auc(card, "qsofa"), 0.607, "auc"))
        C.append(("AUROC SIRS", first_score_auc(card, "sirs"), 0.589, "auc"))
        C.append(("in-hospital mortality", outcome_rate(card), 0.187, "ratio"))
    elif topic == "t3_dysnatremia":
        names = {"severe hypo": (r"severe.*hypo|hypo.*severe|<\s*125|lt125", 1.81),
                 "mild hypo": (r"mild.*hypo|hypo.*mild|125", 1.89),
                 "borderline hypo": (r"border.*hypo|hypo.*border|130", 1.32),
                 "borderline hyper": (r"border.*hyper|hyper.*border|14[5-9]", 1.48),
                 "mild hyper": (r"mild.*hyper|hyper.*mild|15[0-5]", 2.32),
                 "severe hyper": (r"severe.*hyper|hyper.*severe|>\s*155|gt155", 3.64)}
        used = set()
        for nm, (pat, pv) in names.items():
            val = None
            for t in _terms(card):
                if "=" in t["term"] and t["term"] not in used and re.search(pat, _level(t["term"])):
                    val = _f(t.get("or"))
                    used.add(t["term"])
                    break
            C.append((f"aOR {nm}", val, pv, "ratio"))
        lv = exposure_levels(card)
        nrm = [v for k, v in lv.items() if re.search(r"normal|135", k.lower()) and not re.search(r"hypo|hyper", k.lower())]
        C.append(("prevalence normal", nrm[0] if nrm else None, 0.754, "ratio"))
    elif topic == "t4_ne_timing":
        m = card.get("model") or {}
        prim = m.get("primary") or {}
        cont = None
        for t in _terms(card):
            if "=" not in t["term"] and re.search(r"hour|delay|time|_h\b|hrs?", t["term"], re.I):
                cont = _f(t.get("or"))
                break
        if cont is None and prim and "=" not in prim.get("term", ""):
            cont = _f(prim.get("or"))
        C.append(("aOR per hour delay (Bai)", cont, 1.392, "ratio"))
        C.append(("aOR per hour delay (Beck)", cont, 1.02, "ratio"))
        C.append(("mortality", outcome_rate(card), 0.376, "ratio"))
    elif topic == "t5_readmission":
        C.append(("readmission rate", outcome_rate(card), 0.061, "ratio"))
    elif topic == "t6_transfusion":
        m = card.get("model") or {}
        trig = None
        for g in (m.get("by_group") or {}).values():
            trig = _f(g.get("median"))
            if trig:
                break
        if trig is None:
            for col, d in (card.get("table1") or {}).items():
                if re.search(r"trig", col, re.I) and d.get("median") is not None:
                    trig = _f(d["median"]); break
        C.append(("transfusion trigger Hb (median vs mean)", trig, 8.3, "ratio"))
        lv = exposure_levels(card)
        tr = [v for k, v in lv.items() if k.lower() in ("1", "1.0", "true", "yes", "transfused")]
        C.append(("proportion transfused", tr[0] if tr else None, 0.263, "ratio"))
        C.append(("in-hospital mortality", outcome_rate(card), 0.223, "ratio"))
        prim = (m.get("logistic") or m).get("primary") or {}
        C.append(("adjusted effect of transfusion (OR vs HR)", _f(prim.get("or")), 0.98, "ratio"))
    elif topic == "t7_meld":
        C.append(("AUROC MELD", first_score_auc(card, r"^meld$|meld(?!.*na)"), 0.749, "auc"))
        C.append(("AUROC MELD-Na", first_score_auc(card, r"meld.*na"), 0.716, "auc"))
        C.append(("AUROC SOFA", first_score_auc(card, "sofa"), 0.785, "auc"))
        C.append(("in-hospital mortality", outcome_rate(card), 0.523, "ratio"))
    out = []
    for nm, a, p, kind in C:
        e = ratio_err(a, p) if kind == "ratio" else auc_err(a, p)
        out.append({"name": nm, "agent": a, "paper": p, "err": e})
    return out


# ---------------------------------------------------------------------------------------------- code reference
_REF_CACHE: dict[str, Optional[pd.DataFrame]] = {}


def _ref(topic: str) -> Optional[pd.DataFrame]:
    if topic not in _REF_CACHE:
        name = {"t1_aki": "t1_aki", "t2_sofa": "t2_sofa", "t4_ne_timing": "t4_ne", "t7_meld": "t7_meld",
                "t3_dysnatremia": "t3_dysnatremia", "t5_readmission": "t5_readmission",   # own refs (eval/own_refs)
                "t6_transfusion": "t6_transfusion"}.get(topic, "none")
        f = REF / f"{name}.parquet"
        ref = pd.read_parquet(f) if f.exists() else None
        if ref is not None and topic == "t7_meld":
            # mimic-code MELD covers all stays: restrict to cirrhosis by standard ICD codes (evaluator's choice)
            dx = pd.read_parquet(ROOT / ".kcd_cache" / "parquet" / "mimic" / "hosp" / "diagnoses_icd.parquet",
                                 columns=["hadm_id", "icd_code", "icd_version"])
            c = dx["icd_code"].astype(str)
            cirr = ((dx.icd_version == 10) & c.str.match(r"^(K703|K717|K743|K744|K745|K746)")) |                    ((dx.icd_version == 9) & c.str.match(r"^(5712|5715|5716)"))
            hadms = set(dx.loc[cirr, "hadm_id"])
            ref = ref.merge(_stay_map()[["stay_id", "hadm_id"]], on="stay_id", how="left") if "hadm_id" not in ref else ref
            ref = ref[ref["hadm_id"].isin(hadms)]
        if ref is not None and topic == "t1_aki":
            # apply the protocol's eligibility to the reference: ICU stay >= 24 h and KDIGO data present
            icu = _stay_map()
            los = pd.read_parquet(ROOT / ".kcd_cache" / "parquet" / "mimic" / "icu" / "icustays.parquet", columns=["stay_id", "los"])
            ref = ref.merge(los, on="stay_id", how="left")
            ref = ref[(ref["los"] >= 1.0) & ref["max_kdigo_stage_7d"].notna()]
        if topic == "t4_ne_timing":
            ref = None   # mimic-code has no septic-shock-onset cohort: all sepsis-3 NE stays is not comparable
        _REF_CACHE[topic] = ref
    return _REF_CACHE[topic]


_ICU = None


def _stay_map() -> pd.DataFrame:
    global _ICU
    if _ICU is None:
        _ICU = pd.read_parquet(ROOT / ".kcd_cache" / "parquet" / "mimic" / "icu" / "icustays.parquet",
                               columns=["subject_id", "hadm_id", "stay_id", "intime"])
    return _ICU


def code_comparison(topic: str, stays_file: Path, card: dict) -> dict:
    ref = _ref(topic)
    if ref is None or not stays_file.exists():
        return {}
    ag = pd.read_parquet(stays_file)
    ids = pd.to_numeric(ag["unit_id"], errors="coerce").dropna().astype("int64")
    icu = _stay_map()
    ref = ref.copy()
    # restrict reference to first ICU stay if the reference carries the flag and the agent used first stays
    for c in ("first_icu_stay_patient", "first_icu_stay", "first_stay", "is_first_stay"):
        if c in ref.columns:
            ref = ref[ref[c].astype(bool)]
            break
    ref_ids = set(ref["stay_id"].astype("int64"))
    a_ids = set(ids)
    if len(a_ids & set(icu["stay_id"])) < 0.5 * max(len(a_ids), 1):          # agent unit = hadm_id?
        if len(a_ids & set(icu["hadm_id"])) > 0.5 * max(len(a_ids), 1):
            first = icu.sort_values("intime").drop_duplicates("hadm_id")
            a_ids = set(first[first["hadm_id"].isin(a_ids)]["stay_id"])
    inter = len(a_ids & ref_ids)
    union = len(a_ids | ref_ids)
    out = {"n_agent": len(a_ids), "n_ref": len(ref_ids), "jaccard": inter / union if union else None}
    yref = None
    for c in ("readmit", "hospital_expire_flag", "death", "outcome"):   # readmit: t5 own ref outcome
        if c in ref.columns:
            yref = ref[c].astype(float).mean()
            break
    out["ref_outcome_rate"] = yref
    out["rate_err"] = ratio_err(outcome_rate(card), yref)
    # unit-level exposure / score agreement on overlapping units
    ag2 = ag.copy()
    ag2["stay_id"] = pd.to_numeric(ag2["unit_id"], errors="coerce")
    mcol = {"t1_aki": "max_kdigo_stage_7d", "t2_sofa": "first_day_sofa", "t7_meld": "meld",
            "t4_ne_timing": "hours_from_icu_intime_to_ne",
            "t3_dysnatremia": "na_first_24h", "t5_readmission": None, "t6_transfusion": "trigger_hb"}
    rc = mcol.get(topic)
    acols = [c for c in ag2.columns if c not in ("unit_id", "subject_key", "stay_id", "outcome")]
    if rc and rc in ref.columns and acols:
        j = ag2.merge(ref[["stay_id", rc]].rename(columns={rc: "__ref"}), on="stay_id")
        best = None
        for c in acols:
            x = pd.to_numeric(j[c], errors="coerce")
            if x.notna().sum() < 30:   # categorical labels such as 'stage 2': use the first number in the label
                x = pd.to_numeric(j[c].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
                x = x.where(~j[c].astype(str).str.contains("no|none", case=False), 0)
            if x.notna().sum() < 30:
                continue
            r = x.corr(pd.to_numeric(j["__ref"], errors="coerce"), method="spearman")
            if r is not None and not math.isnan(r) and (best is None or abs(r) > abs(best[1])):
                best = (c, r)
        if best:
            out["agreement_col"], out["agreement_spearman"] = best[0], round(best[1], 3)
    parts = [1 - out["jaccard"] if out["jaccard"] is not None else None, out["rate_err"],
             (1 - max(0.0, out["agreement_spearman"])) if out.get("agreement_spearman") is not None else None]
    parts = [p for p in parts if p is not None]
    out["E_code"] = float(np.mean(parts)) if parts else None
    return out


# ---------------------------------------------------------------------------------------------- main
def _previous_code_scores() -> dict:
    """E_code of iterations whose stay sets were pruned (only final iterations keep stays) are reused from the
    last evaluation instead of being lost."""
    f = EXT / "external_errors.csv"
    if not f.exists():
        return {}
    old = pd.read_csv(f)
    cols = [c for c in ("E_code", "jaccard", "agreement_spearman") if c in old.columns]
    return {(r.topic, r.run, r.dataset, int(r.iter)): {c: getattr(r, c) for c in cols} for r in old.itertuples()}


def evaluate() -> pd.DataFrame:
    prev = _previous_code_scores()
    qual = {}
    for qf in sorted((EXT / "qual").glob("*.json")):
        for k, v in json.loads(qf.read_text(encoding="utf-8")).items():
            qual[k] = v.get("E_qual") if isinstance(v, dict) else v
    rows, detail = [], []
    for it_file in sorted(RUNS.glob("*/[rabcd][0-9]*/*/iterations.jsonl")):
        ds = it_file.parent.name
        run = it_file.parent.parent.name
        topic = it_file.parent.parent.parent.name
        for line in it_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            card = rec.get("card") or {}
            if rec.get("blinded"):   # outcome-blinded design phase: the full card lives only in the hidden folder
                hc = EXT / "cards" / topic / run / f"{ds}_iter{rec['iter']}.json"
                if hc.exists():
                    card = json.loads(hc.read_text(encoding="utf-8"))
            crit = rec.get("critique") or {}
            I_v2 = rescore(crit.get("findings", []), hard=bool(card.get("error")) or crit.get("hard", False),
                           gate=crit.get("gate", False)) if crit else rec["internal_error"]
            row = {"topic": topic, "run": run, "dataset": ds, "iter": rec["iter"], "I": I_v2,
                   "I_logged": rec["internal_error"], "scoring_logged": rec.get("scoring", "v1"),
                   "I_deterministic": rec.get("I_deterministic"), "clinical_plausibility": rec.get("clinical_plausibility"),
                   "n_block": rec.get("n_block"), "n_final": rec.get("n_final"), "primary": rec.get("primary")}
            if topic not in PAPERS:
                pass
            elif ds == "mimic" and not card.get("error"):
                comps = paper_comparisons(topic, card)
                errs = [c["err"] for c in comps if c["err"] is not None]
                row["E_paper"] = float(np.mean(errs)) if errs else None
                row["E_paper_coverage"] = f"{len(errs)}/{len(comps)}"
                sf = STAYS / topic / run / f"{ds}_iter{rec['iter']}.parquet"
                cc = code_comparison(topic, sf, card)
                if not sf.exists() and (topic, run, ds, int(rec["iter"])) in prev:
                    cc = {k: (None if pd.isna(v) else v) for k, v in prev[(topic, run, ds, int(rec["iter"]))].items()}
                row["E_code"] = cc.get("E_code")
                row["jaccard"] = cc.get("jaccard")
                row["agreement_spearman"] = cc.get("agreement_spearman")
                detail.append({**row, "paper_comparisons": comps, "code": cc})
            elif ds == "mimic":
                row["E_paper"] = 1.0
                row["E_code"] = 1.0 if _ref(topic) is not None else None
                row["failed"] = True
            q = qual.get(f"{topic}/{run}/{ds}/{rec['iter']}")
            row["E_qual"] = q
            parts = [row.get(k) for k in ("E_paper", "E_code", "E_qual") if row.get(k) is not None]
            row["E"] = float(np.mean(parts)) if parts else None
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(EXT / "external_errors.csv", index=False)
    (EXT / "external_errors_detail.json").write_text(json.dumps(detail, indent=1, default=str), encoding="utf-8")
    return df


if __name__ == "__main__":
    d = evaluate()
    print(d.to_string(index=False) if len(d) else "no runs yet")
