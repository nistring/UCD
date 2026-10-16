"""Compare a finished run with a user-supplied ground-truth cohort (evaluator side, never fed back to the agent).

The ground truth is a CSV file and nothing else, so the evaluation is fully deterministic (no LLM reads it):
  gt.csv      one row per unit: an id column named after the dataset's unit ids (MIMIC-IV: stay_id, hadm_id or
              subject_id; common aliases such as icustay_id are accepted), optionally a 0/1 outcome column
              (outcome, death, hospital_expire_flag, mortality, ...). A single unnamed id column is also accepted.

Cohorts are compared at a common unit level (fine -> coarse, from `units` in ucd/datasets.yaml): n, overlap,
Jaccard, recall (share of the ground truth the agent found), precision, outcome rates and agreement.

  python -m eval.groundtruth --run-dir <run> --gt gt.csv [--dataset D]
Results are printed and saved to eval/gt/<run>.json, outside the run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ucd import datasets                            # noqa: E402
from ucd.cli import _hidden_path, _history, _rel   # noqa: E402
from ucd.gate import Gate                           # noqa: E402

OUT = ROOT / "eval" / "gt"
LEVELS = datasets.LEVELS
GENERIC_ALIASES = {"icu_stay": ["stay_id", "icustay_id", "icu_stay_id", "stayid", "icustay", "patientunitstayid"],
                   "admission": ["hadm_id", "hadmid", "admission_id", "hospital_admission_id", "encounter_id",
                                 "patienthealthsystemstayid"],
                   "patient": ["subject_id", "subjectid", "patient_id", "patientid", "person_id", "uniquepid"]}
OUTCOME_ALIASES = ["outcome", "hospital_expire_flag", "in_hospital_death", "inhospital_death", "hosp_death", "death",
                   "died", "mortality", "expire_flag", "label", "y", "target", "event"]

# ------------------------------------------------------------------ dataset units
def units(dataset: str) -> dict[str, dict[str, str]]:
    """level -> {id, table}, fine -> coarse, for the levels the dataset defines."""
    u = datasets.get(dataset).get("units") or {}
    return {lvl: u[lvl] for lvl in LEVELS if lvl in u}


def links(con, dataset: str) -> dict[str, pd.DataFrame]:
    """level -> its unit table, reduced to the id columns of that level and of every coarser level."""
    u = units(dataset)
    ids = [v["id"] for v in u.values()]
    out = {}
    for lvl, v in u.items():
        if not v.get("table"):
            continue
        cols = [r[0] for r in con.execute(f"DESCRIBE {v['table']}").fetchall()]
        keep = [c for c in ids if c in cols]
        out[lvl] = con.execute(f"SELECT DISTINCT {', '.join(keep)} FROM {v['table']}").df()
    return out


def to_level(ids: pd.Series, frm: str, to: str, link: dict[str, pd.DataFrame], u: dict) -> set:
    """Map ids of level `frm` to the coarser level `to` through frm's unit table."""
    ids = pd.to_numeric(ids, errors="coerce").dropna().astype("int64")
    if frm == to:
        return set(ids)
    a, b = u[frm]["id"], u[to]["id"]
    mp = link[frm][[a, b]].drop_duplicates()
    return set(pd.to_numeric(mp[mp[a].isin(set(ids))][b], errors="coerce").dropna().astype("int64"))


def guess_level(ids: pd.Series, link: dict[str, pd.DataFrame], u: dict) -> str:
    """Which id space a bare id list lives in: the level whose ids overlap it most."""
    s = set(pd.to_numeric(ids, errors="coerce").dropna().astype("int64"))
    return max(u, key=lambda l: len(s & set(link[l][u[l]["id"]])) if l in link else -1)


# ------------------------------------------------------------------ cohort ground truth (CSV)
def load_cohort(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() != ".csv":
        raise SystemExit(f"ground truth {p.name}: only a .csv cohort is accepted")
    return pd.read_csv(p)


def map_columns(df: pd.DataFrame, u: dict | None = None) -> dict[str, Any]:
    u = u if u is not None else units(datasets.default())
    cols = {c.lower().strip(): c for c in df.columns}
    m: dict[str, Any] = {"id_column": None, "level": None, "outcome_column": None}
    for lvl in LEVELS:                                     # finest level present wins
        names = ([u[lvl]["id"]] if lvl in u else []) + GENERIC_ALIASES[lvl]
        hit = next((cols[a.lower()] for a in names if a.lower() in cols), None)
        if hit:
            m.update(id_column=hit, level=lvl)
            break
    if m["id_column"] is None and len(df.columns) == 1:
        m.update(id_column=df.columns[0], level=None)       # a bare list of ids: level guessed later
    m["outcome_column"] = next((cols[a] for a in OUTCOME_ALIASES if a in cols), None)
    if m["id_column"] is None:
        raise SystemExit(f"cannot find an id column in the ground truth (columns: {list(df.columns)}); "
                         f"name it {' / '.join(v['id'] for v in u.values())}")
    return m


def agent_cohort(run_dir: Path, dataset: str) -> tuple[pd.DataFrame, str, int]:
    for h in reversed(_history(run_dir / dataset)):
        f = _hidden_path(run_dir, dataset, h["iter"])
        if f.exists():
            spec = run_dir / dataset / f"iter_{h['iter']}" / "spec.yaml"
            lvl = (yaml.safe_load(spec.read_text(encoding="utf-8")) or {}).get("unit_of_analysis", "icu_stay") \
                if spec.exists() else "icu_stay"
            return pd.read_parquet(f), lvl if lvl in LEVELS else "icu_stay", h["iter"]
    raise SystemExit(f"no stored cohort for {run_dir} (no iteration produced one)")


def compare(run_dir: Path, gt_df: pd.DataFrame, mapping: dict, link: dict, u: dict, dataset: str) -> dict[str, Any]:
    ag, ag_lvl, it = agent_cohort(run_dir, dataset)
    gt_ids = pd.to_numeric(gt_df[mapping["id_column"]], errors="coerce").dropna().astype("int64")
    gt_lvl = mapping["level"] or guess_level(gt_ids, link, u)
    common = max(ag_lvl, gt_lvl, key=LEVELS.index)                       # coarser of the two
    A = to_level(ag["unit_id"], ag_lvl, common, link, u)
    G = to_level(gt_ids, gt_lvl, common, link, u)
    inter, union = len(A & G), len(A | G)
    out: dict[str, Any] = {
        "iteration": it, "agent_level": ag_lvl, "gt_level": gt_lvl, "compared_at": common, "mapping": mapping,
        "n_agent": len(A), "n_gt": len(G), "n_both": inter, "n_agent_only": len(A - G), "n_gt_only": len(G - A),
        "jaccard": round(inter / union, 4) if union else None,
        "recall": round(inter / len(G), 4) if G else None,
        "precision": round(inter / len(A), 4) if A else None}
    oc = mapping.get("outcome_column")
    if oc and "outcome" in ag.columns:
        g = pd.DataFrame({"id": gt_ids.values, "y_gt": pd.to_numeric(gt_df.loc[gt_ids.index, oc], errors="coerce").values})
        a = pd.DataFrame({"id": pd.to_numeric(ag["unit_id"], errors="coerce"),
                          "y_agent": pd.to_numeric(ag["outcome"], errors="coerce")})
        out["outcome_rate_gt"] = round(float(g["y_gt"].mean()), 4) if len(g) else None
        out["outcome_rate_agent"] = round(float(a["y_agent"].mean()), 4) if len(a) else None
        if gt_lvl == ag_lvl:
            j = a.merge(g, on="id").dropna()
            out["outcome_agreement_on_overlap"] = round(float((j.y_agent == j.y_gt).mean()), 4) if len(j) else None
    return out


# ------------------------------------------------------------------ entry points
def report(res: dict[str, Any]) -> str:
    c = res.get("cohort")
    if not c:
        return "(no ground truth compared)"
    L = [f"Cohort vs ground truth (iteration {c['iteration']}, compared at {c['compared_at']} level; columns "
         f"mapped: id={c['mapping']['id_column']}, outcome={c['mapping'].get('outcome_column')})",
         f"  agent n = {c['n_agent']}, ground truth n = {c['n_gt']}, both = {c['n_both']}",
         f"  Jaccard = {c['jaccard']}   recall (GT found) = {c['recall']}   precision = {c['precision']}"]
    if "outcome_rate_gt" in c:
        L.append(f"  outcome rate: agent {c['outcome_rate_agent']} vs ground truth {c['outcome_rate_gt']}"
                 + (f"; agreement on shared units {c['outcome_agreement_on_overlap']}"
                    if c.get("outcome_agreement_on_overlap") is not None else ""))
    return "\n".join(L)


def export_cohort(run_dir: Path, out: Path, dataset: Optional[str] = None) -> int:
    """The run's final cohort as CSV: every registry id of its unit level, coarse to fine (e.g. subject_id, hadm_id,
    stay_id), plus index_time as YYYY-MM-DD HH:MM:SS. Evaluator side: the agent never sees it, and the file holds
    real ids (keep it local). Returns the number of rows."""
    dataset = dataset or datasets.default()
    ag, lvl, _ = agent_cohort(run_dir, dataset)
    u = units(dataset)
    key = u[lvl]["id"]
    df = pd.DataFrame({key: pd.to_numeric(ag["unit_id"], errors="coerce").astype("Int64")})
    df["index_time"] = (pd.to_datetime(ag["index_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                        if "index_time" in ag.columns else None)
    if len(df):
        link = links(Gate(dataset).con, dataset).get(lvl)
        if link is not None:
            df = df.merge(link.astype({c: "Int64" for c in link.columns}), on=key, how="left")
    cols = [u[l]["id"] for l in reversed(LEVELS) if l in u and u[l]["id"] in df.columns]
    df = df[cols + ["index_time"]].sort_values(key)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return len(df)


def evaluate_gt(run_dir: Path, gt: str, log=print, dataset: Optional[str] = None, save: bool = True) -> dict[str, Any]:
    dataset = dataset or datasets.default()
    res: dict[str, Any] = {"run": str(run_dir), "dataset": dataset, "gt_source": str(gt)}
    df = load_cohort(gt)
    u = units(dataset)
    g = Gate(dataset)                                   # evaluator side: plain views, nothing is shown to the agent
    res["cohort"] = compare(run_dir, df, map_columns(df, u), links(g.con, dataset), u, dataset)
    if save:
        f = OUT / f"{_rel(run_dir).as_posix().replace('/', '__')}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    log(report(res))
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval.groundtruth", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gt", required=True, help="ground-truth cohort as CSV (unit id column, optional 0/1 outcome)")
    p.add_argument("--dataset", help="default: the first dataset in ucd/datasets.yaml")
    a = p.parse_args(argv)
    evaluate_gt(Path(a.run_dir), a.gt, dataset=a.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
