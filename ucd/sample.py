"""Sample runs: one folder per study, holding the protocol, the ground truth and (after a run) the results.

  samples/<name>/
    protocol.txt                      the study: a one-line question or a full protocol (free text; this name only)
    gt.csv                            the ground-truth cohort: one row per unit, an id column (e.g. stay_id) and
                                      an optional 0/1 outcome column. CSV only, so no LLM ever reads the reference.
    sample.yaml                       optional: {dataset: mimic, blind: false, max_turns: 30,
                                      gt: <csv elsewhere>, expected_status: [normal, ...]} (task cases)
    result/                           written by a run (an older result/ is kept as result_<date_time>/)
      run/            every file of the agent (protocol, iterations, critic feedback, logs)
      final.md        the agent's summary (or an AUTO one written from the logs)
      report.md       analysis: the question, each design choice and its reason, how the design changed,
                      the result, and the comparison with the ground truth
      gt_comparison.json
      verdict.json    status (normal / partial / unanswerable / contradictory / empty_result), reason, limitation
      cohort.csv      the final cohort's ids and index_time (evaluator side; real ids, git-ignored)
      task_score.json verdict and cohort against the task's expected status and cohort (task cases)

  python -m ucd samples/<name> [--auto]          run one sample
  python -m ucd.sample run samples/a samples/b --jobs 2   several samples in parallel (auto mode)
  python -m ucd.sample report samples/<name>     rebuild report.md from an existing result/ (no LLM calls)

The ground truth is read only after the design loop has finished, and only by the evaluator: the agent never
sees it (it can neither read files nor receive these numbers).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_FILE = "protocol.txt"            # the one input name (user, 2026-10-17)
OTHER_NAMES = ("question.txt", "question.yaml", "question.yml", "question.md", "protocol.yaml", "protocol.yml",
               "protocol.md")
GT_FILE = "gt.csv"
REJECTED = ("gt.txt", "gt.md", "gt.yaml", "gt.yml", "gt.json", "gt.sql", "gt.py", "gt.tsv")   # not accepted


def find(sample: Path) -> tuple[Path, Path | None, dict]:
    q = sample / PROTOCOL_FILE
    if not q.is_file():
        other = [f for f in OTHER_NAMES if (sample / f).is_file()]
        raise SystemExit(f"{sample}: needs {PROTOCOL_FILE}"
                         + (f" (found {', '.join(other)}: save it as text under the name {PROTOCOL_FILE})" if other else ""))
    bad = [f for f in REJECTED if (sample / f).is_file()]
    if bad:
        print(f"{sample.name}: ignoring {', '.join(bad)}; the ground truth must be gt.csv")
    conf = yaml.safe_load((sample / "sample.yaml").read_text(encoding="utf-8")) if (sample / "sample.yaml").is_file() else {}
    conf = conf or {}
    gt = sample / GT_FILE if (sample / GT_FILE).is_file() else None
    if gt is None and conf.get("gt"):                  # a CSV kept elsewhere, e.g. a task's answer key
        ref = (sample / str(conf["gt"])).resolve()
        gt = ref if ref.is_file() and ref.suffix.lower() == ".csv" else None
    return q, gt, conf


def run_sample(sample: Path, auto: bool | None = None, extra: list[str] | None = None, archive: bool = True,
               dataset: str | None = None) -> Path:
    from ucd import chat, datasets
    q, gt, conf = find(sample)
    dataset = dataset or conf.get("dataset") or datasets.default()
    chat._ensure_data(dataset, bool(auto if auto is not None else conf.get("auto")))   # before touching result/
    res = sample / "result"
    if archive and res.exists():                       # keep earlier results
        res.rename(sample / f"result_{time.strftime('%Y%m%d_%H%M%S', time.localtime(res.stat().st_mtime))}")
    run = res / "run"
    argv = [str(q), "--run-dir", str(run), "--dataset", dataset]
    if auto if auto is not None else conf.get("auto"):
        argv.append("--auto")
    if conf.get("blind"):
        argv.append("--blind")
    if conf.get("max_turns"):
        argv += ["--max-turns", str(conf["max_turns"])]
    print(f"Sample {sample.name}: question {q.name}; ground truth: {gt.name if gt else 'none'}")
    t0 = time.time()
    chat.main(argv + list(extra or []))
    if (run / "final.md").exists():
        shutil.copy(run / "final.md", res / "final.md")
    comparison = evaluate(run, gt, dataset)
    if comparison and comparison.get("cohort"):
        chat._diag_gt(sample.name, run, comparison)
    (res / "gt_comparison.json").write_text(json.dumps(comparison, indent=1, default=str), encoding="utf-8")
    finish_task(res, gt, conf, dataset)
    path = write_report(sample, comparison, round(time.time() - t0))
    print(f"\nReport: {path}")
    return path


def finish_task(res: Path, gt: Path | None, conf: dict, dataset: str) -> dict | None:
    """Copy the verdict, export the final cohort (evaluator side) and, for a task case, score both against the
    expected status and cohort (task_score.json). Counts and statuses only; the ids stay in cohort.csv."""
    run = res / "run"
    if (run / "verdict.json").exists():
        shutil.copy(run / "verdict.json", res / "verdict.json")
    try:
        from eval.groundtruth import export_cohort
        export_cohort(run, res / "cohort.csv", dataset)
    except SystemExit:                                 # no design produced a cohort
        (res / "cohort.csv").unlink(missing_ok=True)
    if not conf.get("expected_status"):
        return None
    score = task_score(res, gt, conf)
    (res / "task_score.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
    return score


def task_score(res: Path, gt: Path | None, conf: dict) -> dict:
    """status in the allowed list, and the cohort compared as a set on (stay_id, index_time) when one is expected."""
    import pandas as pd
    v = json.loads(_read(res / "verdict.json") or "{}")
    allowed = list(conf.get("expected_status") or [])
    out = {"status": v.get("status"), "allowed_status": allowed, "status_ok": v.get("status") in allowed,
           "cohort_expected": gt is not None}
    if gt is None:
        out["passed"] = out["status_ok"]
        return out
    exp = pd.read_csv(gt)
    got = pd.read_csv(res / "cohort.csv") if (res / "cohort.csv").exists() else pd.DataFrame(columns=exp.columns)
    key = [c for c in ("stay_id", "index_time") if c in exp.columns and c in got.columns]
    E = set(map(tuple, exp[key].astype(str).values)) if key else set()
    G = set(map(tuple, got[key].astype(str).values)) if key else set()
    both = len(E & G)
    out.update({"compared_on": key, "n_expected": len(E), "n_agent": len(G), "n_both": both,
                "jaccard": round(both / len(E | G), 4) if E | G else 1.0,
                "recall": round(both / len(E), 4) if E else None, "precision": round(both / len(G), 4) if G else None,
                "cohort_exact": E == G})
    out["passed"] = out["status_ok"] and out["cohort_exact"]
    return out


def evaluate(run: Path, gt: Path | None, dataset: str) -> dict | None:
    """The run's cohort against the sample's gt.csv (None without one)."""
    if gt is None:
        return None
    from eval.groundtruth import evaluate_gt
    print(f"\nGround truth {gt.name} (evaluator side; the agent never sees it):")
    try:
        return {"file": gt.name, **evaluate_gt(run, str(gt), dataset=dataset, save=False)}
    except SystemExit as e:
        print(f"  not possible: {e}")
        return {"file": gt.name, "error": str(e)}


# ====================================================================== report
def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1000 else f"{x:,.0f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def _cell(s: Any, n: int = 160) -> str:
    return " ".join(str(s or "").split()).replace("|", "/")[:n]


def write_report(sample: Path, comparison: dict | None = None, seconds: int | None = None) -> Path:
    from ucd import datasets
    res = sample / "result"
    run = res / "run"
    meta = json.loads(_read(run / "meta.json") or "{}")
    q, _, conf = find(sample)
    dataset = conf.get("dataset") or datasets.default()
    hist = _jsonl(run / dataset / "iterations.jsonl")
    ok = [h for h in hist if not (h.get("card") or {}).get("error")]
    fin = ok[-1] if ok else None
    spec = yaml.safe_load(_read(run / dataset / f"iter_{fin['iter']}" / "spec.yaml") or "{}") if fin else {}
    qa = _jsonl(run / "questions.jsonl")
    log = _jsonl(run / "loop_log.jsonl")
    if comparison is None:
        comparison = json.loads(_read(res / "gt_comparison.json") or "null")
        if isinstance(comparison, list):               # results written before 0.6.1 hold a list of one
            comparison = comparison[0] if comparison else None
    try:
        from ucd.llm import ROLES, load_config, role_settings
        cfg = load_config()["llm"]
        models = ", ".join(f"{r} {role_settings(cfg, r)['model']}" for r in ROLES)
    except Exception:
        models = "(LLM settings unavailable)"

    L = [f"# Report: {sample.name}", "",
         f"Dataset: {datasets.get(dataset)['label']} · mode: {meta.get('mode', '–')} · models: {models}"
         + (f" · wall time {seconds // 60} min {seconds % 60} s" if seconds else "")
         + f" · agent steps {len(log)} · design iterations {len(hist)} ({len(ok)} executed)", ""]

    L += ["## 1. Research question", "", f"From `{q.name}`:", "", "> " + _read(q).replace("\n", "\n> "), ""]

    L += ["## 2. Agreed protocol and the choices made at intake", "",
          "The protocol below is the gold standard the agent had to follow. It was produced by the intake step "
          "from the question above, the dataset schema and these answers:", ""]
    intake_q = [x for x in qa if x.get("source") == "intake"]
    if intake_q:
        L += ["| Question | Options | Chosen | By |", "|---|---|---|---|"]
        L += [f"| {_cell(x['question'])} | {_cell('; '.join(x.get('options') or []), 200)} | {_cell(x.get('answer'))} | "
              f"{'agent (auto, recommended option)' if x.get('how') == 'auto' else 'user'} |" for x in intake_q]
    else:
        L.append("(no questions were needed)")
    L += ["", "<details><summary>Full protocol</summary>", "", "```text", _read(run / "protocol.md"), "```",
          "</details>", ""]

    L += ["## 3. The final design and the reason for each choice", ""]
    notes = _read(run / "design_notes.md")
    if notes:
        L += ["**The agent's design notes** (written before the first run):", "", "> " + notes.replace("\n", "\n> "), ""]
    if spec:
        L += ["| Element | Choice |", "|---|---|",
              f"| Unit of analysis | {_cell(spec.get('unit_of_analysis'))} |",
              f"| Time zero | {_cell(spec.get('index_time'), 300)} |"]
        for p in spec.get("population_criteria") or []:
            if isinstance(p, dict):
                L.append(f"| Population: `{p.get('flag')}` | {_cell(p.get('description'), 300)} |")
        ex, oc = spec.get("exposure") or {}, spec.get("outcome") or {}
        if ex:
            L.append(f"| Exposure `{ex.get('column')}` | {_cell(ex.get('description'), 300)} |")
        L.append(f"| Outcome `{oc.get('column')}` | {_cell(oc.get('description'), 300)} |")
        for c in spec.get("covariates") or []:
            if isinstance(c, dict):
                L.append(f"| Covariate `{c.get('column')}` | {_cell(c.get('description'), 200)} "
                         f"(window {c.get('window_start_h')}–{c.get('window_end_h')} h) |")
        m = spec.get("model") or {}
        L.append(f"| Model | {m.get('type')}" + (f", primary term `{m.get('primary_term')}`" if m.get("primary_term") else "")
                 + (", patient-clustered SE" if m.get("cluster_by_patient") else "") + " |")
        for s in spec.get("sensitivity") or []:
            if isinstance(s, dict):
                L.append(f"| Sensitivity analysis | {_cell(s.get('name'))} (rows where `{s.get('flag')}`) |")
        L.append("")
        dev = [p for p in spec.get("protocol_mapping") or [] if isinstance(p, dict) and str(p.get("deviation") or "").strip()]
        if dev:
            L += ["**Deviations from the protocol, with the agent's reasons:**", "", "| Item | Implemented as | Deviation / reason |",
                  "|---|---|---|"]
            L += [f"| {p.get('item')} | {_cell(p.get('implemented_as'))} | {_cell(p.get('deviation'), 300)} |" for p in dev]
            L.append("")
        else:
            L += ["The agent declared no deviation from the protocol.", ""]
        L += undeclared(spec, _read(run / "protocol.md"), dev)
    else:
        L += ["No design executed, so there is no final design.", ""]
    if fin and fin.get("waivers"):
        L += ["**Rule waivers** (granted only when the independent judge quoted the protocol and code verified it):", ""]
        L += [f"- {w.get('rule')}: {'applied' if w.get('applied') else 'refused'} - "
              + _cell(w.get("quote") or w.get("refused_because"), 250) for w in fin["waivers"]]
        L.append("")
    loop_q = [x for x in qa if x.get("source") == "loop"]
    if loop_q:
        L += ["**Questions the agent asked during the design:**", ""]
        L += [f"- {_cell(x['question'], 250)} → {_cell(x.get('answer'))} "
              f"({'agent decided' if x.get('how') == 'auto' else 'user'})" for x in loop_q]
        L.append("")

    L += ["## 4. How the design changed, and why", "",
          "I is the internal error (0 = best); it never rewards effect size. 'Change' is the agent's own reason.", "",
          "| Iter | I | Blocks | n | Events | Primary | Change and reason | Blocking findings |", "|---|---|---|---|---|---|---|---|"]
    for h in hist:
        blocks = [f"{f.get('rule')}: {f.get('message')}" for f in (h.get("critique") or {}).get("findings", [])
                  if f.get("severity") == "block"]
        err = (h.get("card") or {}).get("error")
        L.append(f"| {h['iter']} | {_fmt(h.get('internal_error'))} | {h.get('n_block')} | {_fmt(h.get('n_final'))} | "
                 f"{_fmt(h.get('events'))} | {_fmt(h.get('primary'))} | "
                 f"{_cell(h.get('changes_from_previous') or ('(did not run: ' + _cell(err, 90) + ')' if err else ''), 220)} | "
                 f"{_cell('; '.join(blocks), 220)} |")
    if not hist:
        L.append("| – | – | – | – | – | – | no design was submitted | – |")
    L.append("")

    L += ["## 5. Result", ""]
    if fin:
        card = fin.get("card") or {}
        L += ["| Step | Units | Patients |", "|---|---|---|"]
        L += [f"| {_cell(s.get('step'))} | {_fmt(s.get('n_units'))} | {_fmt(s.get('n_patients'))} |" for s in card.get("flow", [])]
        o = card.get("outcome") or {}
        L += ["", f"Outcome: {_fmt(o.get('events'))} / {_fmt(o.get('n'))} = {_fmt(o.get('rate'))} (95% CI {o.get('ci')})"]
        mdl = card.get("model") or {}
        p = mdl.get("primary") or (mdl.get("logistic") or {}).get("primary") or {}
        if p:
            est = next((k for k in ("or", "hr", "auroc", "value") if p.get(k) is not None), None)
            L.append(f"**Primary estimate:** {(est or '').upper()} {_fmt(p.get(est))}"
                     + (f" (95% CI {_fmt(p.get('ci_low'))}–{_fmt(p.get('ci_high'))})" if p.get("ci_low") is not None else "")
                     + (f", term `{p.get('term')}`" if p.get("term") else ""))
        terms = mdl.get("terms") or []
        if terms:
            L += ["", "| Term | Estimate | 95% CI | p |", "|---|---|---|---|"]
            L += [f"| {t.get('term')} | {_fmt(t.get('or', t.get('hr')))} | {_fmt(t.get('ci_low'))}–{_fmt(t.get('ci_high'))} | "
                  f"{_fmt(t.get('p'))} |" for t in terms[:15]]
        if card.get("blinded"):
            L.append("\n(The result card was outcome-blinded at this iteration.)")
    else:
        L.append("No design executed: there is no result. See section 4 and run/loop_log.jsonl.")
    L.append("")

    L += ["## 6. Comparison with the ground truth", ""]
    c = comparison or {}
    if not c:
        L += ["No ground truth was given for this sample.", ""]
    elif c.get("error"):
        L += [f"`{c['file']}`: not possible: {c['error']}", ""]
    else:
        L += [f"Ground truth: `{c['file']}`.", ""]
        k = c.get("cohort")
        if k:
            L += [f"Cohort compared at the {k['compared_at']} level (agent: {k['agent_level']}, ground truth: "
                  f"{k['gt_level']}; id column `{k['mapping']['id_column']}`):", "",
                  "| n agent | n ground truth | both | agent only | ground truth only | Jaccard | recall | precision |",
                  "|---|---|---|---|---|---|---|---|",
                  f"| {_fmt(k['n_agent'])} | {_fmt(k['n_gt'])} | {_fmt(k['n_both'])} | {_fmt(k['n_agent_only'])} | "
                  f"{_fmt(k['n_gt_only'])} | {_fmt(k['jaccard'])} | {_fmt(k['recall'])} | {_fmt(k['precision'])} |", ""]
            if "outcome_rate_gt" in k:
                L += [f"Outcome rate: agent {_fmt(k['outcome_rate_agent'])}, ground truth {_fmt(k['outcome_rate_gt'])}.", ""]
        L += ["**Analysis:**", ""] + [f"- {x}" for x in analyse(c, (fin or {}).get("card", {}).get("flow") or [])] + [""]

    v = json.loads(_read(res / "verdict.json") or "{}")
    ts = json.loads(_read(res / "task_score.json") or "{}")
    if v or ts:
        L += ["## 6b. Verdict" + (" and task score" if ts else ""), ""]
        if v:
            L += [f"- **Status:** `{v.get('status')}` (set by {v.get('source', '?')})",
                  f"- **Reason:** {_cell(v.get('reason'), 400) or '–'}", f"- **Limitation:** {_cell(v.get('limitation'), 400) or '–'}"]
        if ts:
            L.append(f"- **Expected status:** {', '.join(ts.get('allowed_status') or [])}: "
                     f"{'matches' if ts.get('status_ok') else 'does NOT match'}")
            if ts.get("cohort_expected"):
                L.append(f"- **Cohort** (set on {', '.join(ts.get('compared_on') or [])}): agent {ts.get('n_agent')}, expected "
                         f"{ts.get('n_expected')}, both {ts.get('n_both')}; Jaccard {ts.get('jaccard')}, recall "
                         f"{ts.get('recall')}, precision {ts.get('precision')}; exact match: {'yes' if ts.get('cohort_exact') else 'no'}")
            else:
                L.append("- **Cohort:** none expected for this task (only the status is scored)")
            L.append(f"- **Result: {'PASS' if ts.get('passed') else 'FAIL'}**")
        L.append("")

    L += ["## 7. Remaining issues and limitations", ""]
    if fin:
        open_f = [f for f in (fin.get("critique") or {}).get("findings", []) if f.get("severity") in ("block", "warn")]
        L += [f"- [{f.get('severity')}] {f.get('rule')}: {_cell(f.get('message'), 300)}" for f in open_f[:10]] or \
             ["- No open blocking or warning findings at the final iteration."]
        if fin.get("llm_overall"):
            L.append(f"- Independent clinical critic, overall: {_cell(fin['llm_overall'], 500)}")
    auto_q = [x for x in qa if x.get("how") == "auto" and x.get("source") in ("intake", "loop")]
    if auto_q:
        L.append(f"- {len(auto_q)} choice(s) were made by the agent in auto mode (marked 'agent' above); "
                 "review them before relying on the result.")
    L += ["", "## 8. Files", "", "- `final.md`: the agent's own summary for clinicians",
          "- `run/`: protocol, design notes, every iteration's spec, result card, critic feedback; `loop_log.jsonl` "
          "is the step-by-step audit trail", "- `gt_comparison.json`: the numbers of section 6", ""]
    path = res / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


LANDMARK_WORDS = ("landmark", "alive at", "survived", "survival to", "alive_")


def undeclared(spec: dict, protocol: str, declared: list[dict]) -> list[str]:
    """Design choices that change the population or the outcome window but are neither in the protocol nor declared
    as a deviation (rules, no LLM). The classic one: a survival landmark the protocol never asked for."""
    said = (protocol + " " + " ".join(str(d.get("deviation")) for d in declared)).lower()
    flags = [p for p in spec.get("population_criteria") or [] if isinstance(p, dict)
             and any(w in f"{p.get('flag')} {p.get('description')}".lower() for w in LANDMARK_WORDS)]
    ofh = float(spec.get("outcome_from_h") or 0)
    if (flags or ofh > 0) and not any(w in said for w in ("landmark", "alive", "surviv")):
        what = ", ".join(f"`{p.get('flag')}` ({_cell(p.get('description'), 80)})" for p in flags) or \
            f"outcomes counted only from {ofh:g} h"
        return [f"**Possible undeclared deviation:** the design adds a survival landmark ({what}) that the protocol "
                "does not mention and the agent did not declare. It removes early deaths and changes the population "
                "the estimate refers to.", ""]
    return []


def analyse(c: dict, flow: list[dict] | None = None) -> list[str]:
    """Plain-language reading of a ground-truth comparison (rules, no LLM)."""
    out = []
    k = c.get("cohort")
    if k and k.get("jaccard") is not None:
        if k["jaccard"] >= 0.9:
            out.append(f"The cohorts are nearly identical (Jaccard {_fmt(k['jaccard'])}).")
        else:
            if k["recall"] is not None and k["recall"] < 0.9:
                out.append(f"The agent missed {_fmt(k['n_gt_only'])} ground-truth units (recall {_fmt(k['recall'])}): "
                           "its inclusion criteria are stricter, or it requires data the ground truth does not.")
                # which flow step first fell below the ground-truth size: the most likely culprit
                if k.get("compared_at") == k.get("agent_level"):
                    prev = None
                    for s in flow or []:
                        if prev is not None and s.get("n_units") is not None and s["n_units"] < k["n_gt"] <= prev["n_units"]:
                            out.append(f"The step '{_cell(s.get('step'), 120)}' reduced the cohort from "
                                       f"{_fmt(prev['n_units'])} to {_fmt(s['n_units'])}, below the ground truth's "
                                       f"{_fmt(k['n_gt'])}: compare this criterion with the ground truth's definition first.")
                            break
                        prev = s
            if k["precision"] is not None and k["precision"] < 0.9:
                out.append(f"The agent included {_fmt(k['n_agent_only'])} units outside the ground truth (precision "
                           f"{_fmt(k['precision'])}): its criteria are broader.")
        if k.get("outcome_rate_gt") is not None and k.get("outcome_rate_agent") is not None and \
                abs(k["outcome_rate_gt"] - k["outcome_rate_agent"]) > 0.02:
            out.append("The outcome rates of the two cohorts differ by more than 2 points.")
    if not out:
        out.append("Nothing to compare.")
    out.append("A difference is not automatically an error of the agent: the ground truth may use another "
               "population, period or definition. Section 3 lists the agent's choices and reasons.")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ucd.sample", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or more sample folders")
    r.add_argument("sample", nargs="+"); r.add_argument("--auto", action="store_true")
    r.add_argument("--jobs", type=int, default=1, help="samples run in parallel (auto mode; cores and memory are split)")
    r.add_argument("--no-archive", action="store_true", help=argparse.SUPPRESS)
    w = sub.add_parser("report", help="rebuild report.md from an existing result/"); w.add_argument("sample")
    a = p.parse_args(argv)
    if a.cmd == "run" and a.jobs > 1 and len(a.sample) > 1:
        run_parallel([Path(s) for s in a.sample], a.jobs)
    elif a.cmd == "run":
        for s in a.sample:
            run_sample(Path(s), True if a.auto else None, archive=not a.no_archive)
    else:
        rebuild(Path(a.sample))
    return 0


def run_parallel(samples: list[Path], jobs: int) -> None:
    """Several samples at once, each in its own process (auto mode, console in result/console.log). Cores and
    memory are split between the jobs unless UCD_THREADS / UCD_MEMORY are set."""
    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    from ucd.gate import cores, ram_bytes
    jobs = min(jobs, len(samples))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("UCD_THREADS", str(max(1, cores() // jobs)))
    if ram_bytes() and "UCD_MEMORY" not in env:
        env["UCD_MEMORY"] = f"{max(1, int(0.6 * ram_bytes() / jobs / 2**30))}GB"
    print(f"{len(samples)} samples, {jobs} at a time; per job: {env['UCD_THREADS']} threads, "
          f"memory {env.get('UCD_MEMORY', 'DuckDB default')}")

    def one(s: Path) -> str:
        find(s)                                        # fail early on a bad folder
        res = s / "result"
        if res.exists():
            res.rename(s / f"result_{time.strftime('%Y%m%d_%H%M%S', time.localtime(res.stat().st_mtime))}")
        res.mkdir(parents=True)
        t0 = time.time()
        with open(res / "console.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, "-m", "ucd.sample", "run", str(s), "--auto", "--no-archive"],
                                 stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        return f"{s.name}: exit {rc}, {round(time.time() - t0)} s, report {res / 'report.md'}"

    with ThreadPoolExecutor(jobs) as ex:
        for line in ex.map(one, samples):
            print(line)


def rebuild(sample: Path) -> Path:
    """report.md for an existing result/ (no LLM calls)."""
    from ucd import datasets
    _, gt, conf = find(sample)
    res = sample / "result"
    if not (res / "run").is_dir():
        raise SystemExit(f"{sample}: no result/run folder; run the sample first")
    if (res / "run" / "final.md").exists():
        shutil.copy(res / "run" / "final.md", res / "final.md")
    if not (res / "gt_comparison.json").exists():
        cmp = evaluate(res / "run", gt, conf.get("dataset") or datasets.default())
        (res / "gt_comparison.json").write_text(json.dumps(cmp, indent=1, default=str), encoding="utf-8")
    path = write_report(sample)
    print(path)
    return path


if __name__ == "__main__":
    sys.exit(main())
