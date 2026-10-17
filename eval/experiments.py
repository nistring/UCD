"""Baselines and ablations: run sample studies under several agent configurations ("arms") and compare them.

  python -m eval.experiments list                                   the arms and what each one changes
  python -m eval.experiments run samples/example/t3_sodium_question [more samples] [--arms baselines|ablations|all|a,b,...]
                                 [--reps 1] [--out runs/experiments/<time>] [--max-turns 30]
  python -m eval.experiments summary runs/experiments/<time>        rebuild summary.md from results.csv
  python -m eval.experiments intake samples/tasks/*/* --models a,b  does each intake model flag the invalid tasks?

Planner comparison: --planner MODEL[:think] (repeatable) runs every arm once per planner setting, e.g.
  --planner glm-5-3-ax --planner glm-5-3-ax:think --planner qwen3-8-max-ax
(:think turns on the model's reasoning mode for the planner only, with a 16k reply budget). --jobs N runs N at once.

Every arm gets the same sample request and the same data access (the gate's aggregate API). Arms that use intake
share one protocol per sample, written once in auto mode (<out>/<sample>/protocol.md), so they differ only in what the
arm switches off. Each arm is a set of run_config.json switches that `ucd run` and the design loop honour:

  llm_critic: false     no LLM clinical critic (and no waiver judge)
  rule_feedback: false  the rule critic still scores every design, but its findings are kept on the evaluator side
  waivers: false        waiver requests are ignored
  explore: false        no lookup / explore actions (design from the Data Card only)
  concepts: false       no concept.* library and no topic packs
  notes: false          the Data Card and the critic get the schema only, without the dataset notes
  max_iter: N           at most N executed designs
  blind_outcome: true   outcome-blinded design phase (freeze, then unblind once)

The rule critic scores every executed design in every arm, never shown to the planner where the arm hides it, so
I_rule is comparable across arms. With a sample gt.csv the final cohort is also compared with the ground truth
(Jaccard, recall, precision). Results: <out>/results.csv (one row per run) and <out>/summary.md. Everything stays
aggregate; run folders are git-ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS: dict[str, dict[str, Any]] = {
    # baselines: the same model and data access with less structure around it
    "full": {"group": "baselines", "desc": "UCD as designed: intake, planner with tools, rule critic, LLM critic",
             "config": {}},
    "single_generator": {"group": "baselines", "raw": True, "generator": True,
                         "desc": "one LLM call: the request and the Data Card in, a design out; run once, no feedback",
                         "config": {"llm_critic": False, "rule_feedback": False, "max_iter": 1}},
    "single_agent": {"group": "baselines", "raw": True,
                     "desc": "one LLM agent with tools (lookup, explore, run) and no intake or critics; "
                             "it sees only execution results",
                     "config": {"llm_critic": False, "rule_feedback": False}},
    # ablations: the full system with one part switched off
    "no_intake": {"group": "ablations", "raw": True, "desc": "the raw request goes straight to the planner",
                  "config": {}},
    "no_llm_critic": {"group": "ablations", "desc": "rule critic only", "config": {"llm_critic": False}},
    "no_rule_feedback": {"group": "ablations", "desc": "LLM critic only (rule findings hidden from the planner)",
                         "config": {"rule_feedback": False}},
    "no_critics": {"group": "ablations", "desc": "intake and planner, no critic feedback",
                   "config": {"llm_critic": False, "rule_feedback": False}},
    "no_waivers": {"group": "ablations", "desc": "waiver requests ignored", "config": {"waivers": False}},
    "no_explore": {"group": "ablations", "desc": "no lookup / explore: design from the Data Card only",
                   "config": {"explore": False}},
    "no_concepts": {"group": "ablations", "desc": "no tested concept.* library and no topic packs",
                    "config": {"concepts": False}},
    "no_notes": {"group": "ablations", "desc": "schema only: no dataset notes for the planner and the critic",
                 "config": {"notes": False}},
    "one_design": {"group": "ablations", "desc": "a single executed design (critic feedback cannot be acted on)",
                   "config": {"max_iter": 1}},
    "blinded": {"group": "ablations", "desc": "outcome-blinded design phase (freeze, then unblind once)",
                "config": {"blind_outcome": True}},
    "same_model_critic": {"group": "ablations", "desc": "the clinical critic uses the planner's model",
                          "config": {}, "critic_is_planner": True},
}
GROUPS = {"baselines": [a for a, v in ARMS.items() if v["group"] == "baselines"],
          "ablations": ["full"] + [a for a, v in ARMS.items() if v["group"] == "ablations"],
          "all": list(ARMS)}
FIELDS = ["sample", "arm", "planner", "rep", "status", "seconds", "turns", "designs_run", "design_attempts", "n_final", "events",
          "primary", "I_rule", "rule_blocks", "verdict", "expected", "verdict_ok", "jaccard", "recall", "precision", "n_gt",
          "noop_streak", "bad_turns", "run_dir", "sample_dir"]

ONE_SHOT = """

## ONE-SHOT MODE (this run only; it overrides the procedure and the action protocol above)
You have no actions and no second turn: nothing is looked up, explored or revised. Reply ONCE with exactly two fenced
blocks and nothing else:
1. ```yaml with the whole design/spec.yaml (the DesignSpec fields, without analysis_sql);
2. ```sql with the whole design/query.sql (the analysis SQL, following the analysis_sql contract).
The design is checked and run exactly once, as written."""


# ------------------------------------------------------------------ arms
def arms_from(spec: str) -> list[str]:
    names = GROUPS.get(spec) or [a.strip() for a in spec.split(",") if a.strip()]
    unknown = [a for a in names if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {', '.join(unknown)}; see: python -m eval.experiments list")
    return names


def parse_one_shot(raw: str) -> tuple[str, str]:
    """(spec yaml, sql) from a one-shot reply; analysis_sql inside the yaml is used when the sql block is missing."""
    blocks = re.findall(r"```[ \t]*([A-Za-z]*)[ \t]*\n(.*?)```", raw or "", re.S)
    spec = next((b for k, b in blocks if k.lower() in ("yaml", "yml")), "")
    sql = next((b for k, b in blocks if k.lower() == "sql"), "")
    try:
        data = yaml.safe_load(spec) if spec.strip() else None
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict) and "analysis_sql" in data:
        sql = sql or str(data.pop("analysis_sql") or "")
        spec = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    return spec.strip() + "\n" if spec.strip() else "", sql.strip() + "\n" if sql.strip() else ""


def generate_once(run_dir: Path, question: str, dataset: str, llm) -> str:
    """The single-generator baseline: one call with the planner's own instructions and the Data Card, then one run."""
    from ucd import datacard, loop, verdict
    run_dir.mkdir(parents=True, exist_ok=True)
    loop._seed_design(run_dir)
    system = loop._system_prompt(run_dir, question, dataset) + ONE_SHOT
    user = f"Data Card ({dataset}):\n{datacard.render(dataset)}\n\nWrite the design now (two fenced blocks)."
    t0 = time.time()
    try:
        raw = llm.complete_text(system, user)
    except Exception as e:                              # a failed call is a result of this arm, not a crash
        raw, obs = "", f"LLM call failed: {type(e).__name__}: {str(e)[:300]}"
    else:
        spec, sql = parse_one_shot(raw)
        if spec:
            (run_dir / loop.SPEC_FILE).write_text(spec, encoding="utf-8")
        if sql:
            (run_dir / loop.SQL_FILE).write_text(sql, encoding="utf-8")
        obs = loop._act({"action": "run"}, run_dir, 0, dataset=dataset)
    loop._log(run_dir / "loop_log.jsonl", 0, raw, {"action": "run"}, obs, time.time() - t0)
    loop.write_auto_final(run_dir, "single-generator baseline: one LLM call, the design run once as written")
    verdict.finalize(run_dir)
    return obs


def sample_id(sample: Path) -> str:
    """A short unique name: the folder and its category (answerable/T003, pediatric/T003, dysnatremia/pressure)."""
    return "/".join(Path(sample).resolve().parts[-2:])


def planner_env(spec: str) -> tuple[str, dict[str, str]]:
    """'model[:think]' -> (label, environment for the planner role only); '' = the configured planner."""
    if not spec:
        return "", {}
    model, _, flag = spec.partition(":")
    think = flag.lower() in ("think", "thinking")
    env = {"UCD_PLANNER_MODEL": model, "UCD_PLANNER_THINKING": "1" if think else "0"}
    if think:
        env["UCD_PLANNER_MAX_TOKENS"] = "16384"         # reasoning text counts against the reply budget
    return model + ("+think" if think else ""), env


# ------------------------------------------------------------------ one run
def sample_inputs(sample: Path) -> tuple[str, Optional[Path], str]:
    from ucd import datasets
    from ucd.sample import find
    proposal, gt, conf = find(sample)
    request = proposal.read_text(encoding="utf-8").strip()
    return request, gt, conf.get("dataset") or datasets.default()


def expected_status(sample: Path) -> list[str]:
    """The verdicts a task sample accepts (sample.yaml expected_status); empty for an open study."""
    from ucd.sample import find
    return list(find(sample)[2].get("expected_status") or [])


def shared_protocol(request: str, dataset: str, folder: Path) -> tuple[Optional[str], str]:
    """The intake protocol for one sample, written once in auto mode and reused by every arm that has intake."""
    from ucd.ask import Asker
    from ucd.intake import IntakeStopped, intake
    from ucd.llm import LocalLLM
    f = folder / "protocol.md"
    if f.exists():
        return f.read_text(encoding="utf-8"), ""
    folder.mkdir(parents=True, exist_ok=True)
    try:
        protocol, _, answers = intake(request, LocalLLM(role="intake"), Asker("auto", log=folder / "questions.jsonl"),
                                      dataset=dataset)
    except IntakeStopped as e:
        return None, f"stopped at intake: {e}"
    f.write_text(protocol, encoding="utf-8")
    (folder / "intake_answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    return protocol, ""


def run_arm(arm: str, sample: Path, out: Path, rep: int, max_turns: int, llm=None, planner: str = "") -> dict[str, Any]:
    """One sample under one arm (and planner setting); returns its results row. `llm` replaces the planner (tests)."""
    from ucd import loop
    from ucd.ask import Asker
    from ucd.llm import LocalLLM, load_config, role_settings
    conf = ARMS[arm]
    request, gt, dataset = sample_inputs(sample)
    label, penv = planner_env(planner)
    sid = sample_id(sample)
    run_dir = out / sid / (f"{arm}@{label}" if label else arm) / f"rep{rep}"
    question, why = (request, "") if conf.get("raw") else shared_protocol(request, dataset, out / sid)
    t0 = time.time()
    allowed = expected_status(sample)
    if question is None:                               # stopped at intake: the shared protocol was refused
        v = "unanswerable"
        return {"sample": sid, "sample_dir": str(sample), "arm": arm, "planner": label, "rep": rep, "status": why,
                "verdict": v,
                "run_dir": str(run_dir),
                "expected": "|".join(allowed), "verdict_ok": (v in allowed) if allowed else None}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({"topic": sid, "run": run_dir.name, "question": question,
                                                   "mode": "auto", "dataset": dataset, "arm": arm},
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(conf["config"]), encoding="utf-8")
    env = dict(penv)
    saved = {k: os.environ.get(k) for k in list(env) + ["UCD_CRITIC_MODEL"]}
    os.environ.update(env)
    if conf.get("critic_is_planner"):
        os.environ["UCD_CRITIC_MODEL"] = role_settings(dict(load_config()["llm"]), "planner")["model"]
    try:
        if conf.get("generator"):
            generate_once(run_dir, question, dataset, llm or LocalLLM(role="planner"))
        else:
            loop.run_loop(sid, run_dir, [dataset], max_turns, question=question,
                          asker=Asker("auto", log=run_dir / "questions.jsonl"), llm=llm)
        status = "ok"
    except Exception as e:                              # one broken run must not stop the experiment
        status = f"error: {type(e).__name__}: {str(e)[:200]}"
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    row = {**collect(run_dir, dataset, gt), "sample": sid, "sample_dir": str(sample), "arm": arm, "planner": label, "rep": rep,
           "status": status,
           "seconds": round(time.time() - t0, 1), "run_dir": str(run_dir), "expected": "|".join(allowed)}
    row["verdict_ok"] = (row.get("verdict") in allowed) if allowed else None
    return row


def collect(run_dir: Path, dataset: str, gt: Optional[Path]) -> dict[str, Any]:
    """Aggregate outcomes of one run (evaluator side: hidden critiques and the ground truth are read here only)."""
    from ucd import cli
    hist = cli._history(run_dir / dataset)
    executed = [h for h in hist if not (h.get("card") or {}).get("error")]
    row: dict[str, Any] = {"design_attempts": len(hist), "designs_run": len(executed)}
    log = run_dir / "loop_log.jsonl"
    recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()] if log.exists() else []
    row["turns"] = len(recs)
    obs = [str(r.get("observation", "")) for r in recs]
    row["bad_turns"] = sum(o.startswith(("REFUSED", "INVALID", "ERROR", "LLM call failed")) for o in obs)
    streak = best = 0
    for o in obs:                                      # the longest run of actions that changed nothing
        streak = streak + 1 if o.startswith("NO CHANGE") else 0
        best = max(best, streak)
    row["noop_streak"] = best
    v = run_dir / "verdict.json"
    row["verdict"] = json.loads(v.read_text(encoding="utf-8")).get("status") if v.exists() else None
    if executed:
        last = executed[-1]
        hc = cli.hidden_critique_path(run_dir, dataset, last["iter"])
        crit = json.loads(hc.read_text(encoding="utf-8")) if hc.exists() else last.get("critique") or {}
        hcard = cli.hidden_card_path(run_dir, dataset, last["iter"])
        card = json.loads(hcard.read_text(encoding="utf-8")) if hcard.exists() else last.get("card") or {}
        from ucd.runner import primary_value
        row.update(n_final=card.get("n_final"), events=(card.get("outcome") or {}).get("events"),
                   primary=primary_value(card), I_rule=crit.get("I_deterministic", crit.get("internal_error")),
                   rule_blocks=sum(1 for f in crit.get("findings", [])
                                   if f.get("severity") == "block" and not str(f.get("rule", "")).startswith("L")))
    if gt is not None and executed:
        from eval.groundtruth import evaluate_gt
        try:
            c = evaluate_gt(run_dir, str(gt), log=lambda *_: None, dataset=dataset, save=False)["cohort"]
            row.update(jaccard=c.get("jaccard"), recall=c.get("recall"), precision=c.get("precision"), n_gt=c.get("n_gt"))
        except SystemExit:
            pass
    return row


# ------------------------------------------------------------------ results
def append_row(out: Path, row: dict[str, Any]) -> None:
    f = out / "results.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _num(x: Any) -> Optional[float]:
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _avg(rows: list[dict], k: str, nd: int = 3) -> str:
    vals = [v for v in (_num(r.get(k)) for r in rows) if v is not None]
    return f"{round(mean(vals), nd):g}" if vals else "-"


def summarize(out: Path) -> str:
    """summary.md: one line per arm (means over samples and repeats), then every run."""
    f = out / "results.csv"
    rows = list(csv.DictReader(open(f, encoding="utf-8"))) if f.exists() else []
    keys = list(dict.fromkeys((r["arm"], r.get("planner") or "") for r in rows))
    order = sorted(keys, key=lambda k: (list(ARMS).index(k[0]) if k[0] in ARMS else 99, keys.index(k)))
    L = [f"# Experiment summary ({len(rows)} runs)", "",
         "Means over samples and repeats. I_rule = the rule critic's internal error of the last executed design "
         "(0 = no findings), scored in every arm even where the planner did not see it. Jaccard / recall / precision "
         "compare the final cohort with the sample's gt.csv. Verdict ok = the run's verdict is one the task sample accepts; "
         "invalid rejected = the same, counted over tasks whose accepted verdicts exclude normal.", "",
         "| arm | planner | runs | design ran | designs | turns | bad turns | longest no-op streak | seconds | n final | "
         "I_rule | rule blocks | Jaccard | recall | precision | verdict ok | invalid rejected |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, pl in order:
        rs = [r for r in rows if r["arm"] == a and (r.get("planner") or "") == pl]
        ran = sum(1 for r in rs if (_num(r.get("designs_run")) or 0) > 0)
        judged = [r for r in rs if r.get("expected")]
        bad_q = [r for r in judged if "normal" not in r["expected"].split("|")]
        ok = lambda xs: f"{sum(str(r.get('verdict_ok')) == 'True' for r in xs)}/{len(xs)}" if xs else "-"
        L.append(f"| {a} | {pl or 'configured'} | {len(rs)} | {ran}/{len(rs)} | {_avg(rs, 'designs_run', 2)} | "
                 f"{_avg(rs, 'turns', 1)} | {_avg(rs, 'bad_turns', 1)} | {_avg(rs, 'noop_streak', 1)} | "
                 f"{_avg(rs, 'seconds', 0)} | {_avg(rs, 'n_final', 0)} | {_avg(rs, 'I_rule')} | "
                 f"{_avg(rs, 'rule_blocks', 2)} | {_avg(rs, 'jaccard')} | {_avg(rs, 'recall')} | {_avg(rs, 'precision')} | "
                 f"{ok(judged)} | {ok(bad_q)} |")
    L += ["", "## Arms", ""] + [f"- `{a}`: {ARMS[a]['desc']}" for a in dict.fromkeys(a for a, _ in order) if a in ARMS]
    L += ["", "## Every run", "", "| sample | arm | planner | rep | status | designs | n final | primary | I_rule | verdict | Jaccard |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['sample']} | {r['arm']} | {r.get('planner') or '-'} | {r['rep']} | {r['status']} | {r.get('designs_run') or '-'} | "
          f"{r.get('n_final') or '-'} | {r.get('primary') or '-'} | {r.get('I_rule') or '-'} | {r.get('verdict') or '-'} | "
          f"{r.get('jaccard') or '-'} |" for r in rows]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def intake_check(samples: list[Path], models: list[str], out: Path) -> str:
    """Intake alone, in auto mode, per intake model: does it stop at the invalid tasks and pass the valid ones?"""
    from ucd.ask import Asker
    from ucd.intake import IntakeStopped, intake
    from ucd.llm import LocalLLM
    rows = []
    for m in models:
        for s in samples:
            request, _, dataset = sample_inputs(s)
            allowed = expected_status(s)
            saved = os.environ.get("UCD_INTAKE_MODEL")
            os.environ["UCD_INTAKE_MODEL"] = m
            t0 = time.time()
            try:
                intake(request, LocalLLM(role="intake"), Asker("auto"), log=lambda *_: None, dataset=dataset)
                flagged, kind, why = False, "", ""
            except IntakeStopped as e:
                flagged, kind, why = True, e.kind, str(e)[:160]
            except Exception as e:                      # a gateway or parsing failure is a result too
                flagged, kind, why = None, "error", f"{type(e).__name__}: {str(e)[:120]}"
            finally:
                os.environ.pop("UCD_INTAKE_MODEL", None) if saved is None else os.environ.__setitem__("UCD_INTAKE_MODEL", saved)
            should = bool(allowed) and "normal" not in allowed
            rows.append({"model": m, "sample": sample_id(s), "should_flag": should, "flagged": flagged, "kind": kind,
                         "correct": None if flagged is None else flagged == should, "seconds": round(time.time() - t0, 1),
                         "why": why})
            print(f"  {m} {sample_id(s)}: should flag {should}, flagged {flagged} {kind}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "intake_check.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    L = ["| intake model | correct | invalid flagged | valid passed | errors | median seconds |", "|---|---|---|---|---|---|"]
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        bad = [r for r in rs if r["should_flag"]]
        good = [r for r in rs if not r["should_flag"]]
        secs = sorted(r["seconds"] for r in rs)
        L.append(f"| {m} | {sum(bool(r['correct']) for r in rs)}/{len(rs)} | {sum(r['flagged'] is True for r in bad)}/{len(bad)} | "
                 f"{sum(r['flagged'] is False for r in good)}/{len(good)} | {sum(r['flagged'] is None for r in rs)} | "
                 f"{secs[len(secs) // 2] if secs else '-'} |")
    text = "\n".join(L) + "\n"
    (out / "intake_check.md").write_text(text, encoding="utf-8")
    return text


def _parallel(jobs_list: list[tuple], out: Path, jobs: int, max_turns: int) -> None:
    """Each (sample, arm, rep, planner) in its own process; cores and memory are split between the jobs."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    from ucd.gate import cores, ram_bytes
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.setdefault("UCD_THREADS", str(max(1, cores() // jobs)))
    if ram_bytes() and "UCD_MEMORY" not in env:
        env["UCD_MEMORY"] = f"{max(1, int(0.6 * ram_bytes() / jobs / 2**30))}GB"
    logs = out / "logs"
    logs.mkdir(exist_ok=True)

    def one(job):
        s_, arm, rep, pl = job
        name = f"{sample_id(s_).replace('/', '__')}__{arm}__{(pl or 'configured').replace(':', '-')}__rep{rep}"
        cmd = [sys.executable, "-m", "eval.experiments", "one", str(s_), "--arm", arm, "--rep", str(rep),
               "--out", str(out), "--max-turns", str(max_turns)] + (["--planner", pl] if pl else [])
        with open(logs / f"{name}.log", "w", encoding="utf-8") as fh:
            subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, cwd=str(ROOT))
        text = (logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        line = next((l for l in reversed(text.splitlines()) if l.startswith("ROW ")), None)
        return json.loads(line[4:]) if line else {"sample": sample_id(s_), "arm": arm, "planner": planner_env(pl)[0],
                                                     "rep": rep, "status": "error: no result (see logs/)"}

    with ThreadPoolExecutor(jobs) as ex:
        for row in ex.map(one, jobs_list):
            append_row(out, row)
            print(f"  {row['sample']} / {row['arm']} / {row.get('planner') or 'configured'}: {row['status']}; "
                  f"verdict {row.get('verdict')}; Jaccard {row.get('jaccard')}")


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval.experiments", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the arms")
    r = sub.add_parser("run", help="run samples under several arms")
    r.add_argument("sample", nargs="+")
    r.add_argument("--arms", default="baselines", help="baselines | ablations | all | a comma-separated list")
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--out", default=None, help="default runs/experiments/<date_time>")
    r.add_argument("--max-turns", type=int, default=None)
    r.add_argument("--planner", action="append", default=None, help="MODEL[:think]; repeat to compare planners")
    r.add_argument("--jobs", type=int, default=1, help="runs at once (each in its own process)")
    s = sub.add_parser("summary", help="rebuild summary.md"); s.add_argument("out")
    i = sub.add_parser("intake", help="intake only: does each model flag the invalid tasks?")
    i.add_argument("sample", nargs="+"); i.add_argument("--models", required=True); i.add_argument("--out", required=True)
    o = sub.add_parser("one", help=argparse.SUPPRESS)
    o.add_argument("sample"); o.add_argument("--arm", required=True); o.add_argument("--rep", type=int, default=1)
    o.add_argument("--out", required=True); o.add_argument("--max-turns", type=int, required=True)
    o.add_argument("--planner", default="")
    a = p.parse_args(argv)
    if a.cmd == "one":
        os.environ.setdefault("UCD_CRITIC_BACKEND", "openai_compat")
        row = run_arm(a.arm, Path(a.sample), Path(a.out), a.rep, a.max_turns, planner=a.planner)
        print("ROW " + json.dumps(row, default=str))
        return 0
    if a.cmd == "intake":
        print(intake_check([Path(x) for x in a.sample], [m.strip() for m in a.models.split(",") if m.strip()],
                           Path(a.out)))
        return 0
    if a.cmd == "list":
        for g in ("baselines", "ablations"):
            print(f"\n{g}:")
            for n in GROUPS[g]:
                cfg = ARMS[n]["config"]
                print(f"  {n:18} {ARMS[n]['desc']}" + (f"  {json.dumps(cfg)}" if cfg else ""))
        return 0
    if a.cmd == "summary":
        print(summarize(Path(a.out)))
        return 0
    from ucd import datasets, loop
    os.environ.setdefault("UCD_CRITIC_BACKEND", "openai_compat")      # the critic uses the configured gateway model
    arms, samples = arms_from(a.arms), [Path(x) for x in a.sample]
    for s_ in samples:
        datasets.require_data(sample_inputs(s_)[2])                  # before any LLM call
    out = Path(a.out) if a.out else ROOT / "runs" / "experiments" / time.strftime("%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    planners = a.planner or [""]
    todo = [(s_, arm, rep, pl) for s_ in samples for rep in range(1, a.reps + 1) for pl in planners for arm in arms]
    print(f"{len(todo)} runs ({len(samples)} sample(s) x {len(arms)} arm(s) x {len(planners)} planner setting(s) x "
          f"{a.reps} repeat(s)) -> {out}\nEach run calls the LLM gateway many times; stop with Ctrl+C (finished runs "
          "are kept).")
    if any(not ARMS[arm].get("raw") for arm in arms):  # the shared intake protocols first, once per sample
        for s_ in samples:
            request, _, dataset = sample_inputs(s_)
            shared_protocol(request, dataset, out / sample_id(s_))
    max_turns = a.max_turns or loop.MAX_TURNS
    if a.jobs > 1:
        _parallel(todo, out, a.jobs, max_turns)
    else:
        for k, (s_, arm, rep, pl) in enumerate(todo, 1):
            print(f"\n[{k}/{len(todo)}] {sample_id(s_)} / {arm} / {pl or 'configured planner'} / rep {rep}")
            row = run_arm(arm, s_, out, rep, max_turns, planner=pl)
            append_row(out, row)
            print(f"  {row['status']}; designs run {row.get('designs_run')}; n {row.get('n_final')}; "
                  f"I_rule {row.get('I_rule')}; Jaccard {row.get('jaccard')}")
    print("\n" + summarize(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
