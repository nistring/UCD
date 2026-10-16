"""UCD command line.

  python -m ucd ["question" | protocol.txt] [--auto] [--gt cohort.csv]   # the whole pipeline (ucd/chat.py)

Low-level commands (the agent's data tools; also usable by hand):

  python -m ucd card    --dataset D                               # Data Card (schema + aggregates + docs)
  python -m ucd lookup  --dataset D TERM [--kind items|labs|dx|proc|all]
  python -m ucd explore --dataset D --run-dir R --from "SELECT ..." --group a,b --metric "avg(x) AS m"
  python -m ucd run     --dataset D --run-dir R --spec spec.yaml   # execute + critique + log one iteration
  python -m ucd status  --run-dir R [--dataset D]
  python -m ucd template                                           # empty DesignSpec
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from ucd import critic as C
from ucd import llm_critic
from ucd.blinding import blind_card
from ucd import datacard
from ucd.gate import DATASETS, ROOT, Gate, GateError
from ucd.runner import primary_value, run
from ucd.spec import TEMPLATE, load_spec

HIDDEN = ROOT / "eval" / "stays"
HIDDEN_CARDS = ROOT / "eval" / "cards"


def _standalone(spec, gate, idir: Path) -> dict:
    """A query that uses concept.* is also written with the concepts inlined, over the raw tables only, and
    bind-checked (E-59): an evaluation environment with only the original tables can run it unchanged."""
    from ucd import concepts
    if not concepts.referenced(spec.analysis_sql):
        return {}
    try:
        sql = concepts.inline(spec.analysis_sql, gate.dataset)
        idir.mkdir(parents=True, exist_ok=True)
        (idir / "analysis_standalone.sql").write_text(
            f"-- {spec.title}: analysis_sql with the concept tables inlined; uses only the dataset's raw tables\n{sql}\n",
            encoding="utf-8")
        problem = gate.bind_check(sql)
    except Exception as e:                             # never fail a run over the export
        problem = str(e).splitlines()[0][:200]
    return {"standalone_sql": f"{idir.name}/analysis_standalone.sql (raw tables only; "
                              f"{'binds' if not problem else 'DOES NOT BIND: ' + problem[:200]})"}


def _run_options(run_dir: Path) -> dict:
    opts = {}
    for name in ("critic_config.json", "run_config.json"):
        f = run_dir / name
        if f.exists():
            opts.update(json.loads(f.read_text(encoding="utf-8")))
    return opts


def _rel(run_dir: Path) -> Path:
    rd = run_dir.resolve()
    for base in (ROOT / "runs" / "ucd", ROOT):
        try:
            return rd.relative_to(base.resolve())
        except ValueError:
            pass
    import hashlib
    return Path(f"{rd.name}_{hashlib.sha1(str(rd).encode()).hexdigest()[:8]}")


def _hidden_path(run_dir: Path, dataset: str, it: int) -> Path:
    return HIDDEN / _rel(run_dir) / f"{dataset}_iter{it}.parquet"


def hidden_card_path(run_dir: Path, dataset: str, it: int) -> Path:
    return HIDDEN_CARDS / _rel(run_dir) / f"{dataset}_iter{it}.json"


def design_round(run_dir: Path) -> int:
    """The current design round: it starts at 0 and the loop starts a new one when the protocol changes or the user
    chooses to keep designing (meta.json `round`; E-60). The stop rule counts only this round's designs."""
    meta = run_dir / "meta.json"
    try:
        return int(json.loads(meta.read_text(encoding="utf-8")).get("round", 0)) if meta.exists() else 0
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0


def _history(ddir: Path) -> list[dict]:
    f = ddir / "iterations.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def _fmt_feedback(it: int, card: dict, crit: dict, stop: tuple[bool, str]) -> str:
    """The planner's view of one iteration. Findings and the stop line come FIRST: the loop keeps only the first
    part of each observation, and with the card first the findings were cut off (E-21)."""
    blocks = [f["rule"] for f in crit["findings"] if f["severity"] == "block"]
    L = [f"# Iteration {it}: internal error I = {crit['internal_error']}  (dimensions: {crit['dimensions']})",
         f"**Blocking findings to fix: {', '.join(blocks) if blocks else 'none'}**",
         f"**Stop suggested: {stop[0]}**" + (f" ({stop[1]})" if stop[0] else " - revise the spec and run again."), ""]
    if card.get("error"):
        L += [f"**Execution error:** {card['error']}", ""]
    else:
        L.append("## Flow")
        for s in card.get("flow", []):
            L.append(f"- {s['step']}: units={s['n_units']}, patients={s['n_patients']}")
        o = card.get("outcome", {})
        L += ["", f"Outcome: {o.get('events')} / {o.get('n')} = {o.get('rate')} (95% CI {o.get('ci')})",
              f"Primary estimate: {primary_value(card)}", ""]
    L += ["## Critic findings (deterministic rules + LLM clinical critic 'L*'; fix blocks first)"]
    for f in crit["findings"]:
        L.append(f"- [{f['severity'].upper()} {f['rule']}/{f['dimension']}] {f['message']}"
                 + (f" | evidence: {json.dumps(f['evidence'], default=str)[:300]}" if f.get("evidence") else "")
                 + (f" | fix: {f['fix']}" if f.get("fix") else ""))
    if not crit["findings"]:
        L.append("- none")
    if crit.get("waivers"):
        L += ["", "## Waiver decisions (independent judge + code checks)"]
        for w in crit["waivers"]:
            L.append(f"- {w['rule']}: {'APPLIED' if w['applied'] else 'REFUSED'}"
                     + (f" - {w['refused_because']}" if w.get("refused_because") else f" - quote: \"{w.get('quote', '')[:200]}\""))
    if card.get("sensitivity"):
        L += ["", "## Sensitivity analyses", "```json", json.dumps(card["sensitivity"], ensure_ascii=False, default=str)[:3000], "```"]
    if not card.get("error"):
        L += ["", "## Result card (aggregates; also in iter_<k>/card.json)", "```json",
              json.dumps({k: card[k] for k in card if k not in ("flow", "title", "dataset")}, ensure_ascii=False,
                         indent=1, default=str)[:12000], "```"]
    return "\n".join(L)


def cmd_run(a) -> int:
    run_dir = Path(a.run_dir)
    ddir = run_dir / a.dataset
    ddir.mkdir(parents=True, exist_ok=True)
    hist = _history(ddir)
    it = len(hist) + 1
    idir = ddir / f"iter_{it}"
    idir.mkdir(exist_ok=True)
    shutil.copy(a.spec, idir / "spec.yaml")
    t0 = time.time()
    try:
        spec = load_spec(a.spec, a.dataset)
    except Exception as e:
        card = {"error": f"spec is invalid: {str(e)[:1500]}"}
        crit = C._score(C._F(), hard=True)
        crit["findings"] = [{"rule": "V0", "dimension": "validity", "severity": "block", "message": card["error"],
                             "evidence": None, "fix": "Fix the YAML so it validates against the DesignSpec template."}]
        spec = None
    else:
        gate = Gate.shared(a.dataset, run_dir=run_dir)     # cached WITH steps carry over between iterations
        card, cohort = run(spec, gate)
        if gate.last_reused:
            card["steps_reused"] = gate.last_reused        # step names only: what this revision did not recompute
        card.update(_standalone(spec, gate, idir))
        pf = run_dir / "protocol_items.yaml"
        items = (yaml.safe_load(pf.read_text(encoding="utf-8")) or []) if pf.exists() else None
        if isinstance(items, dict):
            items = items.get("items", [])
        opts = _run_options(run_dir)
        frozen_f = ddir / "FROZEN.json"
        blinded = bool(opts.get("blind_outcome")) and not frozen_f.exists()
        opts["blind"] = blinded
        meta_f = run_dir / "meta.json"
        protocol = json.loads(meta_f.read_text(encoding="utf-8"))["question"] if meta_f.exists() else spec.question
        opts["protocol"] = protocol
        crit = C.critique(spec, card, hist, a.dataset, items, opts)
        full_card = card
        shown_card = blind_card(card) if blinded else card
        crit["options"] = opts
        # independent LLM clinical critic after every design
        rev = llm_critic.review(protocol, items, Path(a.spec).read_text(encoding="utf-8"), shown_card, crit["findings"],
                                [dict(h, primary=None) for h in hist] if blinded else hist,
                                llm_critic.context(a.dataset, protocol))
        (idir / "llm_critic.json").write_text(json.dumps(rev, ensure_ascii=False, indent=1), encoding="utf-8")
        crit = C.merge(crit, llm_critic.to_findings(rev))
        crit["llm_overall"] = rev.get("overall")
        crit["clinical_plausibility"] = rev.get("clinical_plausibility")
        if spec.waivers:
            crit, _ = C.apply_waivers(crit, spec, rev.get("waiver_decisions") or [], protocol, card)
        if items is None:
            crit["findings"].insert(0, {"rule": "P0", "dimension": "validity", "severity": "warn",
                                        "message": "protocol_items.yaml is missing from the run dir",
                                        "evidence": None, "fix": "Write the numbered protocol items first."})
        if frozen_f.exists():
            fz = json.loads(frozen_f.read_text(encoding="utf-8"))
            if Path(a.spec).read_text(encoding="utf-8") != fz.get("spec_text"):
                crit["findings"].append({"rule": "U1", "dimension": "validity", "severity": "warn",
                                         "message": "Design changed after unblinding: allowed only to fix a bug; "
                                                    "describe it in changes_from_previous (it is reported as a "
                                                    "post-unblinding change)", "evidence": None, "fix": ""})
        if blinded:
            hc = hidden_card_path(run_dir, a.dataset, it)
            hc.parent.mkdir(parents=True, exist_ok=True)
            hc.write_text(json.dumps(full_card, ensure_ascii=False, default=str), encoding="utf-8")
            card = shown_card
        if cohort is not None:                          # also when empty: an empty final cohort is the answer
            keep = [c for c in ["unit_id", "subject_key", "index_time", spec.outcome.column] +
                    ([spec.exposure.column] if spec.exposure else []) + spec.model.score_columns
                    if c in cohort.columns]
            hp = _hidden_path(run_dir, a.dataset, it)
            hp.parent.mkdir(parents=True, exist_ok=True)
            cohort[list(dict.fromkeys(keep))].astype({"unit_id": "string", "subject_key": "string"}).to_parquet(hp)
    # only designs that executed count toward the iteration limit and stability; failed attempts are capped apart
    rnd = design_round(run_dir)
    this_round = [h for h in hist if h.get("round", 0) == rnd]   # a protocol change starts a fresh budget (E-60)
    ok = [h for h in this_round if not (h.get("card") or {}).get("error")]
    if card.get("error"):
        stop = (True, f"reached {C.MAX_ATTEMPTS} attempts") if len(this_round) + 1 >= C.MAX_ATTEMPTS else (False, "")
    else:
        stop = C.should_stop([h["internal_error"] for h in ok] + [crit["internal_error"]], crit.get("n_block", 1))
        if stop[0] and crit.get("llm_overall") == "error" and len(ok) + 1 < C.MAX_ITER:
            stop = (False, "")                         # no clinical review this time: not a clean design (R4)
    frozen_f = ddir / "FROZEN.json"
    post_freeze = frozen_f.exists() and spec is not None
    if post_freeze:
        n_post = sum(1 for h in hist if h.get("post_freeze")) + 1
        stop = (True, "design frozen; unblinded result reported") if (crit.get("n_block", 0) == 0 or n_post >= 2) \
            else (False, "")
    elif spec is not None and _run_options(run_dir).get("blind_outcome") and stop[0]:
        stop = (True, "design stable - run `python -m ucd freeze` to lock it, then run once more to unblind")
    rec = {"iter": it, "scoring": C.SCORING_VERSION, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
           "internal_error": crit["internal_error"], "I_deterministic": crit.get("I_deterministic"),
           "llm_overall": crit.get("llm_overall"), "clinical_plausibility": crit.get("clinical_plausibility"),
           "waivers": crit.get("waivers"),
           "dimensions": crit["dimensions"], "n_block": crit.get("n_block"),
           "primary": primary_value(card), "n_final": card.get("n_final"),
           "events": (card.get("outcome") or {}).get("events"), "stop": stop[0], "stop_reason": stop[1],
           "changes_from_previous": spec.changes_from_previous if spec else "", "card": card, "critique": crit,
           "blinded": bool(card.get("blinded")), "post_freeze": post_freeze, "round": rnd}
    if rec["blinded"]:
        rec["primary"] = None
    with open(ddir / "iterations.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    (idir / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (idir / "critique.json").write_text(json.dumps(crit, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    fb = _fmt_feedback(it, card, crit, stop)
    (idir / "feedback.md").write_text(fb, encoding="utf-8")
    print(fb)
    return 0


def cmd_freeze(a) -> int:
    run_dir = Path(a.run_dir)
    ddir = run_dir / a.dataset
    hist = _history(ddir)
    if not hist:
        print("REFUSED: no iteration to freeze")
        return 2
    last = hist[-1]
    if (last.get("card") or {}).get("error") or any(f["severity"] == "block" and C.tier(f["rule"]) == 1
                                                    for f in last["critique"]["findings"]):
        print("REFUSED: the last iteration failed or has a mechanical (tier-1) block; fix it before freezing")
        return 2
    spec_text = (ddir / f"iter_{last['iter']}" / "spec.yaml").read_text(encoding="utf-8")
    (ddir / "FROZEN.json").write_text(json.dumps({"frozen_iter": last["iter"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "spec_text": spec_text}, ensure_ascii=False), encoding="utf-8")
    print(f"Design frozen at iteration {last['iter']}. Run the same spec once more to see the unblinded result. "
          "After unblinding, change the design only to fix a clear bug (logged as a post-unblinding change).")
    return 0


COMMANDS = {"card", "lookup", "explore", "run", "status", "freeze", "template"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:     # a question, a protocol file or chat options: the pipeline
        from ucd.chat import main as chat_main
        return chat_main(argv[1:] if argv and argv[0] == "chat" else argv)
    p = argparse.ArgumentParser(prog="ucd")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("card"); s.add_argument("--dataset", required=True, choices=list(DATASETS))
    s = sub.add_parser("lookup"); s.add_argument("--dataset", required=True, choices=list(DATASETS)); s.add_argument("term")
    s.add_argument("--kind", default="all")
    s = sub.add_parser("explore"); s.add_argument("--dataset", required=True, choices=list(DATASETS)); s.add_argument("--run-dir", required=True)
    s.add_argument("--from", dest="source", help="source SELECT (rows are never returned, only aggregated)")
    s.add_argument("--from-file", dest="source_file")
    s.add_argument("--group", default="", help="comma-separated column names of the source")
    s.add_argument("--metric", action="append", default=[], help="aggregate expression, e.g. 'avg(x) AS mean_x' (repeatable)")
    s = sub.add_parser("run"); s.add_argument("--dataset", required=True, choices=list(DATASETS)); s.add_argument("--run-dir", required=True)
    s.add_argument("--spec", required=True)
    s = sub.add_parser("status"); s.add_argument("--run-dir", required=True); s.add_argument("--dataset")
    s = sub.add_parser("freeze"); s.add_argument("--dataset", required=True, choices=list(DATASETS)); s.add_argument("--run-dir", required=True)
    sub.add_parser("template")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.cmd == "card":
        print(datacard.render(a.dataset))
    elif a.cmd == "lookup":
        print(datacard.lookup(a.dataset, a.term, a.kind))
    elif a.cmd == "explore":
        Path(a.run_dir).mkdir(parents=True, exist_ok=True)
        try:
            if not (a.source or a.source_file):
                raise GateError("give the source SELECT with --from or --from-file")
            g = Gate(a.dataset, run_dir=Path(a.run_dir))
            src = a.source if a.source else Path(a.source_file).read_text(encoding="utf-8")
            groups = [x.strip() for x in a.group.split(",") if x.strip()]
            out = g.aggregate(src, groups, a.metric)
        except GateError as e:
            print(f"REFUSED: {e}")
            return 2
        df = pd.DataFrame(out["rows"], columns=out["columns"]) if out["rows"] else pd.DataFrame(columns=out["columns"])
        print(df.to_markdown(index=False) if len(df) else "(no rows with n >= 10)")
        print(f"\n[suppressed rows (n<10): {out['suppressed_rows']}; truncated: {out['truncated']}; "
              f"budget left: {out.get('budget_left')}]")
    elif a.cmd == "run":
        return cmd_run(a)
    elif a.cmd == "freeze":
        return cmd_freeze(a)
    elif a.cmd == "status":
        rd = Path(a.run_dir)
        for ds in ([a.dataset] if a.dataset else sorted(p.name for p in rd.iterdir() if (p / "iterations.jsonl").exists())):
            for h in _history(rd / ds):
                print(f"{ds} iter {h['iter']}: I={h['internal_error']} blocks={h['n_block']} n={h['n_final']} "
                      f"events={h['events']} primary={h['primary']} stop={h['stop']}")
    elif a.cmd == "template":
        print(TEMPLATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
