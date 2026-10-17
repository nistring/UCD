"""Design loop (0.13.0): a gateway chat model designs the cohort the way the 0.0.0 Claude Code subagent did, by
editing files and running `ucd run` on them, instead of resending the whole design in every reply.

  design/spec.yaml   the DesignSpec fields (no SQL)
  design/query.sql   analysis_sql

Every prompt shows both files as they are now, with line numbers. The model changes them with write or edit (a line
range), and run checks and executes them. Only the gate and the runner touch data.

  python -m ucd.loop --topic t3_dysnatremia --run-dir runs/ucd/t3_dysnatremia/gw1 [--auto] [--max-turns 30]
  python -m ucd.loop --finalize --run-dir runs/ucd/t3_dysnatremia/gw1      # write final.md from the logs

The model may ask the user (action "ask": 2-3 choices plus "edit the protocol myself"; A = the agent decides). Every model reply and observation is
logged to <run_dir>/loop_log.jsonl. If the model ends without final.md, the loop writes one from the logs (AUTO).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from ucd import __version__, datacard, diag, verdict
from ucd.actions import parse_action
from ucd.actions import why_unparsable as _why_unparsable
from ucd.ask import AUTO, MAX_OPTIONS, Asker, clean_options
from ucd.cli import _run_options
from ucd.cli import main as ucd_main
from ucd.critic import MAX_ITER
from ucd.gate import ROOT, Gate, GateError
from ucd.llm import LocalLLM, citations, load_config

WRAP_UP_TURNS = 4          # from this many turns before the limit, the prompt asks for final.md
MAX_TURNS = 30
MAX_ASKS = 5               # questions to the user per run
FALLBACK_DESIGNS = 2       # more designs after one that cannot answer the question, before the user is asked
PROBLEM_OPTIONS = ["Stop: report that this study cannot be done as designed", "Keep designing (the agent tries again)",
                   "Edit the protocol, then keep designing"]
EDIT_OWN = "Edit the protocol myself (opens the editor; my text answers the question)"
STOP_HINT = "To stop the design loop early: type q and press Enter (checked after each step), or press Ctrl+C (at once)."
MAX_LOOKUPS = 12           # dictionary lookups per run, like the gate's budget for aggregate queries
LOOKUP_SYNONYMS = {"lab": "labs", "labitems": "labs", "laboratory": "labs", "item": "items", "chart": "items",
                   "vitals": "items", "diagnosis": "dx", "diagnoses": "dx", "icd": "dx", "procedure": "proc",
                   "procedures": "proc"}
SPEC_FILE, SQL_FILE = "design/spec.yaml", "design/query.sql"
WRITABLE = {"protocol_items.yaml", "design_notes.md", "final.md", "verdict.json", SPEC_FILE, SQL_FILE}
SHOW_LINES = 400           # lines of each design file shown in every prompt

ACTIONS = """\
Reply with exactly ONE JSON object per turn. File content and SQL never go inside the JSON: put them in ONE fenced
block (```sql, ```yaml or ```) right after the JSON. Example (replace lines 12-14 of the query):
{"action": "edit", "file": "design/query.sql", "lines": [12, 14]}
```sql
    AND m.event_time BETWEEN u.start_time AND u.start_time + INTERVAL 24 HOUR
```
Available actions:
{"action": "lookup", "term": ["sodium", "creatinine"], "kind": "all"}      item/diagnosis codes, up to 5 terms; kind: all, or a dictionary named in the Data Card
{"action": "explore", "group": ["col"], "metrics": ["avg(x) AS mean_x"]}   + block = the FROM query; returns aggregates only
{"action": "write", "file": "design/query.sql"}                           + block = the whole new content of the file
{"action": "edit", "file": "design/query.sql", "lines": [a, b]}           + block = what replaces lines a..b as numbered
    in "Your design files"; an empty block deletes them. Prefer edit to rewriting a file.
{"action": "read", "file": "<path in your run directory>"}                e.g. a feedback.md of an earlier iteration
{"action": "run"}                                                          check and execute design/spec.yaml + design/query.sql (one design iteration)
{"action": "status"}                                                       iteration history
{"action": "freeze"}                                                       lock the design (when the run output says so)
{"action": "ask", "question": "...", "options": ["...", "..."], "recommended": 0}   ask the user (max 5 per run, 2-3 options;
    the user may also answer by editing the protocol):
    to settle a protocol definition that is ambiguous or cannot be implemented as written: the question and the
    answer are added to the protocol. Never for SQL errors or things lookup/explore can check. The answer may be
    AUTO: then decide and record why.
{"action": "infeasible", "reason": "...", "status": "unanswerable"}       status: unanswerable|contradictory|partial|empty_result;
    the protocol cannot be answered with this data (exposure or outcome not recorded, impossible population,
    tautology, leakage, reverse causation): the user decides
{"action": "done"}                                                         only after final.md is written
Files you may write: design/spec.yaml, design/query.sql, protocol_items.yaml, design_notes.md, final.md, verdict.json
(and other top-level .md/.txt notes)."""


def _capture(argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            ucd_main(argv)
        except SystemExit:
            pass
    return buf.getvalue()


def _inside(run_dir: Path, name: str) -> tuple[Path, str | None]:
    """(path, relative posix name) of a file inside the run directory; the name is None for a path outside it."""
    f = (run_dir / str(name)).resolve()
    return f, f.relative_to(run_dir.resolve()).as_posix() if run_dir.resolve() in f.parents else None


def _check_content(run_dir: Path, rel: str, content: str) -> str:
    """Why this content may not be written to `rel` ('' when it may). The same checks for write and edit."""
    meta = run_dir / "meta.json"                         # the user's protocol may cite papers; the model may not
    protocol = json.loads(meta.read_text(encoding="utf-8")).get("question", "") if meta.exists() else ""
    cites = citations(content, allowed=protocol)
    if cites:                                            # no appeals to authority in anything the planner writes
        return (f"REFUSED: remove the references to papers or authors ({'; '.join(cites[:5])}). Justify with the "
                "protocol, the run results and explicit reasoning instead.")
    prob = next((p for p in map(lambda d: design_problem(run_dir, d), _datasets(run_dir)) if p), "")
    if rel == "final.md" and prob:
        return (f"REFUSED: the last design cannot answer the question ({prob}). Go back to the design: change it "
                "within the protocol and run again, or use {\"action\": \"infeasible\", \"reason\": ...}.")
    if rel == "final.md" and _n_runs(run_dir) == 0:
        return ("REFUSED: no design has been run yet, so final.md would have no result. Write design/spec.yaml and "
                "design/query.sql, then {\"action\": \"run\"}.")
    if rel == "verdict.json":
        try:
            verdict.parse(content)
        except ValueError as e:
            return f"REFUSED: {e}"
    if rel == "protocol_items.yaml" and not _items_ok(content):   # every run reads it (E-66)
        return ("REFUSED: protocol_items.yaml must be a YAML list of items, one per line, e.g.\n"
                "- {id: P1, kind: population, text: \"Adults (age >= 18) at admission\"}\n"
                "- {id: P2, kind: outcome, text: \"In-hospital mortality\"}\n"
                "Quote the text when it contains a colon.")
    if rel in (SPEC_FILE, SQL_FILE) and re.search(r"^\s*```", content, re.M):
        return (f"REFUSED: the content for {rel} holds a ``` fence line: write one file per action, and put only that "
                "file's content in the block; not changed")
    if rel == SPEC_FILE:
        try:
            spec = yaml.safe_load(content)
        except yaml.YAMLError as e:
            return f"REFUSED: {SPEC_FILE} would not be valid YAML ({str(e).splitlines()[0]} {_yaml_where(e)}); not changed"
        if not isinstance(spec, dict):
            return f"REFUSED: {SPEC_FILE} must be a YAML mapping of DesignSpec fields; not changed"
        if spec.get("analysis_sql"):                 # it would be ignored: the run executes design/query.sql
            return f"REFUSED: {SPEC_FILE} must not hold analysis_sql: the query is {SQL_FILE} (edit that file); not changed"
    return ""


def _yaml_where(e: Exception) -> str:
    mark = getattr(e, "problem_mark", None)
    return f"at line {mark.line + 1}" if mark is not None else ""


def _numbered(text: str, start: int = 1, limit: int = SHOW_LINES) -> str:
    lines = text.splitlines()
    out = [f"{i:>4} | {l}" for i, l in enumerate(lines[start - 1:start - 1 + limit], start)]
    if len(lines) > start - 1 + limit:
        out.append(f"     ... ({len(lines) - (start - 1 + limit)} more lines; use read)")
    return "\n".join(out) or "     (empty)"


def _act(act: dict[str, Any], run_dir: Path, turn: int, asker: Asker | None = None, asked: list | None = None,
         dataset: str | None = None) -> str:
    kind = act.get("action")
    dataset = dataset or act.get("dataset") or (_datasets(run_dir) or [None])[0]
    if kind == "ask":
        asked = asked if asked is not None else []
        if len(asked) >= MAX_ASKS:
            return f"REFUSED: you already asked {MAX_ASKS} questions; decide yourself and record the assumption"
        opts = act.get("options")
        if not isinstance(opts, list) or not 2 <= len([o for o in opts if str(o).strip()]) <= MAX_OPTIONS:
            return f"INVALID: an ask needs 2-{MAX_OPTIONS} options (the user can only pick one); nothing was asked"
        asked.append(act["question"])
        asker = asker or Asker("auto")
        opts, rec = clean_options(opts, act.get("recommended"))
        if asker.mode != "auto":                      # the user may answer in their own words (E-71)
            if len(opts) >= MAX_OPTIONS:
                if rec is not None and rec >= MAX_OPTIONS - 1:
                    opts[MAX_OPTIONS - 2], rec = opts[rec], MAX_OPTIONS - 2
                opts = opts[:MAX_OPTIONS - 1]
            opts = opts + [EDIT_OWN]
        r = asker.ask(act["question"], opts, rec, act.get("why", ""), source="loop")
        if r["answer"] == AUTO:
            return "USER ANSWER: AUTO (the user left this to you). Decide, and record the choice as an assumption."
        if r["answer"] == EDIT_OWN:
            from ucd import intake
            _clarify_protocol(run_dir, act["question"], "answered by the user in the protocol text below", r["how"])
            intake.open_editor(run_dir / "protocol.md", asker, print)
            return ("PROTOCOL UPDATED. The user edited the protocol to answer your question; it is the gold standard "
                    "now (shown above). Change only the parts of the design this affects.")
        _clarify_protocol(run_dir, act["question"], r["answer"], r["how"])
        return (f"PROTOCOL UPDATED. Your question and the answer ({r['how']}) are now part of the protocol: "
                f"{r['answer']}. Change only the parts of the design this affects.")
    opts = _run_options(run_dir)
    if kind in ("lookup", "explore") and not opts.get("explore", True):
        return ("REFUSED: lookup and explore are switched off in this run: design from the Data Card, then run.")
    if kind == "lookup":
        n_look = _n_lookups(run_dir)
        if n_look >= MAX_LOOKUPS:
            return (f"REFUSED: lookup budget used ({MAX_LOOKUPS} terms). You have enough codes: write the "
                    "design files and run.")
        lk, known = str(act.get("kind") or "all").strip().lower(), datacard.lookups(dataset)
        lk = next((k for k in known if lk in (k, k.rstrip("s"), k + "s") or LOOKUP_SYNONYMS.get(lk) == k), lk)
        note = ""
        if lk != "all" and lk not in known:              # a near miss costs no turn: search everything instead
            note, lk = f"(kind {act.get('kind')!r} is not one of {', '.join(known)}; searched all)\n", "all"
        terms = act["term"] if isinstance(act["term"], list) else [act["term"]]
        terms = [str(t) for t in terms if str(t).strip()][:max(1, min(5, MAX_LOOKUPS - n_look))]
        return note + "\n\n".join(f"# {t}\n" + datacard.lookup(dataset, t, lk) for t in terms)
    if kind == "explore":
        groups = act.get("group") or []
        groups = [g.strip() for g in groups.split(",")] if isinstance(groups, str) else groups
        metrics = [m for m in (act.get("metrics") or []) if str(m).strip().lower() not in ("count", "n", "count(*)")]
        source = act.get("from") or act.get("body") or ""
        if not opts.get("concepts", True) and _uses_concepts(source):
            return "REFUSED: concept.* tables are switched off in this run; use the dataset's own tables."
        return json.dumps(Gate.shared(dataset, run_dir=run_dir).aggregate(source, groups, metrics), ensure_ascii=False)
    if kind in ("write", "edit"):
        f, rel = _inside(run_dir, act["file"])
        if rel is None:
            return "REFUSED: files may only be written inside the run directory"
        # the planner writes its design and notes only: never run settings, freeze markers or iteration logs
        if rel not in WRITABLE and not (f.suffix in (".md", ".txt") and "/" not in rel):
            return f"REFUSED: you may write only {', '.join(sorted(WRITABLE))} or another top-level .md/.txt note"
        if kind == "write":
            content = act.get("content", act.get("body"))
            if content is None:
                return "INVALID: write needs the file content in a fenced block after the JSON"
            if not isinstance(content, str):
                content = yaml.safe_dump(content, allow_unicode=True, sort_keys=False)
            done = f"wrote {rel}"
        else:
            if not f.exists():
                return f"INVALID: {rel} does not exist yet; create it with write"
            old = f.read_text(encoding="utf-8").splitlines()
            try:
                a, b = (list(act["lines"]) * 2)[:2] if isinstance(act["lines"], list) else (act["lines"],) * 2
                a, b = int(a), int(b)
            except (TypeError, ValueError, KeyError):
                return "INVALID: edit needs \"lines\": [first, last] (line numbers as shown in your design files)"
            if not (1 <= a <= b <= max(len(old), 1)):
                return f"INVALID: lines {a}-{b} are outside {rel} (it has {len(old)} lines)"
            if act.get("body") is None:                # only an explicit empty block deletes lines
                return ("INVALID: edit needs the replacement lines in a fenced block after the JSON (an empty block "
                        "deletes the lines); nothing was changed")
            new = act["body"].splitlines()
            content = "\n".join(old[:a - 1] + new + old[b:]) + "\n"
            done = (f"edited {rel}: lines {a}-{b} replaced by {len(new)} line(s); it now has "
                    f"{len(content.splitlines())} lines. Around the change:\n"
                    + _numbered(content, max(1, a - 2), len(new) + 4))
        if f.exists() and f.read_text(encoding="utf-8").rstrip("\n") == content.rstrip("\n"):   # E-71: 20 identical edits
            return (f"NO CHANGE: {rel} already holds exactly this; nothing was changed. Run the design to see the "
                    "effect of your earlier changes, or change something else.")
        problem = _check_content(run_dir, rel, content)
        if problem:
            return problem
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return done
    if kind == "read":
        f, rel = _inside(run_dir, act["file"])
        if rel is None or not f.is_file():
            return f"INVALID: no file {act['file']!r} in your run directory"
        start = max(1, int(act.get("from_line") or 1))
        return f"{rel}:\n" + _numbered(f.read_text(encoding="utf-8", errors="replace"), start, 300)
    if kind == "run":
        text, problem = _assembled_spec(run_dir)
        if problem:
            return f"INVALID: {problem}"
        sp = run_dir / dataset / "submitted_spec.yaml"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(text, encoding="utf-8")
        problem = sql_check(run_dir, dataset, text)
        if not problem and not opts.get("concepts", True) and _uses_concepts(text):
            problem = "concept.* tables are switched off in this run; use the dataset's own tables"
        if problem:                                    # an SQL slip: nothing ran, no design attempt used
            diag.event("sql_check", "loop", run=run_dir.name, message=problem.splitlines()[0])
            return (f"SQL CHECK FAILED (nothing was run; no design attempt used): {problem}{_where_in_query(run_dir, problem)}\n"
                    f"Fix {SQL_FILE} with edit and run again.")
        out = _capture(["run", "--dataset", dataset, "--run-dir", str(run_dir), "--spec", str(sp)])
        if "spec is invalid" in out:
            out += f"\n\nFix {SPEC_FILE}: its fields are listed in your instructions (DesignSpec fields)."
        return out
    if kind == "freeze":
        return _capture(["freeze", "--dataset", dataset, "--run-dir", str(run_dir)])
    if kind == "status":
        return _capture(["status", "--run-dir", str(run_dir)]) or "(no iterations yet)"
    if kind == "template":
        return _capture(["template"])
    return f"unknown action {kind!r}. {ACTIONS}"


def _assembled_spec(run_dir: Path) -> tuple[str, str]:
    """(DesignSpec YAML = design/spec.yaml + design/query.sql as analysis_sql, problem)."""
    sp, sq = run_dir / SPEC_FILE, run_dir / SQL_FILE
    if not sq.exists() or not sq.read_text(encoding="utf-8").strip():
        return "", f"{SQL_FILE} is empty: write the query first"
    try:
        spec = yaml.safe_load(sp.read_text(encoding="utf-8")) if sp.exists() else None
    except yaml.YAMLError as e:
        return "", f"{SPEC_FILE} is not valid YAML ({str(e).splitlines()[0]} {_yaml_where(e)})"
    if not isinstance(spec, dict):
        return "", f"{SPEC_FILE} must hold the DesignSpec fields"
    spec["analysis_sql"] = sq.read_text(encoding="utf-8").strip()
    return yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), ""


def _where_in_query(run_dir: Path, problem: str) -> str:
    """DuckDB numbers the lines of the query as the gate received it; name the line of design/query.sql instead."""
    m = re.search(r"LINE \d+:\s*(.+)", problem)
    if not m:
        return ""
    probe = re.sub(r"^\.\.\.\s*", "", m.group(1).strip()).rstrip(".").strip()[:40]
    lines = (run_dir / SQL_FILE).read_text(encoding="utf-8").splitlines() if (run_dir / SQL_FILE).exists() else []
    hit = next((i for i, l in enumerate(lines, 1) if probe and probe in l.strip()), None)
    return f"\n(This is line {hit} of {SQL_FILE}.)" if hit else ""


def _design_files(run_dir: Path) -> str:
    """The design as it is now, numbered: the model edits what it sees instead of recalling what it wrote."""
    parts = ["## Your design files (as they are now; edit uses these line numbers)"]
    for rel in (SPEC_FILE, SQL_FILE):
        f = run_dir / rel
        text = f.read_text(encoding="utf-8") if f.exists() else ""
        parts.append(f"### {rel} ({len(text.splitlines())} lines)\n{_numbered(text)}")
    return "\n\n".join(parts)


def _seed_design(run_dir: Path) -> None:
    """Start the design files from the template (spec fields; the SQL lives in query.sql)."""
    sp, sq = run_dir / SPEC_FILE, run_dir / SQL_FILE
    sp.parent.mkdir(parents=True, exist_ok=True)
    if not sp.exists():
        tpl = _capture(["template"]).split("\nanalysis_sql:")[0].rstrip()
        sp.write_text(tpl + "\n# analysis_sql: the query is design/query.sql\n", encoding="utf-8")
    if not sq.exists():
        sq.write_text("", encoding="utf-8")


def _record_setup(run_dir: Path, topic: str, question: str, system: str, cfg: dict, planner: Any = None) -> None:
    """meta.json: what this run used (version, models, temperatures, prompt hash), so a run can be reproduced. The
    planner entry is the model actually driving the loop (an injected one too, e.g. the self-test's script)."""
    from ucd.llm import role_settings
    meta = run_dir / "meta.json"
    m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {"topic": topic, "run": run_dir.name,
                                                                             "question": question}
    try:
        roles = {r: role_settings(dict(cfg["llm"]), r) for r in ("planner", "critic")}
    except Exception:                                  # a record must never stop the run
        roles = {}
    if planner is not None:
        roles["planner"] = {"model": getattr(planner, "model", None) or type(planner).__name__,
                            "temperature": getattr(planner, "temperature", None)}
    m["setup"] = {"version": __version__, "prompt_sha256": hashlib.sha256(system.encode()).hexdigest()[:16],
                  "models": {r: s.get("model") for r, s in roles.items()},
                  "temperature": {r: s.get("temperature", 0.2) for r, s in roles.items()},
                  "critic_backend": os.environ.get("UCD_CRITIC_BACKEND", "claude_cli")}
    meta.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")


def run_loop(topic: str, run_dir: Path, datasets: list[str], max_turns: int = MAX_TURNS, question: str | None = None,
             asker: Asker | None = None, echo=None, llm=None) -> Path:
    """Run the design loop; returns the path of final.md. `question` overrides the topic's protocol text; `llm` (an
    object with complete_text) replaces the configured model, e.g. for the self-test."""
    from ucd.datasets import require_data
    dataset = datasets[0]
    require_data(dataset)
    t_start = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    budget = int(cfg["llm"].get("context_chars", 24000))      # history budget; the design files are always shown
    if question is None:
        question = yaml.safe_load((ROOT / "ucd" / "topics.yaml").read_text(encoding="utf-8"))[topic]
    asker = asker or Asker("auto", log=run_dir / "questions.jsonl")
    echo = echo or (lambda *_: None)
    asked: list[str] = []
    system = _system_prompt(run_dir, question, dataset)
    llm = llm or LocalLLM(cfg)
    _record_setup(run_dir, topic, question, system, cfg, llm)
    _seed_design(run_dir)
    card = f"Data Card ({dataset}):\n{datacard.render(dataset, notes=_run_options(run_dir).get('notes', True))}"
    log = run_dir / "loop_log.jsonl"
    history: list[str] = []
    stop_said = False                 # the last run said "Stop suggested: True"
    stopped = ""                      # set when the user ends the loop early (q + Enter, Ctrl+C) or stops on a problem
    n_prob = 0                        # consecutive executed designs that cannot answer the question
    verdict_hint: tuple[str, str] = ("", "")   # (status, reason) when the run ends on a declared problem
    echo(f"  {STOP_HINT if sys.stdin.isatty() else 'To stop the design loop early: press Ctrl+C.'}\n"
         "  Stopping keeps everything that has run: final.md then summarises it.")
    try:
        for turn in range(max_turns):
            if stop_key_pressed():
                stopped = f"the user ended the design loop early (q) before turn {turn}"
                echo("\n  q received: stopping the design loop now.")
                break
            hist = "\n\n".join(history)
            if len(hist) > budget:                     # trim the oldest turns
                hist = "...(older turns omitted)...\n" + hist[-budget:]
            user = (f"{card}\n\n{_design_files(run_dir)}\n\n## Your recent actions\n{hist or '(none yet)'}\n\n"
                    f"{_next_step(run_dir, turn, max_turns, stop_said)}\n\nNext action (one JSON object):")
            t0 = time.time()
            try:
                raw = llm.complete_text(system, user)
            except Exception as e:
                raw, act = "", None
                obs = f"LLM call failed: {type(e).__name__}: {str(e)[:300]}"
            else:
                try:
                    act = parse_action(raw)
                except Exception as e:
                    act, obs = None, f"INVALID: {_why_unparsable(raw, e, llm)}"
            if act is not None:
                if act.get("action") == "done":
                    if not (run_dir / "final.md").exists():
                        obs = "REFUSED: write final.md before 'done'."
                    else:
                        _log(log, turn, raw, act, "done", time.time() - t0)
                        break
                elif act.get("action") == "infeasible":
                    reason = str(act.get("reason") or "the agent gave no reason")
                    choice, obs = _problem_question(reason, "the agent says", asker, run_dir, echo)
                    if choice == "stop":
                        verdict_hint = (verdict.infeasible_status(act), reason)
                        stopped = f"the agent declared the study infeasible ({reason}) and the {obs.split()[0].lower()} chose to stop"
                        _log(log, turn, raw, act, obs, time.time() - t0)
                        break
                    n_prob, stop_said = 0, False
                    if choice == "edit":
                        question, system = _reload_protocol(run_dir, question, dataset)
                    else:
                        _new_round(run_dir, "user chose to keep designing")
                else:
                    try:
                        obs = _act(act, run_dir, turn, asker, asked, dataset)
                        if obs.startswith("PROTOCOL UPDATED"):     # back to the protocol: planner and critic re-read it
                            question, system = _reload_protocol(run_dir, question, dataset)
                            n_prob, stop_said = 0, False
                    except GateError as e:
                        obs = f"REFUSED: {e}"
                    except KeyError as e:
                        obs = f"INVALID: missing field {e} for action {act.get('action')!r}.\n{ACTIONS}"
                    except Exception as e:
                        obs = f"ERROR: {type(e).__name__}: {str(e)[:500]}"
            if act and act.get("action") == "run" and "Stop suggested:" in obs:
                stop_said = "Stop suggested: True" in obs
                prob = design_problem(run_dir, dataset)
                if prob:                              # never finish on it: back to the design, then ask the user
                    stop_said = False
                    n_prob += 1
                    if n_prob <= FALLBACK_DESIGNS:
                        obs = (f"BACK TO THE DESIGN LOOP: {prob}. Change the design within the protocol and run again: "
                               "an over-strict criterion, a wrong code, a wrong join or time window is the "
                               f"usual cause ({FALLBACK_DESIGNS - n_prob + 1} more design(s) before the user is asked). "
                               "If the protocol cannot be answered with this data, use the infeasible action.\n\n" + obs)
                    else:
                        choice, note = _problem_question(prob, "after several designs", asker, run_dir, echo)
                        obs = note + "\n\n" + obs
                        if choice == "stop":
                            stopped = f"{prob}; the {note.split()[0].lower()} chose to stop"
                            _log(log, turn, raw, act, obs, time.time() - t0)
                            break
                        n_prob = 0
                        if choice == "edit":
                            question, system = _reload_protocol(run_dir, question, dataset)
                        else:
                            _new_round(run_dir, "user chose to keep designing")
                else:
                    n_prob = 0
            if obs.startswith(("REFUSED", "INVALID", "ERROR")):
                diag.event("error", "loop.action", type=str((act or {}).get("action") or "unparsable"),
                           message=obs.splitlines()[0], turn=turn, run=run_dir.name, raw_chars=len(raw),
                           finish_reason=getattr(llm, "finish_reason", None))
            _log(log, turn, raw, act, obs, time.time() - t0)
            echo(_progress(turn, max_turns, act, obs))
            if act and act.get("action") == "run" and sys.stdin.isatty() and turn < max_turns - 1:
                echo("           (to stop here and keep this result: q + Enter, or Ctrl+C)")
            history.append(f"[turn {turn}] {_brief(act)}\n-> {obs[:6000]}")
    except KeyboardInterrupt:                      # Ctrl+C: end the loop now, keep everything that ran
        stopped = "the user ended the design loop early (Ctrl+C)"
    finally:                                       # whatever happened, leave an honest final.md
        _flush_typed()
        if stopped:
            echo(f"\n  Stopped: {stopped}. Writing final.md from what has run so far ...")
        fm = run_dir / "final.md"
        if fm.exists() and _n_runs(run_dir) == 0:          # a model-written summary of a design that never ran
            fm.replace(run_dir / "final_model_draft_NOT_RUN.md")
        auto_final = not fm.exists()
        if auto_final:
            write_auto_final(run_dir, stopped or ("no design was executed" if _n_runs(run_dir) == 0 else
                                                  f"the planner did not write final.md within {max_turns} turns"))
        v = verdict.finalize(run_dir, *verdict_hint)
        echo(f"  Verdict: {v['status']} ({v['source']})")
        _diag_run(run_dir, topic, datasets, time.time() - t_start, auto_final, stopped)
    return fm


def _brief(act: Any) -> str:
    """An action as the history shows it: the JSON without its block (the files themselves are shown in full)."""
    if not isinstance(act, dict):
        return "ACTION (unparsable reply)"
    body = act.get("body")
    head = json.dumps({k: v for k, v in act.items() if k != "body"}, ensure_ascii=False)[:600]
    return f"ACTION {head}" + (f" + block of {len(body.splitlines())} line(s)" if isinstance(body, str) else "")


def _problem_question(problem: str, who: str, asker: Asker, run_dir: Path, echo) -> tuple[str, str]:
    """Ask the user what to do with a design that cannot answer the question. Returns (stop|continue|edit,
    observation for the planner). Stop is the recommended (and auto) choice."""
    echo(f"\n  Problem ({who}): {problem}")
    r = asker.ask(f"The design cannot answer the question ({who}): {problem}. What now?", PROBLEM_OPTIONS, 0,
                  source="loop")
    by = "AGENT (auto mode)" if r["how"] == "auto" else "USER"
    diag.event("problem", "loop", run=run_dir.name, who=who, message=problem, choice=r["answer"], by=by)
    if r["answer"] == PROBLEM_OPTIONS[0]:
        return "stop", f"{by} chose to stop: {problem}"
    if r["answer"] == PROBLEM_OPTIONS[2]:
        from ucd.intake import open_editor
        open_editor(run_dir / "protocol.md", asker, echo)
        return "edit", f"{by} EDITED THE PROTOCOL (see the protocol above, it is the gold standard now). Keep designing."
    return "continue", f"{by} chose to keep designing: try a different valid design for the protocol."


def _clarify_protocol(run_dir: Path, question: str, answer: str, how: str) -> None:
    """A definition settled during design goes back into the protocol (0.9.0): protocol.md gets a 'Clarifications
    during design' line, which _reload_protocol turns into the gold standard for the planner and the critic."""
    p, meta = run_dir / "protocol.md", run_dir / "meta.json"
    if not p.exists():                                 # loop started without chat: start from the stored question
        q = json.loads(meta.read_text(encoding="utf-8")).get("question", "") if meta.exists() else ""
        p.write_text(q.strip() + "\n", encoding="utf-8")
    text = p.read_text(encoding="utf-8").rstrip()
    head = "Clarifications during design:"
    who = "user" if how in ("choice", "text") else "agent (auto mode, recommended choice)"
    p.write_text(text + ("" if head in text else f"\n\n{head}") + f"\n- {question} -> {answer} ({who})\n",
                 encoding="utf-8")
    items_f = run_dir / "protocol_items.yaml"         # traced like any item: the design must map it (P1)
    if items_f.exists():
        try:
            items = yaml.safe_load(items_f.read_text(encoding="utf-8")) or []
        except yaml.YAMLError:
            return
        lst = items.get("items", []) if isinstance(items, dict) else items
        if isinstance(lst, list):
            lst.append({"id": f"C{sum(1 for i in lst if str((i or {}).get('id', '')).startswith('C')) + 1}",
                        "kind": "clarification", "text": f"{question} -> {answer}"})
            items_f.write_text(yaml.safe_dump(items, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _new_round(run_dir: Path, why: str) -> int:
    """Start a new design round (E-60): the critic's stop rule and attempt cap count only this round's designs, so a
    protocol change or a "keep designing" choice gets its own MAX_ITER designs instead of an immediate stop."""
    meta = run_dir / "meta.json"
    m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    m["round"] = int(m.get("round", 0)) + 1
    m["rounds"] = (m.get("rounds") or []) + [{"round": m["round"], "why": why, "after_designs": _n_runs(run_dir)}]
    meta.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["round"]


def _reload_protocol(run_dir: Path, question: str, dataset: str) -> tuple[str, str]:
    """After a protocol edit: the edited protocol.md becomes the question for the planner and (via meta.json) the
    critic, and a new design round starts. Returns (question, rebuilt system prompt)."""
    p = run_dir / "protocol.md"
    new = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    question = new or question
    meta = run_dir / "meta.json"
    m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    meta.write_text(json.dumps({**m, "question": question}, indent=1, ensure_ascii=False), encoding="utf-8")
    _new_round(run_dir, "protocol changed")
    return question, _system_prompt(run_dir, question, dataset)


_typed: list[str] = []                 # Windows: keys typed during the loop, until Enter


def stop_key_pressed() -> bool:
    """True when the user typed the line q (or quit / stop) and Enter since the last check. Never blocks; needs a real
    terminal (in a terminal that is not a native console, e.g. mintty, use Ctrl+C instead). Other lines typed while
    the loop runs are discarded, on Windows and on Linux alike."""
    if not sys.stdin.isatty():
        return False
    try:
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in "\r\n":
                    line = "".join(_typed).strip().lower()
                    _typed.clear()
                    if line in ("q", "quit", "stop"):
                        return True
                else:
                    _typed.append(ch)
            return False
        import select
        hit = False
        while select.select([sys.stdin], [], [], 0)[0]:  # every line typed meanwhile, not one per turn (E-71)
            line = sys.stdin.readline()
            if not line:                                   # input closed
                break
            hit = hit or line.strip().lower() in ("q", "quit", "stop")
        return hit
    except Exception:
        pass
    return False


def _flush_typed() -> None:
    """Discard keys typed during the loop, so extra q presses do not reach the shell after UCD exits (E-71)."""
    _typed.clear()
    if not sys.stdin.isatty():
        return
    try:
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        else:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def _diag_run(run_dir: Path, topic: str, datasets: list[str], seconds: float, auto_final: bool,
              stopped: str = "") -> None:
    """One 'run' line in the server-side diagnostics log (ucd.diag): outcome of the loop, aggregates only."""
    try:
        from ucd.llm import role_settings
        llm = load_config()["llm"]
        its = [json.loads(l) for d in datasets if (run_dir / d / "iterations.jsonl").exists()
               for l in (run_dir / d / "iterations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        last = its[-1] if its else {}
        crit = run_dir / datasets[0] / f"iter_{last.get('iter')}" / "critique.json"
        blocks = [f["rule"] for f in json.loads(crit.read_text(encoding="utf-8")).get("findings", [])
                  if f.get("severity") == "block"] if crit.exists() else []
        turns = sum(1 for _ in open(run_dir / "loop_log.jsonl", encoding="utf-8")) if (run_dir / "loop_log.jsonl").exists() else 0
        diag.event("run", "loop", name=topic, run=run_dir.name, dataset=",".join(datasets), turns=turns,
                   seconds=round(seconds), designs=_n_runs(run_dir), failed_designs=len(its) - _n_runs(run_dir),
                   I=[i.get("internal_error") for i in its], open_blocks=blocks, stop=last.get("stop_reason"),
                   critic=last.get("llm_overall"), auto_final=auto_final, stopped_by_user=bool(stopped),
                   models={r: role_settings(dict(llm), r).get("model") for r in ("planner", "critic")})
    except Exception as e:
        diag.error("loop.summary", e)


def _system_prompt(run_dir: Path, question: str, dataset: str) -> str:
    from ucd import concepts, packs
    from ucd import datasets as reg
    system = (ROOT / "ucd" / "planner_prompt.md").read_text(encoding="utf-8").format(
        run_dir=run_dir.as_posix(), question=question.strip(), dataset=reg.describe(dataset))
    opts = _run_options(run_dir)
    use_concepts = opts.get("concepts", True)
    lib = concepts.prompt_text(dataset) if use_concepts else ""
    llm_c, rules = opts.get("llm_critic", True), opts.get("rule_feedback", True)
    critics = ("no critic reviews your designs in this run: a run shows only its execution results" if not (llm_c or rules)
               else "the LLM clinical critic is switched off: findings come from the rules only" if not llm_c
               else "rule findings are not shown in this run: findings come from the LLM clinical critic only")
    off = [s for s, on in (("lookup and explore are switched off: design from the Data Card", opts.get("explore", True)),
                          ("concept.* tables are switched off: use the dataset's own tables", use_concepts),
                          (f"at most {opts.get('max_iter')} design run(s)", not opts.get("max_iter")),
                          (critics, llm_c and rules),
                          ("waiver requests are ignored in this run", opts.get("waivers", True)))
           if not on]
    return system + ("\n\n## This run's configuration\n" + "\n".join(f"- {s}" for s in off) if off else "") + (
                     "\n\n## Action protocol\n" + ACTIONS
                     + "\n\n## DesignSpec fields (design/spec.yaml; the query is design/query.sql)\n"
                     + _capture(["template"]).split("\nanalysis_sql:")[0].strip()
                     + (f"\n\n{lib}\nUse these tested concepts instead of re-deriving the same definitions: fewer SQL "
                        "errors, and a concept is computed once and then cached." if lib else "")
                     + (packs.planner_notes(question, dataset) if use_concepts else ""))


def _uses_concepts(sql: str) -> bool:
    from ucd import concepts
    return bool(concepts.referenced(sql or ""))


def _items_ok(content) -> bool:
    """protocol_items.yaml as the run reads it (cli): a list of {id: ...} items, or {items: [...]}."""
    try:
        items = yaml.safe_load(content) if isinstance(content, str) else content
    except yaml.YAMLError:
        return False
    items = items.get("items") if isinstance(items, dict) else items
    return isinstance(items, list) and bool(items) and all(isinstance(i, dict) and i.get("id") for i in items)


def sql_check(run_dir: Path, dataset: str, spec_text: str) -> str:
    """'' when the design's SQL binds and returns every column the spec needs; else the problem (0.9.0). A spec
    that is not a valid DesignSpec is left to the run, which reports it."""
    try:
        from ucd.spec import DesignSpec
        spec = yaml.safe_load(spec_text)
        ds = DesignSpec(**spec) if isinstance(spec, dict) else None
    except Exception:
        return ""
    if ds is None or not ds.analysis_sql.strip():
        return ""
    return Gate.shared(dataset, run_dir=run_dir).bind_check(ds.analysis_sql, ds.columns_needed())


def _datasets(run_dir: Path) -> list[str]:
    return [p.parent.name for p in run_dir.glob("*/iterations.jsonl")]


def design_problem(run_dir: Path, dataset: str) -> str:
    """Why the latest design of `dataset`, if it executed, cannot answer the question: too few units or outcome
    events, or an exposure with the same value for everyone. '' when it can (or when it did not execute)."""
    p = run_dir / dataset / "iterations.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    if not recs or (recs[-1].get("card") or {}).get("error"):
        return ""
    from ucd.critic import MIN_EVENTS, MIN_N, _num
    card = recs[-1].get("card") or {}
    if (card.get("model") or {}).get("type") == "cohort":
        return ""                                      # a cohort task: any size, even empty, is an answer
    n, ev = card.get("n_final"), (card.get("outcome") or {}).get("events")
    if (_num(n) is not None and _num(n) < MIN_N) or (_num(ev) is not None and _num(ev) < MIN_EVENTS):
        return (f"the cohort is too small to answer the question (n = {n}, outcome events = {ev}; at least {MIN_N} "
                f"units and {MIN_EVENTS} events are needed)")
    spec = run_dir / dataset / f"iter_{recs[-1].get('iter')}" / "spec.yaml"
    exp = ((yaml.safe_load(spec.read_text(encoding="utf-8")) or {}).get("exposure") or {}) if spec.exists() else {}
    col = exp.get("column") if isinstance(exp, dict) else None
    if col and ((card.get("table1") or {}).get(col) or {}).get("zero_variance"):
        return f"the exposure '{col}' has the same value for every unit, so its effect cannot be estimated"
    return ""


def _iter_log(run_dir: Path, kind: str):
    log = run_dir / "loop_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if isinstance(r.get("action"), dict) and r["action"].get("action") == kind:
                    yield r


def _n_runs(run_dir: Path, this_round: bool = False) -> int:
    """Designs that actually executed (an invalid spec or an SQL error does not count); with this_round, only those
    of the current design round."""
    from ucd.cli import design_round
    rnd = design_round(run_dir) if this_round else None
    recs = [json.loads(l) for f in run_dir.glob("*/iterations.jsonl") for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return sum(1 for r in recs if not (r.get("card") or {}).get("error") and (rnd is None or r.get("round", 0) == rnd))


def _n_lookups(run_dir: Path) -> int:
    """Lookup terms used so far (one action may search several terms)."""
    return sum(len(t) if isinstance(t := (r.get("action") or {}).get("term"), list) else 1
               for r in _iter_log(run_dir, "lookup"))


def _changes_since_run(run_dir: Path) -> int:
    """Writes and edits that changed a design file since the last run action (E-71: 20 edits, no run)."""
    log, n = run_dir / "loop_log.jsonl", 0
    for line in (log.read_text(encoding="utf-8").splitlines() if log.exists() else []):
        r = json.loads(line) if line.strip() else {}
        a, obs = r.get("action") if isinstance(r.get("action"), dict) else {}, str(r.get("observation", ""))
        if a.get("action") == "run":
            n = 0
        elif obs.startswith(("wrote design/", "edited design/")):
            n += 1
    return n


def _next_step(run_dir: Path, turn: int, max_turns: int, stop_said: bool = False) -> str:
    """Where the run stands, in one line, and what is due near the end."""
    n_runs, rnd = _n_runs(run_dir), _n_runs(run_dir, this_round=True)
    line = (f"Turn {turn + 1}/{max_turns}. Design runs {rnd}/{int(_run_options(run_dir).get('max_iter') or MAX_ITER)} in this round ({n_runs} in all); lookups "
            f"{_n_lookups(run_dir)}/{MAX_LOOKUPS}.")
    changed = _changes_since_run(run_dir)
    if changed:
        line += f" {changed} design change(s) since the last run: run the design to see their effect."
    left = max_turns - turn
    if (run_dir / "final.md").exists():
        return line
    if stop_said:
        return line + " The critic said STOP: do not change the design; write final.md, then reply {\"action\": \"done\"}."
    if left <= WRAP_UP_TURNS:
        if n_runs == 0:
            return line + f" ONLY {left} TURN(S) LEFT AND NO DESIGN HAS RUN: run your design now."
        return line + (f" ONLY {left} TURN(S) LEFT: write final.md (the final design, results and limitations), then "
                       "reply {\"action\": \"done\"}.")
    return line


def _progress(turn: int, max_turns: int, act: Any, obs: str) -> str:
    """One terminal line per turn, so a CLI user can follow the agent."""
    kind = str(act.get("action")) if isinstance(act, dict) else "(invalid)"
    head = obs.strip().splitlines()[0][:110] if obs.strip() else ""
    if kind == "run":
        it = next((l for l in obs.splitlines() if l.startswith("# Iteration")), head)
        stop = next((l for l in obs.splitlines() if "Stop suggested" in l), "")
        head = f"{it[:90]}  {stop.strip('*')[:40]}"
    elif kind == "explore":
        head = "aggregate query" + (" refused" if obs.startswith("REFUSED") else "")
    return f"  [{turn + 1:>2}/{max_turns}] {kind:<9} {head}"


def write_auto_final(run_dir: Path, reason: str) -> Path:
    """Deterministic final.md from the run's own logs (aggregates only). Used when the model never wrote one."""
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    q = (meta.get("question") or "").strip().splitlines()
    out = [f"# Final summary: {meta.get('topic', run_dir.parent.name)} / {run_dir.name}  (AUTO-GENERATED)", "",
           f"> Written by `ucd.loop` from the logs because {reason}. Numbers are copied from the run records; "
           "there is no model-written interpretation.", ""]
    if q:
        out += ["**Question.** " + q[0].replace("Objective:", "").strip(), ""]
    log = run_dir / "loop_log.jsonl"
    if log.exists():
        recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = sum(1 for r in recs if str(r.get("observation", "")).startswith(("REFUSED", "INVALID", "ERROR")))
        kinds: dict[str, int] = {}
        for r in recs:
            k = (r.get("action") or {}).get("action", "(invalid)") if isinstance(r.get("action"), dict) else "(invalid)"
            kinds[k] = kinds.get(k, 0) + 1
        out += [f"**Loop.** {len(recs)} turns; {bad} refused/invalid/error; actions: "
                + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])), ""]
    for f in sorted(run_dir.glob("*/iterations.jsonl")):
        ds = f.parent.name
        hist = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        out += [f"## {ds}", "", "| iter | I | blocks | n | events | primary | stop | change |", "|---|---|---|---|---|---|---|---|"]
        for h in hist:
            ch = str(h.get("changes_from_previous") or "").replace("\n", " ").replace("|", "/")[:120]
            out.append(f"| {h['iter']} | {h.get('internal_error')} | {h.get('n_block')} | {h.get('n_final')} | "
                       f"{h.get('events')} | {h.get('primary')} | {h.get('stop')} | {ch} |")
        last = hist[-1]
        valid = [h for h in hist if h.get("n_final")]
        best = valid[-1] if valid else last
        out += ["", f"**Last iteration {last['iter']}:** stop={last.get('stop')} {last.get('stop_reason') or ''}".rstrip()]
        if best is not last:
            out.append(f"(last iteration that produced a cohort: {best['iter']})")
        card = best.get("card") or {}
        if card.get("error"):
            out.append(f"\nThe design did not run: {str(card['error'])[:300]}")
        oc = card.get("outcome") or {}
        if best.get("n_final"):
            out.append(f"\nCohort n = {best.get('n_final')}, events = {best.get('events')}"
                       + (f", outcome rate = {oc.get('rate')}" if oc.get("rate") is not None else "")
                       + f"; primary estimate: {best.get('primary')}")
        fnd = [x for x in (last.get("critique") or {}).get("findings", []) if x.get("severity") in ("block", "warn")]
        if fnd:
            out += ["", "**Open critic findings (last iteration):**"]
            out += [f"- [{x.get('severity')}] {x.get('rule')}: {str(x.get('message', ''))[:200]}" for x in fnd[:8]]
        spec = run_dir / ds / f"iter_{best['iter']}" / "spec.yaml"
        if spec.exists():
            out += ["", f"<details><summary>Design spec (iteration {best['iter']})</summary>", "", "```yaml",
                    spec.read_text(encoding="utf-8").strip()[:6000], "```", "</details>"]
        out.append("")
    if len(out) < 8:
        out.append("No design iteration was recorded: the model never submitted a runnable spec.")
    path = run_dir / "final.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _log(path: Path, turn: int, raw: str, act: Any, obs: str, sec: float) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"turn": turn, "seconds": round(sec, 1), "raw": raw[:20000], "action": act,
                             "observation": obs[:4000]}, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topic")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--datasets", default=None, help="default: the first dataset in ucd/datasets.yaml")
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--auto", action="store_true", help="never ask the user; the agent decides")
    p.add_argument("--finalize", action="store_true", help="only write final.md from the logs of an existing run")
    a = p.parse_args()
    if a.finalize:
        fm = Path(a.run_dir) / "final.md"
        if fm.exists():
            print(f"{fm} already exists (written by the planner); not overwritten")
        else:
            print(write_auto_final(Path(a.run_dir), "it was requested with --finalize"))
    else:
        if not a.topic:
            p.error("--topic is required")
        rd = Path(a.run_dir)
        from ucd import datasets as reg
        run_loop(a.topic, rd, (a.datasets or reg.default()).split(","), a.max_turns,
                 asker=Asker("auto" if a.auto else "interactive", log=rd / "questions.jsonl"), echo=print)
