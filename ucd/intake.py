"""Proposal step: turn a free-text research request into a complete protocol brief, asking the user when needed.

The standard format (every field must be filled before the design loop starts):
  objective, population, exclusions, time_zero, exposure, outcome, covariates, analysis, primary_estimand
(PICO plus the timing and analysis choices a cohort design needs). The LLM checks the request against it, fills
what the request states, and returns questions (with choices and a recommended one) for what is missing or
ambiguous. The user answers through ucd.ask (2-4 choices, or A = the agent decides the rest of intake). At most
MAX_ROUNDS rounds. Finally the user can edit the protocol file before it becomes the gold standard (confirm).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from ucd.ask import AUTO, MAX_OPTIONS, Asker
from ucd.llm import NO_AUTHORITY, extract_json, strip_citations

DETAILED_CHARS = 400       # a request this long (or with several lines) is treated as a detailed protocol

FIELDS = ["objective", "population", "exclusions", "time_zero", "exposure", "outcome", "covariates", "analysis",
          "primary_estimand"]
LABELS = {"objective": "Objective", "population": "Population", "exclusions": "Exclusions", "time_zero": "Time zero",
          "exposure": "Exposure", "outcome": "Outcome", "covariates": "Covariates", "analysis": "Analysis",
          "primary_estimand": "Primary estimand"}
MAX_ROUNDS = 4                 # scope, details, leftovers (broad to narrow)
BROAD = ("objective", "population", "exposure", "outcome")   # asked before the details (broad to narrow)
MAX_QUESTIONS = 4          # per round
STOP_Q, REPHRASE, CONTINUE = ("Stop: this question cannot be studied as asked", "Rephrase the request",
                              "Continue anyway (the problem is written into the protocol)")


class IntakeStopped(Exception):
    """The user (or auto mode, whose recommended choice it is) stopped at intake because the question is invalid.
    `kind` is the verdict status: unanswerable, contradictory or partial."""
    def __init__(self, problem: str, kind: str = "unanswerable"):
        super().__init__(problem)
        self.kind = kind if kind in ("unanswerable", "contradictory", "partial") else "unanswerable"


SYSTEM = f"""You turn a clinical research request into a complete observational cohort protocol for this dataset:
{{dataset}}
The protocol needs these fields: {", ".join(FIELDS)}.
- analysis must name one explainable model: logistic regression (odds ratios), score evaluation (AUROC and
  calibration), descriptive statistics, or Cox regression (hazard ratios, optionally a time-dependent exposure),
  or "cohort only" when the request asks only to identify a cohort (then outcome and primary estimand are "none").
- Fill a field ONLY from what the request and the user's answers say, or as an obvious standard choice.
- For every field that is missing, vague or has several reasonable readings, ask ONE short question with 2-{MAX_OPTIONS}
  concrete choices (never more, and no open questions) and mark the best one as recommended (0-based index).
- Ask broad to narrow: first only about the scope (objective, population, exposure, outcome). Ask about details
  (time zero, windows, exclusions, covariates, analysis, primary estimand) only once the scope is settled.
- Use the dataset schema below: offer choices that the data can measure. A derived score that is not a
  stored column (e.g. Charlson, SOFA, Elixhauser) must be computed from its components; say so, or offer a
  measurable alternative. Do not ask about things the schema already settles.
- If the request is a detailed protocol, copy its definitions into the fields without shortening them, and ask
  only about real gaps, contradictions, or definitions the data cannot support.
- "problem": if the question cannot be studied as asked, say why in one sentence and do NOT quietly turn it
  into a different, answerable question. Examples: the exposure or outcome is not recorded in this dataset; the
  population cannot exist; the outcome is part of the exposure or of the predictors (tautology, leakage such as
  predicting death from the discharge location); the "effect" precedes its cause (reverse causation); the
  question is not a clinical cohort question. Leave "problem" empty for ordinary gaps (ask questions instead).
  With a problem, set "problem_kind": "unanswerable" (the data cannot answer it), "contradictory" (criteria
  contradict each other) or "partial" (part of it can be answered).
- {NO_AUTHORITY}
Reply with ONE JSON object only:
{{"fields": {{"objective": "...", ...}}, "complete": true|false,
  "questions": [{{"field": "...", "question": "...", "options": ["...", "..."], "recommended": 0, "why": "..."}}],
  "problem": "", "problem_kind": ""}}"""


def is_detailed(request: str) -> bool:
    """A long or multi-line request is a detailed protocol: kept word for word next to the agreed fields."""
    return len(request) > DETAILED_CHARS or request.count("\n") >= 4


def render(fields: dict[str, str]) -> str:
    """The protocol brief in the same plain format as ucd/topics.yaml."""
    return "\n".join(f"{LABELS[k]}: {str(fields.get(k, '')).strip()}" for k in FIELDS if str(fields.get(k, "")).strip())


def missing(fields: dict[str, str]) -> list[str]:
    return [k for k in FIELDS if not str(fields.get(k, "")).strip()]


def _schema(dataset: str, log=print) -> str:
    """The dataset schema for the intake prompt. Without it the model guesses table names from general knowledge
    (on MIMIC it offered OMOP tables such as condition_occurrence; E-71), so a failure is shown and logged."""
    try:
        from ucd.datacard import schema_summary
        return schema_summary(dataset)
    except Exception as e:
        from ucd import diag
        diag.error("intake.schema", e)
        log(f"WARNING: could not read the {dataset} schema ({type(e).__name__}: {str(e)[:200]}). Intake's choices "
            f"may name tables this dataset does not have; check `python -m ucd card --dataset {dataset}`.")
        return "(schema not available)"


def _call(llm, request: str, fields: dict[str, str], answers: list[dict], schema: str = "",
          dataset: str = "") -> dict[str, Any]:
    """One intake call. Gateway errors (LLMError) propagate; a malformed reply becomes {} (fields asked directly)."""
    user = (f"Dataset schema (table (rows): columns):\n{schema}\n\nRequest:\n{request}\n\n"
            f"Fields so far:\n{json.dumps(fields, ensure_ascii=False, indent=1)}\n\n"
            f"User answers so far:\n{json.dumps(answers, ensure_ascii=False, indent=1) if answers else '(none)'}\n\n"
            "Return the updated JSON object.")
    raw = llm.complete_text(SYSTEM.replace("{dataset}", dataset or "(not specified)"), user)          # LLMError -> the caller reports the gateway problem
    try:
        out = extract_json(raw)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def _merge(fields: dict[str, str], out: dict[str, Any], request: str = "") -> None:
    got = out.get("fields")
    if isinstance(got, dict):
        for k, v in got.items():
            if k in FIELDS and v is not None and str(v).strip():
                fields[k] = strip_citations(str(v).strip(), allowed=request)


def _questions(out: dict[str, Any]) -> list[dict]:
    qs = out.get("questions")
    if not isinstance(qs, list):
        return []
    good = []
    for q in qs:
        if isinstance(q, dict) and isinstance(q.get("question"), str) and q["question"].strip():
            opts = q.get("options")
            q["options"] = [str(o) for o in opts if str(o).strip()] if isinstance(opts, list) else []
            if len(q["options"]) >= 2:               # choices only; a missing field is asked directly later
                good.append(q)
    return good


def intake(request: str, llm, asker: Asker, log=print,
           dataset: str | None = None) -> tuple[str, dict[str, str], list[dict]]:
    """Returns (protocol text, fields, answers). Bad LLM output never stops it: missing fields are then asked
    directly and the user's answers fill them. Gateway errors (LLMError) are raised to the caller."""
    fields: dict[str, str] = {}
    answers: list[dict] = []
    from ucd import datasets
    dataset = dataset or datasets.default()
    schema, about = _schema(dataset, log), datasets.describe(dataset)
    seen = -1                                        # number of answers the model has already seen
    problems: list[str] = []                         # flagged problems the user chose to continue with
    for rnd in range(MAX_ROUNDS):
        out = _call(llm, request, fields, answers, schema, about)
        seen = len(answers)
        _merge(fields, out, request)
        prob = out.get("problem").strip() if isinstance(out.get("problem"), str) else ""
        if prob and not problems:                    # ask once; Stop is the recommended (and auto) choice (E-29)
            log(f"\nThe agent thinks this question cannot be studied as asked: {prob}")
            r = asker.ask("What now?", [STOP_Q, REPHRASE, CONTINUE], 0, prob, source="intake")
            if r["answer"] == STOP_Q:
                raise IntakeStopped(prob, str(out.get("problem_kind") or "").strip().lower())
            if r["answer"] == REPHRASE:
                new = asker.ask("Type the new request.", [], source="intake")["answer"]
                if new != AUTO:
                    request, fields, answers, seen = new, {}, [], -1
                    continue
            problems.append(prob)
        if out.get("complete") is True and not missing(fields):
            break
        qs = _questions(out)
        direct = not qs                              # the model gave no usable questions: ask for empty fields
        if direct:
            qs = [{"field": k, "question": f"{LABELS[k]}: what should it be?", "options": []} for k in missing(fields)]
        if not qs:
            break
        broad = [q for q in qs if q.get("field") in BROAD]   # broad to narrow (user, 2026-10-17): scope first,
        qs = broad or qs                                     # details in a later round
        if rnd == 0:
            log("\nThe request does not yet specify a complete protocol (PICO, time zero, analysis). "
                "A few questions:")
        for q in qs[:MAX_QUESTIONS]:
            r = asker.ask(q["question"], q["options"], q.get("recommended"), str(q.get("why") or ""), source="intake")
            auto = r["answer"] == AUTO
            ans = "(user left this to you: choose the standard option and say so)" if auto else r["answer"]
            answers.append({"field": q.get("field"), "question": q["question"], "answer": ans, "how": r["how"]})
            if direct and not auto and q.get("field") in FIELDS:
                fields[q["field"]] = ans             # a direct answer is the field itself
    if seen >= 0 and len(answers) > seen:            # answers of the last round: let the model fold them in
        _merge(fields, _call(llm, request, fields, answers, schema, about), request)
    from ucd import packs                            # the topic's known definition choices (no LLM call)
    for q in packs.intake_questions([request, render(fields)], dataset):     # the request and the agreed scope
        r = asker.ask(q["question"], q["options"], q.get("recommended"), str(q.get("why") or ""), source="intake")
        ans = "(left to the agent: use the topic pack's standard definition and say so)" if r["answer"] == AUTO else r["answer"]
        answers.append({"field": f"pack:{q['pack']}", "question": q["question"], "answer": ans, "how": r["how"]})
    for k in missing(fields):                        # last resort, never start the loop with an empty field
        fields[k] = "not specified by the user: choose a standard definition and record it as an assumption"
    text = render(fields)
    if answers:                                      # the user's own words stay visible to the planner
        text += "\n\nClarifications at intake:\n" + "\n".join(
            f"- {a['question']} -> {a['answer']}" + (" (auto: the agent's recommended choice, not the user's)"
                                                     if a.get("how") == "auto" else " (user)") for a in answers)
    if problems:                                     # never hide a flagged problem from the planner and critic
        text += "\n\nProblem flagged at intake (the user chose to continue; report it in the final summary):\n" + \
            "\n".join(f"- {p}" for p in problems)
    if is_detailed(request):                         # keep a detailed protocol word for word
        text += "\n\nOriginal protocol from the user (authoritative where it is more detailed):\n" + request.strip()
    return text, fields, answers


START, EDIT, STOP = "Start the design loop with this protocol", "Edit the protocol file first", "Stop here"


def open_editor(path: Path, asker: Asker, log=print) -> None:
    """Open `path` in $VISUAL / $EDITOR (notepad on Windows, else nano or vi) and wait until it is closed. Without an
    editor the user edits the file elsewhere and presses Enter."""
    cmd = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    cmd = shlex.split(cmd, posix=os.name != "nt") if cmd else (["notepad"] if os.name == "nt" else
                                                              [e for e in ("nano", "vi") if shutil.which(e)][:1])
    if cmd:
        log(f"Opening {path} in {cmd[0]}; save and close the editor to continue.")
        try:
            if subprocess.call([*cmd, str(path)]) == 0:
                return
        except OSError:
            pass
    log(f"Edit and save {path} in any editor, then press Enter.")
    try:
        asker.read("> ")
    except EOFError:
        pass


def confirm(protocol: str, asker: Asker, path: Path, log=print,
            edit: Callable[[Path, Asker, Callable], None] = open_editor) -> tuple[bool, str]:
    """Final step of intake: the protocol is written to `path` and shown; the user starts the loop, edits the file
    (as often as needed; the edited file is the gold standard), or stops."""
    path.write_text(protocol.strip() + "\n", encoding="utf-8")
    while True:
        log("\n=== Protocol (the gold standard for the design loop) ===\n" + protocol + f"\n\n(file: {path})")
        r = asker.ask("Final check: edit the protocol before the design loop starts?", [START, EDIT, STOP], 0,
                      source="protocol check")
        if r["answer"] != EDIT:
            return r["answer"] != STOP, protocol
        edit(path, asker, log)
        new = path.read_text(encoding="utf-8").strip()
        if new:
            protocol = new
            twice = [f"{n} '{LABELS[k]}:' lines" for k in FIELDS
                     if (n := len(re.findall(rf"(?im)^\s*{re.escape(LABELS[k])}\s*:", protocol))) > 1]
            if twice:                                  # new text pasted above the old one (E-71)
                log(f"WARNING: the protocol now has {', '.join(twice)}: the old text may still be in the file. "
                    "Choose 'Edit' again and keep one version, or start if both are meant.")
        else:
            log("The file is empty; the previous protocol is kept.")
            path.write_text(protocol.strip() + "\n", encoding="utf-8")
