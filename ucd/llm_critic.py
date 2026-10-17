"""LLM clinical critic: an independent LLM call (fresh context, no tools, no data access) after EVERY design.

It sees only: the protocol, the protocol items, the spec (incl. SQL), the aggregate result card and the
deterministic findings. It uses clinical knowledge to judge whether the cohort, definitions and model make
clinical sense and proposes concrete changes (e.g. refining the target population). Its issues join the
deterministic findings in the internal error under the dimension "clinical".

Backends (env UCD_CRITIC_BACKEND):
  claude_cli    - `claude -p` with all tools disabled (development; no API key needed)
  openai_compat - an OpenAI-compatible chat API via ucd.llm (e.g. the KHDP gateway)
  off           - disabled
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

from pathlib import Path

from ucd.llm import strip_citations

SYSTEM = """You are the clinical-methodology CRITIC in a cohort-design agent. You never see patient-level data.
You receive a research protocol (the gold standard), its numbered items, the planner's current design
(DesignSpec YAML including its SQL), the aggregate result card produced by running it, and the findings of a
deterministic rule-based critic.

Your job, using your clinical and epidemiological knowledge:
1. Check that the SQL actually implements each definition (codes/items, units, time windows, joins, age,
   first-stay logic, outcome ascertainment). Point to the exact problem. A common silent error: the first value
   of the whole admission/patient is taken and then required to lie inside the unit's time window, which drops
   every unit with an earlier measurement; the window filter must come before the first-row selection.
2. Judge clinical plausibility of the design and of the aggregates (cohort size, flow losses, outcome rate,
   variable distributions, missingness, directions of associations) for THIS dataset - say what looks wrong
   and the likely cause. Do NOT compare against numbers from published studies, do NOT cite papers, authors,
   journals or 'the literature' as evidence (no 'et al.', no author-year references; that is an appeal to
   authority: argue from the protocol, the aggregates and explicit reasoning), and do NOT push the design toward
   any expected effect size, p-value or AUROC.
3. Propose concrete modifications. Where the protocol is silent or ambiguous you MAY propose clinically
   sensible refinements (e.g. of the target population, windows, covariates). Where the protocol is explicit,
   a disagreement is severity "info" unless following it is clinically or technically wrong for this data.
4. Do not repeat the deterministic findings unless you add a cause or a fix.
   Named definitions: a score, criterion or scale named in the protocol (including a version such as "X-2" or
   "X version 2") means what the protocol says. Never reinterpret it (for example a version number as a second
   measurement time). If the protocol does not define it, raise a "protocol" issue that it is undefined instead of
   redefining it. Outcome timing: do not propose removing a time condition of an outcome (for example "death during
   the unit stay") because a timestamp might be missing, unless the dataset notes or the result card show that it
   is; removing it changes the outcome (death in the unit becomes death anywhere in the hospitalization).
5. WAIVER JUDGE. The DesignSpec may contain `waivers`: requests to waive a rule finding (e.g. V4 look-ahead, V3
   immortal time) because the protocol mandates the design that triggers it. For each request decide:
   - GRANT only if the protocol text explicitly mandates the design element that triggers the rule (e.g. the protocol
     says time zero is admission AND the outcome is all in-hospital deaths, while the predictor is a first-24-h
     score). Quote the shortest protocol passage that proves it, copied character-for-character from the protocol.
   - DENY if the protocol is silent, ambiguous, or asks for a prediction at a later time point, or if the triggering
     design is the planner's own choice. Rules about mechanical errors are never waivable.
   - Being persuaded by the planner's justification alone is not enough: the protocol text must settle it.

Reply ONLY with these tagged lines (no JSON, no markdown), one item per line, fields separated by " | ":
OVERALL: accept | revise
PLAUSIBILITY: <0..1>
ISSUE | <block|warn|info> | <sql|clinical|protocol|statistics|data> | <what is wrong, specific> | <which aggregate or SQL fragment shows it> | <concrete change to the spec or SQL>
WAIVER | <rule id> | <grant|deny> | <verbatim protocol text, required for grant> | <one-sentence reason>
Never use the character "|" inside a field. Quote SQL as it is; no escaping is needed.
Use "block" only for errors that make the result invalid (wrong cohort, wrong outcome, wrong exposure
definition, leakage). At most 8 ISSUE lines, most important first; one WAIVER line per waiver request."""


def _payload(protocol: str, items: Any, spec_text: str, card: dict, findings: list[dict], history: list[dict],
             context: str = "") -> str:
    card_s = json.dumps(card, ensure_ascii=False, default=str)
    if len(card_s) > 18000:
        card_s = card_s[:18000] + " ...[truncated]"
    hist = [{"iter": h["iter"], "I": h["internal_error"], "n": h.get("n_final"), "primary": h.get("primary"),
             "changes": (h.get("changes_from_previous") or "")[:300]} for h in history]
    det = [{k: f.get(k) for k in ("severity", "rule", "message", "fix")} for f in findings]
    ctx = f"## Dataset and topic notes (facts about this data; not evidence from studies)\n{context}\n\n" if context else ""
    return (f"{ctx}## Protocol\n{protocol}\n\n## Protocol items\n{json.dumps(items, ensure_ascii=False)}\n\n"
            f"## Previous iterations\n{json.dumps(hist, ensure_ascii=False)}\n\n## Current DesignSpec\n```yaml\n{spec_text}\n```\n\n"
            f"## Result card (aggregates)\n{card_s}\n\n## Deterministic findings\n{json.dumps(det, ensure_ascii=False)}\n\n"
            "Reply with the tagged lines now.")


# a tag may come wrapped in markdown ("**OVERALL:** revise", "- ISSUE | ..."): leading list/emphasis marks are ignored
_TAG = re.compile(r"^[\s*#>`\-]*(OVERALL|PLAUSIBILITY|ISSUE|WAIVER)[*`]*\s*[:|][*`]*\s*(.*)$", re.I)


def _parse_lines(text: str) -> dict:
    """Tagged-line reply -> the review dict (E-45). Lines that start with no tag continue the previous item (a model
    may wrap a long message)."""
    out: dict[str, Any] = {"issues": [], "waiver_decisions": []}
    last: list[str] | None = None
    for line in text.splitlines():
        m = _TAG.match(line)
        if not m:
            if last is not None and line.strip():
                last[-1] = (last[-1] + " " + line.strip()).strip()
            continue
        tag, rest = m.group(1).upper(), m.group(2).strip()
        last = None
        if tag == "OVERALL":
            out["overall"] = "accept" if rest.lower().startswith("accept") else "revise"
        elif tag == "PLAUSIBILITY":
            num = re.search(r"\d*\.?\d+", rest)
            out["clinical_plausibility"] = max(0.0, min(1.0, float(num.group()))) if num else None
        else:
            f = [x.strip() for x in rest.split("|")]
            if tag == "ISSUE":
                f = (f + [""] * 5)[:5] if len(f) <= 5 else f[:4] + [" | ".join(f[4:])]
                out["issues"].append({"severity": f[0].lower(), "category": f[1].lower() or "clinical",
                                      "message": f[2], "evidence": f[3], "proposed_change": f[4]})
                last = _Tail(out["issues"][-1], "proposed_change")
            else:
                f = (f + [""] * 4)[:4] if len(f) <= 4 else f[:3] + [" | ".join(f[3:])]
                out["waiver_decisions"].append({"rule": f[0], "granted": f[1].lower().startswith("grant"),
                                                "quote": f[2], "reason": f[3]})
                last = _Tail(out["waiver_decisions"][-1], "reason")
    if "overall" not in out and not out["issues"]:
        raise ValueError("no tagged lines in critic output")
    out.setdefault("overall", "revise" if any(i["severity"] == "block" for i in out["issues"]) else "accept")
    return out


class _Tail(list):
    """The last field of the current item, so a wrapped continuation line is appended to it."""
    def __init__(self, item: dict, key: str):
        super().__init__([item.get(key, "")])
        self.item, self.key = item, key

    def __setitem__(self, i, value):
        super().__setitem__(i, value)
        self.item[self.key] = value


def _parse(text: str) -> dict:
    """The critic's reply: tagged lines (E-45), or JSON as before (with the planner's repair; E-30)."""
    from ucd.llm import extract_json, strip_think
    text = strip_think(text)
    tags = [m.group(1).upper() for m in map(_TAG.match, text.splitlines()) if m]
    if "OVERALL" in tags or "ISSUE" in tags or not ("{" in text and "}" in text):
        if tags:
            return _parse_lines(text)
    out = extract_json(text)
    if not isinstance(out, dict):
        raise ValueError("critic output is not a JSON object")
    return out


def _call_claude_cli(user: str) -> str:
    model = os.environ.get("UCD_CRITIC_MODEL", "")
    cmd = ["claude", "-p", "--tools", "", "--no-session-persistence", "--system-prompt", SYSTEM,
           "--output-format", "text"] + (["--model", model] if model else [])
    r = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8", timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed: {r.stderr[:300]}")
    return r.stdout


_LAST: dict[str, Any] = {}                             # the last reply's size and finish reason, for the log


def _call_openai_compat(user: str) -> str:
    from ucd.llm import LocalLLM
    llm = LocalLLM(role="critic")
    raw = llm.complete_text(SYSTEM, user)
    _LAST.update(raw_chars=len(raw), finish_reason=llm.finish_reason)
    return raw


def review(protocol: str, items: Any, spec_text: str, card: dict, findings: list[dict], history: list[dict],
           context: str = "") -> dict:
    backend = os.environ.get("UCD_CRITIC_BACKEND", "claude_cli")
    if backend == "off":
        return {"overall": "n/a", "issues": [], "backend": "off"}
    user = _payload(protocol, items, spec_text, card, findings, history, context)
    last_err = None
    for _ in range(2):
        try:
            raw = _call_claude_cli(user) if backend == "claude_cli" else _call_openai_compat(user)
            out = _parse(raw)
            out["backend"] = backend
            out["issues"] = [i for i in out.get("issues", []) if isinstance(i, dict) and i.get("message")][:8]
            return out
        except Exception as e:  # retry once, then degrade gracefully
            last_err = e
    from ucd import diag
    diag.error("critic", last_err, backend=backend, **_LAST)   # reply length and finish_reason show why (cut off?)
    return {"overall": "error", "issues": [], "backend": backend, "error": str(last_err)[:300]}


def context(dataset: str, protocol: str, notes: bool = True) -> str:
    """What the critic should know about the data and the topic: ucd/docs/<dataset>_notes.md and the critic notes
    of the topic packs that apply (E-58: without them the critic guessed, e.g. SOFA-2 read as a later SOFA).
    notes=False (ablation) leaves the dataset notes out."""
    from ucd import packs
    f = Path(__file__).resolve().parent / "docs" / f"{dataset}_notes.md"
    parts = [f.read_text(encoding="utf-8")[:9000]] if notes and f.exists() else []
    parts += [n for n in packs.critic_notes(protocol, dataset) if n]
    return "\n\n".join(parts)


def to_findings(rev: dict) -> list[dict]:
    out = []
    for k, i in enumerate(rev.get("issues", []), 1):
        sev = i.get("severity", "info")
        sev = sev if sev in ("block", "warn", "info") else "info"
        out.append({"rule": f"L{k}-{i.get('category', 'clinical')}", "dimension": "clinical", "severity": sev,
                    "message": strip_citations(str(i.get("message")))[:600],
                    "evidence": strip_citations(str(i.get("evidence", "")))[:400] or None,
                    "fix": strip_citations(str(i.get("proposed_change", "")))[:600]})
    if rev.get("overall") == "error":
        out.append({"rule": "L0", "dimension": "clinical", "severity": "info",
                    "message": f"LLM critic unavailable: {rev.get('error')}", "evidence": None, "fix": ""})
    return out
