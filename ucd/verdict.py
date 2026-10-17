"""verdict.json: the run's answer in one word, with the reason and the limitation (0.8.0).

  status      normal         the cohort was built as specified
              partial        built, but part of the specification could not be implemented exactly (data limit, approximation)
              unanswerable   the data cannot answer the question (a variable is not recorded, a value is capped or shifted)
              contradictory  the criteria contradict each other
              empty_result   the criteria are valid but no patient meets them

The planner may write verdict.json itself (validated on write). Otherwise, or when its status contradicts the
evidence (status normal for an empty cohort), the loop derives it from what happened: stopped at intake (the
problem's kind), the planner's infeasible action, an empty final cohort, no design executed, or a normal run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATUSES = ("normal", "partial", "unanswerable", "contradictory", "empty_result")


def parse(text: str) -> dict[str, str]:
    """A verdict.json text -> dict, or ValueError naming what is wrong."""
    try:
        v = json.loads(text)
    except ValueError as e:
        raise ValueError(f"verdict.json is not valid JSON ({e})") from None
    if not isinstance(v, dict) or v.get("status") not in STATUSES:
        raise ValueError(f"verdict.json needs \"status\" as one of {', '.join(STATUSES)}")
    return {"status": v["status"], "reason": str(v.get("reason") or ""), "limitation": str(v.get("limitation") or "")}


def _last_record(run_dir: Path) -> dict[str, Any] | None:
    """The iteration record of the last executed design."""
    for p in sorted(run_dir.glob("*/iterations.jsonl")):
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        ok = [r for r in recs if not (r.get("card") or {}).get("error")]
        if ok:
            return ok[-1]
    return None


def _last_card(run_dir: Path) -> dict[str, Any] | None:
    rec = _last_record(run_dir)
    return None if rec is None else rec.get("card") or {}


def finalize(run_dir: Path, hint: str = "", reason: str = "") -> dict[str, str]:
    """Write run_dir/verdict.json: the planner's own verdict if valid and consistent, else a derived one."""
    path, rec = run_dir / "verdict.json", _last_record(run_dir)
    card = None if rec is None else rec.get("card") or {}
    blocks = sorted({str(f.get("rule", "")).split("-")[0] for f in ((rec or {}).get("critique") or {}).get("findings", [])
                     if f.get("severity") == "block"})
    own = None
    if path.exists():
        try:
            own = parse(path.read_text(encoding="utf-8"))
        except ValueError:
            own = None
    empty = bool(card and card.get("empty"))
    if own and not ((empty or blocks) and own["status"] == "normal"):
        v = {**own, "source": "agent"}
    elif hint in STATUSES:
        v = {"status": hint, "reason": reason, "limitation": "", "source": "auto"}
    elif card is None:
        v = {"status": "unanswerable", "reason": reason or "no design could be executed", "limitation": "",
             "source": "auto"}
    elif empty:
        v = {"status": "empty_result", "reason": "the final cohort is empty: no unit meets every criterion",
             "limitation": (own or {}).get("limitation", ""), "source": "auto"}
    elif blocks:                                       # E-76: open blocking findings are not a normal answer
        v = {"status": "partial", "reason": "the last executed design still had blocking findings "
             f"({', '.join(blocks[:8])}); see final.md", "limitation": (own or {}).get("limitation", ""),
             "source": "auto"}
    else:
        v = {"status": "normal", "reason": "a design was executed; see final.md", "limitation": "", "source": "auto"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
    return v


def infeasible_status(act: dict[str, Any]) -> str:
    """The status for the planner's infeasible action: its own, else contradictory when the reason says so."""
    st = str(act.get("status") or "").strip().lower()
    if st in STATUSES and st != "normal":
        return st
    return "contradictory" if "contradict" in str(act.get("reason") or "").lower() else "unanswerable"
