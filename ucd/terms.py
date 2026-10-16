"""Design terms that must survive from the request to the protocol and from the protocol to the planner's items
(0.10.0, critic rule T11). A fixed vocabulary plus every number with a unit; deterministic."""
from __future__ import annotations

import re

VOCAB = {
    "landmark": r"landmark", "target trial": r"target[- ]trial", "emulation": r"emulat",
    "fallback": r"fall[- ]?back",
    "unresolved": r"unresolv|cannot be (?:judged|adjudicated)", "indeterminate": r"indeterminate",
    "first ICU stay": r"first icu stay", "per patient": r"per patient|each patient",
    "per hospitalization": r"per hospitali[sz]ation", "all-cause": r"all[- ]cause",
    "deaths after discharge": r"out-of-hospital|after discharge|post-discharge|any death record",
    "overlap weighting": r"overlap weight", "propensity score": r"propensity", "IPTW": r"\biptw\b",
    "restricted mean survival time": r"restricted mean survival|\brmst\b",
}
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(h|hours?|days?|mmol/l|mg/dl|%)(?![a-z])", re.I)
_UNIT = {"h": "h", "hour": "h", "hours": "h", "day": "d", "days": "d"}


def _numbers(text: str) -> set[str]:
    return {f"{float(n):g} {_UNIT.get(u.lower(), u.lower())}" for n, u in NUMBER.findall(text or "")}


def found(text: str) -> list[str]:
    """The design terms in a text (vocabulary names, then numbers with units)."""
    t = text or ""
    return [k for k, p in VOCAB.items() if re.search(p, t, re.I)] + sorted(_numbers(t))


def missing(source: str, target: str) -> list[str]:
    """Terms of `source` that `target` does not contain (case-insensitive; '24 h' matches '24 hours')."""
    tt, nums = target or "", _numbers(target)
    out = [k for k, p in VOCAB.items() if re.search(p, source or "", re.I) and not re.search(p, tt, re.I)]
    return out + sorted(_numbers(source) - nums)
