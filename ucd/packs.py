"""Topic packs (ucd/packs.yaml): the concepts, notes, intake questions and critic check a topic adds to the
general agent (0.9.0). Detection is by patterns in the request or protocol, so the agent stays universal: a
protocol about anything else simply gets no pack."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

FILE = Path(__file__).resolve().parent / "packs.yaml"


@lru_cache(maxsize=1)
def _all() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(FILE.read_text(encoding="utf-8")) or {}


SCOPE = re.compile(r"^\s*(objective|population|time zero|exposure|outcome|primary estimand|design)\s*:", re.I)


def _scope(text: str) -> str:
    """The lines that define the study (objective, population, time zero, exposure, outcome, estimand); exclusion,
    covariate and other lines do not make a study about a topic (R5). Free text without such lines counts whole."""
    lines = [l for l in (text or "").splitlines() if SCOPE.match(l)]
    return "\n".join(lines) if lines else (text or "")


def detect(text: str, dataset: str) -> list[str]:
    """Names of the packs that apply to this request/protocol and dataset (judged on its scope lines)."""
    t = _scope(text)
    return [n for n, p in _all().items() if dataset in (p.get("datasets") or [dataset])
            and any(re.search(m, t, re.I) for m in p.get("match") or [])]


def planner_notes(question: str, dataset: str) -> str:
    """Section for the planner prompt ('' when no pack applies)."""
    from ucd.concepts import index
    have, out = index(dataset), []
    for n in detect(question, dataset):
        p = _all()[n]
        use = [c for c in p.get("concepts") or [] if c in have]          # only concepts this install has (C6)
        out.append(f"\n\n## Topic pack: {p.get('label', n)}\nUse: {', '.join('concept.' + c for c in use) or '(no tested concepts installed)'}."
                   f"\n{' '.join(str(p.get('notes') or '').split())}")
    return "".join(out)


def critic_notes(protocol: str, dataset: str) -> list[str]:
    """The `critic_notes` of every pack that applies: topic facts the clinical critic must not get wrong (E-58)."""
    return [f"Topic {_all()[n].get('label', n)}: {' '.join(str(_all()[n]['critic_notes']).split())}"
            for n in detect(protocol, dataset) if _all()[n].get("critic_notes")]


def intake_questions(texts: str | list[str], dataset: str) -> list[dict[str, Any]]:
    """The definition questions of every pack that applies to any of the texts (each judged on its own scope)."""
    names = dict.fromkeys(n for t in ([texts] if isinstance(texts, str) else texts) for n in detect(t, dataset))
    qs = []
    for n in names:
        for q in _all()[n].get("questions") or []:
            qs.append({**q, "pack": n})
    return qs


def checks(protocol: str, sql: str, dataset: str) -> list[dict[str, str]]:
    """Critic findings: a pack applies but the design uses none of its tested concepts (warn)."""
    from ucd.concepts import index, referenced
    used, have = set(referenced(sql)), index(dataset)
    out = []
    for n in detect(protocol, dataset):
        c = _all()[n].get("check") or {}
        if c.get("uses") and set(c["uses"]) & set(have) and not used & set(c["uses"]):
            out.append({"rule": f"PK-{n}", "severity": "warn", "fix": f"Use {', '.join('concept.' + u for u in c['uses'])} (see the topic pack).",
                        "message": c.get("message", "the topic's tested concepts are not used")})
    return out
