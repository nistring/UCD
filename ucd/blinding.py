"""Outcome blinding for the design phase (pre-registration inside the agent).

While a run is blinded, the planner and the LLM critic see everything needed to check a design (flow, missingness,
distributions, overall event count and rate, timing of events, model convergence and events-per-parameter) but no
exposure-outcome association or discrimination (ORs, HRs, AUROCs, outcome by group). The full card is stored only in
the hidden evaluation folder. `ucd freeze` locks the design; the next run is unblinded once, and any later change is
logged as a post-unblinding change.
"""
from __future__ import annotations

import copy
from typing import Any

# keys removed from model blocks (associations / performance)
_MODEL_HIDE = {"terms", "primary", "primary_all_levels", "intercept", "auroc_apparent", "auroc_cv", "brier_cv",
               "cv_calibration_intercept", "cv_calibration_slope", "by_group", "or_per_point", "auroc", "auroc_ci",
               "outcome_rate_by_band"}
# keys removed from Table 1 variables (outcome-stratified)
_VAR_HIDE = {"median_outcome1", "median_outcome0"}


def _blind_model(m: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in m.items() if k not in _MODEL_HIDE}
    if "scores" in m:
        out["scores"] = []
        for s in m["scores"]:
            b = {k: v for k, v in s.items() if k not in _MODEL_HIDE}
            a = s.get("auroc")
            if a is not None and a < 0.5:   # sanity alarm without revealing the value
                b["alarm"] = "discrimination below chance: check outcome coding / score direction"
            out["scores"].append(b)
    if "logistic" in m and isinstance(m["logistic"], dict):
        out["logistic"] = _blind_model(m["logistic"])
    out["blinded"] = True
    return out


def blind_card(card: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(card)
    if isinstance(c.get("model"), dict):
        c["model"] = _blind_model(c["model"])
    for v in (c.get("table1") or {}).values():
        for k in _VAR_HIDE:
            v.pop(k, None)
        for lev in (v.get("levels") or {}).values():
            lev.pop("outcome_rate", None)
    for s in c.get("sensitivity") or []:
        s.pop("primary", None)
    c["blinded"] = ("Outcome associations, discrimination and outcome-by-group results are hidden during the design "
                    "phase. Run `python -m ucd freeze` when the design is final; the next run is unblinded once.")
    return c
