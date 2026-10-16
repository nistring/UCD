"""Internal critic: topic-agnostic checks on (spec, result card, previous iterations) -> findings + internal error I.

I uses ONLY the design and its own aggregates. It never uses literature values, and it never rewards effect size,
p-values or high discrimination (so iterating on I cannot p-hack). I in [0, 1]; lower is better.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ucd.runner import primary_value
from ucd.spec import DesignSpec

WEIGHTS = {"validity": 0.30, "feasibility": 0.15, "quality": 0.10, "model": 0.10, "stability": 0.15, "clinical": 0.20}
# "clinical" = issues raised by the independent LLM clinical critic (ucd/llm_critic.py), merged in the CLI
SEV = {"block": 0.5, "warn": 0.15, "info": 0.03}
SCORING_VERSION = "v3"
# v2: noisy-OR aggregation per dimension (no saturation at a few warnings), P3 = info
# v3: rule tiers + waivers; V4 split into V4a (mechanical) and V4 (contextual: outcome counted before predictors are
#     complete); V3 skipped for a Cox time-dependent exposure; S2 (estimate moved) demoted to info.

# Tier 1: mechanical facts about data/code - never waivable.  Tier 2: contextual - a block/warn that an independent
# judge (LLM clinical critic) may waive when quoted protocol text mandates the design.  Everything else: advisory.
TIER1 = {"V0", "V1", "V4a", "V6b", "V8b", "M1", "M2", "M6", "P1", "A1", "A3"}
TIER2 = {"V2", "V3", "V4", "V6", "V7", "V8", "V10", "F1", "F2", "F3", "F4", "Q1", "Q2", "Q4", "M3", "M4", "M5"}
NEEDS_SENSITIVITY = {"V3", "V4"}       # a waiver of these requires a landmark-type sensitivity analysis in the card
MAX_WAIVERS = 2


def tier(rule: str) -> int:
    r = rule.split("-")[0]
    return 1 if r in TIER1 else (2 if r in TIER2 else 3)


MIN_N, MIN_EVENTS = 100, 20
# rules about outcomes, models and estimates, skipped for a cohort task (model.type "cohort")
COHORT_SKIP = {"V2", "V3", "V4", "V4a", "V4b", "V5", "V10", "F1", "F2", "F4", "S1", "S2", "S3", "M1", "M2", "M3", "M4",
               "M5", "A2", "A3", "A4", "A5", "A6", "T2", "T3", "T7"}      # below either, a design cannot answer the question (the loop goes back to design)


def _num(x: Any) -> Optional[float]:
    if isinstance(x, str) and x.startswith("<"):
        return 5.0           # suppressed small count: treat as 5
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class _F:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    def add(self, rule: str, dim: str, sev: str, msg: str, evidence: Any = None, fix: str = "") -> None:
        self.items.append({"rule": rule, "dimension": dim, "severity": sev, "message": msg,
                           "evidence": evidence, "fix": fix})


def critique(spec: DesignSpec, card: dict[str, Any], history: list[dict[str, Any]], dataset: str,
             protocol_items: Optional[list[dict[str, Any]]] = None,
             options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    options = options or {}
    f = _F()
    q = spec.question.lower()

    # ---------------------------------------------------------------- protocol traceability (question = gold standard)
    if protocol_items:
        ids = [str(p.get("id")) for p in protocol_items]
        mapped = {m.item: m for m in spec.protocol_mapping}
        missing = [i for i in ids if i not in mapped]
        if missing:
            f.add("P1", "validity", "block", f"{len(missing)} protocol item(s) are not mapped to the design: {missing}",
                  evidence={"unmapped": missing},
                  fix="Add a protocol_mapping entry per item: implemented_as + deviation (empty if exact).")
        for i, m in mapped.items():
            if i not in ids:
                f.add("P2", "validity", "info", f"protocol_mapping refers to unknown item {i}")
            elif m.deviation.strip():
                f.add("P3", "validity", "info",
                      f"Deviation from protocol item {i}: {m.deviation[:200]}",
                      fix="Implement the item as written if the data allow; otherwise keep the justified deviation.")

    # ---------------------------------------------------------------- hard failures
    if card.get("error"):
        f.add("V0", "validity", "block", f"Design could not be executed: {card['error']}",
              fix="Fix analysis_sql so it runs and returns all required columns.")
        return _score(f, hard=True)

    # ---------------------------------------------------------------- question alignment (gate)
    align_fail = False
    for k in ("population",) if spec.model.type == "cohort" else ("population", "exposure_or_predictor", "outcome"):
        if not getattr(spec.pico, k).strip():
            f.add("A1", "validity", "block", f"PICO element '{k}' is empty", fix="State every PICO element.")
            align_fail = True
    if spec.model.type == "logistic" and not spec.model.primary_term:
        f.add("A2", "validity", "warn", "Logistic model without a primary_term: the main answer is not designated",
              fix="Set model.primary_term to the exposure/predictor column that answers the question.")
    if spec.model.type == "logistic" and spec.exposure and spec.model.predictors and \
            spec.exposure.column not in [p.split("=")[0] for p in spec.model.predictors]:
        f.add("A3", "validity", "block", "The exposure is not in the model predictors", evidence=spec.model.predictors,
              fix="Add the exposure column to model.predictors.")
        align_fail = True
    pt_term = spec.model.primary_term or ""
    cat_cols = {v.column for v in ([spec.exposure] if spec.exposure else []) + spec.covariates if v.type == "categorical"}
    if spec.model.type in ("logistic", "cox") and pt_term in cat_cols and "=" not in pt_term:
        f.add("A6", "validity", "warn", f"primary_term '{pt_term}' is categorical but names no level: the reported "
              "main result is an arbitrary level", fix=f"Set primary_term to '{pt_term}=<level>' (the contrast the "
              "protocol names as the primary estimand).")
    if re.search(r"how well|predict|discriminat|accuracy", q) and spec.model.type == "descriptive":
        f.add("A4", "validity", "warn", "Question asks about prediction but the model is descriptive",
              fix="Use score_eval (existing score) or logistic with discrimination metrics.")
    if re.search(r"associat|effect|risk of|delay", q) and spec.model.type == "score_eval":
        f.add("A5", "validity", "warn", "Question asks about an association but the model only evaluates a score",
              fix="Use logistic regression with the exposure as primary_term and confounders as covariates.")

    # ---------------------------------------------------------------- design validity
    if _num(card.get("duplicate_unit_ids")) and _num(card["duplicate_unit_ids"]) > 0:
        f.add("V1", "validity", "block", "unit_id is duplicated in the final cohort (join fan-out)",
              evidence={"duplicate_unit_ids": card["duplicate_unit_ids"]},
              fix="Aggregate to one row per unit before joining (e.g. GROUP BY unit or window function).")
    pt = spec.prediction_time_h
    for v in spec.covariates:
        if v.window_end_h is not None and v.window_end_h > pt + 1e-9:
            f.add("V2", "validity", "block",
                  f"Look-ahead: covariate '{v.column}' is measured until {v.window_end_h} h, after the prediction "
                  f"time ({pt} h)", evidence={"covariate": v.column, "window_end_h": v.window_end_h},
                  fix="End covariate windows at or before prediction_time_h, or move the time zero/landmark.")
        if v.window_end_h is None and v.window_start_h is None and v.column.lower() not in ("age", "sex", "gender", "female", "male"):
            f.add("V2b", "validity", "info", f"Covariate '{v.column}' has no declared measurement window",
                  fix="Declare window_start_h/window_end_h relative to index.")
    tm = card.get("timing") or {}
    of = float(spec.outcome_from_h or 0.0)
    if spec.exposure and spec.exposure.window_end_h is not None and spec.exposure.window_end_h > max(pt, of) and \
            spec.model.type == "logistic":
        frac = _num(tm.get("events_before_exposure_window_end_frac"))
        sev = "block" if (frac is not None and frac > 0.02) else "warn"
        f.add("V3", "validity", sev,
              f"Immortal-time risk: exposure is assessed until {spec.exposure.window_end_h} h after index while the "
              f"outcome is counted from {pt} h" + (f"; {frac:.1%} of events occur inside the exposure window" if frac is not None else
                                                     "; provide outcome_time_h so this can be measured"),
              evidence={k: tm.get(k) for k in ("events_before_exposure_window_end", "events_before_exposure_window_end_frac")},
              fix="Options: (a) landmark at the end of the exposure window; (b) Cox model with a time-dependent "
                  "exposure; (c) if the protocol itself fixes this exposure window, request a waiver citing it and "
                  "report a landmark sensitivity analysis.")
    # V4a (tier 1): the design says outcomes start at outcome_from_h but earlier events are still counted
    fb = _num(tm.get("events_before_outcome_from_frac"))
    if of > 0 and fb is not None and fb > 0:
        f.add("V4a", "validity", "block",
              f"outcome_from_h = {of} but {fb:.1%} of counted events occur before it",
              evidence={"events_before_outcome_from": tm.get("events_before_outcome_from")},
              fix="Add an inclusion flag that removes units with the outcome before outcome_from_h, or set outcome_from_h = 0.")
    # V4 (tier 2): outcomes are counted while predictors are still being measured
    frac = _num(tm.get("events_before_prediction_time_frac"))
    if pt > of and frac is not None and frac > 0.0:
        if options.get("v4_soft"):
            f.add("V4", "validity", "info",
                  f"{frac:.1%} of outcome events happen before the prediction time ({pt} h). Counting them is standard "
                  "when the protocol defines time zero at admission; a landmark is an optional sensitivity analysis",
                  evidence={"events_before_prediction_time": tm.get("events_before_prediction_time")},
                  fix="Follow the protocol's time zero; report a landmark analysis as sensitivity if relevant.")
        else:
            f.add("V4", "validity", "block" if frac > 0.02 else "warn",
                  f"{frac:.1%} of outcome events occur between {of} h (outcomes counted from) and {pt} h (predictors "
                  "complete): the predictors may reflect the dying process (look-ahead)",
                  evidence={"events_before_prediction_time": tm.get("events_before_prediction_time")},
                  fix=("If the question is a prediction made at hour {0}, landmark (outcome_from_h = {0} plus an "
                       "exclusion flag). If the protocol counts outcomes from time zero (e.g. a day-1 score evaluated "
                       "against all in-hospital deaths), request a waiver citing those protocol items and add the "
                       "landmark as a sensitivity analysis.").format(pt))
    if pt > of and "events_before_prediction_time_frac" not in tm and spec.model.type != "descriptive":
        f.add("V4b", "validity", "warn", "prediction_time_h > 0 but outcome_time_h is not provided, so outcome "
              "events before the prediction time cannot be checked", fix="Return outcome_time_h (hours from index).")
    eaf = _num(tm.get("exposure_after_outcome_frac"))
    if eaf is not None and eaf > 0.01:
        f.add("V5", "validity", "warn", f"{eaf:.1%} of events have exposure after the outcome (reverse timing)",
              evidence={"exposure_after_outcome": tm.get("exposure_after_outcome")},
              fix="Restrict exposure to before the outcome, or use a landmark design.")
    rep = _num(card.get("units_per_patient_gt1_frac")) or 0
    if spec.unit_of_analysis != "patient" and rep > 0.05 and spec.repeated_units == "all_independent":
        f.add("V6", "validity", "warn", f"{rep:.1%} of units are repeat units of the same patient, analysed as independent",
              evidence={"units_per_patient_gt1_frac": rep}, fix="Keep the first unit per patient or justify.")
    if spec.repeated_units == "first_only" and rep > 0.001:
        f.add("V6b", "validity", "block", f"Design says first unit only, but {rep:.1%} of units are repeats",
              evidence={"units_per_patient_gt1_frac": rep}, fix="Apply the first-unit rule in analysis_sql.")
    cont = _num(card.get("contiguous_with_previous_unit_frac"))
    levels = _levels(dataset)
    finest = len(levels) > 1 and spec.unit_of_analysis == levels[0]     # stay-like units can be split by transfers
    if cont is not None and cont > 0.01 and finest:
        f.add("V7", "validity", "warn", f"{cont:.1%} of units start within 1 h of the same patient's previous unit "
              "(transfers split into separate stays?)", evidence={"contiguous_frac": cont},
              fix="Merge contiguous segments into one unit, or use the first segment's start as index.")
    elif "contiguous_with_previous_unit_frac" not in card and finest:
        f.add("V7b", "validity", "info", "unit_end_time not returned: transfer-splitting cannot be checked",
              fix="Return unit_end_time.")
    if "adult" in q and not any(re.search(r"18|adult|age", c.description.lower() + c.flag.lower())
                                for c in spec.population_criteria):
        f.add("V8", "validity", "block", "Question is about adults but no age criterion is applied",
              fix="Add an inclusion flag age >= 18 (computed at admission).")
    age_cols = [c for c in (card.get("table1") or {}) if c.lower() in ("age", "age_years", "age_at_admission")]
    for c in age_cols:
        p01 = _num(card["table1"][c].get("p01"))
        if "adult" in q and p01 is not None and p01 < 18:
            f.add("V8b", "validity", "block", f"Age column '{c}' has values below 18 in an adult cohort",
                  evidence={"p01": p01}, fix="Check the age computation and the adult filter.")
    om = _num(card.get("outcome_missing_frac")) or 0
    if om > 0.05:
        f.add("V10", "validity", "warn", f"Outcome missing for {om:.1%} of units", evidence={"outcome_missing_frac": om},
              fix="Define the outcome for every unit or justify exclusion.")

    # ---------------------------------------------------------------- feasibility
    ev = _num((card.get("outcome") or {}).get("events")) or 0
    nf = _num(card.get("n_final")) or 0
    if ev < MIN_EVENTS:
        f.add("F1", "feasibility", "block", f"Only {card['outcome']['events']} outcome events", fix="Broaden the cohort or outcome window.")
    elif ev < 100:
        f.add("F1", "feasibility", "warn", f"Only {int(ev)} outcome events (<100)")
    m = card.get("model") or {}
    lm = m if spec.model.type == "logistic" else (m.get("logistic") or {})
    epv = _num(lm.get("epv"))
    if epv is not None:
        if epv < 5:
            f.add("F2", "feasibility", "block", f"Events per parameter {epv} < 5", fix="Reduce predictors or merge categories.")
        elif epv < 10:
            f.add("F2", "feasibility", "warn", f"Events per parameter {epv} < 10")
    rf = _num(card.get("retained_frac"))
    if rf is not None and rf < 0.2 and not spec.shrink_justification.strip():
        f.add("F3", "feasibility", "warn", f"Only {rf:.1%} of candidate units retained with no justification",
              evidence={"flow": card.get("flow")}, fix="Justify each restrictive criterion (shrink_justification) or relax it.")
    if nf < 200:
        f.add("F4", "feasibility", "warn", f"Final cohort is small (n={card.get('n_final')})")

    # ---------------------------------------------------------------- data quality
    used = set(spec.model.predictors) | set(spec.model.score_columns) | {spec.model.value_column or ""}
    for col, d in (card.get("table1") or {}).items():
        mf = _num(d.get("missing_frac")) or 0
        if col in {u.split("=")[0] for u in used}:
            if mf > 0.6:
                f.add("Q1", "quality", "block", f"'{col}' is missing in {mf:.0%} of units", fix="Check the mapping/window or drop the variable.")
            elif mf > 0.3:
                f.add("Q1", "quality", "warn", f"'{col}' is missing in {mf:.0%} of units",
                      fix="Check the item-code/window mapping; consider a missing-indicator or a wider window.")
        xf = _num(d.get("extreme_frac"))
        if xf is not None and xf > 0.01:
            f.add("Q2", "quality", "warn", f"'{col}': {xf:.1%} of values are > 10 IQR from the median (unit mix or error values?)",
                  evidence={k: d.get(k) for k in ("p01", "median", "p99")}, fix="Restrict to plausible ranges / one unit of measure.")
        if d.get("zero_variance"):
            f.add("Q3", "quality", "warn", f"'{col}' has no variation in the cohort", fix="Drop it or check the definition.")
    cc = _num(lm.get("complete_case_drop_frac"))
    if cc is not None and cc > 0.2:
        f.add("Q4", "quality", "warn", f"Complete-case analysis drops {cc:.0%} of units",
              fix="Reduce missingness (wider windows, alternative sources) or use missing indicators.")

    # ---------------------------------------------------------------- model adequacy
    for blk in [m, m.get("logistic") or {}]:
        if blk.get("error"):
            f.add("M1", "model", "block", f"Model error: {blk['error']}", fix="Check outcome coding and predictors.")
        if blk.get("converged") is False or blk.get("separation_suspect"):
            f.add("M2", "model", "block", "Model did not converge or shows separation",
                  fix="Remove near-deterministic predictors or merge sparse categories.")
        cs = _num(blk.get("cv_calibration_slope"))
        if cs is not None and not (0.8 <= cs <= 1.2):
            f.add("M3", "model", "warn", f"Cross-validated calibration slope {cs} outside 0.8-1.2 (over/underfitting)",
                  fix="Fewer parameters or more events.")
        p = blk.get("primary") or {}
        if p.get("ci_low") and p.get("ci_high") and p["ci_low"] > 0:
            import math
            w = math.log(p["ci_high"]) - math.log(p["ci_low"])
            if w > 1.5:
                f.add("M4", "model", "warn", f"Primary estimate is imprecise (log-OR CI width {w:.2f})")
    if spec.model.type == "score_eval":
        scores = m.get("scores", [])
        for s in scores[1:]:
            same = _num(s.get("same_as_first"))
            if same is not None and same >= 0.99:          # E-69: a score compared with itself shows no difference
                f.add("M6", "model", "block", f"Score '{s['score']}' equals '{scores[0]['score']}' for {same:.0%} of "
                      "units: the comparison cannot show a difference",
                      fix="Compute each score from its own definition (components and thresholds); the second score "
                          "must not reuse the first one's SQL or thresholds.")
        for s in scores:
            mf = _num(s.get("missing_frac")) or 0
            if mf > 0.2:
                f.add("M5", "model", "warn", f"Score '{s['score']}' missing for {mf:.0%} of units: AUROC is on a selected subset",
                      fix="Define the score for all units (impute missing components as normal, as is standard) or justify.")

    # ---------------------------------------------------------------- stability
    cur = None if options.get("blind") else primary_value(card)
    prev = None
    for h in reversed(history):
        prev = primary_value(h.get("card", {}))
        if prev is not None:
            break
    stab = {}
    if cur is None and not options.get("blind"):
        f.add("S1", "stability", "warn", "No primary estimate was produced", fix="Designate the main result (primary_term / score).")
    elif prev is not None and prev != 0:
        rel = abs(cur - prev) / abs(prev)
        stab = {"current": cur, "previous": prev, "relative_change": round(rel, 4)}
        if rel > 0.03:   # v3: info only - penalising change rewarded settling early
            f.add("S2", "stability", "info", f"Primary estimate changed {rel:.1%} from the previous iteration", evidence=stab)
    nf_prev = None
    for h in reversed(history):
        nf_prev = _num((h.get("card") or {}).get("n_final"))
        if nf_prev:
            break
    if nf_prev and nf:
        ch = abs(nf - nf_prev) / nf_prev
        if ch > 0.2:
            f.add("S3", "stability", "info", f"Cohort size changed {ch:.0%} from the previous iteration",
                  evidence={"n_final": nf, "previous": nf_prev})
    from ucd import packs                              # topic packs: tested concepts not used for a known topic
    protocol = options.get("protocol") or spec.question
    for pk in packs.checks(protocol, spec.analysis_sql, dataset):
        f.add(pk["rule"], "validity", pk.get("severity", "warn"), pk["message"], fix=pk["fix"])
    event_rules(f, spec, card, protocol, protocol_items, dataset)
    if spec.model.type == "cohort":                    # no outcome and no model: those rules do not apply
        f.items = [i for i in f.items if i["rule"] not in COHORT_SKIP]
    out = _score(f, gate=align_fail)
    out["stability"] = stab
    return out


def _levels(dataset: Optional[str]) -> list[str]:
    from ucd import datasets
    try:
        return datasets.levels(dataset) if dataset else []
    except ValueError:                                 # an unregistered dataset name (tests, old runs)
        return []


def _death_fields(dataset: Optional[str]) -> tuple[list[str], list[str]]:
    from ucd import datasets
    try:
        return datasets.death_fields(dataset) if dataset else ([], [])
    except ValueError:
        return [], []


def event_rules(f: "_F", spec: DesignSpec, card: dict[str, Any], protocol: str,
                items: Optional[list[dict[str, Any]]], dataset: Optional[str] = None) -> None:
    """Rules for event-anchored and landmark designs (docs/critic_rules.md): deterministic, topic-independent checks
    on the spec, the SQL text and aggregate counts."""
    import yaml

    from ucd import sqlcte, terms
    sql = re.sub(r"'(?:[^']|'')*'", "''", spec.analysis_sql or "")
    L = float(spec.outcome_from_h or 0)
    if L > 0 and float(spec.prediction_time_h or 0) == L:                       # T2 landmark flag present
        li = f"{L:g}"
        in_sql = re.search(rf"INTERVAL\s*'?\s*{li}\s*(?:HOURS?\b|')|{li}\s*\*\s*3600\b|"
                           rf"date_diff\s*\(\s*''[\s\S]{{0,200}}?\b{li}\b|INTERVAL\s*'?\s*{L / 24:g}\s*DAYS?\b", sql, re.I)
        in_flags = re.search(rf"landmark|\b{li}\s*h|alive", " ".join(c.description for c in spec.population_criteria), re.I)
        if not (in_sql and in_flags):
            f.add("T2", "validity", "block" if not in_flags else "warn", f"A {li} h landmark is declared (outcome_from_h = prediction_time_h = {li}) "
                  f"but no population flag keeps only units alive and still in the unit at time zero + {li} h",
                  fix=f"Add a flag 'alive and in the unit at index_time + {li} h' (unit_end_time >= index_time + "
                      f"INTERVAL {li} HOUR and no death before it) and list it in population_criteria.")
    out_win = _num((card.get("timing") or {}).get("exposed_outside_window"))  # T3 exposure before the landmark
    if out_win:
        f.add("T3", "validity", "block", f"{card['timing']['exposed_outside_window']} exposed units start the exposure "
              f"outside [0, {float(spec.exposure.window_end_h if spec.exposure and spec.exposure.window_end_h is not None else spec.prediction_time_h or 0):g}] h "
              "from time zero", fix="Count only exposure that starts within the "
              "window (after the landmark it is a trajectory, not the exposure).")
    scope = "\n".join(l for l in (protocol or "").splitlines() if re.match(r"\s*(population|objective)\s*:", l, re.I)) or protocol
    upp = _num(card.get("units_per_patient_gt1_frac"))                          # T6 unit level
    if upp and re.search(r"per patient|each patient", scope or "", re.I) and not any(i["rule"] == "V6b" for i in f.items):
        f.add("T6", "validity", "block", f"The protocol asks for one unit per patient, but {upp:.1%} of units are repeat "
              "units of the same patient", fix="Rank the units within the patient (row_number() = 1 partitioned by the "
              "patient id), not within the admission.")
    outcome = "\n".join(l for l in (protocol or "").splitlines() if re.match(r"\s*outcome\s*:", l, re.I))
    in_hosp, after = _death_fields(dataset)                                    # T7, from the registry's `death`
    used = [c for c in in_hosp if re.search(rf"\b{re.escape(c)}\b", sql, re.I)]
    if used and after and re.search(r"all[- ]cause|any death record|after discharge|post-discharge|\d+[- ]day",
                                    outcome, re.I) and not any(re.search(rf"\b{re.escape(c)}\b", sql, re.I) for c in after):
        f.add("T7", "validity", "warn", "The outcome is all-cause death over a fixed horizon, but the SQL uses only "
              f"{', '.join(used)}, which records in-hospital deaths only", fix=f"Also use {', '.join(after)} for deaths "
              "after discharge (see the dataset notes for how it is recorded).")
    parts = sqlcte.split(spec.analysis_sql or "")                              # T10 flow uninformative
    final = re.sub(r"'(?:[^']|'')*'", "''", parts[1] if parts else spec.analysis_sql or "")
    flags = [c.flag for c in spec.population_criteria]
    flow_n = [s.get("n_units") for s in (card.get("flow") or [])]
    filtered = any(re.search(rf"\bWHERE\b[\s\S]*\b{re.escape(fl)}\b", final, re.I) for fl in flags)
    if filtered or (len(flags) >= 2 and len(flow_n) > 2 and len(set(map(str, flow_n))) == 1):
        f.add("T10", "validity", "block" if filtered else "warn",
              "The flow is uninformative: " + ("the final SELECT filters on the flag columns" if filtered else
                                               "no population flag removes any unit"),
              fix="Return every candidate row; the runner applies the flags in order.")
    if items:                                                                  # T11 dropped design terms
        text = yaml.safe_dump(items, allow_unicode=True) + " ".join(
            f"{m.item} {m.implemented_as} {m.deviation}" for m in spec.protocol_mapping)
        from ucd.packs import _scope
        gone = terms.missing(protocol, text)
        core = set(terms.missing(_scope(protocol), text))
        if gone:
            f.add("T11", "validity", "block" if core else "warn", f"Design terms of the protocol are missing from "
                  f"protocol_items.yaml: {', '.join(gone[:12])}", fix="Add a protocol item (or a declared deviation) for each.")
    if re.search(r"\blabel\s+I?LIKE\s*'", spec.analysis_sql or "", re.I):     # T13 dictionary-label matching
        f.add("T13", "validity", "warn", "Items are selected by matching dictionary labels (label LIKE '%...%')",
              fix="Use explicit item codes from lookup, or the tested concept.* table for the concept.")


COHORT_FIRST = ("T", "V", "P", "A", "U", "PK")  # cohort-definition rules are listed (and fixed) before model rules


def _score(f: _F, hard: bool = False, gate: bool = False) -> dict[str, Any]:
    keep = {d: 1.0 for d in WEIGHTS}
    for it in f.items:
        keep[it["dimension"]] *= (1.0 - SEV[it["severity"]])
    dims = {d: 1.0 - v for d, v in keep.items()}
    I = 1.0 if hard else sum(WEIGHTS[d] * v for d, v in dims.items())
    if gate:
        I = max(I, 0.5)
    order = {"block": 0, "warn": 1, "info": 2}
    items = sorted(f.items, key=lambda x: (order[x["severity"]], not x["rule"].startswith(COHORT_FIRST), x["rule"]))
    return {"internal_error": round(I, 4), "dimensions": {k: round(v, 3) for k, v in dims.items()},
            "findings": items, "n_block": sum(i["severity"] == "block" for i in items), "hard": hard, "gate": gate}


QUOTE_MATCH = 0.85   # weak local judges paraphrase quotes; accept a close match, never an invented sentence


def quote_supported(quote: str, protocol: str) -> bool:
    """True if the judge's quote appears in the protocol verbatim, or closely matches one protocol sentence /
    line (difflib ratio >= QUOTE_MATCH over the best-aligned window)."""
    import difflib
    import re as _re
    q, prot = _norm(quote), _norm(protocol)
    if len(q) < 12:
        return False
    if q in prot:
        return True
    units = [_norm(u) for u in _re.split(r"(?<=[.;:])\s+|\n", protocol) if len(_norm(u)) >= 8]
    best = 0.0
    for u in units:
        if len(u) >= len(q):     # slide a window of the quote's length over the longer unit
            for i in range(0, len(u) - len(q) + 1, max(1, len(q) // 8)):
                best = max(best, difflib.SequenceMatcher(None, q, u[i:i + len(q)]).ratio())
        else:
            best = max(best, difflib.SequenceMatcher(None, q, u).ratio())
        if best >= QUOTE_MATCH:
            return True
    return False


def _norm(t: str) -> str:
    t = str(t).lower().replace("\u2265", ">=").replace("\u2264", "<=")
    return " ".join(t.split())


def apply_waivers(crit: dict[str, Any], spec: DesignSpec, decisions: list[dict[str, Any]], protocol: str,
                  card: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the independent judge's waiver decisions to tier-2 findings. Code-side checks: the rule is tier 2 and
    was requested by the planner; the judge's quote appears verbatim in the protocol; landmark-type rules need a
    sensitivity analysis that ran; at most MAX_WAIVERS. A granted waiver turns the finding into 'info' (it is not
    removed, so the report still lists it)."""
    requested = {w.rule.split("-")[0]: w for w in spec.waivers}
    log, granted = [], 0
    prot = _norm(protocol)
    sens_ok = any(r.get("primary") and not r.get("error") for r in (card.get("sensitivity") or []))
    by_rule = {}
    for d in decisions or []:
        if isinstance(d, dict) and d.get("rule"):
            by_rule[str(d["rule"]).split("-")[0]] = d
    for rule, w in requested.items():
        d = by_rule.get(rule, {})
        quote = _norm(d.get("quote", ""))
        rec = {"rule": rule, "requested_items": w.protocol_items, "judge_granted": bool(d.get("granted")),
               "quote": d.get("quote", ""), "reason": d.get("reason", "")}
        why = None
        if tier(rule) != 2:
            why = f"rule {rule} is not waivable (tier {tier(rule)})"
        elif not any(x["rule"].split("-")[0] == rule and x["severity"] in ("block", "warn") for x in crit["findings"]):
            why = "rule did not fire in this iteration"
        elif not d.get("granted"):
            why = "judge denied: " + str(d.get("reason", "no decision returned"))[:300]
        elif not quote_supported(d.get("quote", ""), protocol):
            why = "judge's quote was not found in the protocol (verbatim or close match)"
        elif rule in NEEDS_SENSITIVITY and not sens_ok:
            why = "a landmark-type sensitivity analysis is required and none ran"
        elif granted >= MAX_WAIVERS:
            why = f"waiver cap ({MAX_WAIVERS}) reached"
        rec["applied"] = why is None
        rec["refused_because"] = why
        if why is None:
            granted += 1
            for x in crit["findings"]:
                if x["rule"].split("-")[0] == rule and x["severity"] in ("block", "warn"):
                    x["severity"] = "info"
                    x["message"] = f"[WAIVED: {str(d.get('reason', ''))[:200]}] " + x["message"]
                    x["waived"] = True
        log.append(rec)
    f = _F()
    f.items = crit["findings"]
    out = _score(f, hard=crit.get("hard", False), gate=crit.get("gate", False))
    for k in ("stability", "I_deterministic", "llm_overall", "clinical_plausibility"):
        if k in crit:
            out[k] = crit[k]
    out["waivers"] = log
    return out, log


def rescore(findings: list[dict[str, Any]], hard: bool = False, gate: bool = False) -> float:
    """Recompute I from stored findings with the current scoring (used for post-hoc, version-consistent analysis)."""
    f = _F()
    f.items = [dict(x, severity=("info" if x.get("rule") == "P3" else x["severity"])) for x in findings]
    return _score(f, hard=hard, gate=gate)["internal_error"]


def merge(crit: dict[str, Any], extra: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-score after adding findings (e.g. from the LLM critic)."""
    f = _F()
    f.items = list(crit["findings"]) + list(extra)
    out = _score(f, hard=crit.get("hard", False), gate=crit.get("gate", False))
    out["I_deterministic"] = crit["internal_error"]
    for k in ("stability",):
        if k in crit:
            out[k] = crit[k]
    return out


MAX_ITER = 3      # executed design iterations per run (was 6; reduced 2026-10-17 for speed)
MAX_ATTEMPTS = 6  # all iterations incl. invalid specs / SQL errors, so a model that never gets a spec right stops


def should_stop(I_history: list[float], n_block_last: int = 1, max_iter: int = MAX_ITER) -> tuple[bool, str]:
    """Stop as soon as an executed design has no blocking finding, or after max_iter executed designs (0.9.0).
    The internal error I is still computed and logged for comparison with earlier runs, but no longer decides: it
    barely moved while blocks remained (E-22) and did not track the external error."""
    if n_block_last == 0:
        return True, "no blocking findings"
    if len(I_history) >= max_iter:
        return True, f"reached max iterations ({max_iter})"
    return False, ""
