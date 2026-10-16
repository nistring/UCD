"""Event-anchored designs (0.10.0, topic-independent since 0.13.2): the weighted_binary model and the event-design
critic rules T2-T13. No data, no network."""
import numpy as np
import pandas as pd


def _spec(**model):
    import subprocess
    import sys

    import yaml

    from ucd.spec import DesignSpec
    base = yaml.safe_load(subprocess.run([sys.executable, "-m", "ucd", "template"], capture_output=True, text=True).stdout)
    base.update({"title": "t", "exposure": {"column": "early_tx", "description": "treatment within 24 h", "type": "binary"},
                 "covariates": [{"column": "x", "description": "confounder", "type": "numeric"}],
                 "model": {"type": "weighted_binary", "ps_covariates": ["x"], "bootstrap": 60, **model},
                 "population_criteria": [{"flag": "incl_all", "description": "all"}], "protocol_mapping": [],
                 "analysis_sql": "SELECT 1"})
    return DesignSpec(**base)


def test_weighted_binary_removes_confounding_and_balances_covariates():
    from ucd.runner import fit_weighted_binary
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-1.2 * x)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 1.0 * x))))           # no exposure effect: true RD = 0
    t = np.where(y == 1, 24 + rng.uniform(0, 28 * 24, n), 24 + 28 * 24)  # hours from T0; landmark 24 h
    df = pd.DataFrame({"early_tx": a, "x": x, "t": t})
    spec = _spec(horizon_days=28, time_column="t")
    spec = spec.model_copy(update={"outcome_from_h": 24})
    out = fit_weighted_binary(df, spec, pd.Series(y))
    crude = out["crude"]["risk_exposed"] - out["crude"]["risk_unexposed"]
    assert abs(crude) > 0.1                                          # confounded
    assert abs(out["primary"]["value"]) < 0.04                       # weighting removes it
    lo, hi = out["primary"]["ci_low"], out["primary"]["ci_high"]
    assert lo < 0 < hi
    assert out["balance"]["max_abs_smd_after"] < 0.02 < out["balance"]["max_abs_smd_before"]
    assert abs(out["rmst_difference_days"]["value"]) < 1.5 and out["weights"] == "overlap"


def test_weighted_binary_needs_both_groups():
    from ucd.runner import fit_weighted_binary
    df = pd.DataFrame({"early_tx": [1] * 5 + [0] * 100, "x": np.arange(105.0)})
    assert "fewer than 10" in fit_weighted_binary(df, _spec(), pd.Series([0, 1] * 52 + [0]))["error"]


def _event_spec(sql, **over):
    s = _spec()
    return s.model_copy(update={"analysis_sql": sql, "prediction_time_h": 24, "outcome_from_h": 24, **over})


def test_event_design_rules_fire_on_the_usual_failure_modes():
    """T2, T3, T6, T7, T10, T11, T13 on the mistakes of the first event-anchored run."""
    from ucd import critic as C
    from ucd.spec import Criterion as PopCriterion
    protocol = ("Objective: early treatment after an in-stay event\n"
                "Population: adults; unit = each patient's first ICU stay.\n"
                "Outcome: all-cause death within 28 days after the 24 h landmark (any death record).")
    bad_sql = ("WITH k AS (SELECT stay_id, charttime FROM concept.sofa WHERE sofa_24hours >= 2), "
               "c AS (SELECT stay_id FROM icu.procedureevents p JOIN icu.d_items d USING (itemid) WHERE d.label ILIKE '%vent%') "
               "SELECT k.stay_id AS unit_id, a.hospital_expire_flag AS outcome, TRUE AS incl_a, TRUE AS incl_b FROM k "
               "WHERE incl_a")
    spec = _event_spec(bad_sql).model_copy(update={"population_criteria": [
        PopCriterion(flag="incl_a", description="adults"), PopCriterion(flag="incl_b", description="event")]})
    card = {"flow": [{"n_units": 1518}, {"n_units": 1518}, {"n_units": 1518}], "units_per_patient_gt1_frac": 0.2,
            "timing": {"exposed_outside_window": 12}}
    f = C._F()
    C.event_rules(f, spec, card, protocol, [{"id": "P1", "text": "adults"}])
    assert {"T2", "T3", "T6", "T7", "T10", "T11", "T13"} <= {i["rule"] for i in f.items}


def test_cohort_definition_findings_come_first():
    from ucd import critic as C
    f = C._F()
    f.add("M3", "model", "block", "model"); f.add("T10", "validity", "block", "flow"); f.add("Q1", "quality", "block", "q")
    assert [i["rule"] for i in C._score(f)["findings"]][0] == "T10"


def test_heuristic_event_rules_block_only_on_clear_evidence():
    """E-61: a false block costs a design attempt, so T2/T10/T11 block only on clear evidence."""
    from ucd import critic as C
    from ucd.spec import Criterion as PopCriterion
    crit = [PopCriterion(flag="incl_a", description="adults"),
            PopCriterion(flag="incl_alive", description="alive and in the ICU at the 24 h landmark")]
    sql = "SELECT 1 AS unit_id, date_diff('hour', intime, outtime) >= 24 AS incl_alive, TRUE AS incl_a FROM x"
    spec = _event_spec(sql).model_copy(update={"population_criteria": crit})
    f = C._F()
    C.event_rules(f, spec, {"flow": [{"n_units": 9}, {"n_units": 9}, {"n_units": 9}]},
                  "Objective: x\nBackground: an earlier study used propensity scores.", [{"id": "P1", "text": "x"}])
    sev = {i["rule"]: i["severity"] for i in f.items}
    assert "T2" not in sev                                   # landmark declared, date_diff form recognised
    assert sev.get("T10") == "warn"                          # flat flow without filtering in the final SELECT
    assert sev.get("T11") == "warn"                          # the term is only in a background line
