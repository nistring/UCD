"""UCD unit tests: gate refusals, runner arithmetic, critic rules (no real data needed)."""
import math

import numpy as np
import pandas as pd
import pytest

from ucd import critic as C
from ucd.gate import Gate, GateError, check_select
from ucd.runner import run
from ucd.spec import DesignSpec


# ------------------------------------------------------------------ gate
@pytest.mark.parametrize("sql", [
    "DELETE FROM hosp.patients",
    "SELECT 1; SELECT 2",
    "COPY (SELECT 1) TO 'x.csv'",
    "SELECT * FROM read_parquet('C:/x.parquet')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "ATTACH 'x.db'",
])
def test_gate_rejects_statements(sql):
    with pytest.raises(GateError):
        check_select(sql)


def test_gate_rejects_identifier_columns():
    with pytest.raises(GateError):
        Gate.sanitize(pd.DataFrame({"subject_id": [1], "n": [100]}))


def test_gate_rejects_datetime_columns():
    with pytest.raises(GateError):
        Gate.sanitize(pd.DataFrame({"t": pd.to_datetime(["2150-01-01"]), "n": [100]}))


def test_gate_requires_n_and_suppresses_small_cells():
    with pytest.raises(GateError):
        Gate.sanitize(pd.DataFrame({"x": [1]}))
    out = Gate.sanitize(pd.DataFrame({"grp": ["a", "b", "c"], "n": [100, 9, 10]}))
    assert [r["grp"] for r in out["rows"]] == ["a", "c"]
    assert out["suppressed_rows"] == 1


def test_gate_row_dump_is_truncated():
    out = Gate.sanitize(pd.DataFrame({"v": range(80), "n": [50] * 80}))
    assert len(out["rows"]) == 50 and out["truncated"]


# ------------------------------------------------------------------ runner
class FakeGate:
    dataset = "mimic"

    def __init__(self, df):
        self.df = df

    def materialize(self, sql):
        return self.df.copy()


def _toy(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    age = rng.normal(60, 15, n)
    p = 1 / (1 + np.exp(-(-2 + math.log(2.0) * x + 0.02 * (age - 60))))
    y = rng.binomial(1, p)
    t0 = pd.Timestamp("2150-01-01")
    return pd.DataFrame({"unit_id": np.arange(n), "subject_key": np.arange(n), "index_time": t0,
                         "unit_end_time": t0 + pd.Timedelta(hours=48), "incl_adult": age >= 18,
                         "outcome": y, "exposed": x, "age": age, "outcome_time_h": np.where(y == 1, 72.0, np.nan)})


def _spec(**kw):
    base = dict(title="t", question="Is exposure associated with death in adult ICU patients?",
                pico={"population": "adult ICU", "exposure_or_predictor": "exposed", "outcome": "death"},
                unit_of_analysis="icu_stay", index_time="ICU admission",
                population_criteria=[{"flag": "incl_adult", "description": "age >= 18"}],
                exposure={"column": "exposed", "description": "x", "type": "binary", "window_start_h": -24, "window_end_h": 0},
                outcome={"column": "outcome", "description": "death"},
                covariates=[{"column": "age", "description": "age", "window_start_h": 0, "window_end_h": 0}],
                model={"type": "logistic", "predictors": ["exposed", "age"], "primary_term": "exposed"},
                analysis_sql="SELECT 1")
    base.update(kw)
    return DesignSpec.model_validate(base)


def test_runner_recovers_known_or():
    df = _toy()
    card, _ = run(_spec(), FakeGate(df))
    import statsmodels.api as sm
    d = df[df.incl_adult]
    fit = sm.Logit(d.outcome, sm.add_constant(d[["exposed", "age"]])).fit(disp=0)
    assert card["model"]["primary"]["or"] == pytest.approx(math.exp(fit.params["exposed"]), rel=1e-3)
    assert 1.5 < card["model"]["primary"]["or"] < 2.7


def test_critic_blocks_a_score_compared_with_itself():
    """M6 (E-69): sepsis_rough compared 'SOFA-2' built with the SOFA-1 thresholds (once copied) with SOFA-1."""
    df = _toy()
    df["s1"] = (df.age // 10).astype(int)
    df["s2"], df["s3"] = df.s1, df.s1 + (df.exposed * 2)
    for second, blocked in (("s2", True), ("s3", False)):
        spec = _spec(model={"type": "score_eval", "score_columns": ["s1", second]}, covariates=[],
                     exposure={"column": "s1", "description": "x", "type": "continuous", "window_start_h": 0, "window_end_h": 0})
        card, _ = run(spec, FakeGate(df))
        assert card["model"]["scores"][1]["same_as_first"] == (1.0 if blocked else pytest.approx(0.5, abs=0.05))
        m6 = [f for f in C.critique(spec, card, [], "mimic")["findings"] if f["rule"] == "M6"]
        assert bool(m6) is blocked and (not m6 or m6[0]["severity"] == "block")


def test_critic_clean_design_has_low_error():
    card, _ = run(_spec(), FakeGate(_toy()))
    crit = C.critique(_spec(), card, [], "mimic")
    assert crit["n_block"] == 0, crit["findings"]


def test_critic_flags_lookahead_covariate():
    s = _spec(covariates=[{"column": "age", "description": "max lab over stay", "window_start_h": 0, "window_end_h": 168}])
    card, _ = run(s, FakeGate(_toy()))
    rules = {f["rule"] for f in C.critique(s, card, [], "mimic")["findings"]}
    assert "V2" in rules


def test_critic_flags_immortal_time_exposure():
    df = _toy()
    df["outcome_time_h"] = np.where(df.outcome == 1, 10.0, np.nan)   # deaths inside the exposure window
    s = _spec(exposure={"column": "exposed", "description": "x within 72h", "type": "binary",
                        "window_start_h": 0, "window_end_h": 72})
    card, _ = run(s, FakeGate(df))
    f = [x for x in C.critique(s, card, [], "mimic")["findings"] if x["rule"] == "V3"]
    assert f and f[0]["severity"] == "block"


def test_critic_flags_repeat_units_and_missing_adult_filter():
    df = _toy()
    df["subject_key"] = df["unit_id"] // 2
    df["incl_all"] = True
    s = _spec(population_criteria=[{"flag": "incl_all", "description": "all ICU stays"}])
    card, _ = run(s, FakeGate(df))
    rules = {f["rule"] for f in C.critique(s, card, [], "mimic")["findings"]}
    assert {"V6b", "V8"} <= rules


def test_stop_rule():
    assert C.should_stop([0.5, 0.3, 0.29, 0.29])[0]
    assert not C.should_stop([0.5, 0.3])[0]
    assert C.should_stop([0.1, 0.1], n_block_last=0)[0]
    assert not C.should_stop([0.5, 0.5], n_block_last=1)[0]
    assert C.should_stop([0.9] * 6, n_block_last=2)[0]


# ------------------------------------------------------------------ v3: cox, sensitivity, waivers
def _cox_toy(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    texp = np.where(rng.random(n) < 0.4, rng.uniform(0, 48, n), np.nan)      # time-dependent exposure
    t0 = pd.Timestamp("2150-01-01")
    # constant baseline hazard 0.005/h, exposure HR 2 after it starts
    base_t = rng.exponential(1 / 0.005, n)
    t = base_t.copy()
    exp_mask = ~np.isnan(texp) & (base_t > texp)
    t[exp_mask] = texp[exp_mask] + (base_t[exp_mask] - texp[exp_mask]) / 2.0
    cens = np.full(n, 240.0)
    ev = (t <= cens).astype(int)
    follow = np.minimum(t, cens)
    return pd.DataFrame({"unit_id": np.arange(n), "subject_key": np.arange(n), "index_time": t0,
                         "unit_end_time": t0 + pd.Timedelta(hours=240), "incl_adult": True, "outcome": ev,
                         "followup_h": follow, "texp_h": texp, "age": age,
                         "outcome_time_h": np.where(ev == 1, follow, np.nan),
                         "incl_late": follow > 24})


def test_cox_time_dependent_recovers_hr():
    s = _spec(exposure=None, covariates=[{"column": "age", "description": "age", "window_start_h": 0, "window_end_h": 0}],
              model={"type": "cox", "time_column": "followup_h", "td_exposure_time_column": "texp_h",
                     "predictors": ["age"], "primary_term": "td_exposure"},
              sensitivity=[{"name": "alive at 24 h", "flag": "incl_late"}])
    card, _ = run(s, FakeGate(_cox_toy()))
    hr = card["model"]["primary"]["hr"]
    assert 1.6 < hr < 2.5, card["model"]
    assert card["sensitivity"][0]["primary"]["hr"] > 1.3


def _v4_case():
    df = _toy()
    df["outcome_time_h"] = np.where(df.outcome == 1, 10.0, np.nan)          # all deaths before 24 h
    df["incl_alive24"] = df.outcome == 0
    s = _spec(prediction_time_h=24, outcome_from_h=0,
              covariates=[{"column": "age", "description": "age", "window_start_h": 0, "window_end_h": 24}],
              waivers=[{"rule": "V4", "protocol_items": ["P1"], "justification": "protocol counts all deaths from admission"}],
              sensitivity=[{"name": "landmark 24 h", "flag": "incl_alive24"}])
    return df, s


PROTOCOL = "Time zero: ICU admission. Outcome: in-hospital death. Score window: first 24 h of ICU stay."


def test_v4_fires_as_tier2_and_waiver_applies_with_verbatim_quote():
    df, s = _v4_case()
    card, _ = run(s, FakeGate(df))
    crit = C.critique(s, card, [], "mimic")
    assert any(f["rule"] == "V4" and f["severity"] == "block" for f in crit["findings"])
    good = [{"rule": "V4", "granted": True, "quote": "Time zero: ICU admission. Outcome: in-hospital death.", "reason": "protocol"}]
    out, log = C.apply_waivers(crit, s, good, PROTOCOL, card)
    assert log[0]["applied"] is False   # the sensitivity flag refits on survivors only -> no events -> no estimate
    card["sensitivity"] = [{"name": "x", "primary": {"or": 1.2}}]
    crit = C.critique(s, card, [], "mimic")
    out, log = C.apply_waivers(crit, s, good, PROTOCOL, card)
    assert log[0]["applied"] and out["internal_error"] < crit["internal_error"]


def test_waiver_refused_for_fake_quote_or_tier1():
    df, s = _v4_case()
    card, _ = run(s, FakeGate(df))
    card["sensitivity"] = [{"name": "x", "primary": {"or": 1.2}}]
    crit = C.critique(s, card, [], "mimic")
    fake = [{"rule": "V4", "granted": True, "quote": "Deaths before 24 hours are part of the outcome.", "reason": "x"}]
    _, log = C.apply_waivers(crit, s, fake, PROTOCOL, card)
    assert not log[0]["applied"] and "verbatim" in log[0]["refused_because"]
    s2 = s.model_copy(update={"waivers": [{"rule": "V1", "protocol_items": ["P1"], "justification": "x"}]})
    s2 = DesignSpec.model_validate(s2.model_dump())
    _, log = C.apply_waivers(crit, s2, [{"rule": "V1", "granted": True, "quote": PROTOCOL, "reason": "x"}], PROTOCOL, card)
    assert not log[0]["applied"] and "not waivable" in log[0]["refused_because"]


def test_v4a_mechanical_block_when_landmark_declared_but_not_applied():
    df, s = _v4_case()
    s = s.model_copy(update={"outcome_from_h": 24.0})
    card, _ = run(s, FakeGate(df))
    rules = {f["rule"] for f in C.critique(s, card, [], "mimic")["findings"]}
    assert "V4a" in rules and "V4" not in rules


# ------------------------------------------------------------------ structured explore (needs local MIMIC-IV parquet)
from ucd.gate import PARQUET
needs_mimic = pytest.mark.skipif(not (PARQUET / "mimic" / "hosp" / "patients.parquet").exists(), reason="no MIMIC parquet")


@needs_mimic
def test_aggregate_gate_computes_n_and_refuses_leaks():
    g = Gate("mimic")
    out = g.aggregate("SELECT gender, hospital_expire_flag AS d FROM hosp.patients JOIN hosp.admissions USING (subject_id)",
                      ["gender"], ["avg(d) AS mortality"])
    assert out["rows"] and all(r["n"] >= 10 for r in out["rows"])
    for src, grp, met in [("SELECT subject_id AS s FROM hosp.patients", ["s"], []),            # aliased identifier
                          ("SELECT anchor_age FROM hosp.patients", [], ["max(anchor_age) AS m"]),  # single value
                          ("SELECT anchor_age FROM hosp.patients", [], ["quantile_cont(anchor_age, 0.99)"]),
                          ("SELECT anchor_age FROM hosp.patients", [], ["string_agg(anchor_age, ',')"])]:
        with pytest.raises(GateError):
            g.aggregate(src, grp, met)


def test_unknown_dataset_is_refused():
    with pytest.raises(ValueError, match="available: mimic"):
        Gate("kmimic")


def test_runner_never_prints_small_category_labels():
    df = _toy()
    df["cat"] = ["rare_%d" % i if i < 5 else "common" for i in range(len(df))]
    s = _spec(covariates=[{"column": "cat", "description": "c", "type": "categorical", "window_start_h": 0, "window_end_h": 0}],
              model={"type": "logistic", "predictors": ["exposed"], "primary_term": "exposed"})
    card, _ = run(s, FakeGate(df))
    assert not any(k.startswith("rare_") for k in card["table1"]["cat"]["levels"])


def test_cluster_robust_se_widens_ci_with_repeat_units():
    df = _toy()
    df = pd.concat([df, df.assign(unit_id=df.unit_id + 10**6)], ignore_index=True)   # every patient twice
    s_plain = _spec(repeated_units="all_independent")
    s_cl = _spec(repeated_units="all_independent",
                 model={"type": "logistic", "predictors": ["exposed", "age"], "primary_term": "exposed",
                        "cluster_by_patient": True})
    p1 = run(s_plain, FakeGate(df))[0]["model"]["primary"]
    p2 = run(s_cl, FakeGate(df))[0]["model"]["primary"]
    assert p2["or"] == pytest.approx(p1["or"], rel=1e-6)
    assert (p2["ci_high"] - p2["ci_low"]) > (p1["ci_high"] - p1["ci_low"])


def test_blind_card_hides_associations_but_keeps_design_checks():
    from ucd.blinding import blind_card
    card, _ = run(_spec(), FakeGate(_toy()))
    b = blind_card(card)
    txt = str(b["model"])
    assert "terms" not in b["model"] and "primary" not in b["model"] and "auroc" not in txt
    assert b["outcome"]["events"] == card["outcome"]["events"] and b["model"]["epv"] == card["model"]["epv"]
    assert all("median_outcome1" not in v for v in b["table1"].values())
