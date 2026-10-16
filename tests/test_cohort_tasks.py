"""Cohort-only tasks, verdict.json and task scoring. No data, no network."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from ucd import loop, verdict


def _spec(sql="SELECT 1"):
    from ucd.spec import DesignSpec
    base = yaml.safe_load(subprocess.run([sys.executable, "-m", "ucd", "template"], capture_output=True, text=True).stdout)
    base.update({"title": "t", "model": {"type": "cohort"}, "exposure": None, "covariates": [], "protocol_mapping": [],
                 "pico": {"population": "adults", "exposure_or_predictor": "", "comparator": "", "outcome": ""},
                 "outcome": {"column": "outcome", "description": "none"},
                 "population_criteria": [{"flag": "incl_x", "description": "x"}], "analysis_sql": sql})
    return DesignSpec(**base)


def test_cohort_task_needs_no_outcome_and_accepts_an_empty_cohort():
    from ucd import critic as C
    from ucd.runner import run
    spec = _spec()
    assert "outcome" not in spec.columns_needed()
    for n_true in (0, 40):
        df = pd.DataFrame({"unit_id": range(50), "subject_key": range(50), "index_time": pd.Timestamp("2150-01-01"),
                           "incl_x": [i < n_true for i in range(50)]})
        card, _ = run(spec, SimpleNamespace(dataset="mimic", materialize=lambda sql: df.copy()))
        assert "error" not in card and card["empty"] is (n_true == 0) and card["model"]["type"] == "cohort"
        rules = {f["rule"] for f in C.critique(spec, card, [], "mimic")["findings"]}
        assert not rules & C.COHORT_SKIP and "V0" not in rules


def _run_dir(tmp_path, empty):
    d = tmp_path / "mimic"
    d.mkdir(parents=True)
    (d / "iterations.jsonl").write_text(json.dumps({"iter": 1, "card": {"n_final": 0 if empty else 500, "empty": empty}})
                                        + "\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("empty, own, hint, want", [
    (True, None, "", "empty_result"),
    (True, {"status": "normal"}, "", "empty_result"),          # an empty cohort is never "normal"
    (False, {"status": "partial", "reason": "r"}, "", "partial"),
    (False, None, "", "normal"),
    (False, None, "contradictory", "contradictory"),
])
def test_verdict_is_derived_from_what_happened(tmp_path, empty, own, hint, want):
    rd = _run_dir(tmp_path, empty)
    if own:
        (rd / "verdict.json").write_text(json.dumps(own), encoding="utf-8")
    assert verdict.finalize(rd, hint, "why")["status"] == want
    assert verdict.finalize(tmp_path / "nothing_ran")["status"] == "unanswerable"


def test_verdict_json_is_validated_when_the_planner_writes_it(tmp_path):
    bad = loop._act({"action": "write", "file": "verdict.json", "content": '{"status": "fine"}'}, tmp_path, 0)
    assert bad.startswith("REFUSED") and "status" in bad
    ok = loop._act({"action": "write", "file": "verdict.json", "content": '{"status": "partial", "reason": "x"}'}, tmp_path, 0)
    assert ok == "wrote verdict.json"
    assert verdict.infeasible_status({"reason": "the criteria contradict each other"}) == "contradictory"


def test_task_score_checks_status_and_cohort(tmp_path):
    from ucd.sample import task_score
    gt = tmp_path / "expected.csv"
    gt.write_text("stay_id,index_time\n1,2150-01-01 00:00:00\n2,2150-01-02 00:00:00\n", encoding="utf-8")
    res = tmp_path / "result"
    res.mkdir()
    (res / "verdict.json").write_text('{"status": "normal"}', encoding="utf-8")
    (res / "cohort.csv").write_text("subject_id,hadm_id,stay_id,index_time\n9,8,1,2150-01-01 00:00:00\n", encoding="utf-8")
    s = task_score(res, gt, {"expected_status": ["normal"]})
    assert s["status_ok"] and s["n_both"] == 1 and s["recall"] == 0.5 and not s["cohort_exact"] and not s["passed"]
    assert task_score(res, None, {"expected_status": ["empty_result"]})["passed"] is False
