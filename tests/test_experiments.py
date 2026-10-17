"""Baselines and ablations (eval/experiments.py) and the run_config switches they rely on."""
import json
from pathlib import Path

import pytest

from eval import experiments as X
from ucd import loop
from ucd.gate import PARQUET

needs_mimic = pytest.mark.skipif(not (PARQUET / "mimic" / "hosp" / "patients.parquet").exists(), reason="no MIMIC parquet")
SWITCHES = {"llm_critic", "rule_feedback", "waivers", "explore", "concepts", "notes", "max_iter", "blind_outcome"}
SELFTEST = Path(__file__).resolve().parents[1] / "ucd" / "docs" / "selftest_mimic"


def test_every_arm_uses_known_switches_and_the_groups_resolve():
    for name, arm in X.ARMS.items():
        assert set(arm["config"]) <= SWITCHES, name
    assert X.arms_from("baselines") == ["full", "single_generator", "single_agent"]
    assert X.arms_from("ablations")[0] == "full" and "no_llm_critic" in X.arms_from("all")
    with pytest.raises(SystemExit):
        X.arms_from("full,nonsense")


def test_one_shot_reply_is_split_into_spec_and_query():
    raw = "Here it is.\n```yaml\ntitle: t\nunit_of_analysis: icu_stay\n```\n```sql\nSELECT 1 AS unit_id\n```\n"
    spec, sql = X.parse_one_shot(raw)
    assert spec.startswith("title: t") and sql == "SELECT 1 AS unit_id\n"
    spec, sql = X.parse_one_shot("```yaml\ntitle: t\nanalysis_sql: SELECT 2\n```")
    assert "analysis_sql" not in spec and sql == "SELECT 2\n"
    assert X.parse_one_shot("no blocks at all") == ("", "")


def test_switched_off_tools_are_refused_and_named_in_the_prompt(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps({"explore": False, "concepts": False, "llm_critic": False,
                                                          "rule_feedback": False}), encoding="utf-8")
    assert loop._act({"action": "lookup", "term": "sodium"}, tmp_path, 0, dataset="mimic").startswith("REFUSED")
    (tmp_path / "run_config.json").write_text(json.dumps({"concepts": False}), encoding="utf-8")
    out = loop._act({"action": "explore", "from": "SELECT * FROM concept.sofa", "group": ["x"]}, tmp_path, 0,
                    dataset="mimic")
    assert out.startswith("REFUSED") and "concept" in out
    (tmp_path / "run_config.json").write_text(json.dumps({"llm_critic": False, "rule_feedback": False, "max_iter": 1}),
                                              encoding="utf-8")
    system = loop._system_prompt(tmp_path, "question", "mimic")
    assert "no critic reviews your designs" in system and "at most 1 design run(s)" in system


def test_summary_averages_per_arm(tmp_path):
    for arm, j in (("full", 0.9), ("full", 0.7), ("single_generator", 0.2)):
        X.append_row(tmp_path, {"sample": "s", "arm": arm, "rep": 1, "status": "ok", "designs_run": 1, "jaccard": j})
    text = X.summarize(tmp_path)
    assert "| full |" in text and "| 0.8 |" in text and "| single_generator |" in text


@needs_mimic
def test_single_generator_runs_once_and_hides_the_rule_critic(tmp_path, monkeypatch):
    monkeypatch.setenv("UCD_CRITIC_BACKEND", "off")
    monkeypatch.setattr(loop.datacard, "render", lambda d, **k: "(card)")      # the full card scans chartevents
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "protocol.txt").write_text("Is age 65 or older linked to in-hospital death in adults?\n", encoding="utf-8")

    class OneShot:
        def complete_text(self, system, user):
            assert "ONE-SHOT MODE" in system
            spec = (SELFTEST / "spec.yaml").read_text(encoding="utf-8")
            return f"```yaml\n{spec}\n```\n```sql\n{(SELFTEST / 'query.sql').read_text(encoding='utf-8')}\n```"

    row = X.run_arm("single_generator", sample, tmp_path / "out", 1, 5, llm=OneShot())
    assert row["status"] == "ok" and row["designs_run"] == 1 and row["turns"] == 1
    assert row["n_final"] > 1000 and row["I_rule"] is not None
    run = Path(row["run_dir"])
    shown = json.loads((run / "mimic" / "iter_1" / "critique.json").read_text(encoding="utf-8"))
    assert shown["internal_error"] is None and not [f for f in shown["findings"] if not f["rule"].startswith("L")]
    assert (run / "final.md").exists() and (run / "verdict.json").exists()


def test_summary_counts_correct_rejections(tmp_path):
    rows = [("T001", "empty_result|unanswerable", "True"), ("T007", "contradictory|empty_result", "False"),
            ("T002", "normal", "True")]
    for s, exp, ok in rows:
        X.append_row(tmp_path, {"sample": s, "arm": "full", "rep": 1, "status": "ok", "expected": exp, "verdict_ok": ok})
    assert "| 2/3 | 1/2 |" in X.summarize(tmp_path)


def test_planner_settings_and_the_stuck_metric(tmp_path, monkeypatch):
    assert X.planner_env("") == ("", {})
    label, env = X.planner_env("deepseek-v4-pro-ax:think")
    assert label == "deepseek-v4-pro-ax+think" and env["UCD_PLANNER_THINKING"] == "1" and env["UCD_PLANNER_MAX_TOKENS"]
    from ucd.llm import load_config, role_settings
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    rs = role_settings(dict(load_config()["llm"]), "planner")
    assert rs["model"] == "deepseek-v4-pro-ax" and rs["max_tokens"] == 16384
    assert rs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    critic_cfg = bool(load_config()["llm"]["roles"]["critic"].get("thinking", False))    # the planner's switch stays
    assert role_settings(dict(load_config()["llm"]), "critic")["extra_body"]["chat_template_kwargs"]["enable_thinking"] \
        is critic_cfg                                                                    # out of the other roles
    log = tmp_path / "loop_log.jsonl"
    log.write_text("\n".join(json.dumps({"observation": o}) for o in
                             ["wrote x", "NO CHANGE: a", "NO CHANGE: a", "INVALID: y", "NO CHANGE: a"]), encoding="utf-8")
    row = X.collect(tmp_path, "mimic", None)
    assert row["noop_streak"] == 2 and row["bad_turns"] == 1 and row["turns"] == 5
