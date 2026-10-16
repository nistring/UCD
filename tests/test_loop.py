"""Enclave loop against a scripted fake LLM: actions, error handling, done-guard (no real model, MIMIC-IV parquet)."""
import json

import pytest

from ucd import loop
from ucd.gate import PARQUET

needs_mimic = pytest.mark.skipif(not (PARQUET / "mimic" / "hosp" / "patients.parquet").exists(), reason="no MIMIC parquet")


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete_text(self, system, user):
        assert "Action protocol" in system
        return self.replies.pop(0) if self.replies else '{"action": "done"}'


@needs_mimic
def test_loop_runs_actions_and_survives_bad_turns(tmp_path, monkeypatch):
    replies = [
        "not json at all",
        '{"action": "explore", "dataset": "mimic", "from": "SELECT gender FROM hosp.patients", "group": ["gender"]}',
        '{"action": "explore", "dataset": "mimic", "from": "SELECT subject_id AS s FROM hosp.patients", "group": ["s"]}',
        '{"action": "lookup", "dataset": "mimic"}',
        '{"action": "done"}',
        '{"action": "write", "file": "final.md", "content": "# final"}',
        '{"action": "done"}',
    ]
    monkeypatch.setattr(loop, "LocalLLM", lambda cfg=None: ScriptedLLM(replies))
    monkeypatch.setattr(loop.datacard, "render", lambda d: f"(Data Card for {d})")   # full MIMIC card scans chartevents
    loop.run_loop("t9_trap_dayparity", tmp_path, ["mimic"], max_turns=10)
    log = [json.loads(l) for l in (tmp_path / "loop_log.jsonl").read_text(encoding="utf-8").splitlines()]
    obs = [r["observation"] for r in log]
    assert obs[0].startswith("INVALID")
    assert '"n"' in obs[1] and "rows" in obs[1]
    assert obs[2].startswith("REFUSED")          # identifier-like group refused by the gate
    assert obs[3].startswith("INVALID: missing field")
    assert obs[4].startswith("REFUSED: write final.md")
    assert obs[5].startswith("REFUSED: no design has been run yet")   # final.md needs a result
    # nothing ran: the loop writes an honest AUTO final.md instead of the model's text
    assert "AUTO-GENERATED" in (tmp_path / "final.md").read_text(encoding="utf-8")
    assert "no design was executed" in (tmp_path / "final.md").read_text(encoding="utf-8")


def test_lookup_budget_and_next_step(tmp_path):
    log = tmp_path / "loop_log.jsonl"
    log.write_text("\n".join(json.dumps({"action": {"action": "lookup"}}) for _ in range(loop.MAX_LOOKUPS)),
                   encoding="utf-8")
    assert loop._act({"action": "lookup", "dataset": "mimic", "term": "x"}, tmp_path, 0).startswith("REFUSED: lookup budget")
    assert f"lookups {loop.MAX_LOOKUPS}/{loop.MAX_LOOKUPS}" in loop._next_step(tmp_path, 0, 30)
    assert "NO DESIGN HAS RUN" in loop._next_step(tmp_path, 27, 30)


def test_design_files_are_written_edited_and_read_by_line(tmp_path):
    """0.13.0: the design is two files the planner edits by line range, as shown numbered in every prompt."""
    w = loop._act({"action": "write", "file": "design/query.sql", "body": "SELECT 1 AS a,\n  2 AS b\nFROM t"}, tmp_path, 0)
    assert w == "wrote design/query.sql"
    e = loop._act({"action": "edit", "file": "design/query.sql", "lines": [2, 2], "body": "  3 AS b,\n  4 AS c"}, tmp_path, 0)
    assert e.startswith("edited design/query.sql: lines 2-2 replaced by 2 line(s)") and "   3 |   4 AS c" in e
    assert (tmp_path / "design/query.sql").read_text(encoding="utf-8") == "SELECT 1 AS a,\n  3 AS b,\n  4 AS c\nFROM t\n"
    assert loop._act({"action": "edit", "file": "design/query.sql", "lines": [9, 9], "body": "x"}, tmp_path, 0).startswith("INVALID")
    assert loop._act({"action": "edit", "file": "design/query.sql", "lines": [1, 3], "body": ""}, tmp_path, 0).startswith("edited")
    assert (tmp_path / "design/query.sql").read_text(encoding="utf-8") == "FROM t\n"     # an empty block deletes
    assert "   1 | FROM t" in loop._act({"action": "read", "file": "design/query.sql"}, tmp_path, 0)
    assert loop._act({"action": "read", "file": "../x"}, tmp_path, 0).startswith("INVALID")
    loop._act({"action": "write", "file": "design/spec.yaml", "body": "title: t\nmodel: {type: logistic}"}, tmp_path, 0)
    bad = loop._act({"action": "edit", "file": "design/spec.yaml", "lines": [2, 2], "body": "model: {type: [}"}, tmp_path, 0)
    assert bad.startswith("REFUSED") and "valid YAML" in bad          # the spec stays parseable
    assert "model: {type: logistic}" in (tmp_path / "design/spec.yaml").read_text(encoding="utf-8")
    shown = loop._design_files(tmp_path)
    assert "### design/spec.yaml (2 lines)" in shown and "   2 | model: {type: logistic}" in shown


def test_replies_as_models_really_write_them(tmp_path):
    """Review of 0.13.0: a ```json wrapper, CRLF, an unclosed block and an edit without a block."""
    P = loop.parse_action
    assert P('```json\n{"action": "run"}\n```') == {"action": "run"}
    act = P('Here:\n```json\n{"action": "edit", "file": "design/query.sql", "lines": [1, 1]}\n```\n```sql\nSELECT 2\n```')
    assert act["action"] == "edit" and act["body"] == "SELECT 2"
    assert P('{"action": "write", "file": "f.md"}\r\n```sql\r\nSELECT 1\r\n```')["body"] == "SELECT 1"
    with pytest.raises(ValueError, match="not closed"):
        P('{"action": "edit", "file": "design/query.sql", "lines": [2, 4]}\n```sql\nB1\nB2')
    (tmp_path / "design").mkdir()
    (tmp_path / "design" / "query.sql").write_text("a\nb\nc\n", encoding="utf-8")
    out = loop._act({"action": "edit", "file": "design/query.sql", "lines": [2, 2]}, tmp_path, 0)
    assert out.startswith("INVALID") and (tmp_path / "design" / "query.sql").read_text(encoding="utf-8") == "a\nb\nc\n"
    two = loop._act({"action": "write", "file": "design/query.sql", "body": "SELECT 1\n```\nand\n```sql\nSELECT 2"}, tmp_path, 0)
    assert two.startswith("REFUSED") and "one file" in two
    sql_in_spec = loop._act({"action": "write", "file": "design/spec.yaml", "body": "title: t\nanalysis_sql: SELECT 1"}, tmp_path, 0)
    assert sql_in_spec.startswith("REFUSED") and "design/query.sql" in sql_in_spec
    assert "   1 | a" in loop._act({"action": "read", "file": "design/query.sql", "from_line": 0}, tmp_path, 0)


def test_run_assembles_the_files_and_names_the_failing_line(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(loop, "_capture", lambda argv: calls.append(argv) or "# Iteration 1\n**Stop suggested: False**")
    assert loop._act({"action": "run"}, tmp_path, 0, dataset="mimic").startswith("INVALID: design/query.sql is empty")
    loop._act({"action": "write", "file": "design/spec.yaml", "body": "title: t"}, tmp_path, 0)
    loop._act({"action": "write", "file": "design/query.sql", "body": "WITH a AS (\n  SELECT x.nope\n  FROM t x\n)\nSELECT * FROM a"},
              tmp_path, 0)
    monkeypatch.setattr(loop, "sql_check", lambda *a: 'Binder Error: no "nope"\n\nLINE 2:   SELECT x.nope\n         ^')
    out = loop._act({"action": "run"}, tmp_path, 0, dataset="mimic")
    assert out.startswith("SQL CHECK FAILED") and "(This is line 2 of design/query.sql.)" in out and not calls
    monkeypatch.setattr(loop, "sql_check", lambda *a: "")
    assert loop._act({"action": "run"}, tmp_path, 0, dataset="mimic").startswith("# Iteration 1")
    import yaml
    sent = yaml.safe_load((tmp_path / "mimic" / "submitted_spec.yaml").read_text(encoding="utf-8"))
    assert sent["title"] == "t" and sent["analysis_sql"].startswith("WITH a AS (") and calls[0][0] == "run"


@needs_mimic
def test_selftest_drives_every_planner_path_on_real_data(tmp_path):
    """The self-test the server runs first (python -m ucd.selftest): no LLM, the real gate, runner and rule critic."""
    from ucd.selftest import selftest
    assert selftest("mimic", tmp_path / "st", echo=lambda *_: None) == []
    setup = json.loads((tmp_path / "st" / "meta.json").read_text(encoding="utf-8"))["setup"]
    assert setup["models"]["planner"] == "ScriptedPlanner" and setup["critic_backend"] == "off"   # what really ran


@needs_mimic
def test_user_can_end_the_loop_early(tmp_path, monkeypatch):
    class CtrlC:
        calls = 0

        def complete_text(self, system, user):
            CtrlC.calls += 1
            if CtrlC.calls == 2:
                raise KeyboardInterrupt                      # Ctrl+C while waiting for the model
            return '{"action": "status"}'
    monkeypatch.setattr(loop, "LocalLLM", lambda cfg=None: CtrlC())
    monkeypatch.setattr(loop.datacard, "render", lambda d: "(card)")
    fm = loop.run_loop("t9_trap_dayparity", tmp_path / "a", ["mimic"], max_turns=10)
    assert CtrlC.calls == 2 and "Ctrl+C" in fm.read_text(encoding="utf-8")

    monkeypatch.setattr(loop, "stop_key_pressed", lambda: True)  # q + Enter before the first turn
    monkeypatch.setattr(loop, "LocalLLM", lambda cfg=None: ScriptedLLM([]))
    shown = []
    fm = loop.run_loop("t9_trap_dayparity", tmp_path / "b", ["mimic"], max_turns=10, echo=shown.append)
    assert "early (q)" in fm.read_text(encoding="utf-8") and not (tmp_path / "b" / "loop_log.jsonl").exists()
    text = "\n".join(shown)                                       # the user is told how to stop, and that it worked
    assert "To stop the design loop early" in text and "Ctrl+C" in text and "q received" in text


def _fake_runner(n_final, events, zero_var=False):
    """Replaces the in-process `ucd run`: records an executed design with the given size."""
    def capture(argv):
        if argv[0] != "run":
            return ""
        rd = __import__("pathlib").Path(argv[argv.index("--run-dir") + 1]) / "mimic"
        it = len(rd.joinpath("iterations.jsonl").read_text().splitlines()) + 1 if rd.joinpath("iterations.jsonl").exists() else 1
        (rd / f"iter_{it}").mkdir(parents=True, exist_ok=True)
        (rd / f"iter_{it}" / "spec.yaml").write_text("exposure: {column: low_na}\n", encoding="utf-8")
        card = {"n_final": n_final, "outcome": {"events": events}, "table1": {"low_na": {"zero_variance": zero_var}}}
        with open(rd / "iterations.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"iter": it, "card": card}) + "\n")
        return f"# Iteration {it}\n**Stop suggested: True** (reached max iterations (3))"
    return capture


RUN = '{"action": "run"}'


def _design(run_dir):
    (run_dir / "design").mkdir(parents=True, exist_ok=True)
    (run_dir / "design" / "spec.yaml").write_text("title: x\n", encoding="utf-8")
    (run_dir / "design" / "query.sql").write_text("SELECT 1 AS unit_id\n", encoding="utf-8")


def test_design_problem_detects_small_cohorts_and_constant_exposure(tmp_path):
    assert loop.design_problem(tmp_path, "mimic") == ""
    for runner, want in ((_fake_runner(40, 5), "too small"), (_fake_runner(5000, 600, zero_var=True), "same value"),
                         (_fake_runner(5000, 600), "")):
        runner(["run", "--run-dir", str(tmp_path)])
        got = loop.design_problem(tmp_path, "mimic")
        assert (want in got) if want else got == ""


@needs_mimic
def test_small_cohort_goes_back_to_design_then_asks(tmp_path, monkeypatch):
    from ucd.ask import Asker
    monkeypatch.setattr(loop, "_capture", _fake_runner(40, 5))
    monkeypatch.setattr(loop.datacard, "render", lambda d: "(card)")
    _design(tmp_path)
    replies = [RUN, '{"action": "write", "file": "final.md", "content": "# done"}', RUN, RUN]
    monkeypatch.setattr(loop, "LocalLLM", lambda cfg=None: ScriptedLLM(replies))
    fm = loop.run_loop("t9_trap_dayparity", tmp_path, ["mimic"], max_turns=10, asker=Asker("auto"))
    obs = [json.loads(l)["observation"] for l in (tmp_path / "loop_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert obs[0].startswith("BACK TO THE DESIGN LOOP: the cohort is too small")
    assert obs[1].startswith("REFUSED: the last design cannot answer")             # final.md refused meanwhile
    assert obs[2].startswith("BACK TO THE DESIGN LOOP") and "1 more design" in obs[2]
    assert obs[3].startswith("AGENT (auto mode) chose to stop") and len(obs) == 4   # 3rd small design: asked
    text = fm.read_text(encoding="utf-8")
    assert "AUTO-GENERATED" in text and "too small" in text


def test_protocol_items_must_be_a_yaml_list_when_written(tmp_path):
    """E-66: `P1: population: Adult ...` lines were accepted, and every later run crashed on them after its query
    had executed (ScannerError), losing the first design whose SQL passed."""
    bad = loop._act({"action": "write", "file": "protocol_items.yaml",
                     "content": "P1: population: Adult patients (age >= 18).\nP2: outcome: death\n"}, tmp_path, 0)
    assert bad.startswith("REFUSED") and "id: P1" in bad and not (tmp_path / "protocol_items.yaml").exists()
    ok = loop._act({"action": "write", "file": "protocol_items.yaml",
                    "content": "- {id: P1, kind: population, text: 'Adults (age >= 18)'}\n"}, tmp_path, 0)
    assert ok == "wrote protocol_items.yaml"


@needs_mimic
def test_infeasible_action_lets_the_user_decide(tmp_path, monkeypatch):
    from ucd.ask import Asker
    answers = iter(["2", "1"])                                    # keep designing, then stop
    asker = Asker("interactive", read=lambda p="": next(answers), write=lambda *_: None)
    monkeypatch.setattr(loop.datacard, "render", lambda d: "(card)")
    inf = '{"action": "infeasible", "reason": "coffee intake is not recorded"}'
    monkeypatch.setattr(loop, "LocalLLM", lambda cfg=None: ScriptedLLM([inf, '{"action": "status"}', inf]))
    fm = loop.run_loop("t9_trap_dayparity", tmp_path, ["mimic"], max_turns=10, asker=asker)
    obs = [json.loads(l)["observation"] for l in (tmp_path / "loop_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert obs[0].startswith("USER chose to keep designing") and obs[2].startswith("USER chose to stop")
    assert "coffee intake is not recorded" in fm.read_text(encoding="utf-8")
