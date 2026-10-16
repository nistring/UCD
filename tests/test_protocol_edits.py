"""Protocol edits, intake warnings and action parsing (E-71). No data, no network."""
import json

import pytest

from ucd import intake, loop
from ucd.actions import parse_action
from ucd.ask import Asker


def test_intake_warns_when_the_schema_cannot_be_read(monkeypatch):
    """Without the schema the intake model offered OMOP tables (condition_occurrence) for MIMIC, silently."""
    import ucd.datacard as D
    monkeypatch.setattr(D, "schema_summary", lambda ds: (_ for _ in ()).throw(RuntimeError("card failed")))
    shown = []
    assert "not available" in intake._schema("mimic", shown.append)
    assert any("WARNING" in s and "mimic" in s and "card failed" in s for s in shown)


def test_a_protocol_with_repeated_fields_is_flagged_after_editing(tmp_path):
    """The user pasted the new protocol above the old one: three Objective lines went into the design loop."""
    answers = iter(["2", "1"])                              # edit, then start
    asker = Asker("interactive", read=lambda p="": next(answers), write=lambda *_: None)
    p = tmp_path / "protocol.md"

    def paste_on_top(path, asker, log):
        path.write_text("Objective: new\nOutcome: x\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    shown = []
    ok, text = intake.confirm("Objective: old\nOutcome: y", asker, p, shown.append, edit=paste_on_top)
    assert ok and text.count("Objective:") == 2
    assert any("2 'Objective:' lines" in s for s in shown)


def test_an_edit_that_changes_nothing_says_so(tmp_path):
    loop._act({"action": "write", "file": "design/spec.yaml", "body": "title: t\nprediction_time_h: 24"}, tmp_path, 0)
    same = loop._act({"action": "edit", "file": "design/spec.yaml", "lines": [2, 2], "body": "prediction_time_h: 24"},
                     tmp_path, 0)
    assert same.startswith("NO CHANGE")
    assert loop._act({"action": "write", "file": "design/spec.yaml", "body": "title: t\nprediction_time_h: 24"},
                     tmp_path, 0).startswith("NO CHANGE")
    log = tmp_path / "loop_log.jsonl"                       # 2 design changes after the last run
    rows = [{"action": {"action": "run"}, "observation": "# Iteration 1"},
            {"action": {"action": "edit", "file": "design/spec.yaml"}, "observation": "edited design/spec.yaml: ..."},
            {"action": {"action": "edit", "file": "design/spec.yaml"}, "observation": "NO CHANGE: ..."},
            {"action": {"action": "write", "file": "design/query.sql"}, "observation": "wrote design/query.sql"}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert "2 design change(s) since the last run" in loop._next_step(tmp_path, 5, 30)


def test_a_second_action_inside_the_block_is_refused():
    """The model wrote final.md and verdict.json in one reply; the verdict action ended up inside final.md."""
    raw = ('{"action": "write", "file": "final.md"}\n```\n# Final\n\n{"action": "write", "file": "verdict.json"}\n'
           '```json\n{"status": "partial"}\n```')
    with pytest.raises(ValueError, match="one action per reply"):
        parse_action(raw)


def test_the_user_can_answer_a_question_by_editing_the_protocol(tmp_path, monkeypatch):
    """The agent offered 'I will provide the SOFA-2 criteria' but the user could not type them."""
    (tmp_path / "protocol.md").write_text("Objective: x\n", encoding="utf-8")

    def write_definition(path, asker, log=print):
        path.write_text(path.read_text(encoding="utf-8") + "SOFA-2: components and thresholds ...\n", encoding="utf-8")
    monkeypatch.setattr(intake, "open_editor", write_definition)
    seen = {}
    asker = Asker("interactive", read=lambda p="": "4", write=lambda *a: seen.setdefault("text", " ".join(map(str, a))))
    act = {"action": "ask", "question": "How should SOFA-2 be defined?", "options": ["a", "b", "c", "d"], "recommended": 3}
    out = loop._act(act, tmp_path, 0, asker, [])
    assert out.startswith("PROTOCOL UPDATED") and "edited the protocol" in out
    assert "SOFA-2: components" in (tmp_path / "protocol.md").read_text(encoding="utf-8")
    auto = loop._act({**act, "question": "q2"}, tmp_path, 1, Asker("auto"), [])
    assert auto.startswith("PROTOCOL UPDATED") and "d" in auto                     # auto keeps the recommended choice
