"""Interactive pieces without a real model or terminal: asker, intake, the loop's ask action, ground-truth mapping."""
import json

import pandas as pd

from ucd import critic as C
from ucd import loop
from ucd.ask import AUTO, Asker
from ucd.intake import FIELDS, confirm, intake


def _asker(answers, mode="interactive", log=None):
    it = iter(answers)
    out = []
    return Asker(mode, log=log, read=lambda prompt="": next(it), write=out.append), out


def test_asker_choices_only_and_A_lasts_one_step(tmp_path):
    a, out = _asker(["t", "a", "my own words", "2", "A", "1"], log=tmp_path / "q.jsonl")
    assert a.ask("Q1", ["x", "y"], source="intake") == {"answer": "y", "how": "choice"}   # t, a, text refused
    assert sum("please enter a number 1-2 or A" in o for o in out) == 3
    assert a.ask("Q2", ["x", "y"], recommended=0, source="intake") == {"answer": "x", "how": "auto"}
    assert a.ask("Q3", ["x", "y"], source="intake") == {"answer": AUTO, "how": "auto"}    # same step: no prompt
    assert a.mode == "interactive"
    assert a.ask("Q4", ["x", "y"], source="loop") == {"answer": "x", "how": "choice"}     # new step: asked again
    assert len((tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_asker_caps_choices_and_keeps_recommended():
    a, out = _asker(["5", "4"])
    r = a.ask("Q", ["a", "b", "c", "d", "e", "f"], recommended=5)
    assert r == {"answer": "f", "how": "choice"} and not any("5)" in o for o in out)
    assert any(o.startswith("  4) f") and "recommended" in o for o in out)


def test_asker_open_question_takes_text():
    a, _ = _asker(["Is low sodium linked to death?"])
    assert a.ask("What would you like to study?", []) == {"answer": "Is low sodium linked to death?", "how": "text"}


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete_text(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else "{}"


def test_intake_asks_for_missing_fields_then_completes():
    first = {"fields": {"objective": "AKI and death", "outcome": "in-hospital death"}, "complete": False,
             "questions": [{"field": "population", "question": "Which patients?", "options": ["All adult ICU stays",
                            "First ICU stay per adult"], "recommended": 1}]}
    full = {k: f"{k} text" for k in FIELDS}
    llm = FakeLLM([json.dumps(first), json.dumps({"fields": full, "complete": True, "questions": []})])
    a, _ = _asker(["2", "1", "2"])                 # population, then the AKI pack's two definition questions
    text, fields, answers = intake("does AKI kill?", llm, a)
    assert answers[0]["answer"] == "First ICU stay per adult"
    assert [x["field"] for x in answers[1:]] == ["pack:aki", "pack:aki"] and answers[2]["answer"] == "Creatinine only"
    assert "Which creatinine baseline defines AKI?" in text                # pack choices are part of the protocol
    assert "First ICU stay per adult" in llm.prompts[1]                    # the answer goes back to the model
    assert all(fields[k] for k in FIELDS) and text.startswith("Objective:")


def test_intake_survives_garbage_and_fills_every_field():
    a, _ = _asker([], mode="auto")
    text, fields, _ = intake("something vague", FakeLLM(["not json", "nope", "still not"]), a)
    assert all(fields[k] for k in FIELDS) and "assumption" in fields["population"]


def test_confirm_lets_the_user_edit_the_protocol_file(tmp_path):
    a, _ = _asker(["2", "1"])
    path = tmp_path / "protocol.md"

    def edit(p, asker, log):
        p.write_text(p.read_text(encoding="utf-8") + "Exclusions: readmissions within 30 days\n", encoding="utf-8")

    ok, prot = confirm("Objective: x", a, path, log=lambda *_: None, edit=edit)
    assert ok and prot.endswith("Exclusions: readmissions within 30 days")
    assert path.read_text(encoding="utf-8").strip() == prot
    ok, _ = confirm("Objective: x", Asker("auto"), tmp_path / "p2.md", log=lambda *_: None)
    assert ok                                                               # auto: starts with the protocol as is


def test_loop_ask_action_and_limit(tmp_path):
    a, _ = _asker(["1"])
    asked = []
    act = {"action": "ask", "question": "Use 24 h or 48 h window?", "options": ["24 h", "48 h"]}
    assert loop._act(act, tmp_path, 0, a, asked).startswith("PROTOCOL UPDATED")      # back into the protocol
    assert "Use 24 h or 48 h window? -> 24 h (user)" in (tmp_path / "protocol.md").read_text(encoding="utf-8")
    assert loop._act(act, tmp_path, 1, Asker("auto"), asked).startswith("USER ANSWER: AUTO")
    assert loop._act({**act, "options": []}, tmp_path, 1, a, asked).startswith("INVALID")
    assert loop._act({**act, "options": list("abcde")}, tmp_path, 1, a, asked).startswith("INVALID")
    asked += ["q"] * loop.MAX_ASKS
    assert loop._act(act, tmp_path, 2, a, asked).startswith("REFUSED")


def test_feedback_puts_findings_before_the_card():
    """E-21: the loop keeps the first 6000 characters of an observation; the findings must be inside them."""
    from ucd.cli import _fmt_feedback
    card = {"flow": [{"step": "all", "n_units": 100, "n_patients": 90}], "outcome": {"events": 10, "n": 100},
            "model": {"terms": [{"term": f"x{i}", "or": 1.0} for i in range(500)]}}
    crit = {"internal_error": 0.5, "dimensions": {}, "findings": [
        {"severity": "block", "rule": "V6b", "dimension": "validity", "message": "repeats", "fix": "dedupe"}]}
    fb = _fmt_feedback(1, card, crit, (False, ""))
    assert len(fb) > 6000 and "V6b" in fb[:600] and "[BLOCK V6b" in fb[:6000] and "Stop suggested" in fb[:600]


def test_should_stop_is_faster_now():
    assert C.should_stop([0.5, 0.4, 0.35], 1)[0]                           # 3 iterations max
    assert C.should_stop([0.6], 0) == (True, "no blocking findings")                 # I no longer decides
    assert not C.should_stop([0.4], 1)[0]


# ------------------------------------------------------------------ ground truth
from eval import groundtruth as GT   # noqa: E402

ICU = pd.DataFrame({"stay_id": [31, 32, 33, 34], "hadm_id": [21, 21, 22, 23], "subject_id": [11, 11, 12, 13]})
ADM = pd.DataFrame({"hadm_id": [21, 22, 23, 24], "subject_id": [11, 12, 13, 14]})       # 24: no ICU stay
LINK = {"icu_stay": ICU, "admission": ADM}
U = GT.units("mimic")


def test_gt_column_rules():
    m = GT.map_columns(pd.DataFrame({"ICUSTAY_ID": [1], "Died": [0]}), U)
    assert m["id_column"] == "ICUSTAY_ID" and m["level"] == "icu_stay" and m["outcome_column"] == "Died"
    with pytest.raises(SystemExit, match="stay_id"):
        GT.map_columns(pd.DataFrame({"pt_unit": [31], "flag": [0]}), U)       # no guessing, no LLM


def test_gt_compare_across_levels(tmp_path, monkeypatch):
    agent = pd.DataFrame({"unit_id": ["31", "33"], "subject_key": ["11", "12"], "outcome": [1, 0]})
    monkeypatch.setattr(GT, "agent_cohort", lambda rd, ds, it=None: (agent, "icu_stay", 2))
    gt = pd.DataFrame({"hadm_id": [21, 23], "death": [1, 1]})              # admission-level ground truth
    res = GT.compare(tmp_path, gt, GT.map_columns(gt, U), LINK, U, "mimic")
    assert res["compared_at"] == "admission" and res["n_agent"] == 2 and res["n_gt"] == 2
    assert res["n_both"] == 1 and res["jaccard"] == round(1 / 3, 4) and res["recall"] == 0.5
    bare = pd.DataFrame({"ids": [33, 34]})                                   # bare id list: level guessed
    res = GT.compare(tmp_path, bare, GT.map_columns(bare, U), LINK, U, "mimic")
    assert res["gt_level"] == "icu_stay" and res["n_both"] == 1


# ------------------------------------------------------------------ regressions from the 2026-10-17 code review
import pytest  # noqa: E402

from ucd import intake as I  # noqa: E402
from ucd.llm import LLMError, role_settings  # noqa: E402


def test_planner_cannot_write_run_settings_or_logs(tmp_path):
    for f in ("run_config.json", "critic_config.json", "mimic/FROZEN.json", "mimic/iterations.jsonl", "x.py"):
        out = loop._act({"action": "write", "file": f, "content": "{}"}, tmp_path, 0)
        assert out.startswith("REFUSED"), f
        assert not (tmp_path / f).exists()
    assert loop._act({"action": "write", "file": "design_notes.md", "content": "ok"}, tmp_path, 0) == "wrote design_notes.md"
    assert loop._act({"action": "write", "file": "./final.md", "content": "x"}, tmp_path, 0).startswith("REFUSED: no design")


def test_invalid_specs_do_not_count_as_runs(tmp_path):
    (tmp_path / "mimic").mkdir()
    (tmp_path / "mimic" / "iterations.jsonl").write_text(
        json.dumps({"card": {"error": "spec is invalid"}}) + "\n" + json.dumps({"card": {"n_final": 5}}) + "\n",
        encoding="utf-8")
    assert loop._n_runs(tmp_path) == 1


def test_null_action_does_not_crash():
    assert "None" in loop._progress(0, 5, {"action": None}, "x")


def test_intake_raises_gateway_errors_and_fills_direct_answers():
    class Down:
        def complete_text(self, s, u):
            raise LLMError("HTTP 503", 503)
    with pytest.raises(LLMError):
        I.intake("q", Down(), Asker("auto"))
    a, _ = _asker(["adults", "first ICU stay", "ICU admission", "Na < 135", "t", "late answer"] * 3)
    text, fields, _ = I.intake("q", FakeLLM(["garbage"] * 5), a)
    assert fields["objective"] == "adults" and "Clarifications at intake" in text


def test_malformed_intake_reply_is_tolerated():
    bad = {"fields": ["not", "a", "dict"], "questions": [{"question": "Q?", "options": "x", "recommended": "zero"}]}
    a, _ = _asker([], mode="auto")
    text, fields, _ = I.intake("q", FakeLLM([json.dumps(bad)] * 5), a)
    assert all(fields[k] for k in FIELDS)
    a2, _ = _asker(["1"])
    assert a2.ask("Q", ["x", "y"], recommended="1")["answer"] == "x"        # "1" coerced, choice still works


def test_khdp_token_never_sent_to_other_hosts(monkeypatch):
    monkeypatch.delenv("UCD_LLM_API_KEY", raising=False)
    monkeypatch.setenv("UCD_LLM_API_KEY_FILE", "")
    llm = {"base_url": "https://internal-api.khdp.net/v1", "model": "m", "api_key": "khdp" + "_pat_secret",
           "api_key_config": "none", "roles": {"critic": {"base_url": "http://other.example/v1"}}}
    assert role_settings(llm, "critic")["api_key"] == "none"
    assert role_settings(llm, "planner")["api_key"] == "khdp" + "_pat_secret"


def test_gt_admission_level_keeps_admissions_without_icu():
    assert GT.to_level(pd.Series([24, 21]), "admission", "patient", LINK, U) == {14, 11}


def test_registry_drives_dataset_details():
    from ucd import datasets
    assert datasets.default() in datasets.names() and U["icu_stay"]["id"] == "stay_id"
    with pytest.raises(ValueError, match="available"):
        datasets.get("no_such_dataset")


def test_gt_accepts_only_csv(tmp_path):
    for name in ("gt.txt", "gt.sql", "gt.py", "gt.yaml"):
        (tmp_path / name).write_text("x", encoding="utf-8")
        with pytest.raises(SystemExit, match="only a .csv"):
            GT.load_cohort(str(tmp_path / name))
    (tmp_path / "gt.csv").write_text("stay_id,death\n31,1\n", encoding="utf-8")
    assert GT.load_cohort(str(tmp_path / "gt.csv")).shape == (1, 2)


def test_proposal_is_txt_only(tmp_path):
    from ucd.chat import read_proposal
    (tmp_path / "p.txt").write_text("Is sodium linked to death?", encoding="utf-8")
    assert read_proposal(tmp_path / "p.txt") == "Is sodium linked to death?"
    for name in ("p.yaml", "p.yml", "p.md"):
        (tmp_path / name).write_text("objective: x", encoding="utf-8")
        with pytest.raises(ValueError, match=".txt"):
            read_proposal(tmp_path / name)


def test_sample_needs_a_txt_proposal(tmp_path):
    from ucd.sample import find
    for other in ("protocol.yaml", "question.txt"):               # protocol.txt is the only accepted name
        (tmp_path / other).write_text("objective: x", encoding="utf-8")
        with pytest.raises(SystemExit, match="under the name protocol.txt"):
            find(tmp_path)
    (tmp_path / "protocol.txt").write_text("Objective: x", encoding="utf-8")
    assert find(tmp_path)[0].name == "protocol.txt"


def test_json_repair_handles_sql_quotes_and_newlines():
    from ucd.llm import extract_json
    raw = ('{"action": "run", "spec": {"analysis_sql": "SELECT "sofa".score, x\\nFROM t\n'      # a raw newline too
           'WHERE a = \'b\' AND c IN (1, 2)", "n": 1}}')
    act = extract_json(raw)
    assert act["spec"]["analysis_sql"].startswith('SELECT "sofa".score, x') and act["spec"]["n"] == 1
    with pytest.raises(ValueError):
        extract_json('{"action": "run", "spec": {"analysis_sql": "SELECT 1')     # truncated: still an error


def test_lookup_kind_near_misses_are_accepted(tmp_path, monkeypatch):
    from ucd import datacard
    seen = []
    monkeypatch.setattr(datacard, "lookups", lambda d, g=None: {"items": 0, "labs": 0, "dx": 0, "proc": 0})
    monkeypatch.setattr(datacard, "lookup", lambda d, term, kind: seen.append(kind) or "ok")
    for kind, want in (("lab", "labs"), ("Labs", "labs"), ("diagnosis", "dx"), ("item", "items"), ("vitals", "items"),
                       ("nonsense", "all")):
        out = loop._act({"action": "lookup", "dataset": "mimic", "term": "x", "kind": kind}, tmp_path, 0)
        assert seen[-1] == want and not out.startswith("INVALID")


def test_missing_parquet_stops_before_any_llm_call(tmp_path, monkeypatch):
    from ucd import datasets
    monkeypatch.setenv("UCD_MIMIC_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(SystemExit, match="build_parquet"):
        datasets.require_data("mimic")


def test_diag_log_redacts_and_summarises(tmp_path):
    from ucd import diag
    try:
        raise ValueError("Conversion Error: could not convert 'Smith' to INT for id 12345678 key " + "khdp" + "_pat_abc123")
    except ValueError as e:
        diag.error("gate.materialize", e, run="r1")
    diag.event("run", "loop", name="t", designs=0, open_blocks=["V6b"], I=[1.0], seconds=10, turns=30, auto_final=True)
    text = (diag.log_dir() / diag.FILE).read_text(encoding="utf-8")
    assert "Smith" not in text and "12345678" not in text and "_pat_abc" not in text and "'<v>'" in text
    assert "test_chat.py" in text                                    # code location kept for fixing
    s = diag.summary()
    assert "gate.materialize ValueError" in s and "no design executed: 1/1" in s and "V6b (1)" in s
    out = diag.export()
    import zipfile
    assert set(zipfile.ZipFile(out).namelist()) == {diag.FILE, "summary.txt"}


def test_report_flags_undeclared_landmark():
    from ucd.sample import undeclared
    spec = {"population_criteria": [{"flag": "incl_alive_24h", "description": "alive at 24 h (landmark)"}]}
    assert "undeclared" in undeclared(spec, "Outcome: in-hospital death.", [])[0]
    assert undeclared(spec, "Outcome: death after a 24 h landmark.", []) == []


def test_intake_problem_offers_stop_recommended():
    bad = json.dumps({"fields": {}, "complete": False, "questions": [], "problem": "coffee intake is not recorded"})
    with pytest.raises(I.IntakeStopped, match="coffee"):
        I.intake("Does coffee reduce mortality?", FakeLLM([bad] * 5), Asker("auto"), log=lambda *_: None)
    full = {k: "x" for k in FIELDS}
    a, out = _asker(["3"])                                         # continue anyway: asked once, then recorded
    replies = [json.dumps({"fields": full, "complete": True, "questions": [], "problem": "coffee is not recorded"})] * 3
    text, _, _ = I.intake("Does coffee reduce mortality?", FakeLLM(replies), a, log=lambda *_: None)
    assert "Problem flagged at intake" in text and "coffee is not recorded" in text
    assert sum(o.startswith("? What now?") for o in out) == 1


def test_critic_reply_is_repaired_like_planner_actions():
    """E-30: a raw newline inside a string used to make the clinical critic drop out of the iteration."""
    from ucd.llm_critic import _parse
    out = _parse('<think>x</think>{"overall": "revise", "issues": [{"message": "line one\nline two"}]}')
    assert out["overall"] == "revise" and "line two" in out["issues"][0]["message"]


def test_sample_folder_refuses_options_it_would_ignore(tmp_path, capsys):
    """E-31: --gt / --run-dir / --topic / --file used to be dropped silently with a sample folder."""
    from ucd import chat
    (tmp_path / "protocol.txt").write_text("q", encoding="utf-8")
    with pytest.raises(SystemExit):
        chat.main([str(tmp_path), "--gt", str(tmp_path / "x.csv")])
    assert "not used" in capsys.readouterr().err


def test_report_reads_old_list_comparisons(tmp_path):
    from ucd.sample import write_report
    s = tmp_path / "s"
    (s / "result" / "run").mkdir(parents=True)
    (s / "protocol.txt").write_text("q", encoding="utf-8")
    (s / "result" / "gt_comparison.json").write_text(json.dumps([{"file": "gt.csv", "error": "no ids"}]), encoding="utf-8")
    text = write_report(s).read_text(encoding="utf-8")
    assert "`gt.csv`: not possible: no ids" in text


def test_no_appeals_to_authority(tmp_path):
    """User rule: models may not cite papers or authors as evidence. Enforced on what they write."""
    from ucd.llm import citations
    from ucd.llm_critic import to_findings
    assert citations("as shown by Smith et al. (2019)") and citations("doi:10.1001/x") and citations("Previous studies found")
    assert not citations("KDIGO stage 3, SOFA >= 2, n = 2016, age 65 (years), ICU stays 2008-2019")
    out = loop._act({"action": "write", "file": "design_notes.md", "content": "Window from Seymour et al."}, tmp_path, 0)
    assert out.startswith("REFUSED: remove the references") and not (tmp_path / "design_notes.md").exists()
    f = to_findings({"issues": [{"message": "Too strict (Vincent, 1996)", "severity": "warn"}]})[0]
    assert "Vincent" not in f["message"] and "[citation removed]" in f["message"]
    fields = {}
    I._merge(fields, {"fields": {"exposure": "lactate > 2 as in prior trials (Rivers 2001)"}})
    assert "Rivers" not in fields["exposure"]


def test_users_own_citations_are_kept(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"question": "Sepsis as defined by Singer et al."}), encoding="utf-8")
    out = loop._act({"action": "write", "file": "protocol_items.yaml",
                     "content": "- {id: P1, text: Sepsis as defined by Singer et al.}"}, tmp_path, 0)
    assert out == "wrote protocol_items.yaml"
    fields = {}
    I._merge(fields, {"fields": {"population": "sepsis (Singer et al.)"}}, request="sepsis, Singer et al. definition")
    assert "Singer et al." in fields["population"]


def test_missing_parquet_is_built_on_request(tmp_path, monkeypatch):
    """E-36: with raw files present, chat builds the parquet (auto mode) instead of only printing the command."""
    from ucd import chat, datasets
    import ucd.build_parquet as B
    built = []
    state = {"miss": ["icu.icustays"]}
    monkeypatch.setattr(datasets, "missing_tables", lambda name: (tmp_path, state["miss"]))
    monkeypatch.setattr(datasets, "source_dir", lambda name: tmp_path)
    monkeypatch.setattr(datasets, "require_data", lambda name: None)
    monkeypatch.setattr(B, "build", lambda name, jobs=None: built.append(name) or state.update(miss=[]))
    chat._ensure_data("mimic", auto=True)
    assert built == ["mimic"]


def test_binder_error_names_the_right_table():
    from ucd.gate import Gate, GateError
    g = Gate.__new__(Gate)
    g.schema = lambda: {"hosp": {"admissions": [("hadm_id", "INT")]}, "icu": {"icustays": [("intime", "TIMESTAMP")]}}
    h = g.column_hint('Binder Error: Values list "a" does not have a column named "intime"',
                      "SELECT a.intime FROM hosp.admissions a")
    assert "'a' is hosp.admissions" in h and "exists in: icu.icustays" in h


def test_json_repair_server_cases():
    """E-34: pretty-printed run actions whose SQL has quoted identifiers followed by ':' or ','."""
    from ucd.llm import extract_json
    raw = ('{\n  "action": "run",\n  "spec": {\n    "analysis_sql": "SELECT "sofa"::INTEGER AS s, "a", "b"\n'
           'FROM t WHERE c = \'d\'",\n    "n": 1,\n    "model": {"predictors": ["age", "sex"]}\n  }\n}')
    act = extract_json(raw)
    assert act["spec"]["analysis_sql"].startswith('SELECT "sofa"::INTEGER AS s, "a", "b"\nFROM')
    assert act["spec"]["n"] == 1 and act["spec"]["model"]["predictors"] == ["age", "sex"]


def test_enter_takes_the_recommended_choice_never_auto(tmp_path, monkeypatch):
    """User, 2026-10-17: the default answer is the recommended option, not automation; a non-terminal stdin does not
    switch to auto by itself (only --auto or the end of input does)."""
    import io
    import sys
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert Asker("interactive").mode == "interactive"
    a, out = _asker(["", "", "2"])
    r = a.ask("Window?", ["6 h", "12 h", "24 h"], 1, source="intake")
    assert r == {"answer": "12 h", "how": "choice"} and a.auto_step is None and any("Enter = 2" in l for l in out)
    r = a.ask("Unit?", ["stay", "admission"], None, source="intake")       # no recommendation: Enter asks again
    assert r == {"answer": "admission", "how": "choice"}
