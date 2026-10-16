"""Reply parsing, JSON repair, redaction, citation checks and other loop regressions (E-37 to E-42). No data, no network."""
import os
import sys
import types

import pytest

from ucd import loop
from ucd.llm import citations, extract_json, strip_citations, strip_think


def test_repair_keeps_sql_with_quoted_identifiers_inside_arrays():
    """E-38: a quote followed by ',' inside an open '(' is SQL, not the end of the array item."""
    act = extract_json('{"action":"explore","from":"SELECT 1","metrics": ["corr("age", "sofa") AS r", "avg(x) AS m"]}')
    assert act["metrics"] == ['corr("age", "sofa") AS r', "avg(x) AS m"]
    assert extract_json('{"q": "x IN (\'a\', \'b\') AND "y" = 1", "n": 2}') == {"q": "x IN ('a', 'b') AND \"y\" = 1", "n": 2}


def test_redaction_hides_single_quoted_values_but_keeps_identifiers():
    """E-39: DuckDB puts data values in single quotes, identifiers in double quotes."""
    from ucd.diag import redact
    out = redact("Could not convert string 'vancomycin' to INT32; 'kim'; column named \"intime\" in \"hosp.admissions\"")
    assert "vancomycin" not in out and "kim" not in out and '"intime"' in out and '"hosp.admissions"' in out


@pytest.mark.parametrize("text", ["As Jones et al. found", "(van Walraven 2009)", "Singer and colleagues",
                                  "(Seymour, 2016)", "Vincent (1996)", "doi:10.1001/jama.2016.0287", "a seminal study"])
def test_citations_are_caught(text):
    assert citations(text)


@pytest.mark.parametrize("text", ["KDIGO (2012)", "SAPS II (1993)", "Cox (1972)", "Hb (2019)", "ICU (2019)",
                                  "January (2019)", "Chest 2019", "a landmark study at 24 h", "10.12345/kg",
                                  "age (years)", "Sepsis-3 (2016)", "MIMIC-IV (2008-2019)"])
def test_clinical_text_is_not_a_citation(text):
    """E-40: concepts, units and calendar words in parentheses with a year are not author citations."""
    assert citations(text) == []


def test_allowed_citation_does_not_whitelist_other_authors():
    """E-40: the user's 'Vincent et al.' must not let the model's 'Jones et al.' through."""
    assert citations("As Jones et al. found", allowed="SOFA as in Vincent et al. 1996") == ["Jones et al."]
    assert citations("SOFA as in Vincent et al.", allowed="SOFA as in Vincent et al. 1996") == []
    assert strip_citations("as Smith et al. showed") == "as [citation removed] showed"


def test_column_hint_ignores_expression_aliases_and_schema_errors():
    """E-41: 'i.stay_id AS a' is not a table; a failing schema lookup returns no hint instead of a new error."""
    from ucd.gate import Gate
    g = Gate.__new__(Gate)
    g.schema = lambda: {"hosp": {"admissions": [("hadm_id", "INT")]}, "icu": {"icustays": [("intime", "TS")]}}
    err = 'Binder Error: Table "a" does not have a column named "intime"'
    assert "i.stay_id" not in g.column_hint(err, "SELECT i.stay_id AS a, a.intime FROM hosp.admissions x JOIN (SELECT 1) a ON 1")
    assert "'a' is hosp.admissions" in g.column_hint(err, "SELECT a.intime FROM hosp.admissions AS a")

    def boom():
        raise RuntimeError("DESCRIBE failed")
    g.schema = boom
    assert g.column_hint(err, "") == ""


def test_unparsable_reply_says_why():
    """E-37: cut off at the token limit, empty, or prose: the planner is told which."""
    cut = types.SimpleNamespace(finish_reason="length", max_tokens=8192)
    ok = types.SimpleNamespace(finish_reason="stop")
    assert "cut off at the output limit (8192 tokens)" in loop._why_unparsable('{"action": "run", "spec": {', ValueError(), cut)
    assert "empty" in loop._why_unparsable("", ValueError(), ok)
    assert "no JSON object" in loop._why_unparsable("I will now design the cohort.", ValueError(), ok)
    assert strip_think("<think>reasoning that never closed") == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows console keys")
def test_windows_stop_key_needs_a_whole_line(monkeypatch):
    """E-42: 'q' inside other text, or q without Enter, does not stop the loop; the line 'q' + Enter does."""
    keys = []
    fake = types.SimpleNamespace(kbhit=lambda: bool(keys), getwch=lambda: keys.pop(0))
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    loop._typed.clear()
    keys.extend("quality\r")
    assert not loop.stop_key_pressed()
    keys.extend("q")
    assert not loop.stop_key_pressed()                    # no Enter yet
    keys.extend("\r")
    assert loop.stop_key_pressed()


def test_sample_result_is_not_archived_when_the_run_stops_at_the_data_check(tmp_path, monkeypatch):
    """E-42: stopping at the parquet prompt must leave the previous result/ in place."""
    from ucd import chat
    from ucd.sample import run_sample
    (tmp_path / "protocol.txt").write_text("q", encoding="utf-8")
    (tmp_path / "result").mkdir()

    def stop(dataset, auto):
        raise SystemExit("no parquet")
    monkeypatch.setattr(chat, "_ensure_data", stop)
    with pytest.raises(SystemExit):
        run_sample(tmp_path, auto=True)
    assert (tmp_path / "result").is_dir() and not list(tmp_path.glob("result_*"))


def test_sql_travels_outside_the_json():
    """E-43: file content and SQL in a fenced block after the action are never escaped, so they cannot break the
    JSON; the block runs to the last fence, so a final.md with its own fenced blocks arrives whole (0.13.0)."""
    raw = ('{"action": "write", "file": "design/query.sql"}\n```sql\n'
           'SELECT "sofa"::INT AS s, corr("a", "b") AS r, \'{"x": 1}\' AS lit\nFROM t\n```')
    act = loop.parse_action(raw)
    assert act["body"].startswith('SELECT "sofa"::INT') and act["body"].endswith("FROM t") and act["file"] == "design/query.sql"
    final = '{"action": "write", "file": "final.md"}\n```\n# Final\n```yaml\na: 1\n```\nend\n```'
    assert loop.parse_action(final)["body"] == "# Final\n```yaml\na: 1\n```\nend"
    assert loop.parse_action('{"action": "edit", "file": "f", "lines": [3, 4]}\n```sql\n```')["body"] == ""
    assert loop.parse_action('{"action": "status"}') == {"action": "status"}


def test_one_lookup_action_searches_several_terms(tmp_path, monkeypatch):
    """Fewer turns: several terms per lookup; each term counts toward the budget."""
    import json
    from ucd import datacard
    monkeypatch.setattr(datacard, "lookups", lambda d, g=None: {"labs": 0})
    monkeypatch.setattr(datacard, "lookup", lambda d, term, kind: f"hits for {term}")
    out = loop._act({"action": "lookup", "dataset": "mimic", "term": ["sodium", "creatinine"]}, tmp_path, 0)
    assert "hits for sodium" in out and "hits for creatinine" in out
    (tmp_path / "loop_log.jsonl").write_text(json.dumps({"action": {"action": "lookup", "term": ["a", "b", "c"]}}) + "\n",
                                            encoding="utf-8")
    assert loop._n_lookups(tmp_path) == 3


def test_thinking_off_survives_unrelated_bad_requests(monkeypatch):
    """E-44: a 422 for another reason must not switch reasoning on for the rest of the run."""
    from ucd.llm import LLMError, LocalLLM
    llm = LocalLLM.__new__(LocalLLM)
    llm.model, llm.temperature, llm.max_tokens, llm.base_url, llm.role = "m", 0.2, 100, "http://x", "planner"
    llm.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def bad(method, url, body):
        raise LLMError("HTTP 422: prompt too long", 422)
    monkeypatch.setattr(llm, "_request", bad)
    with pytest.raises(LLMError):
        llm._chat("s", "u")
    assert llm.extra_body                                       # still there
