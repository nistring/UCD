"""Incremental design iterations (WITH-step cache), critic tagged lines, broad-first intake,
usage-error logging. No data, no network."""
import json

import duckdb
import pytest

from ucd import sqlcte

SQL = """WITH a AS (SELECT 1 AS x, ')' AS p -- a ) in a comment
), b(y) AS (SELECT x FROM a WHERE '(' <> 'z'), c AS (SELECT y * 10 AS v FROM b)
SELECT v FROM c"""


def test_split_respects_literals_comments_and_column_lists():
    steps, final = sqlcte.split(SQL)
    assert [h for h, _ in steps] == ["a", "b(y)", "c"] and final == "SELECT v FROM c"
    assert sqlcte.references(steps[2][1], ["a", "b", "c"]) == ["b"]
    assert sqlcte.split("SELECT 1") is None and sqlcte.split("WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r") is None
    joined = sqlcte.join(steps, final)
    assert sqlcte.split(joined) == (steps, final) and duckdb.connect().execute(joined).fetchall() == [(10,)]


def _gate():
    from ucd.gate import Gate
    g = Gate.__new__(Gate)
    g.con, g._steps, g._prev_keys, g.last_reused = duckdb.connect(), {}, set(), []
    return g


def test_step_cache_reuses_unchanged_steps_and_stays_correct():
    g = _gate()
    assert g.con.execute(g._with_cached_steps(SQL)).fetchall() == [(10,)] and g.last_reused == []
    changed = SQL.replace("y * 10 AS v", "y * 100 AS v")      # an edit of one line of step c
    assert g.con.execute(g._with_cached_steps(changed)).fetchall() == [(100,)]
    assert g.last_reused == ["a", "b"]                         # only c and the final SELECT recomputed
    upstream = changed.replace("SELECT 1 AS x", "SELECT 2 AS x").replace("y * 100 AS v", "y * 10 AS v")
    assert g.con.execute(g._with_cached_steps(upstream)).fetchall() == [(20,)]
    assert g.last_reused == []                                 # b and c read from a: recomputed too


def test_critic_tagged_lines_parse_and_fall_back_to_json():
    """E-45: the critic replies in tagged lines; quotes and SQL need no escaping, wrapped lines are joined."""
    from ucd.llm_critic import _parse
    r = _parse('OVERALL: revise\nPLAUSIBILITY: 0.6\nISSUE | block | sql | first sodium per "hadm_id" | '
               "PARTITION BY l.hadm_id | filter the window first,\nthen row_number() per stay\n"
               "WAIVER | V4 | grant | Outcome: in-hospital death. | mandated\n")
    assert r["overall"] == "revise" and r["clinical_plausibility"] == 0.6
    assert r["issues"][0]["proposed_change"] == "filter the window first, then row_number() per stay"
    assert r["waiver_decisions"][0] == {"rule": "V4", "granted": True, "quote": "Outcome: in-hospital death.",
                                        "reason": "mandated"}
    assert _parse('{"overall": "accept", "issues": []}')["overall"] == "accept"


def test_intake_asks_broad_scope_before_details():
    from ucd import intake as I
    from ucd.ask import Asker
    asked = []

    class LLM:
        def complete_text(self, system, user):
            return json.dumps({"fields": {}, "complete": False, "questions": [
                {"field": "time_zero", "question": "Time zero?", "options": ["ICU admission", "hospital admission"]},
                {"field": "population", "question": "Which patients?", "options": ["adults", "all"]}]})
    a = Asker("auto", write=lambda *_: None)
    real = a.ask
    a.ask = lambda q, *x, **k: (asked.append(q), real(q, *x, **k))[1]
    I.intake("sodium and death", LLM(), a, log=lambda *_: None)
    assert asked[0] == "Which patients?" and "Time zero?" not in asked[:1]


def test_usage_errors_are_logged_with_their_message():
    """E-46: the log used to say only '2'."""
    from ucd import chat, diag
    with pytest.raises(SystemExit):
        chat.main(["--no-such-option"])
    ev = [e for e in diag.read() if e["kind"] == "usage"]
    assert ev and "no-such-option" in ev[-1]["message"]
