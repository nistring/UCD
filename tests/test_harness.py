"""SQL bind-check, concept library, topic packs, stop rule, protocol clarifications. No data, no network."""
import duckdb
import pytest

from ucd import loop


def _gate():
    from ucd.gate import Gate
    g = Gate.__new__(Gate)
    g.con, g._steps, g._prev_keys, g.last_reused, g.dataset = duckdb.connect(), {}, set(), [], "toy"
    g.con.execute("CREATE TABLE patients AS SELECT 1 AS subject_id, 60 AS anchor_age")
    g.con.execute("CREATE TABLE admissions AS SELECT 1 AS subject_id, 10 AS hadm_id")
    g.schema = lambda: {"main": {"patients": [("subject_id", "INT"), ("anchor_age", "INT")],
                                 "admissions": [("subject_id", "INT"), ("hadm_id", "INT")]}}
    return g


def test_bind_check_finds_sql_slips_without_running_anything():
    g = _gate()
    assert g.bind_check("SELECT subject_id AS unit_id FROM patients", ["unit_id"]) == ""
    slip = g.bind_check("SELECT a.anchor_age FROM main.admissions a")
    assert "anchor_age" in slip and "'a' is main.admissions" in slip and "exists in: main.patients" in slip
    assert "does not return" in g.bind_check("SELECT subject_id FROM patients", ["unit_id", "outcome"])
    assert "single SELECT" in g.bind_check("DELETE FROM patients")


def test_column_hint_lists_the_columns_of_a_named_table_or_concept():
    """E-64: for `FROM concept.chemistry che`, the hint said 'no table has a column ...' (concepts are not in the raw
    schema) and never showed what `che` has, so the model guessed again."""
    g = _gate()
    g.con.execute("CREATE SCHEMA derived; CREATE VIEW derived.chemistry AS SELECT 10 AS hadm_id, 1.0 AS creatinine")
    msg = g.bind_check("SELECT che.creatinine_value FROM derived.chemistry che")    # not in the raw schema, like a concept
    assert "'che' is derived.chemistry (columns: hadm_id, creatinine)" in msg
    assert "'a' is main.admissions (columns: subject_id, hadm_id)" in g.bind_check("SELECT a.anchor_age FROM main.admissions a")


def test_bind_check_explains_window_and_aggregate_aliases_in_where():
    """E-62: DuckDB lets WHERE name a SELECT alias, so the Postgres habit `... AS rn ... WHERE rn = 1` fails with
    'WHERE clause cannot contain window functions'; the bare message made small models repeat the slip."""
    g = _gate()
    sql = ("WITH firsts AS (SELECT subject_id, row_number() OVER (PARTITION BY subject_id ORDER BY hadm_id) AS rn "
           "FROM admissions WHERE rn = 1) SELECT subject_id AS unit_id FROM firsts")
    msg = g.bind_check(sql, ["unit_id"])
    assert "window functions" in msg and "QUALIFY rn = 1" in msg
    agg = g.bind_check("SELECT subject_id, max(anchor_age) AS top FROM patients WHERE top > 50 GROUP BY subject_id")
    assert "aggregates" in agg and "HAVING" in agg
    fixed = sql.replace("WHERE rn = 1", "QUALIFY rn = 1")                       # the advice really binds
    assert g.bind_check(fixed, ["unit_id"]) == ""


def test_concepts_build_once_with_dependencies_then_reuse_the_cache(tmp_path, monkeypatch):
    from ucd import concepts as K
    d = tmp_path / "sql"
    d.mkdir()
    (d / "adults.sql").write_text("DROP TABLE IF EXISTS src_derived.adults; CREATE TABLE src_derived.adults AS "
                                  "SELECT subject_id FROM src_main.patients WHERE anchor_age >= 18", encoding="utf-8")
    (d / "adult_admissions.sql").write_text("CREATE TABLE src_derived.adult_admissions AS SELECT a.hadm_id FROM "
                                            "src_main.admissions a JOIN src_derived.adults USING (subject_id)",
                                            encoding="utf-8")
    monkeypatch.setattr(K, "_conf", lambda ds: {"sql_dir": str(d), "schemas": {"src_main": "main", "src_derived": "concept"}})
    monkeypatch.setattr(K, "CACHE", tmp_path / "cache")
    K.index.cache_clear()
    g = _gate()
    assert K.dependencies("toy", "adult_admissions") == ["adults"]
    assert K.ensure(g.con, "toy", ["adult_admissions"]) == ["adults", "adult_admissions"]      # dependency first
    assert g.con.execute("SELECT * FROM concept.adult_admissions").fetchall() == [(10,)]
    g2 = _gate()
    assert K.ensure(g2.con, "toy", ["adult_admissions"]) == []                                  # cached: nothing built
    with pytest.raises(ValueError, match="unknown concept"):
        K.ensure(g2.con, "toy", ["nope"])
    K.index.cache_clear()


def test_topic_packs_detect_ask_and_check():
    from ucd import packs
    assert packs.detect("maximum KDIGO stage of acute kidney injury", "mimic") == ["aki"]
    assert packs.detect("platelets below 50 within 48 h", "mimic") == []
    assert {q["pack"] for q in packs.intake_questions("septic shock onset", "mimic")} == {"sepsis"}
    assert packs.checks("acute kidney injury", "SELECT * FROM concept.kdigo_stages", "mimic") == []
    assert packs.checks("acute kidney injury", "SELECT creat FROM hosp.labevents", "mimic")[0]["rule"] == "PK-aki"


def test_stop_rule_is_no_blocking_findings_or_the_cap():
    from ucd import critic as C
    assert C.should_stop([0.9], 0) == (True, "no blocking findings")             # whatever I is
    assert C.should_stop([0.05, 0.04], 1) == (False, "")                         # a block left: go on
    assert C.should_stop([0.5, 0.4, 0.3], 2)[1].startswith("reached max iterations")


def test_a_settled_definition_goes_back_into_the_protocol(tmp_path):
    (tmp_path / "protocol.md").write_text("Population: platelets low early in the ICU stay\n", encoding="utf-8")
    loop._clarify_protocol(tmp_path, "What does early mean?", "within 48 h of ICU admission", "choice")
    q, system = loop._reload_protocol(tmp_path, "old", "mimic")
    assert "Clarifications during design:\n- What does early mean? -> within 48 h of ICU admission (user)" in q
    assert "within 48 h of ICU admission" in system                              # the planner's prompt has it
    import json
    assert "within 48 h" in json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["question"]   # critic


def test_critic_tags_wrapped_in_markdown_still_parse():
    from ucd.llm_critic import _parse
    r = _parse("Review:\n**OVERALL:** revise\n**PLAUSIBILITY:** 0.7\n- ISSUE | warn | sql | x | y | z\n")
    assert r["overall"] == "revise" and r["clinical_plausibility"] == 0.7 and r["issues"][0]["severity"] == "warn"


def test_review_fixes_v090(tmp_path, monkeypatch):
    """B2, R2, R3, R5, C2 from the v0.9.0 review (B1, revise after a failed check, went with revise in 0.13.0)."""
    from ucd import concepts as K, packs
    from ucd.llm_critic import _parse
    # B2: prose that starts with "Overall" does not swallow a JSON verdict and its block
    r = _parse('- Overall the design is close, but:\n{"overall": "revise", "issues": [{"severity": "block", "message": "x"}]}')
    assert r["issues"] and r["issues"][0]["severity"] == "block"
    # R5: an exclusion line does not make a study about sepsis or AKI
    assert packs.detect("Objective: delirium after surgery\nExclusions: sepsis, chronic renal failure", "mimic") == []
    # C2: a concept name inside a string literal is not a concept reference
    assert K.referenced("SELECT 'concept.foo' AS label FROM concept.sofa") == ["sofa"]
    # R2 + R3: declaring computes nothing; a changed SQL makes the cached parquet stale
    sql = tmp_path / "sql"
    sql.mkdir()
    (sql / "one.sql").write_text("CREATE TABLE d.one AS SELECT 1 AS v", encoding="utf-8")
    monkeypatch.setattr(K, "_conf", lambda ds: {"sql_dir": str(sql), "schemas": {"d": "concept"}})
    monkeypatch.setattr(K, "CACHE", tmp_path / "cache")
    K.index.cache_clear()
    con = duckdb.connect()
    K.declare(con, "toy", ["one"])
    assert con.execute("SELECT v FROM concept.one").fetchall() == [(1,)] and not (tmp_path / "cache").exists()
    assert K.ensure(con, "toy", ["one"]) == ["one"] and K.ensure(con, "toy", ["one"]) == []
    (sql / "one.sql").write_text("CREATE TABLE d.one AS SELECT 2 AS v", encoding="utf-8")
    assert K.ensure(con, "toy", ["one"]) == ["one"]                   # stale cache rebuilt
    assert con.execute("SELECT v FROM concept.one").fetchall() == [(2,)]
    assert not list((tmp_path / "cache").rglob("*.part"))
    (sql / "one.sql").write_text("CREATE TABLE d.one AS SELECT 3 AS v", encoding="utf-8")
    with pytest.raises(ValueError, match="concepts build"):
        K.ensure(con, "toy", ["one"], max_new=0)                      # a query's cap
    assert K.ensure(con, "toy", ["one"], max_new=None) == ["one"]     # the build command: no cap (E-54)
    K.index.cache_clear()


def test_cte_alias_hint_lists_the_step_columns():
    from ucd.gate import Gate
    g = Gate.__new__(Gate)
    import duckdb
    g.con = duckdb.connect()
    sql = "WITH adm AS (SELECT 1 AS hadm_id, 2 AS subject_id) SELECT a.intime FROM adm a"
    try:
        g.con.execute(f"DESCRIBE {sql}")
    except duckdb.Error as e:
        err = str(e)
    g.schema = lambda: {}
    hint = g.column_hint(err, sql)
    assert "step 'adm' returns: hadm_id, subject_id" in hint, hint


def test_cap_counts_named_concepts_not_dependencies_and_inline(tmp_path, monkeypatch):
    """E-57: a concept with many dependencies (sepsis3: 21) builds under the per-query cap; E-59: inline() gives one
    query over the raw tables only."""
    import duckdb
    from ucd import concepts as K
    sql = tmp_path / "sql"
    sql.mkdir()
    (sql / "base.sql").write_text("CREATE TABLE d.base AS SELECT 1 AS v", encoding="utf-8")
    for i in range(1, 5):
        (sql / f"c{i}.sql").write_text(f"CREATE TABLE d.c{i} AS SELECT v + 1 AS v FROM d.{'base' if i == 1 else f'c{i-1}'}",
                                       encoding="utf-8")
    monkeypatch.setattr(K, "_conf", lambda ds: {"sql_dir": str(sql), "schemas": {"d": "concept"}})
    monkeypatch.setattr(K, "CACHE", tmp_path / "cache")
    K.index.cache_clear()
    q = "WITH x AS (SELECT v FROM concept.c4) SELECT v * 10 AS w FROM x"
    standalone = K.inline(q, "toy")
    assert "concept." not in standalone and duckdb.sql(standalone).fetchall() == [(50,)]
    assert K.inline("SELECT 1", "toy") == "SELECT 1"
    con = duckdb.connect()
    assert K.ensure(con, "toy", ["c4"], max_new=1) == ["base", "c1", "c2", "c3", "c4"]
    assert con.execute(q).fetchall() == [(50,)]
    K.index.cache_clear()


def test_exposure_type_aliases():
    from ucd.spec import Variable
    assert Variable(column="x", description="", type="continuous").type == "numeric"
    assert Variable(column="x", description="", type="Dichotomous").type == "binary"
    assert Variable(column="x", description="", type="nominal").type == "categorical"
    with pytest.raises(Exception):
        Variable(column="x", description="", type="banana")


def test_critic_context_has_dataset_notes_and_topic_notes():
    from ucd import llm_critic
    ctx = llm_critic.context("mimic", "Objective: does SOFA-2 at ICU admission predict ICU mortality in sepsis?")
    assert "deathtime" in ctx and "SOFA-2 are two different scoring systems" in ctx
    assert "SOFA-2" not in llm_critic.context("mimic", "Objective: sodium and death")
    assert "Named definitions" in llm_critic.SYSTEM


def test_protocol_change_starts_a_new_design_round(tmp_path):
    """E-60: after 3 designs the user edits the protocol; the next design must not be stopped at once for "reached
    max iterations" - the stop rule counts only the current round."""
    import json
    from ucd import cli, loop
    from ucd.critic import MAX_ITER
    (tmp_path / "mimic").mkdir()
    (tmp_path / "meta.json").write_text(json.dumps({"question": "old"}), encoding="utf-8")
    (tmp_path / "protocol.md").write_text("Objective: new", encoding="utf-8")
    recs = [{"iter": i, "round": 0, "card": {}, "internal_error": 0.3} for i in range(1, MAX_ITER + 1)]
    (tmp_path / "mimic" / "iterations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert cli.design_round(tmp_path) == 0 and loop._n_runs(tmp_path, this_round=True) == MAX_ITER
    q, _ = loop._reload_protocol(tmp_path, "old", "mimic")
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert q == "Objective: new" and meta["round"] == 1 and meta["rounds"][0]["after_designs"] == MAX_ITER
    assert loop._n_runs(tmp_path) == MAX_ITER and loop._n_runs(tmp_path, this_round=True) == 0
    assert "Design runs 0/3 in this round (3 in all)" in loop._next_step(tmp_path, 0, 30)
    loop._new_round(tmp_path, "user chose to keep designing")
    assert cli.design_round(tmp_path) == 2
