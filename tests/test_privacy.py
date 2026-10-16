"""The aggregate API never returns identifiers, under any column name (E-68). No data, no network."""
import duckdb
import pytest

from ucd import gate as G
from ucd.gate import Gate, GateError


def _gate(tmp_path, monkeypatch):
    g = Gate.__new__(Gate)
    g.con, g._steps, g._prev_keys, g.last_reused, g.dataset, g.run_dir = duckdb.connect(), {}, set(), [], "toy", None
    # 300 patients with 40 lab rows each: an aliased patient id is neither an id name nor near-unique
    g.con.execute("CREATE TABLE patients AS SELECT 10000000 + i AS subject_id, i % 2 AS sex FROM range(300) t(i)")
    g.con.execute("CREATE TABLE labs AS SELECT 10000000 + (i % 300) AS subject_id, i % 7 AS item, i * 1.0 AS value "
                  "FROM range(12000) t(i)")
    g.con.execute("CREATE TABLE person AS SELECT i AS person_id, i % 90 AS age FROM range(300) t(i)")
    monkeypatch.setattr(G, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(G, "GLOBAL_LOG", tmp_path / "gate.log")
    monkeypatch.setattr(Gate, "_unit_tables", lambda self: [("subject_id", "patients"), ("person_id", "person")])
    return g


def test_aliased_or_renamed_ids_are_refused(tmp_path, monkeypatch):
    g = _gate(tmp_path, monkeypatch)
    for src, col in (("SELECT subject_id AS s, value FROM labs", "s"),            # aliased: passed before E-68
                     ("SELECT CAST(subject_id AS VARCHAR) AS s, value FROM labs", "s"),
                     ("SELECT person_id AS p FROM person, range(20)", "p")):
        with pytest.raises(GateError, match="identifier"):
            g.aggregate(src, [col], [])
    with pytest.raises(GateError, match="identifier"):                          # an OMOP id by its own name
        g.aggregate("SELECT person_id, age FROM person, range(20)", ["person_id"], [])
    with pytest.raises(GateError, match="identifier"):
        Gate.sanitize(__import__("pandas").DataFrame({"visit_occurrence_id": [1], "n": [10]}))


def test_categories_and_codes_still_group(tmp_path, monkeypatch):
    g = _gate(tmp_path, monkeypatch)
    assert len(g.aggregate("SELECT item, value FROM labs", ["item"], ["avg(value) AS m"])["rows"]) == 7
    assert len(g.aggregate("SELECT sex FROM patients, range(20)", ["sex"], [])["rows"]) == 2
    assert G.is_id_name("visit_occurrence_id") and G.is_id_name("STAY_ID") and not G.is_id_name("measurement_concept_id")
    assert not G.is_id_name("itemid") and not G.is_id_name("valid")
