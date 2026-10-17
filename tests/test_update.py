"""ucd.update: only changed files are written; user edits, data and unknown files survive."""
import json
import os
import zipfile

import pytest

from ucd import update as U


def _install(root, files, version="0.1"):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    hashes = {r: U.digest(t.encode()) for r, t in files.items()}
    (root / U.MANIFEST).write_text(json.dumps({"version": version, "files": hashes}), encoding="utf-8")


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for rel, text in files.items():
            z.writestr(f"ucd_project/{rel}", text)
    return path


OLD = {"ucd/__init__.py": '__version__ = "0.1"\n', "ucd/same.py": "a\n", "ucd/changed.py": "old\n",
       "ucd/config.yaml": "model: x\n", "ucd/both.py": "old\n", "ucd/gone.py": "x\n", "ucd/gone_edited.py": "x\n"}
NEW = {"ucd/__init__.py": '__version__ = "0.2"\n', "ucd/same.py": "a\n", "ucd/changed.py": "new\n",
       "ucd/config.yaml": "model: x\n", "ucd/both.py": "new\n", "ucd/added.py": "n\n"}


def test_update_touches_only_what_changed(tmp_path):
    root = tmp_path / "inst"
    _install(root, OLD)
    (root / "ucd/config.yaml").write_text("model: mine\n", encoding="utf-8")     # user edit, version unchanged
    (root / "ucd/both.py").write_text("mine\n", encoding="utf-8")                # user edit, version changed too
    (root / "ucd/gone_edited.py").write_text("mine\n", encoding="utf-8")
    (root / "samples/s/gt.csv").parent.mkdir(parents=True)
    (root / "samples/s/gt.csv").write_text("stay_id\n1\n", encoding="utf-8")      # never part of a version
    same = root / "ucd/same.py"
    os.utime(same, (1_000_000, 1_000_000))
    counts = U.apply(root, U.read_source(_zip(tmp_path / "b.zip", NEW)), log=lambda *_: None)

    assert counts == {"same": 1, "add": 1, "update": 2, "conflict": 1, "keep": 1, "remove": 1, "keep-removed": 1}
    assert same.stat().st_mtime == 1_000_000                                    # unchanged file not rewritten
    assert (root / "ucd/changed.py").read_text() == "new\n" and (root / "ucd/added.py").exists()
    assert (root / "ucd/config.yaml").read_text() == "model: mine\n"           # your edit kept
    assert (root / "ucd/both.py").read_text() == "new\n"
    backup = next((root / U.BACKUP).iterdir())
    assert (backup / "ucd/both.py").read_text() == "mine\n"                    # your conflicting copy saved
    assert not (root / "ucd/gone.py").exists() and (backup / "ucd/gone.py").exists()
    assert (root / "ucd/gone_edited.py").read_text() == "mine\n"
    assert (root / "samples/s/gt.csv").exists()
    man = json.loads((root / U.MANIFEST).read_text(encoding="utf-8"))
    assert man["version"] == "0.2" and "ucd/gone.py" not in man["files"]
    # a second update with the same bundle changes nothing, and your config edit is still recognised as yours
    again = U.apply(root, U.read_source(tmp_path / "b.zip"), log=lambda *_: None)
    assert again["keep"] == 1 and again["update"] == again["add"] == again["conflict"] == 0


def test_crlf_only_difference_counts_as_same(tmp_path):
    root = tmp_path / "inst"
    _install(root, {"ucd/__init__.py": '__version__ = "0.1"\n', "ucd/a.py": "x\ny\n"})
    (root / "ucd/a.py").write_bytes(b"x\r\ny\r\n")
    counts = U.apply(root, {"ucd/__init__.py": b'__version__ = "0.1"\n', "ucd/a.py": b"x\ny\n"}, log=lambda *_: None)
    assert counts["same"] == 2 and (root / "ucd/a.py").read_bytes() == b"x\r\ny\r\n"


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "inst"
    _install(root, OLD)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    U.apply(root, U.read_source(_zip(tmp_path / "b.zip", NEW)), dry_run=True, log=lambda *_: None)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_unsafe_paths_are_refused(tmp_path):
    for bad in ("../evil.py", "/etc/x", "C:/x.py"):
        with zipfile.ZipFile(tmp_path / "e.zip", "w") as z:
            z.writestr("ucd/__init__.py", "")
            z.writestr(bad, "x")
        with pytest.raises(SystemExit):
            U.read_source(tmp_path / "e.zip")


def test_failed_write_keeps_the_old_manifest(tmp_path, monkeypatch):
    root = tmp_path / "inst"
    _install(root, OLD)
    old = (root / U.MANIFEST).read_text(encoding="utf-8")
    real = type(root).write_bytes

    def flaky(self, data):
        if self.name == "changed.py":
            raise OSError(36, "File name too long")
        return real(self, data)
    monkeypatch.setattr(type(root), "write_bytes", flaky)
    counts = U.apply(root, U.read_source(_zip(tmp_path / "b.zip", NEW)), log=lambda *_: None)
    assert counts["failed"] == 1 and (root / U.MANIFEST).read_text(encoding="utf-8") == old
    assert (root / "ucd/added.py").exists()                                  # the rest was still applied


def test_parallel_parquet_build(tmp_path, monkeypatch):
    """Several tables converted at once by separate processes; finished tables are skipped on a re-run."""
    from ucd import build_parquet as B
    src, out = tmp_path / "raw", tmp_path / "pq"
    for module, table, n in (("hosp", "a", 50), ("hosp", "b", 5), ("icu", "c", 20)):
        (src / module).mkdir(parents=True, exist_ok=True)
        (src / module / f"{table}.csv").write_text("x,y\n" + "".join(f"{i},v{i}\n" for i in range(n)), encoding="utf-8")
    monkeypatch.setattr(B.datasets, "source_dir", lambda name: src)
    monkeypatch.setattr(B.datasets, "parquet_dir", lambda name: out)
    monkeypatch.setattr(B.datasets, "is_direct", lambda name: False, raising=False)   # a toy name is not registered
    assert [d.name for _, d in B.plan("toy")] == ["a.parquet", "c.parquet", "b.parquet"]   # largest first
    B.build("toy", jobs=3)
    import duckdb
    assert duckdb.sql(f"SELECT count(*) FROM '{(out / 'hosp' / 'a.parquet').as_posix()}'").fetchone()[0] == 50
    assert not list(out.rglob("*.part")) and B.plan("toy") == []
