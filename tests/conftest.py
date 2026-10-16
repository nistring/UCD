import pytest


@pytest.fixture(autouse=True)
def _diag_to_tmp(tmp_path, monkeypatch):
    """Tests never write to the real diagnostics log (logs/ucd_events.jsonl)."""
    monkeypatch.setenv("UCD_LOG_DIR", str(tmp_path / "diag_logs"))
    monkeypatch.delenv("UCD_DIAG", raising=False)
