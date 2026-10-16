"""Diagnostics log for improving UCD: errors and run outcomes, gathered on the machine where UCD runs (e.g. the
KHDP server, shared by everyone who runs it there).

Every UCD process appends one JSON line per event to <UCD_LOG_DIR, default <project>/logs>/ucd_events.jsonl:
  error   an exception, gateway error, clinical-critic failure, failed design SQL, refused or invalid agent action
  run     one line per finished design loop: version, models, turns, designs executed and failed, internal error
          per iteration, open blocking rules, stop reason, seconds, whether final.md had to be written automatically
  gt      ground-truth agreement of a run (Jaccard, recall, precision: aggregates only)
Never logged: data rows, query results, the planner's SQL, the research question text, quoted values and long
numbers inside error messages (redacted, they can carry data values), API keys. The file is still produced next to
the data: treat it as PHI-equivalent and follow the platform's export rules.

  python -m ucd.diag summary [--days N]   errors by place and type, failure rates, recent runs
  python -m ucd.diag export               logs/ucd_diag_<host>_<time>.zip to bring back for fixing (redacted
                                          again and scanned for keys)
Logging never raises: a broken log must not stop a run. UCD_DIAG=off turns it off.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FILE = "ucd_events.jsonl"
_REDACT = [(re.compile(r"khdp_pat_[A-Za-z0-9_\-]+"), "<key>"), (re.compile(r"(?i)bearer\s+\S+"), "Bearer <key>"),
           (re.compile(r"'[^'\n]{0,200}'"), "'<v>'"), (re.compile(r'"[^"\n]{0,200}"'), '"<v>"'),
           (re.compile(r"\b\d{5,}\b"), "<n>")]
TEXT_KEYS = ("message", "detail", "observation")


def log_dir() -> Path:
    return Path(os.environ.get("UCD_LOG_DIR") or ROOT / "logs")


def redact(text: Any, limit: int = 400) -> str:
    s = str(text)
    for rx, sub in _REDACT:
        s = rx.sub(lambda m: m.group(0) if sub == '"<v>"' and _schema_token(m.group(0)) else sub, s)
    return s[:limit]


def _schema_token(quoted: str) -> bool:
    """A double-quoted token that is schema or syntax, not data, and may stay readable in the log: an SQL identifier
    ("intime", "hosp.admissions") or punctuation. Only double quotes qualify: DuckDB writes identifiers in double
    quotes and data values in single quotes ('kim', 'vancomycin'), which are always redacted. Digits are redacted."""
    inner = quoted[1:-1]
    if re.fullmatch(r"[^A-Za-z0-9\s]{1,3}", inner):
        return True
    return bool(re.fullmatch(r"[a-z_][a-z_.]{0,63}", inner))


def _version() -> str:
    try:
        from ucd import __version__
        return __version__
    except Exception:
        return "?"


def event(kind: str, where: str, **fields: Any) -> None:
    """Append one event. Text fields are redacted; never raises."""
    if os.environ.get("UCD_DIAG", "").lower() == "off":
        return
    try:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "where": where, "version": _version(),
               "host": socket.gethostname(), "pid": os.getpid()}
        rec.update({k: (redact(v) if k in TEXT_KEYS else v) for k, v in fields.items()})
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def error(where: str, err: Any, **fields: Any) -> None:
    """An error: exception type, redacted message and, for exceptions, the code location (file:line function)."""
    tb = None
    if isinstance(err, BaseException) and err.__traceback__ is not None:
        tb = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_tb(err.__traceback__)][-6:]
    event("error", where, type=type(err).__name__ if isinstance(err, BaseException) else "message",
          message=str(err), traceback=tb, **fields)


def read(days: float | None = None) -> list[dict]:
    p = log_dir() / FILE
    if not p.is_file():
        return []
    cut = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - days * 86400)) if days else ""
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("time", "") >= cut:
            out.append(r)
    return out


def summary(days: float | None = None) -> str:
    ev = read(days)
    if not ev:
        return f"no events in {log_dir() / FILE}"
    errs = [e for e in ev if e["kind"] == "error"]
    runs = [e for e in ev if e["kind"] == "run"]
    L = [f"{len(ev)} events ({ev[0]['time']} .. {ev[-1]['time']}), {len(errs)} errors, {len(runs)} runs; "
         f"versions {dict(Counter(e.get('version') for e in ev))}", "", "Errors by place and type (most frequent first):"]
    for (w, t), n in Counter((e["where"], e.get("type")) for e in errs).most_common(15):
        ex = next(e for e in reversed(errs) if (e["where"], e.get("type")) == (w, t))
        L.append(f"  {n:>4}  {w} {t}: {ex.get('message', '')[:120]}")
    if runs:
        n = len(runs)
        L += ["", "Runs:",
              f"  no design executed: {sum(r.get('designs', 0) == 0 for r in runs)}/{n}; "
              f"stopped with blocking findings open: {sum(bool(r.get('open_blocks')) for r in runs)}/{n}; "
              f"AUTO final.md: {sum(bool(r.get('auto_final')) for r in runs)}/{n}",
              f"  median seconds {sorted(r.get('seconds', 0) for r in runs)[n // 2]}, "
              f"median turns {sorted(r.get('turns', 0) for r in runs)[n // 2]}",
              "  most frequent open blocking rules: "
              + ", ".join(f"{k} ({v})" for k, v in Counter(b for r in runs for b in r.get('open_blocks') or []).most_common(8))]
        for r in runs[-8:]:
            L.append(f"  {r['time']} {r.get('name', '?')[:30]:<30} v{r.get('version')} designs={r.get('designs')} "
                     f"failed={r.get('failed_designs')} I={r.get('I')} stop={str(r.get('stop'))[:40]}")
    gts = [e for e in ev if e["kind"] == "gt"]
    if gts:
        L += ["", "Ground truth (latest 8):"] + [f"  {g['time']} {g.get('name', '?')[:30]:<30} J={g.get('jaccard')} "
                                                 f"recall={g.get('recall')} precision={g.get('precision')}" for g in gts[-8:]]
    return "\n".join(L)


def export() -> Path:
    ev = read()
    for e in ev:                                   # redact again: older versions may have logged more
        for k in TEXT_KEYS:
            if k in e:
                e[k] = redact(e[k])
    body = "\n".join(json.dumps(e, ensure_ascii=False, default=str) for e in ev) + "\n"
    if re.search(r"khdp_pat_[A-Za-z0-9]", body):
        raise SystemExit("refusing to export: a KHDP key pattern survived redaction")
    out = log_dir() / f"ucd_diag_{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(FILE, body)
        z.writestr("summary.txt", summary())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ucd.diag", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("--days", type=float)
    sub.add_parser("export")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.cmd == "summary":
        print(summary(a.days))
    else:
        out = export()
        print(f"{out}  ({len(read())} events). Check the platform's export rules before taking it off the server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
