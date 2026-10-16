"""Privacy / fairness audit of planner transcripts: flags any tool call that touches forbidden paths or bypasses
the ucd CLI. Usage: python -m eval.audit <transcript_dir>   -> eval/audit.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"physionet", r"Synthetic_KMIMIC", r"KMIMIC Sample", r"\.kcd_cache",
             r"(?<![\w])eval[/\\]+(external|references|bench|variants)", r"(?<![\w])prep[/\\]+[a-z]",
             r"(?<![\w])kcd_agent[/\\]+[a-z]", r"ucd[/\\][a-z_]+\.py", r"\.parquet", r"read_csv|read_parquet|pd\.read|duckdb\.connect",
             r"WebFetch|WebSearch"]
PAT = re.compile("|".join(FORBIDDEN))


def scan(f: Path) -> dict:
    run_dir, hits, n_calls, tools = None, [], 0, {}
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        msg = rec.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text" and run_dir is None:
                m = re.search(r"runs/ucd/(t\d_\w+/[ra]\d)", c.get("text", ""))
                if m:
                    run_dir = m.group(1)
            if c.get("type") != "tool_use":
                continue
            n_calls += 1
            name = c.get("name")
            tools[name] = tools.get(name, 0) + 1
            if name in ("WebFetch", "WebSearch"):
                hits.append({"tool": name, "input": str(c.get("input"))[:200]})
                continue
            s = json.dumps(c.get("input"), ensure_ascii=False)
            if run_dir is None:
                m = re.search(r"runs/ucd/(t\d_\w+/[ra]\d)", s)
                if m:
                    run_dir = m.group(1)
            # the planner's own instruction file mentions forbidden paths only as rules: ignore reads of prompt.md
            if name == "Read" and s.endswith('prompt.md"}'):
                continue
            # paths inside the run dir / other runs
            other = re.findall(r"runs/ucd/(t\d_\w+/[ra]\d)", s)
            if run_dir and any(o != run_dir for o in other):
                hits.append({"tool": name, "why": "other run", "input": s[:300]})
            if PAT.search(s) and not re.search(r"python -m ucd", s):
                hits.append({"tool": name, "why": PAT.search(s).group(0), "input": s[:300]})
    return {"transcript": f.name, "run": run_dir, "tool_calls": n_calls, "tools": tools, "violations": hits}


def main(d: str) -> None:
    out = []
    for f in sorted(Path(d).glob("*.jsonl")):
        r = scan(f)
        if r["run"]:
            out.append(r)
    (ROOT / "eval" / "audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in out:
        print(f"{r['run']:<22} calls={r['tool_calls']:<4} violations={len(r['violations'])}")
        for h in r["violations"][:5]:
            print("     ", h.get("why", h["tool"]), "|", h["input"][:160])


if __name__ == "__main__":
    main(sys.argv[1])
