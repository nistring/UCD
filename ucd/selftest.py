"""Self-test of the design loop on the real data, with a scripted model instead of an LLM (0.13.0).

  python -m ucd.selftest [--dataset mimic] [--critic]

It drives every path a planner uses: write the protocol items and both design files (the query with a deliberate
slip), run (the SQL check must name the line), edit that line, explore, run (a design must execute), write final.md,
done. It needs the dataset's parquet and ucd/docs/selftest_<dataset>/{spec.yaml,query.sql}; no LLM is called unless
--critic is given (then the configured clinical critic reviews the design too). Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ucd.gate import ROOT

SLIP = ("gender", "sex_no_such_column")          # one column name made wrong in the first version of the query


class ScriptedPlanner:
    """Replies like a planner would, one JSON action (+ fenced block) per turn."""

    def __init__(self, spec: str, query: str):
        line = next(i for i, l in enumerate(query.splitlines(), 1) if f"p.{SLIP[0]}" in l)
        good = query.splitlines()[line - 1]
        items = ("- {id: P1, kind: population, text: 'adults, first ICU stay per patient'}\n"
                 "- {id: P2, kind: exposure, text: 'age 65 or older at admission'}\n"
                 "- {id: P3, kind: outcome, text: 'in-hospital death'}")
        self.slip_line = line
        self.replies = [
            '{"action": "write", "file": "protocol_items.yaml"}\n```yaml\n' + items + "\n```",
            '{"action": "write", "file": "design/spec.yaml"}\n```yaml\n' + spec.strip() + "\n```",
            '{"action": "write", "file": "design/query.sql"}\n```sql\n'
            + query.strip().replace(f"p.{SLIP[0]}", f"p.{SLIP[1]}", 1) + "\n```",
            '{"action": "run"}',
            f'{{"action": "edit", "file": "design/query.sql", "lines": [{line}, {line}]}}\n```sql\n{good}\n```',
            '{"action": "explore", "group": ["age_65"], "metrics": ["avg(outcome) AS death_rate"]}\n```sql\n'
            + query.strip() + "\n```",
            '{"action": "run"}',
            '{"action": "write", "file": "final.md"}\n```\n# Self-test\n\nThe pipeline ran the scripted design.\n```',
            '{"action": "done"}',
        ]
        self.finish_reason = None

    def complete_text(self, system: str, user: str) -> str:
        assert "design/query.sql" in user, "the prompt must show the design files"
        return self.replies.pop(0) if self.replies else '{"action": "done"}'


def selftest(dataset: str, run_dir: Path, critic: bool = False, echo=print) -> list[str]:
    """Run the scripted loop; returns the failed checks ([] = all passed)."""
    from ucd import loop
    from ucd.ask import Asker
    src = ROOT / "ucd" / "docs" / f"selftest_{dataset}"
    if not (src / "spec.yaml").exists() or not (src / "query.sql").exists():
        return [f"no self-test design for {dataset!r}: add {src.relative_to(ROOT).as_posix()}/spec.yaml and query.sql"]
    planner = ScriptedPlanner((src / "spec.yaml").read_text(encoding="utf-8"), (src / "query.sql").read_text(encoding="utf-8"))
    t0 = time.time()
    before = os.environ.get("UCD_CRITIC_BACKEND")
    if not critic:
        os.environ["UCD_CRITIC_BACKEND"] = "off"
    try:
        fm = loop.run_loop("selftest", run_dir, [dataset], max_turns=12, question="Self-test of the UCD pipeline.",
                           asker=Asker("auto", log=run_dir / "questions.jsonl"), echo=echo, llm=planner)
    finally:                                       # leave the caller's environment as it was
        if before is None:
            os.environ.pop("UCD_CRITIC_BACKEND", None)
        else:
            os.environ["UCD_CRITIC_BACKEND"] = before
    obs = [json.loads(l)["observation"] for l in (run_dir / "loop_log.jsonl").read_text(encoding="utf-8").splitlines()]
    its = [json.loads(l) for l in (run_dir / dataset / "iterations.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (run_dir / dataset / "iterations.jsonl").exists() else []
    n = (its[-1].get("n_final") if its else None) or 0
    checks = {
        "protocol items and design files written": all(o.startswith("wrote") for o in obs[:3]),
        f"SQL check caught the slip and named line {planner.slip_line}":
            len(obs) > 3 and obs[3].startswith("SQL CHECK FAILED") and f"line {planner.slip_line} of" in obs[3],
        "edit by line range applied": len(obs) > 4 and obs[4].startswith("edited design/query.sql"),
        "aggregate query answered": len(obs) > 5 and '"n"' in obs[5],
        "design executed (cohort n > 0)": len(obs) > 6 and "# Iteration" in obs[6] and n > 0,
        "final.md written by the planner": "AUTO-GENERATED" not in fm.read_text(encoding="utf-8"),
    }
    echo(f"\nSelf-test on {dataset} ({time.time() - t0:.0f} s, run dir {run_dir}): cohort n = {n}")
    for name, ok in checks.items():
        echo(f"  [{'ok' if ok else 'FAILED'}] {name}")
    return [name for name, ok in checks.items() if not ok]


def main(argv=None) -> int:
    from ucd import datasets
    p = argparse.ArgumentParser(prog="python -m ucd.selftest", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default=None, help="default: the first dataset in ucd/datasets.yaml")
    p.add_argument("--critic", action="store_true", help="also call the configured LLM clinical critic")
    p.add_argument("--run-dir", default=None, help="default: runs/ucd/selftest/<time>")
    a = p.parse_args(argv)
    ds = a.dataset or datasets.default()
    rd = Path(a.run_dir) if a.run_dir else ROOT / "runs" / "ucd" / "selftest" / time.strftime("%Y%m%d_%H%M%S")
    failed = selftest(ds, rd, a.critic)
    print("\nSELF-TEST PASSED" if not failed else f"\nSELF-TEST FAILED: {len(failed)} check(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
