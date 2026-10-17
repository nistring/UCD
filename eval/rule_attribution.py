"""Do critic findings move the cohort toward the ground truth or away from it? One row per critic rule.

  python -m eval.rule_attribution runs/experiments/<out> [more experiment folders] [--samples samples]

For every run in an experiment folder whose sample has a gt.csv, each revision (executed design k -> k+1) is compared
with the ground truth on both sides (Jaccard). The findings shown at design k that are gone at design k+1 count as
acted on, and the change in Jaccard is credited to each of them. LLM-critic findings are grouped by category
(L-sql, L-clinical, ...), since their numbers differ between runs. Rules that are hidden from the planner in an arm
(rule_feedback: false) are not counted for that arm: the planner could not act on them.

The attribution is associative, not causal: the agent may change a design for other reasons in the same revision,
and a revision that resolves several findings credits all of them. Output: <first folder>/rule_attribution.{csv,md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rule_key(f: dict) -> str:
    r = str(f.get("rule", ""))
    return f"L-{r.split('-', 1)[1]}" if r.startswith("L") and "-" in r else r


def shown_findings(run_dir: Path, dataset: str, rec: dict) -> set[str]:
    """Rule keys of the findings the planner saw for one iteration (block or warn)."""
    return {rule_key(f) for f in (rec.get("critique") or {}).get("findings", []) if f.get("severity") in ("block", "warn")}


def revisions(run_dir: Path, dataset: str, gt: Path) -> list[dict[str, Any]]:
    """(k -> k+1, Jaccard before and after, findings acted on) for every pair of consecutive executed designs."""
    from eval import groundtruth as G
    from ucd import cli
    from ucd.gate import Gate
    hist = [h for h in cli._history(run_dir / dataset) if not (h.get("card") or {}).get("error")]
    if len(hist) < 2:
        return []
    df = G.load_cohort(str(gt))
    u = G.units(dataset)
    link = G.links(Gate(dataset).con, dataset)
    mapping = G.map_columns(df, u)

    def jac(it: int) -> Optional[float]:
        try:
            return G.compare(run_dir, df, mapping, link, u, dataset, iteration=it).get("jaccard")
        except SystemExit:
            return None

    out = []
    for a, b in zip(hist, hist[1:]):
        ja, jb = jac(a["iter"]), jac(b["iter"])
        if ja is None or jb is None:
            continue
        before, after = shown_findings(run_dir, dataset, a), shown_findings(run_dir, dataset, b)
        out.append({"from": a["iter"], "to": b["iter"], "j_before": ja, "j_after": jb, "delta": round(jb - ja, 4),
                    "acted_on": sorted(before - after), "introduced": sorted(after - before)})
    return out


def find_sample(name: str, roots: list[Path]) -> Optional[Path]:
    for r in roots:
        hits = [p for p in r.rglob(name) if (p / "protocol.txt").exists()]
        if hits:
            return hits[0]
    return None


def analyse(folders: list[Path], sample_roots: list[Path]) -> tuple[list[dict], list[dict]]:
    from eval.experiments import ARMS
    from ucd.sample import find
    revs, per_rule = [], defaultdict(list)
    for folder in folders:
        f = folder / "results.csv"
        for row in (csv.DictReader(open(f, encoding="utf-8")) if f.exists() else []):
            s = Path(row["sample_dir"]) if row.get("sample_dir") else find_sample(row["sample"].split("/")[-1],
                                                                                  sample_roots)
            if s is None:
                continue
            _, gt, conf = find(s)
            run_dir = Path(row["run_dir"])
            if gt is None or not run_dir.exists():
                continue
            hidden_rules = ARMS.get(row["arm"], {}).get("config", {}).get("rule_feedback") is False
            for r in revisions(run_dir, conf.get("dataset") or "mimic", gt):
                acted = [k for k in r["acted_on"] if not (hidden_rules and not k.startswith("L-"))]
                revs.append({**r, "acted_on": acted, "sample": row["sample"], "arm": row["arm"],
                             "planner": row.get("planner") or ""})
                for k in acted:
                    per_rule[k].append(r["delta"])
    rules = [{"rule": k, "revisions": len(v), "mean_delta_jaccard": round(mean(v), 4),
              "improved": sum(d > 0.005 for d in v), "worsened": sum(d < -0.005 for d in v),
              "unchanged": sum(abs(d) <= 0.005 for d in v)} for k, v in per_rule.items()]
    rules.sort(key=lambda r: r["mean_delta_jaccard"])
    return revs, rules


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval.rule_attribution", description=__doc__.split("\n\n")[0])
    p.add_argument("folder", nargs="+", help="experiment output folders (with results.csv)")
    p.add_argument("--samples", action="append", default=None, help="where the sample folders are (default samples)")
    a = p.parse_args(argv)
    folders = [Path(x) for x in a.folder]
    revs, rules = analyse(folders, [Path(x) for x in (a.samples or ["samples"])])
    out = folders[0]
    with open(out / "rule_attribution.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rule", "revisions", "mean_delta_jaccard", "improved", "worsened", "unchanged"])
        w.writeheader()
        w.writerows(rules)
    (out / "rule_attribution_revisions.json").write_text(json.dumps(revs, indent=1), encoding="utf-8")
    L = [f"# Rule attribution ({len(revs)} revisions with a ground truth)", "",
         "Change in Jaccard with the ground truth after a revision that resolved the finding (associative: other "
         "changes in the same revision are credited too). Most harmful first.", "",
         "| rule | revisions | mean change in Jaccard | improved | worsened | unchanged |", "|---|---|---|---|---|---|"]
    L += [f"| {r['rule']} | {r['revisions']} | {r['mean_delta_jaccard']:+.4f} | {r['improved']} | {r['worsened']} | "
          f"{r['unchanged']} |" for r in rules]
    text = "\n".join(L) + "\n"
    (out / "rule_attribution.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
