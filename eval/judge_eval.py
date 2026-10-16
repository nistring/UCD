"""How often does the LLM waiver judge grant a waiver it should deny (false waiver) or deny one it should grant?

Planted cases pair a protocol with a waiver request; the expected decision is fixed by construction. Each case is
run `--reps` times through ucd.llm_critic.review, and the code-side checks of ucd.critic.apply_waivers (verbatim
quote) are applied too, so the reported rates are for the full waiver mechanism.

  python -m eval.judge_eval [--reps 3]   -> eval/judge_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ucd import llm_critic
from ucd.critic import quote_supported

ROOT = Path(__file__).resolve().parents[1]
T = yaml.safe_load((ROOT / "ucd" / "topics.yaml").read_text(encoding="utf-8"))

PREDICT24 = T["t2_sofa"].replace("Outcome: in-hospital death.",
    "Prediction is made 24 h after ICU admission for patients alive and still in the ICU; outcome: in-hospital "
    "death after that prediction time.")
SILENT = ("Objective: Does the first-day lactate level predict death in adult ICU patients?\n"
          "Population: adults in the ICU. Predictor: lactate in the first 24 h. Outcome: death.\n"
          "Analysis: AUROC of lactate.")

V4_SPEC = """prediction_time_h: 24
outcome_from_h: 0
waivers:
  - {rule: V4, protocol_items: [P5, P9], justification: "%s"}
sensitivity:
  - {name: "24 h landmark", flag: incl_alive_24h}
"""
V3_SPEC = """prediction_time_h: 0
outcome_from_h: 0
exposure: {column: %s, window_start_h: 0, window_end_h: %d}
waivers:
  - {rule: V3, protocol_items: [P4, P8], justification: "%s"}
sensitivity:
  - {name: "landmark at end of exposure window", flag: incl_alive_landmark}
model: {type: logistic, primary_term: %s}
"""
F_V4 = [{"severity": "block", "rule": "V4", "message": "12.8% of outcome events occur between 0 h and 24 h (look-ahead)"}]
F_V3 = [{"severity": "block", "rule": "V3", "message": "Immortal-time risk: exposure assessed until the window end while "
         "outcome counted from 0 h; 30% of events occur inside the exposure window"}]

CASES = [
    # (name, protocol, spec, findings, expected_grant)
    ("t2 SOFA, protocol counts all deaths from admission", T["t2_sofa"],
     V4_SPEC % "time zero = ICU admission and outcome = in-hospital death per protocol", F_V4, True),
    ("t3 sodium, protocol counts all deaths from admission", T["t3_dysnatremia"],
     V4_SPEC % "time zero = ICU admission, first sodium within 24 h, outcome = in-hospital death", F_V4, True),
    ("t1 AKI, protocol fixes a 0-7 day exposure window", T["t1_aki"],
     V3_SPEC % ("aki_stage", 168, "protocol defines exposure as max KDIGO stage between ICU admission and day 7", "aki_stage=3"), F_V3, True),
    ("same SOFA design but protocol predicts at 24 h", PREDICT24,
     V4_SPEC % "time zero = ICU admission per protocol", F_V4, False),
    ("protocol silent on time zero and outcome window", SILENT,
     V4_SPEC % "the protocol counts all deaths from admission", F_V4, False),
    ("t6 transfusion: protocol requires time-dependent handling", T["t6_transfusion"],
     V3_SPEC % ("rbc_any", 240, "protocol defines exposure as any RBC transfusion during the ICU stay", "rbc_any"), F_V3, False),
    ("planner misquotes the protocol (claims a 7-day window for t3)", T["t3_dysnatremia"],
     V3_SPEC % ("na_cat", 168, "the protocol says the exposure is the worst sodium over the first 7 ICU days", "na_cat"), F_V3, False),
]
CARD = {"n_final": 40000, "outcome": {"events": 6000, "rate": 0.15}, "sensitivity": [{"name": "landmark", "n": 39000}],
        "model": {"blinded": True}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    results = []
    for name, prot, spec, finds, expect in CASES:
        rule = finds[0]["rule"]
        for k in range(a.reps):
            rev = llm_critic.review(prot, None, spec, CARD, finds, [])
            d = next((x for x in rev.get("waiver_decisions") or [] if str(x.get("rule", "")).startswith(rule)), {})
            judge = bool(d.get("granted"))
            quote_ok = bool(d.get("quote")) and quote_supported(d["quote"], prot)
            applied = judge and quote_ok
            results.append({"case": name, "rep": k + 1, "expected_grant": expect, "judge_granted": judge, "quote": d.get("quote", ""),
                            "quote_verbatim": quote_ok, "applied": applied, "reason": str(d.get("reason", rev.get("error", "")))[:200]})
            print(f"{name:<62} rep {k+1}: expected {'GRANT' if expect else 'DENY '} | judge {'GRANT' if judge else 'DENY '}"
                  f" | quote ok {quote_ok} | applied {applied}", flush=True)
    pos = [r for r in results if r["expected_grant"]]
    neg = [r for r in results if not r["expected_grant"]]
    summary = {"n": len(results),
               "false_waiver_rate_judge": sum(r["judge_granted"] for r in neg) / max(len(neg), 1),
               "false_waiver_rate_applied": sum(r["applied"] for r in neg) / max(len(neg), 1),
               "false_denial_rate_applied": sum(not r["applied"] for r in pos) / max(len(pos), 1)}
    print(json.dumps(summary, indent=1))
    import os
    backend = os.environ.get("UCD_CRITIC_BACKEND", "claude_cli")
    name = "judge_eval.json" if backend == "claude_cli" else f"judge_eval_{backend}_{os.environ.get('UCD_LLM_MODEL', 'local').replace(':', '-')}.json"
    (ROOT / "eval" / name).write_text(json.dumps({"summary": summary, "results": results},
                                                                           indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
