"""Minimal pairs for the rejection test: each answerable task gets invalid twins that differ from it in one detail.

  python -m eval.minimal_pairs samples/tasks/answerable/T003 [more tasks] [--out samples/tasks/pairs]

A pair is (the answerable task, its twin). Running both separates "noticed the problem" from "refuses everything":
a model that rejects a twin but answers its original has detected the change. Each twin is written to
<out>/<kind>/<task>/ with protocol.txt and a sample.yaml that lists the accepted verdicts. The first four kinds
mirror the hand-labelled invalid tasks (T001, T005, T007, T018); the accepted verdicts are taken from them.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ADULT = "is 18 years or older"
# kind -> (how the population line changes, accepted verdicts, what makes it invalid)
KINDS = {
    "pediatric": (("replace", ADULT, "is under 18 years"), ["empty_result", "unanswerable"],
                  "the dataset holds adults only"),
    "capped_age": (("replace", ADULT, "is 95 years or older"), ["partial", "unanswerable"],
                   "ages above the dataset's cap are not recorded exactly"),
    "impossible_prior_stay": (("append", "Then keep only the patients who had an ICU discharge within the 30 days "
                                         "immediately before that patient-level first ICU admission."),
                              ["contradictory", "empty_result"],
                              "a discharge before the patient's first ICU admission cannot exist"),
    "long_mortality": (("append", "Then keep only those who died within 3 years from the ICU discharge time of their "
                                  "patient-level first ICU stay. Deaths include deaths in the ICU."),
                       ["partial", "unanswerable"],
                       "deaths are followed for one year after discharge only"),
    "unrecorded_variable": (("append", "Then keep only the patients with a documented smoking history of more than "
                                       "20 pack-years."),
                            ["unanswerable", "partial"],
                            "smoking pack-years are not recorded in the structured tables"),
}


def twin(protocol: str, kind: str) -> str | None:
    """The protocol with one detail changed, or None when the task has no place for this change."""
    (op, *args), _, _ = KINDS[kind]
    lines = protocol.splitlines()
    i = next((k for k, l in enumerate(lines) if l.startswith("Population:")), None)
    if i is None:
        return None
    if op == "replace":
        if args[0] not in lines[i]:
            return None
        lines[i] = lines[i].replace(args[0], args[1])
    else:
        lines[i] = lines[i].rstrip() + " " + args[0]
    return "\n".join(lines) + "\n"


def make(tasks: list[Path], out: Path) -> list[Path]:
    made = []
    for t in tasks:
        text = (t / "protocol.txt").read_text(encoding="utf-8")
        conf = yaml.safe_load((t / "sample.yaml").read_text(encoding="utf-8")) if (t / "sample.yaml").exists() else {}
        if "normal" not in ((conf or {}).get("expected_status") or []):
            continue                                    # twins are made from answerable tasks only
        for kind, (_, accepted, why) in KINDS.items():
            new = twin(text, kind)
            if new is None:
                continue
            d = out / kind / t.name
            d.mkdir(parents=True, exist_ok=True)
            (d / "protocol.txt").write_text(new, encoding="utf-8", newline="\n")
            (d / "sample.yaml").write_text(yaml.safe_dump(
                {"dataset": (conf or {}).get("dataset", "mimic"), "expected_status": accepted,
                 "pair_of": t.name, "change": kind, "why_invalid": why}, sort_keys=False), encoding="utf-8",
                newline="\n")
            made.append(d)
    return made


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval.minimal_pairs", description=__doc__.split("\n\n")[0])
    p.add_argument("task", nargs="+")
    p.add_argument("--out", default="samples/tasks/pairs")
    a = p.parse_args(argv)
    made = make([Path(x) for x in a.task], Path(a.out))
    print(f"{len(made)} invalid twins in {a.out}: " + ", ".join(sorted({m.parent.name for m in made})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
