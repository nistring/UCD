"""UCD: design an observational cohort study from a research question (dataset: ucd/datasets.yaml).

  python -m ucd                                            asks what you want to study
  python -m ucd "Is low sodium at ICU admission linked to death?"     a short question
  python -m ucd my_proposal.txt                            a proposal file: .txt only (free text, any length)
  python -m ucd samples/<name>                             a sample folder: question + ground truth -> report.md
  python -m ucd --topic t3_dysnatremia                     a stored protocol (ucd/topics.yaml)

  options:  --auto      never ask me; the agent takes its recommended choices
            --gt FILE   compare the final cohort with my ground-truth cohort (.csv)
            --blind     hide outcome associations until the design is frozen
            --dataset D another registered dataset (default: the first in ucd/datasets.yaml)

Steps: 1 the request is checked against the protocol format (PICO + time zero + analysis) and you are asked about
gaps; 2 you check the protocol and can edit the file (final step of intake); 3 the agent designs, runs and revises the cohort (at most 3 iterations; it may ask
you); 4 final.md is written to the run folder.

Other tools:  python -m ucd.llm check            (LLM and per-agent models)
              python -m eval.groundtruth --run-dir R --gt cohort.csv   (evaluate a finished run)
              python -m ucd card|lookup|explore|run|status|freeze|template -h   (low-level agent tools)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

from ucd.ask import AUTO, Asker
from ucd.gate import ROOT

NO_INPUT = ("\nNo keyboard input reached UCD (e.g. the `!` prefix in Claude Code, or a pipe). Run it in your own "
            "terminal, or pass the question and use auto mode:\n"
            '  python -m ucd "Is low sodium at ICU admission linked to death?" --auto')
STOP = {"the", "a", "an", "of", "in", "is", "and", "with", "does", "do", "for", "to", "among", "are", "what", "how",
        "objective", "protocol", "study"}


def _slug(text: str) -> str:
    return "_".join(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP)[:40] or "request"


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):              # Korean Windows editors often save CP949
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def read_proposal(path: Path) -> str:
    """A proposal file: .txt only (user, 2026-10-17), free text of any length (a question or a full protocol)."""
    if path.suffix.lower() != ".txt":
        raise ValueError(f"{path.name}: a proposal must be a .txt file (free text)")
    return _read_text(path)


def _read_request(parts: list[str], file: str | None) -> tuple[str, str]:
    """(text, source). A single argument that names an existing file is read as a proposal."""
    path = Path(file) if file else (Path(parts[0]) if len(parts) == 1 and len(parts[0]) < 260
                                    and Path(parts[0]).is_file() else None)
    if path:
        return read_proposal(path), path.name
    return " ".join(parts).strip(), "command line"


def main(argv=None) -> int:
    """Entry point; unexpected failures also go to the server-side diagnostics log (ucd.diag)."""
    from ucd import diag
    try:
        return _main(argv)
    except KeyboardInterrupt:
        diag.event("interrupted", "chat")
        raise
    except SystemExit as e:
        if e.code not in (0, None, 2):                 # 2 = a usage error, already logged with its message
            diag.error("chat.exit", str(e.code))
        raise
    except Exception as e:
        diag.error("chat", e)
        raise


class _Parser(argparse.ArgumentParser):
    """argparse that also logs usage errors with their message (E-46: the log said only "2")."""
    def error(self, message):
        from ucd import diag
        diag.event("usage", "chat", message=message)
        super().error(message)


def _main(argv=None) -> int:
    p = _Parser(prog="python -m ucd", description=__doc__.split("\n\n")[0],
                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("request", nargs="*", help="a research question, or the path of a protocol file")
    p.add_argument("--file", help="protocol file (same as giving its path as the request)")
    p.add_argument("--topic", help="a stored protocol from ucd/topics.yaml (no intake questions)")
    p.add_argument("--auto", action="store_true", help="never ask; the agent takes its recommended choices")
    p.add_argument("--gt", help="ground-truth cohort (.csv: a unit id column, optional 0/1 outcome) to compare with")
    p.add_argument("--blind", action="store_true", help="outcome-blinded design phase (freeze, then unblind once)")
    p.add_argument("--dataset", help="a dataset from ucd/datasets.yaml (default: the first one)")
    p.add_argument("--run-dir", help="output folder (default runs/ucd/<question>/<date_time>)")
    p.add_argument("--max-turns", type=int, default=None, help="agent steps in the design loop (default 30)")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ.setdefault("UCD_CRITIC_BACKEND", "openai_compat")   # the critic uses the configured gateway model
    if len(a.request) == 1 and Path(a.request[0]).is_dir():          # a sample folder
        from ucd.sample import run_sample
        if a.gt or a.run_dir or a.topic or a.file:                  # never drop an option silently (E-31)
            p.error("with a sample folder, --gt, --run-dir, --topic and --file are not used: the folder has its own "
                    "gt.csv and writes to <folder>/result")
        extra = (["--blind"] if a.blind else []) + (["--max-turns", str(a.max_turns)] if a.max_turns else [])
        run_sample(Path(a.request[0]), True if a.auto else None, extra, dataset=a.dataset)
        return 0
    from ucd import datasets, loop
    from ucd.intake import IntakeStopped, confirm, intake, is_detailed
    from ucd.llm import LLMError, LocalLLM

    topics = yaml.safe_load((ROOT / "ucd" / "topics.yaml").read_text(encoding="utf-8"))
    if a.topic and a.topic not in topics:
        p.error(f"unknown topic {a.topic!r}; available: {', '.join(topics)}")
    if a.gt and (not Path(a.gt).is_file() or Path(a.gt).suffix.lower() != ".csv"):   # fail now, not after the loop
        p.error(f"the ground truth must be an existing .csv file: {a.gt}")
    try:
        dataset = a.dataset or datasets.default()
        label = datasets.get(dataset)["label"]
    except ValueError as e:
        p.error(str(e))
    _ensure_data(dataset, a.auto)                                    # before any LLM call
    try:
        request, source = _read_request(a.request, a.file)
    except (OSError, ValueError) as e:
        p.error(f"cannot read the proposal: {e}")
    asker = Asker("auto" if a.auto else "interactive")
    print(f"UCD - cohort design agent. Dataset: {label}.")
    if asker.mode == "interactive":
        print("When asked: type a number, or A to let the agent decide the rest of that step (the next step asks again).")
    else:                                            # E-71: a sample's sample.yaml can switch it on unnoticed
        print("AUTO MODE (--auto, or `auto: true` in the sample's sample.yaml): the agent answers every question with "
              "its recommended choice, marked [auto]. Remove the flag to be asked.")
    print("During the design loop: type q and press Enter, or press Ctrl+C, to stop early and keep the results so far.")
    if not a.topic and not request:
        if asker.mode == "auto":
            p.error("give a question or a protocol file, e.g.\n"
                    '  python -m ucd "Is low sodium at ICU admission linked to death?" --auto')
        r = asker.ask("What would you like to study? A one-line question is enough.", [], source="request")
        request, source = (r["answer"] if r["answer"] != AUTO else ""), "typed"
        if not request:
            print("No question given; stopping." + (NO_INPUT if asker.eof else ""))
            return 1

    name = a.topic or _slug(request.strip().splitlines()[0] if request.strip() else request)
    run_dir = Path(a.run_dir) if a.run_dir else ROOT / "runs" / "ucd" / name / time.strftime("%Y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    asker.log = run_dir / "questions.jsonl"
    print(f"Run folder: {run_dir}")

    if a.topic:
        protocol = topics[a.topic].strip()
    else:
        detailed = is_detailed(request)
        print(f"\n[1/3] Checking your {'protocol' if detailed else 'question'} ({source}) against the protocol format "
              f"and the {label} schema ...")
        try:
            protocol, _, answers = intake(request, LocalLLM(role="intake"), asker, dataset=dataset)
        except LLMError as e:
            print(f"The LLM gateway failed: {str(e)[:300]}\nCheck it with: python -m ucd.llm check")
            return 2
        except IntakeStopped as e:
            (run_dir / "request.md").write_text(request.strip() + "\n", encoding="utf-8")
            (run_dir / "final.md").write_text(
                f"# Stopped at intake: the question cannot be studied as asked\n\n**Request.** {request.strip()}\n\n"
                f"**Problem.** {e}\n\nNo protocol was agreed and no design was run.\n", encoding="utf-8")
            (run_dir / "verdict.json").write_text(json.dumps(
                {"status": e.kind, "reason": str(e), "limitation": "stopped at intake; no cohort was built",
                 "source": "intake"}, indent=1, ensure_ascii=False), encoding="utf-8")
            from ucd import diag
            diag.event("run", "intake", name=name, run=run_dir.name, designs=0, stopped_at_intake=True, message=str(e))
            print(f"\nStopped: {e}\nSummary: {run_dir / 'final.md'}")
            return 0
        (run_dir / "request.md").write_text(request.strip() + "\n", encoding="utf-8")
        (run_dir / "intake_answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
    ok, protocol = confirm(protocol, asker, run_dir / "protocol.md")
    if not ok:
        print(f"Stopped before the design loop. The protocol is saved in {run_dir / 'protocol.md'}")
        return 0
    (run_dir / "meta.json").write_text(json.dumps({"topic": name, "run": run_dir.name, "question": protocol,
                                                   "mode": asker.mode, "dataset": dataset}, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.blind:
        (run_dir / "run_config.json").write_text('{"blind_outcome": true}', encoding="utf-8")

    print("\n[2/3] Designing the cohort (the agent sees schema and counts only; at most 3 design iterations) ...")
    final = loop.run_loop(name, run_dir, [dataset], a.max_turns or loop.MAX_TURNS, question=protocol,
                          asker=asker, echo=print)
    text = final.read_text(encoding="utf-8") if final.exists() else ""
    print(f"\n[3/3] Result: {final}\n")
    print(text[:4000] + ("\n... (see the file for the rest)" if len(text) > 4000 else ""))
    if a.gt:
        print("\nGround-truth comparison (the agent never sees it):")
        from eval.groundtruth import evaluate_gt
        try:
            _diag_gt(name, run_dir, evaluate_gt(run_dir, a.gt, dataset=dataset))
        except SystemExit as e:
            print(f"  not possible: {e}")
    print(f"\nAll files of this run: {run_dir}")
    return 0


def _ensure_data(dataset: str, auto: bool) -> None:
    """The dataset's parquet must exist. When it is missing but the raw files are there, offer to build it now
    (recommended, so auto mode builds it unattended; E-36) instead of only printing the command."""
    from ucd import datasets
    pq, miss = datasets.missing_tables(dataset)
    if not miss:
        return
    try:
        raw = datasets.source_dir(dataset)
    except SystemExit:
        datasets.require_data(dataset)                 # no raw files either: stop with the full explanation
        return
    print(f"{dataset}: the parquet is missing ({len(miss)} required tables) in {pq}; raw files found in {raw}.")
    r = Asker("auto" if auto else "interactive").ask(
        "Build the parquet now? (once; all cores, about as long as the largest table)", ["Build it now", "Stop"], 0,
        source="setup")
    if r["answer"] != "Build it now":
        datasets.require_data(dataset)
    from ucd.build_parquet import build
    build(dataset)
    datasets.require_data(dataset)


def _diag_gt(name: str, run_dir: Path, res) -> None:
    """Ground-truth agreement (aggregates only) for the diagnostics log."""
    from ucd import diag
    c = (res or {}).get("cohort", res) or {}
    diag.event("gt", "groundtruth", name=name, run=run_dir.name, **{k: c.get(k) for k in
               ("jaccard", "recall", "precision", "n_agent", "n_gt", "compared_at")})


if __name__ == "__main__":
    sys.exit(main())
