"""Ask the user a question in the terminal: numbered choices (at most MAX_OPTIONS) or A.

  1..N   pick a choice
  Enter  the recommended choice (when there is one; otherwise the question is asked again). Never automation.
  A      let the agent decide every remaining question of THIS step (intake, protocol check, design loop).
         The next step asks the user again. Only when typed.

A question without choices (e.g. "What would you like to study?") takes a typed answer.

Modes: "interactive" (the default, also when stdin is not a terminal) or "auto" (only with --auto, or after the
input has ended, e.g. stdin is /dev/null: never prompts; the agent's recommended choice, or "decide yourself"). Every question and answer is appended to
<run_dir>/questions.jsonl.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

AUTO = "AUTO"
MAX_OPTIONS = 4


def clean_options(options: Any, recommended: Any) -> tuple[list[str], int | None]:
    """At most MAX_OPTIONS non-empty choices; `recommended` (LLM-supplied: may be "0", the option text or nonsense)
    becomes a valid index or None. A recommended choice beyond the limit is kept in place of the last one."""
    options = [str(o).strip() for o in (options if isinstance(options, list) else []) if str(o).strip()]
    try:
        rec = options.index(recommended) if isinstance(recommended, str) and recommended in options else int(recommended)
    except (TypeError, ValueError):
        rec = None
    if rec is not None and not 0 <= rec < len(options):
        rec = None
    if len(options) > MAX_OPTIONS:
        if rec is not None and rec >= MAX_OPTIONS:
            options[MAX_OPTIONS - 1], rec = options[rec], MAX_OPTIONS - 1
        options = options[:MAX_OPTIONS]
    return options, rec


class Asker:
    def __init__(self, mode: str = "interactive", log: Path | None = None,
                 read: Callable[[str], str] = input, write: Callable[[str], None] = print):
        self.mode, self.log, self.read, self.write = mode, log, read, write
        self.eof = False                      # input ended (no keyboard): later questions are answered in auto
        self.auto_step: str | None = None     # the step in which the user pressed A

    def ask(self, question: str, options: list[str] | None = None, recommended: int | None = None,
            why: str = "", source: str = "") -> dict[str, Any]:
        """Returns {"answer": str, "how": "choice"|"text"|"auto"}. In auto, answer is the recommended option if
        one is given, else AUTO (the caller tells the agent to decide and record its assumption)."""
        options, recommended = clean_options(options, recommended)
        if self.auto_step is not None and source != self.auto_step:
            self.write(f"\n  (A applied to the {self.auto_step} step only; this is a new step, so you decide again)")
            self.auto_step = None
        if self.mode == "auto" or self.auto_step is not None:
            res = self._auto(options, recommended)
            self.write(f"  [auto] {question} -> " + (res["answer"] if res["answer"] != AUTO else "agent decides"))
        else:
            res = self._prompt(question, options, recommended, why, source)
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source, "question": question,
                                     "why": why, "options": options, "recommended": recommended, **res},
                                    ensure_ascii=False) + "\n")
        return res

    @staticmethod
    def _auto(options: list[str], recommended: int | None) -> dict[str, Any]:
        if recommended is not None:
            return {"answer": options[recommended], "how": "auto"}
        return {"answer": AUTO, "how": "auto"}

    def _prompt(self, question: str, options: list[str], recommended: int | None, why: str,
                source: str) -> dict[str, Any]:
        w = self.write
        step = source or "current"
        w("")
        w("? " + question)
        if why:
            w("  (" + why + ")")
        for i, o in enumerate(options, 1):
            w(f"  {i}) {o}" + ("   <- recommended" if recommended == i - 1 else ""))
        w(f"  A) let the agent decide the rest of the {step} step" if options else
          f"  (type your answer, or A to let the agent decide the rest of the {step} step)")
        if options and recommended is not None:
            w(f"  Enter = {recommended + 1} (recommended)")
        while True:
            try:
                r = self.read("> ").strip()
            except (EOFError, OSError):                  # input ended (stdin closed or /dev/null): nobody can answer
                self.mode, self.eof = "auto", True
                w("  (no keyboard input: switching to auto mode)")
                return self._auto(options, recommended)
            if r == "A":
                self.auto_step = source
                w(f"  the agent decides the remaining questions of the {step} step")
                return self._auto(options, recommended)
            if options and not r and recommended is not None:
                return {"answer": options[recommended], "how": "choice"}      # Enter: the user accepted it
            if options and r.isdigit() and 1 <= int(r) <= len(options):
                return {"answer": options[int(r) - 1], "how": "choice"}
            if not options and r:
                return {"answer": r, "how": "text"}
            w("  please enter " + (f"a number 1-{len(options)} or A" if options else "your answer or A"))
