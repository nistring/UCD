"""The planner's reply -> an action (0.13.0): one JSON object, and file content or SQL in ONE fenced block after it
(never inside the JSON, so nothing needs escaping). The block runs from the first fence to the last one, so a final.md
that holds its own fenced blocks arrives whole. The JSON itself may come in a ```json fence."""
from __future__ import annotations

import json
import re

from ucd.llm import extract_json

OPEN = re.compile(r"```[ \t]*[A-Za-z0-9_-]*[ \t]*\n")
SECOND_ACTION = re.compile(r'^\s*\{\s*"action"\s*:', re.M)
JSON_FENCE = re.compile(r"```[ \t]*json[ \t]*\n(.*?)```", re.S | re.I)
BLOCK_ACTIONS = {"write", "edit", "explore"}       # actions whose content comes in the block after the JSON


def _leading_json(text: str) -> tuple[dict | None, str]:
    """(the JSON object the reply starts with, the text after it), or (None, text) when it does not start with one."""
    i = text.find("{")
    if i < 0 or text[:i].strip():
        return None, text
    try:
        obj, k = json.JSONDecoder().raw_decode(text[i:])
    except ValueError:
        return None, text
    return (obj, text[i + k:]) if isinstance(obj, dict) and "action" in obj else (None, text)


def parse_action(raw: str) -> dict:
    """The JSON action, with the fenced block (if any) as `body`: the FROM query of an explore, the content of a write,
    the replacement lines of an edit. A block that is opened but never closed is an error, never an empty body."""
    raw = (raw or "").replace("\r\n", "\n")
    first = raw.find("```")                        # a bare JSON before the first fence wins over a ```json block
    j = JSON_FENCE.search(raw) if first >= 0 and "{" not in raw[:first] else None
    rest = raw[:j.start()] + raw[j.end():] if j else raw
    m, end = OPEN.search(rest), rest.rfind("```")
    if m and end < m.end():
        raise ValueError("the fenced block after the JSON is not closed (``` missing), so nothing was changed")
    body, head = (rest[m.end():end], rest[:m.start()] + rest[end + 3:]) if m else (None, rest)
    if m is None and not j:                         # E-80: JSON, then the file or SQL without its fence
        lead, tail = _leading_json(head)
        if lead is not None and tail.strip():
            if SECOND_ACTION.search(tail):
                raise ValueError("the reply holds a second action; send one action per reply (the next one in the next "
                                 "turn)")
            if lead.get("action") in BLOCK_ACTIONS:
                body = tail.strip("\n")
            head = json.dumps(lead)                 # text after any other action is commentary: ignored
    act = extract_json(j.group(1) if j else head)
    if not isinstance(act, dict) or "action" not in act:
        raise ValueError("JSON object without an 'action' field")
    if body is not None and SECOND_ACTION.search(body):   # E-71: final.md swallowed the verdict.json action
        raise ValueError("the block holds a second action; send one action per reply (the next one in the next turn)")
    if body is not None:
        act["body"] = body.rstrip("\n")
    return act


def why_unparsable(raw: str, err: Exception, llm) -> str:
    """The planner's reply was not a JSON action: say which of the usual causes it was (E-37), so the next turn can
    fix it instead of repeating it."""
    if getattr(llm, "finish_reason", None) == "length":
        return (f"your reply was cut off at the output limit ({getattr(llm, 'max_tokens', '?')} tokens). Change the "
                "design files in smaller pieces: edit a few lines at a time instead of rewriting a whole file.")
    if not raw.strip():
        return "your reply was empty (only reasoning, or nothing). Reply with exactly one JSON action."
    if "{" not in raw:
        return "your reply contained no JSON object (only text). Reply with exactly one JSON action."
    return (f"{str(err)[:200]}. Reply with exactly one JSON object; put file content or SQL in ONE fenced block after "
            "it, not inside the JSON.")
