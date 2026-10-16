"""Split a WITH query into its named steps (CTEs), and put it back together.

Used for incremental design iterations (0.7.5): the gate caches each step's result, so a run after a small edit
recomputes only the steps that change. The parser respects string literals, quoted identifiers and comments, so a
parenthesis or a comma inside them never splits a step.
"""
from __future__ import annotations

import re

_NAME = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*\s*(?:\([^)]*\))?)\s+AS\s*(?:NOT\s+)?(?:MATERIALIZED\s*)?\(", re.I)


def _skip(sql: str, i: int) -> int:
    """If a literal, quoted identifier or comment starts at i, return the index after it; else i."""
    c = sql[i]
    if c in "'\"":
        j = i + 1
        while j < len(sql):
            if sql[j] == c:
                if j + 1 < len(sql) and sql[j + 1] == c:     # doubled quote inside a literal
                    j += 2
                    continue
                return j + 1
            j += 1
        return len(sql)
    if sql.startswith("--", i):
        j = sql.find("\n", i)
        return len(sql) if j < 0 else j + 1
    if sql.startswith("/*", i):
        j = sql.find("*/", i + 2)
        return len(sql) if j < 0 else j + 2
    return i


def _close(sql: str, i: int) -> int:
    """sql[i] is '(' ; return the index of its matching ')'."""
    depth = 0
    while i < len(sql):
        j = _skip(sql, i)
        if j != i:
            i = j
            continue
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced parentheses")


def _steps_from(s: str, i: int) -> tuple[list[tuple[str, str]], str] | None:
    """`name AS (body)[, name AS (body)]* rest` starting at i -> (steps, rest); None if no step starts there."""
    steps = []
    try:
        while True:
            nm = _NAME.match(s, i)
            if not nm:
                return (steps, s[i:].strip()) if steps else None
            open_at = nm.end() - 1
            close_at = _close(s, open_at)
            steps.append((" ".join(nm.group(1).split()), s[open_at + 1:close_at].strip()))   # name, or name(cols)
            i = close_at + 1
            rest = re.match(r"\s*,", s[i:])
            if not rest:
                return steps, s[i:].strip()
            i += rest.end()
    except ValueError:
        return None


def split(sql: str) -> tuple[list[tuple[str, str]], str] | None:
    """(steps [(name, body)], final SELECT) for a top-level WITH query; None when there is no WITH, it is
    WITH RECURSIVE, or the text cannot be split safely."""
    s = sql.strip()
    m = re.match(r"(?is)\s*(?:--[^\n]*\n\s*|/\*.*?\*/\s*)*WITH\b", s)
    if not m or re.match(r"(?is)\s*WITH\s+RECURSIVE\b", s[m.end() - 4:]):
        return None
    got = _steps_from(s, m.end())
    return got if got and got[1] else None


def bare(header: str) -> str:
    """A step's name without its column list: 'b(x, y)' -> 'b'."""
    return header.split("(")[0].strip()


def join(steps: list[tuple[str, str]], final: str) -> str:
    """The WITH query for the given steps and final SELECT."""
    if not steps:
        return final
    return "WITH " + ",\n".join(f"{n} AS (\n{b}\n)" for n, b in steps) + "\n" + final


def references(body: str, names: list[str]) -> list[str]:
    """The step names (from `names`) that `body` refers to, outside literals and comments."""
    code, i = [], 0
    while i < len(body):
        j = _skip(body, i)
        if j != i:
            code.append(" ")
            i = j
            continue
        code.append(body[i])
        i += 1
    text = "".join(code)
    return [n for n in names if re.search(rf"\b{re.escape(n)}\b", text, re.I)]
