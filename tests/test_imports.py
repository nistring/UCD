"""Every `from <project module> import <name>` resolves, including imports inside functions (E-63).

Lazy imports run only when their function runs, so test collection never sees them: `ucd.sample.finish_task`
imported `export_cohort` from a groundtruth.py that never had it, and every sample run crashed after its ground-truth
comparison. No data, no network."""
import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ("ucd", "eval")


def _imports():
    for f in sorted([*ROOT.glob("ucd/*.py"), *ROOT.glob("eval/*.py")]):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] in PROJECT:
                for a in node.names:
                    yield f.relative_to(ROOT).as_posix(), node.lineno, node.module, a.name


def test_every_project_import_resolves():
    missing = []
    for where, line, module, name in _imports():
        mod = importlib.import_module(module)
        if name != "*" and not hasattr(mod, name):
            try:
                importlib.import_module(f"{module}.{name}")      # a submodule (from ucd import gate)
            except ImportError:
                missing.append(f"{where}:{line} from {module} import {name}")
    assert not missing, missing
