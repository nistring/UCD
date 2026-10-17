"""Update an installed UCD to a new version, touching only what changed.

  python -m ucd.update ../ucd_bundle_v<version>_<time>.zip [--dry-run]
  python -m ucd.update path/to/unpacked/ucd_project      [--dry-run]

For every file of the new version (contents compared with line endings ignored):
  same as here                                  -> left alone (not rewritten, timestamp kept)
  new file                                      -> added
  changed in the new version only               -> replaced
  changed by you here, not in the new version   -> your copy is kept
  changed by you AND in the new version         -> your copy is backed up, the new one written (listed as conflict)
Files of the old version that the new one no longer has are removed (moved to the backup), unless you edited them.
Files that were never part of a version (data, gt.csv, runs/, results, config you added, caches, ~/.khdp_pat) are
never touched. Everything replaced or removed is first copied to .ucd_update_backup/<time>/; to roll back, copy it
back. The installed version's file hashes are kept in .ucd_manifest.json (shipped in the bundle, rewritten by every
update); without it, the first update cannot tell your edits from the old version and backs up every file it
replaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ".ucd_manifest.json"
BACKUP = ".ucd_update_backup"


def digest(data: bytes) -> str:
    """Content hash; text files compare equal across CRLF/LF."""
    if b"\0" not in data:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def version_of(files: dict[str, bytes]) -> str:
    m = re.search(rb'__version__\s*=\s*"([^"]+)"', files.get("ucd/__init__.py", b""))
    return m.group(1).decode() if m else "?"


def _safe(rel: str) -> str:
    p = PurePosixPath(rel)
    if not p.parts or p.is_absolute() or ".." in p.parts or ":" in p.parts[0] or "\\" in rel:
        raise SystemExit(f"refusing unsafe path in the new version: {rel!r}")
    return p.as_posix()


def read_source(src: Path) -> dict[str, bytes]:
    """The new version's files {relative path: bytes}. A zip may have one top folder (ucd_project/); a folder must
    be an unpacked bundle (it holds .ucd_manifest.json, and only the files listed there are used)."""
    if src.is_dir():
        mf = src / MANIFEST
        if not mf.is_file():
            raise SystemExit(f"{src} has no {MANIFEST}: give a bundle zip or an unpacked bundle folder")
        listed = json.loads(mf.read_text(encoding="utf-8"))["files"]
        return {_safe(r): (src / r).read_bytes() for r in listed}
    with zipfile.ZipFile(src) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names)
        out = {}
        for n in names:
            rel = n.split("/", 1)[1] if strip else n
            out[_safe(rel)] = z.read(n)
    if "ucd/__init__.py" not in out:
        raise SystemExit(f"{src} does not look like a UCD bundle (no ucd/__init__.py)")
    return out


def plan(root: Path, new: dict[str, bytes]) -> tuple[list[tuple[str, str]], dict, dict]:
    """[(action, path)] with action in same, add, update, keep, conflict, remove, keep-removed."""
    mf = root / MANIFEST
    old = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else {}
    old_h: dict[str, str] = old.get("files", {})
    new_files = {r: b for r, b in new.items() if r != MANIFEST}
    new_h = {r: digest(b) for r, b in new_files.items()}
    acts = []
    for rel, nh in sorted(new_h.items()):
        loc = root / rel
        if not loc.is_file():
            acts.append(("add", rel))
            continue
        lh = digest(loc.read_bytes())
        oh = old_h.get(rel)
        if lh == nh:
            acts.append(("same", rel))
        elif oh is not None and lh != oh and nh == oh:
            acts.append(("keep", rel))                 # only you changed it
        elif oh is not None and lh != oh:
            acts.append(("conflict", rel))             # you and the new version changed it
        else:
            acts.append(("update", rel))               # unedited old version (or no manifest to tell)
    for rel in sorted(set(old_h) - set(new_h)):
        loc = root / rel
        if loc.is_file():
            acts.append(("remove" if digest(loc.read_bytes()) == old_h[rel] else "keep-removed", rel))
    return acts, {"version": old.get("version") or version_of({"ucd/__init__.py": _read(root / "ucd/__init__.py")}),
                  "has_manifest": bool(old_h)}, {"version": version_of(new_files), "files": new_h}


def _read(p: Path) -> bytes:
    return p.read_bytes() if p.is_file() else b""


LABEL = {"add": "added", "update": "replaced", "conflict": "CONFLICT: you and the new version changed it; yours is "
         "in the backup", "keep": "kept yours (only you changed it)", "remove": "removed (no longer part of UCD)",
         "keep-removed": "kept: no longer part of UCD, but you edited it"}


def apply(root: Path, new: dict[str, bytes], dry_run: bool = False, log=print) -> dict[str, int]:
    acts, old, man = plan(root, new)
    counts = {a: sum(1 for x, _ in acts if x == a) for a in ("same", *LABEL)}
    log(f"UCD update {old['version']} -> {man['version']} in {root}"
        + ("" if old["has_manifest"] else f"\n  (no {MANIFEST} here: every replaced file is backed up)"))
    for a, rel in acts:
        if a != "same":
            log(f"  {rel}: {LABEL[a]}")
    log("  " + ", ".join(f"{n} {a}" for a, n in counts.items() if n) + (" (dry run: nothing written)" if dry_run else ""))
    if dry_run:
        return counts
    backup = root / BACKUP / time.strftime("%Y%m%d_%H%M%S")
    failed = []
    for a, rel in acts:
        loc = root / rel
        try:
            if a in ("update", "conflict", "remove"):
                (backup / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(loc, backup / rel)
            if a in ("add", "update", "conflict"):
                loc.parent.mkdir(parents=True, exist_ok=True)
                loc.write_bytes(new[rel])
            elif a == "remove":
                loc.unlink()
        except OSError as e:
            failed.append(f"{rel}: {e.strerror or e}")
    if backup.exists():
        log(f"  backup of replaced and removed files: {backup}")
    if failed:                                         # keep the old manifest: the next run retries these files and
        log(f"  {len(failed)} file(s) could not be written (on Windows: path too long? enable long paths):")
        for f in failed:                               # never mistakes an old copy for your own edit
            log(f"    {f}")
        counts["failed"] = len(failed)
        return counts
    (root / MANIFEST).write_text(json.dumps(man, indent=0, sort_keys=True), encoding="utf-8")
    log("  next: python -m pytest tests -q")
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ucd.update", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("source", help="new bundle zip, or an unpacked bundle folder")
    p.add_argument("--dry-run", action="store_true", help="only list what would change")
    p.add_argument("--root", default=str(ROOT), help="the installation to update (default: this one)")
    a = p.parse_args(argv)
    counts = apply(Path(a.root), read_source(Path(a.source)), a.dry_run)
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
