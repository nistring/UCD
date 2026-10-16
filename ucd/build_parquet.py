"""One-time conversion of a dataset's raw tables to parquet for the UCD gate, in parallel.

  python -m ucd.build_parquet [dataset] [--jobs N]     (default: the first dataset in ucd/datasets.yaml)

Reads <source>/<module>/<table>.csv or .csv.gz (source from ucd/datasets.yaml or UCD_<NAME>_SOURCE) and writes
.kcd_cache/parquet/<dataset>/<module>/<table>.parquet (or the dataset's parquet_env folder). Types are auto-detected
over the whole file (sample_size=-1) so no row is silently dropped. Finished tables are skipped on re-runs.

Parallel: one process per table, largest first, so the biggest file starts at once and the small ones fill the other
cores. A gzip file is decompressed by one thread, so the build takes about as long as its largest table alone.
--jobs defaults to as many tables as the cores and memory allow (2 GB per job); the cores (UCD_THREADS, default all)
and memory (default 80 % of RAM; UCD_MEMORY sets the limit per job) are split between the jobs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb

from ucd import datasets

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".kcd_cache" / "duckdb_tmp"


def _convert(job: tuple[str, str, int, str]) -> tuple[str, float]:
    """Worker (own process): one CSV -> parquet. Written to .part and renamed, so an interrupted build never leaves
    a table that looks finished."""
    src, dst, threads, memory = job
    t0 = time.time()
    tmp = TMP / str(os.getpid())
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmp.as_posix()}'; SET memory_limit='{memory}'; SET threads={threads}; "
                "SET preserve_insertion_order=false;")
    part = Path(dst).with_suffix(".part")
    con.execute(f"COPY (SELECT * FROM read_csv('{Path(src).as_posix()}', sample_size=-1)) "
                f"TO '{part.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    con.close()
    part.replace(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    return dst, time.time() - t0


def plan(name: str) -> list[tuple[Path, Path]]:
    """(raw file, parquet target) for every table not converted yet, largest first."""
    src, out = datasets.source_dir(name), datasets.parquet_dir(name)
    files = [f for f in src.glob("*/*") if f.name.endswith((".csv", ".csv.gz"))]
    if not files:
        raise SystemExit(f"no <module>/<table>.csv[.gz] files under {src}")
    todo = []
    for f in sorted(files, key=lambda f: f.stat().st_size, reverse=True):
        table = f.name[:-7] if f.name.endswith(".csv.gz") else f.name[:-4]
        dst = out / f.parent.name / f"{table}.parquet"
        if not dst.exists():
            todo.append((f, dst))
    return todo


def build(name: str, jobs: int | None = None) -> None:
    from ucd.gate import cores, ram_bytes
    todo = plan(name)
    print(f"{name} source: {datasets.source_dir(name)}\n{name} parquet: {datasets.parquet_dir(name)}", flush=True)
    if not todo:
        print("every table is already converted")
        return
    total = int(os.environ.get("UCD_THREADS") or cores())
    ram_gb = ram_bytes() / 2**30 or 16
    jobs = max(1, min(jobs or len(todo), len(todo), total, max(1, int(ram_gb * 0.8 // 2)),
                      61 if os.name == "nt" else len(todo)))       # Windows: at most 61 worker processes
    threads = max(1, total // jobs)
    memory = os.environ.get("UCD_MEMORY") or f"{max(1, int(ram_gb * 0.8 / jobs))}GB"
    print(f"{len(todo)} tables, {jobs} in parallel; per job {threads} threads, memory {memory}", flush=True)
    t0 = time.time()
    for _, dst in todo:
        dst.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(jobs) as ex:
        futs = {ex.submit(_convert, (str(f), str(dst), threads, memory)): f for f, dst in todo}
        try:
            for fut in as_completed(futs):
                dst, sec = fut.result()
                print(f"{name} {Path(dst).parent.name}.{Path(dst).stem}: {sec:.0f}s", flush=True)
        except BaseException:                          # a failed table (or Ctrl+C): do not start the queued ones
            ex.shutdown(cancel_futures=True)
            raise
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ucd.build_parquet", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("dataset", nargs="?", default=None)
    p.add_argument("--jobs", type=int, default=None, help="tables converted at the same time (default: automatic)")
    a = p.parse_args(argv)
    build(a.dataset or datasets.default(), a.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
