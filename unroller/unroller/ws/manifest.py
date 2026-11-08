"""Run manifest: one ``manifest.json`` per workdir (PLAN section 4.4), atomic writes, parameter hashes, resume.

Every (stage, slab) entry records the parameter hash, the input hashes, the outputs, an optional
``table_hash``, the git commit, wall and CPU seconds, peak RSS and any extra numbers (e.g. bytes fetched).
A stage is *done* (``Manifest.is_done``) only if its entry says so with the same parameter hash, the same
input hashes and all its output paths still present; ``run.py`` (WP13) skips such stages on resume.

Several processes may share one workdir: every read-modify-write of ``manifest.json`` holds an exclusive
``fcntl`` lock on ``manifest.json.lock`` and replaces the file atomically.
"""
from __future__ import annotations

import contextlib
import enum
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import time

import numpy as np


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, enum.Enum):
        return o.value
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def canonical_json(obj) -> str:
    """Sorted keys, no whitespace, numpy scalars/arrays as plain JSON: equal parameters give equal text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default, allow_nan=True)


def param_hash(params) -> str:
    """First 16 hex digits of sha256(canonical_json(params))."""
    return hashlib.sha256(canonical_json(params).encode()).hexdigest()[:16]


def file_hash(path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit(repo: str | None = None) -> str:
    """HEAD commit of the repository holding this package (``-dirty`` if tracked files changed), or
    'unknown'."""
    repo = repo or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        c = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
        if not c:
            return "unknown"
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def atomic_write_bytes(path, data: bytes) -> str:
    path = str(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def atomic_write_json(path, obj, indent: int | None = 1) -> str:
    return atomic_write_bytes(path, json.dumps(obj, indent=indent, sort_keys=True, default=_default).encode())


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (ru_maxrss is KiB on Linux)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class Manifest:
    def __init__(self, workdir):
        self.workdir = os.path.abspath(str(workdir))
        os.makedirs(self.workdir, exist_ok=True)
        self.path = os.path.join(self.workdir, "manifest.json")

    @staticmethod
    def key(stage: str, slab=None) -> str:
        return stage if slab is None else f"{stage}/{int(slab):03d}"

    @contextlib.contextmanager
    def _locked(self):
        with open(self.path + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def _read(self) -> dict:
        if not os.path.exists(self.path):
            return {"stages": {}}
        with open(self.path) as f:
            return json.load(f)

    def load(self) -> dict:
        with self._locked():
            return self._read()

    def entry(self, stage: str, slab=None) -> dict | None:
        return self.load()["stages"].get(self.key(stage, slab))

    def _outputs_exist(self, e) -> bool:
        return all(os.path.exists(p if os.path.isabs(p) else os.path.join(self.workdir, p))
                   for p in e.get("outputs", []))

    def is_done(self, stage: str, slab=None, params=None, inputs: dict | None = None) -> bool:
        e = self.entry(stage, slab)
        return bool(e and e.get("status") == "done" and e.get("param_hash") == param_hash(params or {})
                    and e.get("inputs", {}) == dict(inputs or {}) and self._outputs_exist(e))

    def record(self, stage: str, slab=None, params=None, inputs: dict | None = None, outputs=(), status="done",
               table_hash: str | None = None, **stats) -> dict:
        e = {"stage": stage, "slab": slab, "status": status, "param_hash": param_hash(params or {}),
             "params": json.loads(canonical_json(params or {})), "inputs": dict(inputs or {}),
             "outputs": [str(p) for p in outputs], "table_hash": table_hash, "git_commit": git_commit(),
             "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        e.update(json.loads(json.dumps(stats, default=_default)))
        with self._locked():
            m = self._read()
            m["stages"][self.key(stage, slab)] = e
            atomic_write_json(self.path, m)
        return e

    @contextlib.contextmanager
    def stage(self, stage: str, slab=None, params=None, inputs: dict | None = None):
        """``with man.stage("s1", slab=3, params=p, inputs=h) as rec: ...; rec["outputs"] = [...]``.
        Records wall / CPU seconds and peak RSS; status 'done' on success, 'failed' (with the error) if the
        block raises (the exception propagates).  ``rec`` may carry outputs, table_hash and extra numbers."""
        rec = {"outputs": [], "table_hash": None}
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            yield rec
        except BaseException as ex:
            self.record(stage, slab, params, inputs, rec.pop("outputs", []), status="failed",
                        table_hash=rec.pop("table_hash", None), error=repr(ex)[:500],
                        wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0,
                        peak_rss_bytes=peak_rss_bytes(), **rec)
            raise
        self.record(stage, slab, params, inputs, rec.pop("outputs", []), status="done",
                    table_hash=rec.pop("table_hash", None), wall_s=time.perf_counter() - t0,
                    cpu_s=time.process_time() - c0, peak_rss_bytes=peak_rss_bytes(), **rec)
