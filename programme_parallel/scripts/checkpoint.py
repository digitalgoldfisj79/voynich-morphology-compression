#!/usr/bin/env python3
"""Atomic recoverable checkpoint writer for programme state."""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, tempfile
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--stage", required=True)
    p.add_argument("--job-id")
    args = p.parse_args()
    root = Path(args.root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "programme_state.pkl":
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    state = {"schema": "vms-script-origin-programme-state-v1", "updated_utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage, "job_id": args.job_id, "files": files}
    target = root / "programme_state.pkl"
    fd, tmp = tempfile.mkstemp(prefix="programme_state.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(state, f, protocol=5)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(json.dumps({"checkpoint": str(target), "sha256": sha256(target), "files": len(files)}, sort_keys=True))

if __name__ == "__main__":
    main()

