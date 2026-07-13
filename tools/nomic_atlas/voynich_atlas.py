#!/usr/bin/env python3
"""Prepare or publish Voynich DINOv3 embeddings as a Nomic Atlas map."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

PREFERRED = ("word_embeddings", "proposal_embeddings", "embeddings", "features", "vectors")
META_NAMES = {
    "word": ("word_metadata.jsonl", "words.jsonl", "word_index.jsonl", "mapped_words.jsonl"),
    "proposal": ("proposal_metadata.jsonl", "proposals.jsonl", "proposal_index.jsonl"),
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def field(name: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "field"
    return ("source" + value if value.startswith("_") else value)[:120]


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return str(value)


def read_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                if line.strip():
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        fail(f"{path}:{number} is not a JSON object")
                    rows.append(item)
        return rows
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("records", "data", "items", "words", "proposals"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
            fail(f"Unsupported JSON structure in {path}")
        return data
    fail(f"Unsupported metadata format: {path.suffix}")


def locate(run_dir: Path, level: str) -> tuple[Path, Path | None]:
    name = f"{level}_embeddings.npz"
    candidates = [run_dir / "outputs" / name, run_dir / name, *run_dir.rglob(name)]
    found = []
    for path in candidates:
        if path.exists() and path not in found:
            found.append(path)
    if len(found) != 1:
        fail(f"Expected exactly one {name} under {run_dir}; found {found}")
    for root in (run_dir / "outputs", run_dir):
        for meta_name in META_NAMES[level]:
            path = root / meta_name
            if path.exists():
                return found[0], path
    return found[0], None


def load(npz_path: Path, metadata_path: Path | None, embedding_key: str | None):
    with np.load(npz_path, allow_pickle=True) as archive:
        inventory = {k: {"shape": list(archive[k].shape), "dtype": str(archive[k].dtype)} for k in archive.files}
        key = embedding_key
        if key is None:
            lookup = {k.lower(): k for k in archive.files}
            key = next((lookup[k] for k in PREFERRED if k in lookup), None)
        if key is None:
            matrices = [k for k in archive.files if archive[k].ndim == 2 and np.issubdtype(archive[k].dtype, np.number)]
            if len(matrices) != 1:
                fail(f"Cannot select embedding array; pass --embedding-key. Candidates: {matrices}")
            key = matrices[0]
        if key not in archive.files:
            fail(f"Embedding key {key!r} is absent")
        embeddings = np.asarray(archive[key])
        if embeddings.ndim != 2 or not np.issubdtype(embeddings.dtype, np.number):
            fail(f"{key} is not a 2-D numeric array")
        if not np.all(np.isfinite(embeddings)):
            fail("Embedding matrix contains non-finite values")
        rows = [dict() for _ in range(len(embeddings))]
        for name in archive.files:
            arr = np.asarray(archive[name])
            if name != key and arr.ndim == 1 and len(arr) == len(rows):
                for i, value in enumerate(arr):
                    rows[i][field(name)] = scalar(value)
    if metadata_path:
        external = read_metadata(metadata_path)
        if len(external) != len(rows):
            fail(f"Metadata rows {len(external):,} != embedding rows {len(rows):,}")
        for row, extra in zip(rows, external, strict=True):
            for name, value in extra.items():
                clean = field(name)
                if clean in row and row[clean] != scalar(value):
                    clean = "metadata_" + clean
                row[clean] = scalar(value)
    return embeddings, rows, key, inventory


def prepare(args: argparse.Namespace) -> None:
    if args.run_dir:
        embeddings_path, discovered_meta = locate(args.run_dir, args.level)
        metadata_path = args.metadata or discovered_meta
    else:
        embeddings_path, metadata_path = args.embeddings, args.metadata
    embeddings, rows, key, inventory = load(embeddings_path, metadata_path, args.embedding_key)
    n = len(rows)
    max_rows = args.max_rows if args.max_rows is not None else (50_000 if args.level == "proposal" else n)
    if max_rows < n:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(n, size=max_rows, replace=False))
    else:
        indices = np.arange(n)
    vectors = np.asarray(embeddings[indices], dtype=np.float32)
    zero_vectors = 0
    if args.normalize == "l2":
        norms = np.linalg.norm(vectors, axis=1)
        zero_vectors = int(np.count_nonzero(norms == 0))
        norms[norms == 0] = 1
        vectors /= norms[:, None]
    prepared_rows = []
    for source_row in indices.tolist():
        row = dict(rows[source_row])
        payload = f"{args.run_id}|{args.level}|{source_row}".encode()
        row.update({
            "atlas_id": f"{args.level}-{hashlib.sha256(payload).hexdigest()[:24]}",
            "source_row": source_row,
            "run_id": args.run_id,
            "atlas_level": args.level,
        })
        token = next((row.get(k) for k in ("eva", "token", "transcription", "word", "label") if row.get(k)), None)
        place = next((row.get(k) for k in ("folio", "page_id", "panel_id", "page") if row.get(k)), None)
        row["display_text"] = " | ".join(str(x) for x in (token, place) if x is not None) or f"{args.level} row {source_row}"
        prepared_rows.append(row)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    meta_out, emb_out, idx_out = out / "atlas_metadata.jsonl", out / "atlas_embeddings.npy", out / "source_indices.npy"
    with meta_out.open("w", encoding="utf-8") as fh:
        for row in prepared_rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    np.save(emb_out, vectors, allow_pickle=False)
    np.save(idx_out, indices, allow_pickle=False)
    manifest = {
        "schema_version": "voynich-nomic-atlas-manifest/1",
        "run_id": args.run_id,
        "level": args.level,
        "source": {
            "embeddings_path": str(embeddings_path.resolve()),
            "embeddings_sha256": sha256(embeddings_path),
            "metadata_path": str(metadata_path.resolve()) if metadata_path else None,
            "metadata_sha256": sha256(metadata_path) if metadata_path else None,
            "embedding_key": key,
            "arrays": inventory,
            "rows": n,
            "dimensions": int(embeddings.shape[1]),
        },
        "prepared": {
            "rows": len(prepared_rows), "dimensions": int(vectors.shape[1]),
            "normalization": args.normalize, "zero_vectors": zero_vectors,
            "seed": args.seed, "metadata_file": meta_out.name,
            "embeddings_file": emb_out.name, "source_indices_file": idx_out.name,
        },
        "integrity": {
            "metadata_sha256": sha256(meta_out), "embeddings_sha256": sha256(emb_out),
            "source_indices_sha256": sha256(idx_out),
        },
    }
    (out / "atlas_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def bundle(path: Path):
    manifest = json.loads((path / "atlas_manifest.json").read_text(encoding="utf-8"))
    meta_path = path / manifest["prepared"]["metadata_file"]
    emb_path = path / manifest["prepared"]["embeddings_file"]
    if sha256(meta_path) != manifest["integrity"]["metadata_sha256"] or sha256(emb_path) != manifest["integrity"]["embeddings_sha256"]:
        fail("Prepared bundle SHA-256 mismatch")
    rows = read_metadata(meta_path)
    vectors = np.load(emb_path, allow_pickle=False)
    if len(rows) != len(vectors) or len({r.get("atlas_id") for r in rows}) != len(rows):
        fail("Prepared metadata/embedding alignment or atlas_id uniqueness failure")
    return manifest, rows, vectors


def publish(args: argparse.Namespace) -> None:
    token = os.getenv("NOMIC_API_KEY")
    if not token:
        fail("NOMIC_API_KEY is not set")
    manifest, rows, vectors = bundle(args.bundle_dir)
    from nomic import AtlasDataset, login
    login(token)
    dataset = AtlasDataset(args.dataset, description=args.description, unique_id_field="atlas_id", is_public=args.public)
    if dataset.total_datums and not args.allow_nonempty:
        fail(f"{dataset.identifier} already contains {dataset.total_datums:,} rows; refusing to append")
    dataset.add_data(data=rows, embeddings=vectors)
    projection = dataset.create_index(
        name=args.index_name, modality="embedding", topic_model=False, duplicate_detection=False
    )
    result = {
        "dataset": dataset.identifier, "rows_uploaded": len(rows),
        "dataset_link": getattr(projection, "dataset_link", None),
        "map_link": getattr(projection, "map_link", None), "source_manifest": manifest,
    }
    (args.bundle_dir / "atlas_publish_result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--embeddings", type=Path)
    prep.add_argument("--metadata", type=Path)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--level", choices=("word", "proposal"), default="word")
    prep.add_argument("--embedding-key")
    prep.add_argument("--normalize", choices=("l2", "none"), default="l2")
    prep.add_argument("--max-rows", type=int)
    prep.add_argument("--seed", type=int, default=20260713)
    pub = commands.add_parser("publish")
    pub.add_argument("--bundle-dir", type=Path, required=True)
    pub.add_argument("--dataset", required=True)
    pub.add_argument("--index-name", default="voynich-dinov3-word-map")
    pub.add_argument("--description", default="Voynich DINOv3 embedding map with provenance metadata")
    pub.add_argument("--public", action="store_true")
    pub.add_argument("--allow-nonempty", action="store_true")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else publish(args)


if __name__ == "__main__":
    main()
