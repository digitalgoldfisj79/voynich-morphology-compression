#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import gc
import hashlib
import io
import json
import os
import random
import time
import urllib.parse

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor, AutoModel

PROTOCOL = "P586-VMS-ROOT-0.1-20260803"
RUN_ID = "ef6a0302-8269-4166-8c1c-63b14abb9c47"
SEED = 20260803
DINO_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
SUPABASE = "https://ymaqlcfjmdwncdbjprmw.supabase.co"
REST = SUPABASE + "/rest/v1"
STORAGE = SUPABASE + "/storage/v1/object/public/manuscripts/"
BRIDGE = SUPABASE + "/storage/v1/object/public/bridge/"
CHECKPOINT_URL = BRIDGE + "p586_root_v01/full_run/checkpoint_recovered.json"
UPLOAD_EP = SUPABASE + "/functions/v1/p586-root-upload-v01"
ANON = "sb_publishable_BOm91KbAPOZDCQ7H3yLFzw_VtNPk2ap"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "VoynichRootResearch/0.1-comparison-continuation"})


def log(event: str, **kwargs) -> None:
    print(json.dumps({"event": event, "time": time.time(), **kwargs}, sort_keys=True), flush=True)


def get(url: str, tries: int = 5, timeout: int = 120, headers: dict | None = None, params: dict | None = None) -> requests.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            response = SESSION.get(url, timeout=timeout, headers=headers, params=params)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            time.sleep(min(15, 1.6**attempt))
    raise RuntimeError(f"GET failed {url}: {last}")


def upload(path: str, content_type: str, data: bytes) -> dict:
    payload = {
        "path": path,
        "content_type": content_type,
        "data_b64": base64.b64encode(data).decode(),
    }
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = SESSION.post(
                UPLOAD_EP,
                headers={
                    "x-upload-token": os.environ["UPLOAD_TOKEN"],
                    "content-type": "application/json",
                },
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last = exc
            time.sleep(min(15, 1.7**attempt))
    raise RuntimeError(f"UPLOAD failed {path}: {last}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_from_url(url: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(get(url).content))).convert("RGB")


def image_from_storage(path: str) -> Image.Image:
    return image_from_url(STORAGE + urllib.parse.quote(path, safe="/"))


def jpg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def fetch_controls() -> list[dict]:
    params = {
        "select": "id,manuscript_id,slug,seq,obj_index,crop_path",
        "part": "eq.root",
        "crop_path": "not.is.null",
        "crop_qa": "is.null",
        "manuscript_id": "in.(voynich,bsb1784)",
        "order": "manuscript_id,slug,obj_index",
        "limit": "1000",
    }
    response = get(
        REST + "/herbal_objects",
        headers={"apikey": ANON, "authorization": "Bearer " + ANON, "accept": "application/json"},
        params=params,
    )
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected control response: {type(rows).__name__}")
    return rows


def embed_images(model, processor, images: list[Image.Image], batch_size: int = 4) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(model.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            output = model(**inputs)
            vector = output.last_hidden_state[:, 0].float()
            vector = torch.nn.functional.normalize(vector, dim=1)
        vectors.append(vector.cpu().numpy().astype("float32"))
        log("embedding_progress", done=min(start + batch_size, len(images)), total=len(images))
    return np.vstack(vectors)


def reciprocal_pairs(first_indices: np.ndarray, second_indices: np.ndarray, embeddings: np.ndarray) -> list[tuple[int, int]]:
    similarities = embeddings[first_indices] @ embeddings[second_indices].T
    first_to_second = np.argmax(similarities, axis=1)
    second_to_first = np.argmax(similarities, axis=0)
    return [
        (int(first_indices[first_position]), int(second_indices[second_position]))
        for first_position, second_position in enumerate(first_to_second)
        if second_to_first[second_position] == first_position
    ]


def corpus_analysis(
    include_partial: bool,
    embeddings: np.ndarray,
    corpora: np.ndarray,
    groups: np.ndarray,
    statuses: np.ndarray,
    repetitions: int = 10_000,
) -> dict:
    voynich = np.where(corpora == "voynich")[0]
    bsb = np.where(corpora == "bsb1784")[0]
    palatino = np.where(
        (corpora == "p586")
        & ((statuses == "accept") | (include_partial & (statuses == "partial")))
    )[0]
    counts = {"voynich": int(len(voynich)), "p586": int(len(palatino)), "bsb1784": int(len(bsb))}
    if not len(voynich) or not len(palatino) or not len(bsb):
        return {"available": False, "counts": counts}

    targets = np.concatenate([palatino, bsb])
    labels = np.concatenate([np.ones(len(palatino), dtype=int), np.zeros(len(bsb), dtype=int)])
    similarities = embeddings[voynich] @ embeddings[targets].T
    metrics: dict[str, object] = {}
    top_indices: dict[int, np.ndarray] = {}
    for k in (1, 5, 10):
        effective = min(k, len(targets))
        top = np.argpartition(-similarities, effective - 1, axis=1)[:, :effective]
        top_indices[k] = top
        metrics[f"palatino_share_top{k}"] = float(labels[top].mean())

    voynich_to_palatino = embeddings[voynich] @ embeddings[palatino].T
    voynich_to_bsb = embeddings[voynich] @ embeddings[bsb].T
    best_palatino = voynich_to_palatino.max(axis=1)
    best_bsb = voynich_to_bsb.max(axis=1)
    observed_difference = float((best_palatino - best_bsb).mean())
    metrics["mean_best_palatino"] = float(best_palatino.mean())
    metrics["mean_best_bsb"] = float(best_bsb.mean())
    metrics["mean_best_difference"] = observed_difference

    null_values: dict[str, list[float]] = {
        "palatino_share_top1": [],
        "palatino_share_top5": [],
        "palatino_share_top10": [],
        "mean_best_difference": [],
    }
    for _ in range(repetitions):
        permuted = np.zeros(len(targets), dtype=int)
        permuted[RNG.choice(len(targets), len(palatino), replace=False)] = 1
        for k in (1, 5, 10):
            null_values[f"palatino_share_top{k}"].append(float(permuted[top_indices[k]].mean()))
        labelled_palatino = np.where(permuted == 1)[0]
        labelled_bsb = np.where(permuted == 0)[0]
        null_values["mean_best_difference"].append(
            float(
                (
                    similarities[:, labelled_palatino].max(axis=1)
                    - similarities[:, labelled_bsb].max(axis=1)
                ).mean()
            )
        )

    null_summary = {}
    for key, values in null_values.items():
        array = np.asarray(values, dtype=float)
        observed = float(metrics[key])
        null_summary[key] = {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
            "p_upper": float((1 + np.sum(array >= observed - 1e-12)) / (repetitions + 1)),
            "p_lower": float((1 + np.sum(array <= observed + 1e-12)) / (repetitions + 1)),
        }

    reciprocal_palatino = reciprocal_pairs(voynich, palatino, embeddings)
    reciprocal_bsb = reciprocal_pairs(voynich, bsb, embeddings)
    metrics["reciprocal_voynich_palatino"] = int(len(reciprocal_palatino))
    metrics["reciprocal_voynich_bsb"] = int(len(reciprocal_bsb))

    voynich_groups = np.unique(groups[voynich])
    palatino_groups = np.unique(groups[palatino])
    bsb_groups = np.unique(groups[bsb])
    bootstrap = []
    for _ in range(3_000):
        sampled_voynich_groups = RNG.choice(voynich_groups, len(voynich_groups), replace=True)
        sampled_palatino_groups = RNG.choice(palatino_groups, len(palatino_groups), replace=True)
        sampled_bsb_groups = RNG.choice(bsb_groups, len(bsb_groups), replace=True)
        sampled_voynich = np.concatenate([voynich[groups[voynich] == group] for group in sampled_voynich_groups])
        sampled_palatino = np.concatenate([palatino[groups[palatino] == group] for group in sampled_palatino_groups])
        sampled_bsb = np.concatenate([bsb[groups[bsb] == group] for group in sampled_bsb_groups])
        bootstrap.append(
            float(
                (
                    (embeddings[sampled_voynich] @ embeddings[sampled_palatino].T).max(axis=1)
                    - (embeddings[sampled_voynich] @ embeddings[sampled_bsb].T).max(axis=1)
                ).mean()
            )
        )
    metrics["bootstrap_difference_ci95"] = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]

    matched_size = min(len(palatino), len(bsb))
    matched_differences = []
    matched_win_fractions = []
    for _ in range(repetitions):
        palatino_sample = RNG.choice(palatino, matched_size, replace=False)
        bsb_sample = RNG.choice(bsb, matched_size, replace=False)
        palatino_best = (embeddings[voynich] @ embeddings[palatino_sample].T).max(axis=1)
        bsb_best = (embeddings[voynich] @ embeddings[bsb_sample].T).max(axis=1)
        difference = palatino_best - bsb_best
        matched_differences.append(float(difference.mean()))
        matched_win_fractions.append(float((difference > 0).mean()))
    matched_array = np.asarray(matched_differences)
    metrics["matched_pool_audit"] = {
        "secondary_posthoc": True,
        "reference_size_each": int(matched_size),
        "repetitions": repetitions,
        "mean_best_difference": float(matched_array.mean()),
        "reference_subsample_interval": [
            float(np.quantile(matched_array, 0.025)),
            float(np.quantile(matched_array, 0.975)),
        ],
        "voynich_query_win_fraction": float(np.mean(matched_win_fractions)),
        "win_fraction_interval": [
            float(np.quantile(matched_win_fractions, 0.025)),
            float(np.quantile(matched_win_fractions, 0.975)),
        ],
    }

    flattened = np.argsort(-voynich_to_palatino.ravel())
    top_pairs = []
    used_voynich: set[int] = set()
    for flat_index in flattened:
        voynich_position, palatino_position = np.unravel_index(flat_index, voynich_to_palatino.shape)
        if int(voynich_position) in used_voynich:
            continue
        used_voynich.add(int(voynich_position))
        score = float(voynich_to_palatino[voynich_position, palatino_position])
        bsb_similarities = voynich_to_bsb[voynich_position]
        decoy_position = int(np.argmin(np.abs(bsb_similarities - score)))
        top_pairs.append(
            {
                "voynich_index": int(voynich[voynich_position]),
                "palatino_index": int(palatino[palatino_position]),
                "bsb_index": int(bsb[decoy_position]),
                "palatino_similarity": score,
                "bsb_similarity": float(bsb_similarities[decoy_position]),
            }
        )
        if len(top_pairs) >= 12:
            break

    return {
        "available": True,
        "include_partial": include_partial,
        "counts": counts,
        "metrics": metrics,
        "null": null_summary,
        "top_pairs": top_pairs,
    }


def create_blind_audit(records: list[dict], pair_source: list[dict]) -> dict | None:
    if not pair_source:
        return None
    sheet = Image.new("RGB", (1200, 360 * len(pair_source)), "white")
    draw = ImageDraw.Draw(sheet)
    key = []
    for trial, pair in enumerate(pair_source, start=1):
        query = records[pair["voynich_index"]]["image"]
        palatino = records[pair["palatino_index"]]["image"]
        bsb = records[pair["bsb_index"]]["image"]
        flip = bool(RNG.integers(0, 2))
        option_a, option_b = (bsb, palatino) if flip else (palatino, bsb)
        key.append(
            {
                "trial": trial,
                "A": "bsb1784" if flip else "p586",
                "B": "p586" if flip else "bsb1784",
                "voynich_key": records[pair["voynich_index"]]["key"],
                "palatino_key": records[pair["palatino_index"]]["key"],
                "bsb_key": records[pair["bsb_index"]]["key"],
                "palatino_similarity": pair["palatino_similarity"],
                "bsb_similarity": pair["bsb_similarity"],
            }
        )
        for column, (label, image) in enumerate([("QUERY", query), ("A", option_a), ("B", option_b)]):
            x = column * 400
            y = (trial - 1) * 360
            thumbnail = image.copy()
            thumbnail.thumbnail((370, 300))
            sheet.paste(thumbnail, (x + (400 - thumbnail.width) // 2, y + 45 + (300 - thumbnail.height) // 2))
            draw.rectangle((x, y, x + 399, y + 359), outline="black")
            draw.text((x + 8, y + 8), f"Trial {trial} {label}", fill="black")
    sheet_upload = upload("p586_root_v01/full_run/audit/blind_triptychs_recovered.jpg", "image/jpeg", jpg_bytes(sheet))
    key_upload = upload(
        "p586_root_v01/full_run/audit/blind_key_recovered.json",
        "application/json",
        json.dumps(key, indent=2, sort_keys=True).encode(),
    )
    return {
        "sheet": sheet_upload,
        "key": key_upload,
        "trials": len(key),
        "status": "PACK_CREATED_NOT_YET_ADJUDICATED",
    }


def main() -> None:
    checkpoint = get(CHECKPOINT_URL).json()
    proposals = checkpoint.get("proposals", [])
    retained = [proposal for proposal in proposals if proposal.get("qa_status") in {"accept", "partial"}]
    control_rows = fetch_controls()
    log(
        "inputs_resolved",
        proposals=len(proposals),
        retained=len(retained),
        accept=sum(proposal.get("qa_status") == "accept" for proposal in retained),
        partial=sum(proposal.get("qa_status") == "partial" for proposal in retained),
        voynich=sum(row.get("manuscript_id") == "voynich" for row in control_rows),
        bsb=sum(row.get("manuscript_id") == "bsb1784" for row in control_rows),
    )

    records: list[dict] = []
    for proposal in retained:
        try:
            image = image_from_storage(proposal["crop_path"])
        except Exception as exc:
            log("palatino_crop_fail", path=proposal.get("crop_path"), error=str(exc))
            continue
        records.append(
            {
                "key": f"p586:c{int(proposal['canvas_index']):03d}:r{int(proposal['root_index']):02d}",
                "corpus": "p586",
                "group": str(proposal["canvas_index"]),
                "status": proposal["qa_status"],
                "crop_path": proposal["crop_path"],
                "image": image,
                "meta": proposal,
            }
        )
    for row in control_rows:
        try:
            image = image_from_storage(row["crop_path"])
        except Exception as exc:
            log("control_crop_fail", id=row.get("id"), path=row.get("crop_path"), error=str(exc))
            continue
        records.append(
            {
                "key": f"{row['manuscript_id']}:{row['id']}",
                "corpus": row["manuscript_id"],
                "group": str(row.get("slug") or row.get("seq")),
                "status": "accept",
                "crop_path": row["crop_path"],
                "image": image,
                "meta": row,
            }
        )

    log("dino_load_start", model=DINO_MODEL, records=len(records))
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModel.from_pretrained(
        DINO_MODEL,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    revision = getattr(model.config, "_commit_hash", None)
    embeddings = embed_images(model, processor, [record["image"] for record in records], batch_size=4)
    dimension = int(embeddings.shape[1])
    keys = np.array([record["key"] for record in records], dtype=object)
    corpora = np.array([record["corpus"] for record in records], dtype=object)
    groups = np.array([record["group"] for record in records], dtype=object)
    statuses = np.array([record["status"] for record in records], dtype=object)
    paths = np.array([record["crop_path"] for record in records], dtype=object)

    embedding_buffer = io.BytesIO()
    np.savez_compressed(
        embedding_buffer,
        embeddings=embeddings,
        keys=keys,
        corpora=corpora,
        groups=groups,
        statuses=statuses,
        paths=paths,
        model=np.array([DINO_MODEL], dtype=object),
        revision=np.array([revision], dtype=object),
    )
    embedding_upload = upload(
        "p586_root_v01/embeddings/all_roots_dinov3_vit7b16_recovered.npz",
        "application/octet-stream",
        embedding_buffer.getvalue(),
    )

    primary = corpus_analysis(False, embeddings, corpora, groups, statuses)
    sensitivity = corpus_analysis(True, embeddings, corpora, groups, statuses)
    audit = create_blind_audit(records, sensitivity.get("top_pairs", []) if sensitivity.get("available") else [])

    comparison = {
        "protocol_id": PROTOCOL,
        "run_id": RUN_ID,
        "seed": SEED,
        "model": DINO_MODEL,
        "model_revision": revision,
        "embedding_dim": dimension,
        "embedding_bundle": embedding_upload,
        "checkpoint_sha256": checkpoint.get("result_sha256"),
        "recovery_canvases": checkpoint.get("recovery_canvases"),
        "primary": primary,
        "sensitivity": sensitivity,
        "visual_audit": audit,
        "limitations": [
            "Exploratory first pass.",
            "Palatino crops are model-localised and model-QAed, not human-verified.",
            "Positive similarity does not establish exemplar, lineage, provenance or botanical identity.",
            "Matched-pool audit is a secondary post-hoc check added after the interrupted run exposed reference-pool imbalance as a material issue.",
        ],
    }
    canonical = json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
    comparison["result_sha256"] = sha256(canonical)
    upload(
        "p586_root_v01/full_run/comparison_result_recovered.json",
        "application/json",
        json.dumps(comparison, indent=2, sort_keys=True).encode(),
    )

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "canvas_index",
            "plant_index",
            "root_index",
            "qa_status",
            "qa_confidence",
            "root_fraction",
            "detector_score",
            "visibility",
            "crop_path",
            "context_crop_path",
            "image_sha256",
            "qa_reason",
        ],
    )
    writer.writeheader()
    for proposal in proposals:
        writer.writerow({key: proposal.get(key) for key in writer.fieldnames})
    upload(
        "p586_root_v01/full_run/root_proposals_recovered.csv",
        "text/csv",
        csv_buffer.getvalue().encode(),
    )

    print(
        "RESULT_JSON="
        + json.dumps(
            {
                "records": len(records),
                "embedding_dim": dimension,
                "primary": primary,
                "sensitivity": sensitivity,
                "sha256": comparison["result_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
