#!/usr/bin/env python3
"""Frozen DINOv3 root/flower comparison for BSB Clm 5905.

This phase opens the target labels only after the 198-folio extraction manifest
has frozen. It reuses parent flower embeddings, deterministically derives parent
root crops from frozen colour masks/root boundaries, embeds all new crops with
the frozen DINOv3-7B representation, runs manuscript-level statistics, creates
blinded triptychs, and persists a human-readable result bundle.
"""
from __future__ import annotations

import base64
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModel

PROTOCOL = "CLM5905-VMS-RF-0.1-20260804"
SEED = 20260804
MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
REV = "b80367753773648a6793235ab9c65cdbb029506f"
SUPA = "https://ymaqlcfjmdwncdbjprmw.supabase.co"
BRIDGE = SUPA + "/storage/v1/object/public/bridge/"
UPLOAD_URL = SUPA + "/functions/v1/clm5905-upload-v01"
BLIND_URL = SUPA + "/functions/v1/clm5905-blind-v01"
EXTRACT = BRIDGE + "clm5905_v01/extraction/extraction_manifest_frozen.json"
CORE = BRIDGE + "p586_plant_v01/bundles/P586_PLANT_MORPHOLOGY_HUMAN_READABLE_CORE.zip"
ROOT_BUNDLE = BRIDGE + "p586_root_v01/embeddings/all_roots_dinov3_vit7b16_recovered.npz"
PREFIX = "clm5905_v01/analysis"
OUT = Path(os.environ.get("CLM_ANALYSIS_OUT", "/work/clm5905_analysis"))
TARGET = "clm5905"
VOYNICH = "voynich"
CONTROLS = ["bnf_lat_6862", "bnf_gr_2179", "herb_18f0aa144a2b", "herb_78e2bbc79062", "bsb1784"]
PARENT_ORDER = ["bncf_palatino_586", "voynich", "bnf_lat_6862", "bnf_gr_2179", "herb_18f0aa144a2b", "herb_78e2bbc79062", "bsb1784", "herb_205bfb89efbc"]
S = requests.Session()
S.headers["User-Agent"] = "CLM5905DINO/0.1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csha(obj: Any) -> str:
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def get(url: str, tries: int = 6, timeout: int = 300) -> requests.Response:
    last: Exception | None = None
    for k in range(tries):
        try:
            r = S.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            time.sleep(min(20.0, 1.8**k))
    raise RuntimeError(f"GET failed {url}: {last}")


def post_json(url: str, payload: dict[str, Any], tries: int = 5, timeout: int = 300) -> dict[str, Any]:
    last: Exception | None = None
    for k in range(tries):
        try:
            r = S.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            ans = r.json()
            if isinstance(ans, dict) and ans.get("error"):
                raise RuntimeError(str(ans))
            return ans
        except Exception as exc:
            last = exc
            time.sleep(min(20.0, 1.8**k))
    raise RuntimeError(f"POST failed {url}: {last}")


def upload(path: str, content_type: str, data: bytes) -> None:
    post_json(UPLOAD_URL, {"path": path, "content_type": content_type, "data_b64": base64.b64encode(data).decode()}, timeout=360)


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.convert("RGB").save(b, "PNG", optimize=True)
    return b.getvalue()


def load_json_from_zip(z: zipfile.ZipFile, name: str) -> dict[str, Any]:
    return json.loads(z.read(name))


def parent_prefix(cid: str) -> str:
    return "p586_plant_v01/target" if cid == "bncf_palatino_586" else f"p586_plant_v01/controls/{cid}"


def reconstruct_parent_items(manifests: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for cid in PARENT_ORDER:
        m = manifests.get(cid)
        if not m:
            continue
        for p in m.get("plants", []):
            broad = p.get("qa_status") in {"accept", "partial"}
            strict = p.get("qa_status") == "accept"
            valid = bool(p.get("mask_valid", False))
            base = {"corpus": cid, "plant_id": p.get("plant_id"), "qa_status": p.get("qa_status"), "strict": strict, "broad": broad}
            if broad and p.get("crop_path"):
                items.append({**base, "kind": "whole_raw", "path": p["crop_path"]})
            if broad and valid and p.get("masked_crop_path"):
                items.append({**base, "kind": "whole_masked", "path": p["masked_crop_path"]})
            if broad and valid and p.get("above_strict_path"):
                items.append({**base, "kind": "above_strict", "path": p["above_strict_path"]})
            if broad and valid and p.get("above_context_path"):
                items.append({**base, "kind": "above_context", "path": p["above_context_path"]})
            for z in p.get("reproductive_structures", []):
                if z.get("qa_status") not in {"accept", "partial"} or not z.get("crop_path"):
                    continue
                cls = z.get("qa_class") or z.get("proposed_class")
                items.append(
                    {
                        **base,
                        "kind": "reproductive",
                        "path": z["crop_path"],
                        "repro_id": z.get("repro_id"),
                        "repro_status": z.get("qa_status"),
                        "repro_class": cls,
                    }
                )
    for i, item in enumerate(items):
        item["item_id"] = f"i{i:06d}"
    return items


def load_parent() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    raw = get(CORE).content
    z = zipfile.ZipFile(io.BytesIO(raw))
    analysis = load_json_from_zip(z, "01_RESULTS/dinov3_analysis.json")
    manifests: dict[str, dict[str, Any]] = {}
    masks: dict[str, dict[str, Any]] = {}
    for cid in PARENT_ORDER:
        ch = f"02_CORPORA/{cid}/channels_manifest_frozen.json"
        cm = f"02_CORPORA/{cid}/color_masks_frozen.json"
        if ch in z.namelist():
            manifests[cid] = load_json_from_zip(z, ch)
        if cm in z.namelist():
            masks[cid] = load_json_from_zip(z, cm)
    items = reconstruct_parent_items(manifests)
    meta = {x["item_id"]: x for x in items}
    parent_vecs: dict[str, dict[str, np.ndarray]] = {}
    for rec in analysis["embedding_files"]:
        cid = rec["corpus"]
        data = get(BRIDGE + rec["path"]).content
        if sha(data) != rec["sha256"]:
            raise RuntimeError(f"parent embedding hash mismatch: {cid}")
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        ids = [str(x) for x in arr["item_id"]]
        vec = arr["embedding"].astype(np.float32)
        if any(i not in meta for i in ids):
            raise RuntimeError(f"cannot reconstruct parent item metadata for {cid}")
        parent_vecs[cid] = {i: v for i, v in zip(ids, vec)}
    return analysis, manifests, masks, parent_vecs, items


def parent_flower_data(parent_vecs: dict[str, dict[str, np.ndarray]], items: list[dict[str, Any]], broad: bool) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classes = {"flower", "flower_head", "inflorescence", "bud"} if broad else {"flower", "flower_head", "inflorescence"}
    for item in items:
        if item["kind"] != "reproductive" or item.get("repro_class") not in classes:
            continue
        if broad:
            if item.get("repro_status") not in {"accept", "partial"}:
                continue
        elif item.get("repro_status") != "accept":
            continue
        v = parent_vecs.get(item["corpus"], {}).get(item["item_id"])
        if v is None:
            continue
        vectors[item["corpus"]].append(v)
        metadata[item["corpus"]].append(item)
    return {k: np.stack(v) for k, v in vectors.items() if v}, dict(metadata)


def red_root_crop(source: Image.Image, quantized: Image.Image) -> tuple[Image.Image, Image.Image, dict[str, Any]] | None:
    src = np.asarray(source.convert("RGB"))
    q = np.asarray(quantized.convert("RGB").resize(source.size, Image.Resampling.NEAREST))
    red = (q[:, :, 0] >= 150) & (q[:, :, 0] >= q[:, :, 1] + 50) & (q[:, :, 0] >= q[:, :, 2] + 50)
    ys, xs = np.where(red)
    if len(xs) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x, pad_y = max(3, round((x1 - x0) * 0.12)), max(3, round((y1 - y0) * 0.12))
    x0, x1 = max(0, x0 - pad_x), min(source.width, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(source.height, y1 + pad_y)
    ordinary = source.crop((x0, y0, x1, y1))
    local_src = src[y0:y1, x0:x1]
    local_red = red[y0:y1, x0:x1]
    masked = np.full_like(local_src, 255)
    masked[local_red] = local_src[local_red]
    return ordinary, Image.fromarray(masked, "RGB"), {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "red_pixels": int(red.sum())}


def derive_parent_roots(manifests: dict[str, dict[str, Any]], masks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cid in [VOYNICH] + CONTROLS:
        if cid not in manifests:
            continue
        mask_map = {r.get("plant_id"): r for r in masks.get(cid, {}).get("records", [])}
        for p in manifests[cid].get("plants", []):
            status = p.get("qa_status")
            if status not in {"accept", "partial"} or not p.get("crop_path"):
                continue
            mr = mask_map.get(p.get("plant_id"), {})
            quant_path = mr.get("quantized_mask_path") or p.get("quantized_mask_path")
            method = "colour_red"
            ordinary: Image.Image | None = None
            masked_im: Image.Image | None = None
            diagnostics: dict[str, Any] = {}
            try:
                source_raw = get(BRIDGE + p["crop_path"]).content
                source = Image.open(io.BytesIO(source_raw)).convert("RGB")
                if quant_path:
                    q = Image.open(io.BytesIO(get(BRIDGE + quant_path).content)).convert("RGB")
                    result = red_root_crop(source, q)
                else:
                    result = None
                if result:
                    ordinary, masked_im, diagnostics = result
                else:
                    method = "frozen_boundary_fallback"
                    y1000 = p.get("root_boundary_y_1000")
                    if not p.get("has_visible_roots") or y1000 is None or float(y1000) >= 970:
                        continue
                    y = max(0, min(source.height - 10, round(float(y1000) * source.height / 1000)))
                    y = max(0, y - round(source.height * 0.04))
                    ordinary = source.crop((0, y, source.width, source.height))
                    masked_im = ordinary.copy()
                    diagnostics = {"x": 0, "y": y, "w": source.width, "h": source.height}
                root_id = f"{cid}__{p.get('plant_id')}"
                odata, mdata = png(ordinary), png(masked_im)
                opath = f"{PREFIX}/derived_parent_roots/ordinary/{root_id}.png"
                mpath = f"{PREFIX}/derived_parent_roots/masked/{root_id}.png"
                upload(opath, "image/png", odata)
                upload(mpath, "image/png", mdata)
                out.append(
                    {
                        "item_id": root_id,
                        "corpus": cid,
                        "plant_id": p.get("plant_id"),
                        "qa_status": status,
                        "strict": status == "accept",
                        "broad": True,
                        "ordinary_path": opath,
                        "masked_path": mpath,
                        "ordinary_sha256": sha(odata),
                        "masked_sha256": sha(mdata),
                        "derivation_method": method,
                        "diagnostics": diagnostics,
                        "source_path": p.get("crop_path"),
                    }
                )
            except Exception as exc:
                print(json.dumps({"event": "parent_root_error", "corpus": cid, "plant": p.get("plant_id"), "error": str(exc)}), flush=True)
    return out


def extraction_items(extract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    roots: list[dict[str, Any]] = []
    flowers: list[dict[str, Any]] = []
    whole: list[dict[str, Any]] = []
    for r in extract["records"]:
        n = int(r["illustration_number"])
        if r.get("whole_crop_path") and r.get("whole_status") in {"accept", "partial"}:
            whole.append({"item_id": f"clm_w_{n:03d}", "corpus": TARGET, "qa_status": r.get("whole_status"), "strict": r.get("whole_status") == "accept", "broad": True, "path": r["whole_crop_path"], "illustration_number": n, "folio": r.get("folio"), "label": r.get("label")})
        if r.get("root_crop_path") and r.get("root_status") in {"accept", "partial"}:
            roots.append({"item_id": f"clm_root_{n:03d}", "corpus": TARGET, "qa_status": r.get("root_status"), "strict": r.get("root_status") == "accept", "broad": True, "ordinary_path": r["root_crop_path"], "masked_path": r.get("root_masked_path"), "illustration_number": n, "folio": r.get("folio"), "label": r.get("label"), "description": r.get("root_description")})
        for z in r.get("reproductive", []):
            if z.get("crop_path") and z.get("status") in {"accept", "partial"}:
                flowers.append({"item_id": z.get("repro_id"), "corpus": TARGET, "qa_status": z.get("status"), "strict": z.get("status") == "accept", "broad": True, "ordinary_path": z.get("crop_path"), "masked_path": z.get("masked_path"), "repro_class": z.get("class"), "illustration_number": n, "folio": r.get("folio"), "label": r.get("label"), "description": z.get("description")})
    return roots, flowers, whole


def preprocess(im: Image.Image) -> torch.Tensor:
    im = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1))


def embed(items: list[dict[str, Any]], path_key: str, model: Any, device: str, batch_size: int = 6) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    vec: dict[str, np.ndarray] = {}
    errors: list[dict[str, Any]] = []
    batch: list[torch.Tensor] = []
    keys: list[str] = []

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.inference_mode():
            o = model(pixel_values=x)
            v = o.last_hidden_state[:, 0, :].float()
            v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-12)
        for key, value in zip(keys, v.cpu().numpy()):
            vec[key] = value.astype(np.float32)
        batch.clear()
        keys.clear()

    for i, item in enumerate(items, 1):
        path = item.get(path_key)
        if not path:
            continue
        try:
            raw = get(BRIDGE + path).content
            item[path_key + "_sha256_verified"] = sha(raw)
            batch.append(preprocess(Image.open(io.BytesIO(raw))))
            keys.append(item["item_id"])
        except Exception as exc:
            errors.append({"item_id": item["item_id"], "path": path, "error": f"{type(exc).__name__}: {exc}"})
        if len(batch) >= batch_size:
            flush()
        if i % 50 == 0:
            print(json.dumps({"event": "embed_progress", "path_key": path_key, "n": i, "total": len(items), "vectors": len(vec), "errors": len(errors)}), flush=True)
    flush()
    return vec, errors


def save_npz(path: str, items: list[dict[str, Any]], vec: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = [x for x in items if x["item_id"] in vec]
    ids = np.array([x["item_id"] for x in rows])
    corpora = np.array([x["corpus"] for x in rows])
    mat = np.stack([vec[x["item_id"]] for x in rows]).astype(np.float32) if rows else np.zeros((0, 4096), np.float32)
    b = io.BytesIO()
    np.savez_compressed(b, item_id=ids, corpus=corpora, embedding=mat, model=np.array([MODEL]), revision=np.array([REV]))
    data = b.getvalue()
    upload(path, "application/octet-stream", data)
    return {"path": path, "sha256": sha(data), "bytes": len(data), "n": len(rows)}


def best(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (q @ r.T).max(axis=1)


def analyse(name: str, data: dict[str, np.ndarray], meta: dict[str, list[dict[str, Any]]], rng: np.random.Generator) -> dict[str, Any]:
    counts = {c: len(v) for c, v in data.items()}
    eligible = [c for c in CONTROLS if c in data and len(data[c]) >= 8]
    out: dict[str, Any] = {"channel": name, "counts": counts, "eligible_controls": eligible}
    if TARGET not in data or VOYNICH not in data or len(data[TARGET]) == 0 or len(data[VOYNICH]) == 0 or not eligible:
        out["available"] = False
        out["reason"] = "target/reference/control sample gate failed"
        return out
    out["available"] = True
    full: dict[str, Any] = {}
    for cid, v in data.items():
        if cid == VOYNICH:
            continue
        b = best(v, data[VOYNICH])
        full[cid] = {"mean_best": float(b.mean()), "median_best": float(np.median(b)), "n": len(v)}
    out["full_scores"] = full
    participants = [TARGET] + eligible
    m = min([len(data[VOYNICH])] + [len(data[c]) for c in participants])
    out["balanced_n"] = m
    if m < 2:
        out["available"] = False
        out["reason"] = "balanced sample below 2"
        return out
    reps = {c: [] for c in participants}
    for _ in range(500):
        ref = data[VOYNICH][rng.choice(len(data[VOYNICH]), m, replace=False)]
        for cid in participants:
            q = data[cid][rng.choice(len(data[cid]), m, replace=False)]
            reps[cid].append(float(best(q, ref).mean()))
    means = {c: float(np.mean(v)) for c, v in reps.items()}
    effect = means[TARGET] - float(np.mean([means[c] for c in eligible]))
    out["balanced_mean_best"] = means
    out["primary_effect_target_minus_control_mean"] = effect
    boot = []
    for _ in range(10000):
        tv = reps[TARGET][rng.integers(500)]
        chosen = rng.choice(eligible, len(eligible), replace=True)
        cv = np.mean([reps[c][rng.integers(500)] for c in chosen])
        boot.append(tv - cv)
    out["hierarchical_bootstrap_95ci"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    out["target_rank_descending"] = 1 + sum(means[c] > means[TARGET] for c in eligible)
    out["manuscript_label_rank_p_upper"] = (1 + sum(means[c] >= means[TARGET] for c in eligible)) / (1 + len(eligible))
    out["leave_one_control_out"] = {
        c: means[TARGET] - float(np.mean([means[d] for d in eligible if d != c])) if len(eligible) > 1 else None for c in eligible
    }
    shares: dict[tuple[int, str], list[float]] = defaultdict(list)
    for _ in range(100):
        pools, labels = [], []
        for cid in participants:
            x = data[cid][rng.choice(len(data[cid]), m, replace=False)]
            pools.append(x)
            labels.extend([cid] * m)
        pool = np.concatenate(pools)
        labels_arr = np.array(labels)
        order = np.argsort(-(data[VOYNICH] @ pool.T), axis=1)
        for k in (1, 5, 10):
            flat = labels_arr[order[:, : min(k, order.shape[1])]].reshape(-1)
            for cid in participants:
                shares[(k, cid)].append(float(np.mean(flat == cid)))
    out["topk_source_shares"] = {str(k): {c: float(np.mean(shares[(k, c)])) for c in participants} for k in (1, 5, 10)}
    sim = data[TARGET] @ data[VOYNICH].T
    pairs = []
    for qi, ri in sorted(((i, int(np.argmax(sim[i]))) for i in range(len(data[TARGET]))), key=lambda z: sim[z[0], z[1]], reverse=True)[:12]:
        pairs.append({"target_index": qi, "voynich_index": ri, "similarity": float(sim[qi, ri]), "target_item": meta[TARGET][qi], "voynich_item": meta[VOYNICH][ri]})
    out["strongest_target_pairs"] = pairs
    return out


def data_from_items(items: list[dict[str, Any]], vec: dict[str, np.ndarray], strict: bool, path_label: str = "ordinary") -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    d: dict[str, list[np.ndarray]] = defaultdict(list)
    m: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["item_id"] not in vec:
            continue
        if strict and not item.get("strict"):
            continue
        if not strict and not item.get("broad"):
            continue
        d[item["corpus"]].append(vec[item["item_id"]])
        m[item["corpus"]].append({**item, "embedding_variant": path_label})
    return {k: np.stack(v) for k, v in d.items() if v}, dict(m)


def merge_target_with_parent(target_items: list[dict[str, Any]], target_vec: dict[str, np.ndarray], parent_data: dict[str, np.ndarray], parent_meta: dict[str, list[dict[str, Any]]], strict: bool) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    data = dict(parent_data)
    meta = {k: list(v) for k, v in parent_meta.items()}
    rows = [x for x in target_items if x["item_id"] in target_vec and ((x.get("strict")) if strict else x.get("broad"))]
    if rows:
        data[TARGET] = np.stack([target_vec[x["item_id"]] for x in rows])
        meta[TARGET] = rows
    return data, meta


def open_image(path: str) -> Image.Image:
    return Image.open(io.BytesIO(get(BRIDGE + path).content)).convert("RGB")


def item_path(item: dict[str, Any]) -> str:
    return str(item.get("ordinary_path") or item.get("path") or item.get("crop_path"))


def create_blind_pack(channel: str, result: dict[str, Any], data: dict[str, np.ndarray], meta: dict[str, list[dict[str, Any]]], rng: np.random.Generator, start_trial: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[bytes]]:
    trials: list[dict[str, Any]] = []
    key: dict[str, Any] = {}
    sheet_bytes: list[bytes] = []
    eligible = result.get("eligible_controls", [])
    for offset, pair in enumerate(result.get("strongest_target_pairs", [])[:6]):
        trial = start_trial + offset
        ri = int(pair["voynich_index"])
        ref_vec = data[VOYNICH][ri]
        target_sim = float(pair["similarity"])
        candidates = []
        for cid in eligible:
            sims = data[cid] @ ref_vec
            for ci, score in enumerate(sims):
                candidates.append((abs(float(score) - target_sim), cid, ci, float(score)))
        if not candidates:
            continue
        _, ccid, ci, cscore = min(candidates, key=lambda x: x[0])
        target_item = pair["target_item"]
        ref_item = pair["voynich_item"]
        control_item = meta[ccid][ci]
        target_side = "A" if rng.integers(2) == 0 else "B"
        a_item, b_item = (target_item, control_item) if target_side == "A" else (control_item, target_item)
        ref = ImageOps.contain(open_image(item_path(ref_item)), (560, 330))
        a = ImageOps.contain(open_image(item_path(a_item)), (420, 300))
        b = ImageOps.contain(open_image(item_path(b_item)), (420, 300))
        sheet = Image.new("RGB", (920, 690), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 12), f"Blind {channel} trial {trial}", fill="black")
        draw.text((430, 38), "Reference", fill="black")
        sheet.paste(ref, ((920 - ref.width) // 2, 62))
        draw.text((210, 376), "A", fill="black")
        draw.text((680, 376), "B", fill="black")
        sheet.paste(a, (20 + (420 - a.width) // 2, 405))
        sheet.paste(b, (480 + (420 - b.width) // 2, 405))
        raw = png(sheet)
        path = f"clm5905_v01/blind/{channel}_{trial:02d}.png"
        upload(path, "image/png", raw)
        url = BRIDGE + path
        try:
            adjudication = post_json(BLIND_URL, {"imageUrl": url, "trial": trial, "channel": channel}, tries=5, timeout=300)
        except Exception as exc:
            adjudication = {"choice": "abstain", "confidence": 0.0, "reason": f"adjudication error: {exc}"}
        choice = adjudication.get("choice", "abstain")
        outcome = "confirmation" if choice == target_side else "contradiction" if choice in {"A", "B"} else choice
        public = {"trial": trial, "channel": channel, "sheet_path": path, "choice": choice, "confidence": adjudication.get("confidence"), "reason": adjudication.get("reason"), "outcome": outcome}
        trials.append(public)
        key[str(trial)] = {"target_side": target_side, "target_item": target_item, "control_corpus": ccid, "control_item": control_item, "reference_item": ref_item, "target_similarity": target_sim, "control_similarity": cscore}
        sheet_bytes.append(raw)
    return trials, key, sheet_bytes


def crosscheck_dedicated_root(target_data: np.ndarray) -> dict[str, Any]:
    raw = get(ROOT_BUNDLE).content
    z = np.load(io.BytesIO(raw), allow_pickle=True)
    e = z["embeddings"].astype(np.float32)
    corp = np.array([str(x) for x in z["corpora"]])
    status = np.array([str(x) for x in z["statuses"]])
    v = e[corp == "voynich"]
    b = e[corp == "bsb1784"]
    if len(target_data) == 0 or len(v) == 0 or len(b) == 0:
        return {"available": False}
    mt = float(best(v, target_data).mean())
    mb = float(best(v, b).mean())
    return {"available": True, "counts": {"target": len(target_data), "voynich": len(v), "bsb1784": len(b)}, "mean_best_target": mt, "mean_best_bsb1784": mb, "target_minus_bsb1784": mt - mb, "bundle_sha256": sha(raw), "status_counts": {s: int(np.sum(status == s)) for s in sorted(set(status))}}


def supported(result: dict[str, Any], blind: list[dict[str, Any]]) -> bool:
    if not result.get("available"):
        return False
    ci = result.get("hierarchical_bootstrap_95ci") or [0, 0]
    contradictions = sum(t.get("outcome") == "contradiction" for t in blind)
    confirmations = sum(t.get("outcome") == "confirmation" for t in blind)
    return result.get("primary_effect_target_minus_control_mean", 0) > 0 and ci[0] > 0 and result.get("target_rank_descending") == 1 and contradictions <= confirmations


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    extract_raw = get(EXTRACT).content
    extract = json.loads(extract_raw)
    if extract.get("protocol_id") != PROTOCOL or len(extract.get("records", [])) != 198:
        raise RuntimeError("target extraction is not the frozen 198-record protocol manifest")
    target_roots, target_flowers, target_whole = extraction_items(extract)
    parent_analysis, manifests, masks, parent_vecs, parent_items = load_parent()
    parent_roots = derive_parent_roots(manifests, masks)
    all_new = target_roots + parent_roots + target_flowers + target_whole

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(json.dumps({"event": "load_model", "device": device, "new_items": len(all_new)}), flush=True)
    model = AutoModel.from_pretrained(MODEL, revision=REV, token=os.environ.get("HF_TOKEN"), torch_dtype=dtype).to(device).eval()

    root_items = target_roots + parent_roots
    root_ord, err_ro = embed(root_items, "ordinary_path", model, device)
    root_mask, err_rm = embed(root_items, "masked_path", model, device)
    flower_ord, err_fo = embed(target_flowers, "ordinary_path", model, device)
    flower_mask, err_fm = embed(target_flowers, "masked_path", model, device)
    whole_vec, err_w = embed(target_whole, "path", model, device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embedding_files = [
        save_npz(f"{PREFIX}/embeddings/root_ordinary.npz", root_items, root_ord),
        save_npz(f"{PREFIX}/embeddings/root_masked.npz", root_items, root_mask),
        save_npz(f"{PREFIX}/embeddings/flower_ordinary_target.npz", target_flowers, flower_ord),
        save_npz(f"{PREFIX}/embeddings/flower_masked_target.npz", target_flowers, flower_mask),
        save_npz(f"{PREFIX}/embeddings/whole_target.npz", target_whole, whole_vec),
    ]

    rng = np.random.default_rng(SEED)
    results: dict[str, Any] = {}
    datasets: dict[str, tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]] = {}

    for strict in (True, False):
        suffix = "strict" if strict else "broad"
        d, m = data_from_items(root_items, root_ord, strict, "ordinary")
        datasets[f"roots_ordinary_{suffix}"] = (d, m)
        results[f"roots_ordinary_{suffix}"] = analyse(f"roots_ordinary_{suffix}", d, m, rng)
        d, m = data_from_items(root_items, root_mask, strict, "masked")
        datasets[f"roots_masked_{suffix}"] = (d, m)
        results[f"roots_masked_{suffix}"] = analyse(f"roots_masked_{suffix}", d, m, rng)

        pf, pm = parent_flower_data(parent_vecs, parent_items, broad=not strict)
        td, tm = merge_target_with_parent(target_flowers, flower_ord, pf, pm, strict)
        datasets[f"flowers_ordinary_{suffix}"] = (td, tm)
        results[f"flowers_ordinary_{suffix}"] = analyse(f"flowers_ordinary_{suffix}", td, tm, rng)
        td, tm = merge_target_with_parent(target_flowers, flower_mask, pf, pm, strict)
        datasets[f"flowers_masked_{suffix}"] = (td, tm)
        results[f"flowers_masked_{suffix}"] = analyse(f"flowers_masked_{suffix}", td, tm, rng)

    root_strict_data, root_strict_meta = datasets["roots_ordinary_strict"]
    flower_strict_data, flower_strict_meta = datasets["flowers_ordinary_strict"]
    root_blind, root_key, root_sheets = create_blind_pack("roots", results["roots_ordinary_strict"], root_strict_data, root_strict_meta, rng, 1)
    flower_blind, flower_key, flower_sheets = create_blind_pack("flowers", results["flowers_ordinary_strict"], flower_strict_data, flower_strict_meta, rng, 7)
    blind = root_blind + flower_blind
    blind_key = {**root_key, **flower_key}

    target_strict_root = root_strict_data.get(TARGET, np.zeros((0, 4096), np.float32))
    root_crosscheck = crosscheck_dedicated_root(target_strict_root)
    root_ok = supported(results["roots_ordinary_strict"], root_blind)
    flower_ok = supported(results["flowers_ordinary_strict"], flower_blind)
    if root_ok and flower_ok:
        classification = "dual_channel_affinity"
    elif root_ok:
        classification = "root_only_affinity"
    elif flower_ok:
        classification = "flower_only_affinity"
    else:
        r = results["roots_ordinary_strict"].get("primary_effect_target_minus_control_mean")
        f = results["flowers_ordinary_strict"].get("primary_effect_target_minus_control_mean")
        classification = "no_corpus_level_affinity" if r is not None and f is not None and r <= 0 and f <= 0 else "mixed_or_underpowered"

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL,
        "target_extraction_sha256": sha(extract_raw),
        "target_extraction_manifest_sha256": extract.get("extraction_manifest_sha256"),
        "parent_result_sha256": parent_analysis.get("result_sha256"),
        "model_id": MODEL,
        "model_revision": REV,
        "representation": "L2-normalised CLS float32",
        "preprocessing": {"resize": [224, 224], "resample": "bilinear", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "rescale": 1 / 255},
        "seed": SEED,
        "target_counts": {"roots": len(target_roots), "flowers_all": len(target_flowers), "whole": len(target_whole)},
        "derived_parent_root_counts": {c: sum(x["corpus"] == c for x in parent_roots) for c in [VOYNICH] + CONTROLS},
        "derived_parent_root_methods": {m: sum(x["derivation_method"] == m for x in parent_roots) for m in sorted({x["derivation_method"] for x in parent_roots})},
        "embedding_files": embedding_files,
        "embedding_errors": err_ro + err_rm + err_fo + err_fm + err_w,
        "results": results,
        "blind_adjudication": blind,
        "blind_key_sha256": csha(blind_key),
        "dedicated_root_crosscheck": root_crosscheck,
        "root_supported": root_ok,
        "flower_supported": flower_ok,
        "final_classification": classification,
        "claim_limit": "Image-morphology comparison only; no claim of exemplar, lineage, provenance, textual relationship or botanical identity.",
    }
    report["result_sha256"] = csha(report)
    result_data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False).encode()
    key_data = json.dumps(blind_key, indent=2, sort_keys=True, ensure_ascii=False).encode()
    upload(f"{PREFIX}/RESULT.json", "application/json", result_data)
    upload(f"{PREFIX}/blind_adjudication.json", "application/json", json.dumps(blind, indent=2, sort_keys=True).encode())
    upload(f"{PREFIX}/blind_key.json", "application/json", key_data)

    rows = []
    for name, r in results.items():
        rows.append(
            {
                "channel": name,
                "available": r.get("available"),
                "balanced_n": r.get("balanced_n"),
                "effect": r.get("primary_effect_target_minus_control_mean"),
                "ci_low": (r.get("hierarchical_bootstrap_95ci") or [None, None])[0],
                "ci_high": (r.get("hierarchical_bootstrap_95ci") or [None, None])[1],
                "rank": r.get("target_rank_descending"),
                "rank_p_upper": r.get("manuscript_label_rank_p_upper"),
                "eligible_controls": ";".join(r.get("eligible_controls", [])),
            }
        )
    csv_buf = io.StringIO()
    w = csv.DictWriter(csv_buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    summary_csv = csv_buf.getvalue().encode()
    upload(f"{PREFIX}/channel_summary.csv", "text/csv", summary_csv)

    def line(channel: str) -> str:
        r = results[channel]
        ci = r.get("hierarchical_bootstrap_95ci") or [None, None]
        return f"- {channel}: effect `{r.get('primary_effect_target_minus_control_mean')}`, 95% CI `{ci[0]}` to `{ci[1]}`, rank `{r.get('target_rank_descending')}`, rank p-upper `{r.get('manuscript_label_rank_p_upper')}`, balanced n `{r.get('balanced_n')}`."

    final_md = "\n".join(
        [
            "# BSB Clm 5905 ↔ Voynich roots and flowers — final report",
            "",
            f"Protocol: `{PROTOCOL}`",
            f"Final classification: **`{classification}`**",
            "",
            "## Primary ordinary-crop results",
            "",
            line("roots_ordinary_strict"),
            line("flowers_ordinary_strict"),
            "",
            "## Broad and masked sensitivities",
            "",
            line("roots_ordinary_broad"),
            line("roots_masked_strict"),
            line("flowers_ordinary_broad"),
            line("flowers_masked_strict"),
            "",
            "## Blind stage",
            "",
            f"- Root trials: {len(root_blind)}; confirmations {sum(x.get('outcome') == 'confirmation' for x in root_blind)}, contradictions {sum(x.get('outcome') == 'contradiction' for x in root_blind)}, ties/abstentions {sum(x.get('outcome') in {'tie','abstain'} for x in root_blind)}.",
            f"- Flower trials: {len(flower_blind)}; confirmations {sum(x.get('outcome') == 'confirmation' for x in flower_blind)}, contradictions {sum(x.get('outcome') == 'contradiction' for x in flower_blind)}, ties/abstentions {sum(x.get('outcome') in {'tie','abstain'} for x in flower_blind)}.",
            "",
            "## Interpretation limit",
            "",
            "This is a controlled image-morphology result. It does not establish a source, exemplar, historical lineage, provenance, textual relationship or botanical identity. Individual attractive pairs remain descriptive unless the manuscript-level result supports them.",
            "",
            f"Result SHA-256: `{report['result_sha256']}`",
        ]
    )
    upload(f"{PREFIX}/FINAL_REPORT.md", "text/markdown", final_md.encode())

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("00_PROTOCOL/PROTOCOL_ID.txt", PROTOCOL + "\n")
        zf.writestr("01_RESULTS/RESULT.json", result_data)
        zf.writestr("01_RESULTS/channel_summary.csv", summary_csv)
        zf.writestr("01_RESULTS/FINAL_REPORT.md", final_md)
        zf.writestr("02_BLIND/blind_adjudication.json", json.dumps(blind, indent=2, sort_keys=True))
        zf.writestr("02_BLIND/blind_key.json", key_data)
        for i, raw in enumerate(root_sheets + flower_sheets, 1):
            zf.writestr(f"02_BLIND/sheet_{i:02d}.png", raw)
        zf.writestr("03_MANIFESTS/target_extraction_manifest.json", extract_raw)
        zf.writestr("03_MANIFESTS/derived_parent_roots.json", json.dumps(parent_roots, indent=2, sort_keys=True, ensure_ascii=False))
        zf.writestr("README.md", final_md)
    bundle_data = bundle.getvalue()
    bundle_path = f"{PREFIX}/CLM5905_ROOT_FLOWER_RESULTS_CORE.zip"
    upload(bundle_path, "application/zip", bundle_data)
    print(
        "RESULT_JSON="
        + json.dumps(
            {
                "protocol_id": PROTOCOL,
                "classification": classification,
                "result_sha256": report["result_sha256"],
                "bundle_path": bundle_path,
                "bundle_url": BRIDGE + bundle_path,
                "bundle_sha256": sha(bundle_data),
                "bundle_bytes": len(bundle_data),
                "root_primary": results["roots_ordinary_strict"],
                "flower_primary": results["flowers_ordinary_strict"],
                "blind": blind,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
