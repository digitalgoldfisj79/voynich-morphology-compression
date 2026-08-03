#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROTOCOL = "P586-VMS-ROOT-0.1-20260803"
RUN_ID = "ef6a0302-8269-4166-8c1c-63b14abb9c47"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MANIFEST_URL = "https://iiif.archive.org/iiif/bncf-pal.-586-images/manifest.json"
SUPABASE = "https://ymaqlcfjmdwncdbjprmw.supabase.co"
UPLOAD_EP = SUPABASE + "/functions/v1/p586-root-upload-v01"
BRIDGE = SUPABASE + "/storage/v1/object/public/bridge/"
CHECKPOINT_URL = BRIDGE + "p586_root_v01/full_run/checkpoint.json"
RECOVERY_CANVASES = [67, 80, 81, 82, 83, 84, 85]
SEED = 20260803

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "VoynichRootResearch/0.1-recovery"})

PLANT_PROMPT = '''Return JSON only. Identify every botanical plant illustration on this medieval manuscript page and give normalized 0-1000 bounding boxes. Preferred schema: {"plants":[{"plant_index":0,"bbox_1000":[x0,y0,x1,y1],"has_visible_root":true,"confidence":0.0,"description":"brief"}],"page_description":"brief"}. Box each complete illustrated plant from topmost foliage to lowest root. Do not return boxes for handwritten text, initials, borders, stains, human or animal figures, or decoration. Include monochrome and stylised plants. Set has_visible_root true only when a distinct root, bulb, rhizome or branching underground form is actually drawn.'''
ROOT_PROMPT = '''This image is only the lower 60% of one medieval botanical illustration. Return strict JSON only: {"root_bbox_1000":[x0,y0,x1,y1],"visibility":"clear|partial|none","confidence":0.0,"description":"brief"}. Tightly box the entire coherent root, bulb, rhizome or branching underground form and only a minimal stem junction. Exclude all handwritten text, leaves, hands, birds, human or animal figures, stains and page background. If no morphologically interpretable root is visible, return null.'''
QA_PROMPT = '''Image 1 is the complete plant and Image 2 is the tightened proposed root crop. Return strict JSON only: {"status":"accept|partial|reject","confidence":0.0,"root_fraction":0.0,"attached_to_plant":true,"text_fraction":0.0,"description":"brief","reason":"brief"}. Accept if Image 2 contains the coherent interpretable root, bulb or rhizome belonging to Image 1, even with minor parchment or minimal stem. Partial only for significant truncation, obscuration or contamination. Reject non-roots, writing, leaves, hands, birds, figures, plain stems or unrelated fragments.'''


def log(event: str, **kwargs) -> None:
    print(json.dumps({"event": event, "time": time.time(), **kwargs}, sort_keys=True), flush=True)


def get(url: str, tries: int = 5, timeout: int = 120) -> requests.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            response = SESSION.get(url, timeout=timeout)
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
                timeout=240,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last = exc
            time.sleep(min(15, 1.7**attempt))
    raise RuntimeError(f"UPLOAD failed {path}: {last}")


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def jpg_bytes(image: Image.Image, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json(text: str):
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    for opener, closer in [("{", "}"), ("[", "]")]:
        starts = [m.start() for m in re.finditer(re.escape(opener), cleaned)]
        for start in starts:
            depth = 0
            in_string = False
            escaped = False
            for index, char in enumerate(cleaned[start:], start):
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == opener:
                        depth += 1
                    elif char == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cleaned[start : index + 1])
                            except Exception:
                                break
    return None


def unwrap(value):
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], (dict, list)):
        value = value[0]
    return value


def numeric(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def infer(model, processor, images: list[Image.Image], prompt: str, max_new_tokens: int):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return unwrap(parse_json(raw)), raw


def canvases(manifest: dict) -> list:
    return manifest.get("items") or (manifest.get("sequences") or [{}])[0].get("canvases") or []


def canvas_label(canvas: dict, index: int) -> str:
    value = canvas.get("label") or {}
    if isinstance(value, dict):
        return " ".join(str(item) for entries in value.values() for item in (entries if isinstance(entries, list) else [entries]))
    return str(value or index)


def image_url(canvas: dict, width: int = 1600) -> str | None:
    for mode in (3, 2):
        try:
            body = canvas["items"][0]["items"][0]["body"] if mode == 3 else canvas["images"][0]["resource"]
            service = body.get("service")
            if isinstance(service, list):
                service = service[0] if service else None
            if isinstance(service, dict):
                base = service.get("id") or service.get("@id")
                if base:
                    return base.rstrip("/") + f"/full/{width},/0/default.jpg"
            direct = body.get("id") or body.get("@id")
            if direct:
                return direct
        except Exception:
            pass
    return None


def normalise_box(value, width: int, height: int, minimum_pixels: int = 30):
    numbers: list[float] = []
    if isinstance(value, str):
        numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", value)[:4]]
    elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", value[0])[:4]]
    elif isinstance(value, (list, tuple)):
        for item in value:
            try:
                numbers.append(float(item))
            except Exception:
                pass
            if len(numbers) == 4:
                break
    if len(numbers) != 4:
        return None
    if max(numbers) <= 1.5:
        numbers = [x * 1000 for x in numbers]
    x0, y0, x1, y1 = [max(0, min(1000, x)) for x in numbers]
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return None
    box = (
        round(x0 * width / 1000),
        round(y0 * height / 1000),
        round(x1 * width / 1000),
        round(y1 * height / 1000),
    )
    if box[2] - box[0] < minimum_pixels or box[3] - box[1] < minimum_pixels:
        return None
    return box


def expand(box, width: int, height: int, horizontal: float = 0.05, top: float = 0.03, bottom: float = 0.12):
    x0, y0, x1, y1 = box
    box_width = x1 - x0
    box_height = y1 - y0
    return (
        max(0, round(x0 - box_width * horizontal)),
        max(0, round(y0 - box_height * top)),
        min(width, round(x1 + box_width * horizontal)),
        min(height, round(y1 + box_height * bottom)),
    )


def pad(box, width: int, height: int, amount: float = 0.08):
    x0, y0, x1, y1 = box
    dx = (x1 - x0) * amount
    dy = (y1 - y0) * amount
    return (
        max(0, round(x0 - dx)),
        max(0, round(y0 - dy)),
        min(width, round(x1 + dx)),
        min(height, round(y1 + dy)),
    )


def intersection_over_union(first, second) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union else 0.0


def box_json(box) -> dict:
    return {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]}


def plants_from(answer) -> list[dict]:
    value = unwrap(answer)
    if isinstance(value, dict):
        value = value.get("plants", value.get("objects", []))
    value = unwrap(value)
    if isinstance(value, dict):
        value = value.get("plants", value.get("objects", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def qa_normalise(answer) -> tuple[str, float, float, str]:
    if not isinstance(answer, dict):
        return "reject", 0.0, 0.0, "non-object QA response"
    raw_status = str(answer.get("status", "reject")).lower()
    confidence = numeric(answer.get("confidence"))
    root_fraction = numeric(answer.get("root_fraction"))
    text = (str(answer.get("description", "")) + " " + str(answer.get("reason", ""))).lower()
    contradictions = [
        "does not contain",
        "no visible root",
        "no coherent",
        "not the root",
        "shows a close-up of a leaf",
        "shows a hand",
        "shows a bird",
        "human figure",
        "animal figure",
        "unrelated fragment",
        "plain stem",
        "no discernible root",
    ]
    if any(phrase in text for phrase in contradictions):
        return "reject", confidence, root_fraction, str(answer.get("reason") or answer.get("description") or "explicit contradiction")
    if raw_status == "accept" and confidence >= 0.80 and root_fraction >= 0.30:
        return "accept", confidence, root_fraction, str(answer.get("reason") or "")
    if raw_status in {"accept", "partial", "accept|partial"} and confidence >= 0.50 and root_fraction >= 0.25:
        return "partial", confidence, root_fraction, str(answer.get("reason") or "")
    return "reject", confidence, root_fraction, str(answer.get("reason") or "failed QA gate")


def process_canvas(index: int, canvas: dict, model, processor) -> tuple[dict, list[dict]]:
    source_url = image_url(canvas)
    record = {
        "canvas_index": index,
        "canvas_label": canvas_label(canvas, index),
        "source_image_url": source_url,
        "clip_score": None,
        "included": False,
        "inclusion_reason": "no accepted or partial roots",
        "metadata": {"recovered_after_auth_failure": True},
    }
    proposals: list[dict] = []
    raw_page = get(source_url).content
    page = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_page))).convert("RGB")
    record.update(image_width=page.width, image_height=page.height, sha256=sha256(raw_page))
    page_answer, _ = infer(model, processor, [page], PLANT_PROMPT, 750)
    plants = plants_from(page_answer)
    record["metadata"]["plants_detected_raw"] = len(plants)
    seen = []
    root_index = 0
    for plant_position, plant_record in enumerate(plants):
        plant_box = normalise_box(plant_record.get("bbox_1000") or plant_record.get("bbox"), page.width, page.height, 60)
        if not plant_box or any(intersection_over_union(plant_box, existing) > 0.65 for existing in seen):
            continue
        seen.append(plant_box)
        context_box = expand(plant_box, page.width, page.height)
        plant_image = page.crop(context_box)
        width, height = plant_image.size
        lower_y = round(height * 0.40)
        lower = plant_image.crop((0, lower_y, width, height))
        root_answer, _ = infer(model, processor, [lower], ROOT_PROMPT, 400)
        relative_box = normalise_box(root_answer.get("root_bbox_1000") if isinstance(root_answer, dict) else None, lower.width, lower.height, 20)
        if not relative_box:
            continue
        relative_box = pad(relative_box, lower.width, lower.height, 0.08)
        root_image = lower.crop(relative_box)
        qa_answer, _ = infer(model, processor, [plant_image, root_image], QA_PROMPT, 400)
        status, confidence, root_fraction, reason = qa_normalise(qa_answer)
        root_in_plant = (
            relative_box[0],
            lower_y + relative_box[1],
            relative_box[2],
            lower_y + relative_box[3],
        )
        root_page_box = (
            context_box[0] + root_in_plant[0],
            context_box[1] + root_in_plant[1],
            context_box[0] + root_in_plant[2],
            context_box[1] + root_in_plant[3],
        )
        stem = f"c{index:03d}_p{plant_position:02d}_r{root_index:02d}"
        crop_path = f"p586_root_v01/full_run/crops/{stem}.png"
        context_path = f"p586_root_v01/full_run/context/{stem}_plant.jpg"
        crop_bytes = png_bytes(root_image)
        upload(crop_path, "image/png", crop_bytes)
        upload(context_path, "image/jpeg", jpg_bytes(plant_image))
        proposals.append(
            {
                "canvas_index": index,
                "plant_index": int(plant_record.get("plant_index", plant_position)) if str(plant_record.get("plant_index", plant_position)).lstrip("-").isdigit() else plant_position,
                "root_index": root_index,
                "bbox": box_json(root_page_box),
                "context_bbox": box_json(context_box),
                "detector": MODEL_ID + "/whole-plant-lower60-tight-v1",
                "detector_score": numeric(root_answer.get("confidence")) if isinstance(root_answer, dict) else 0.0,
                "crop_path": crop_path,
                "context_crop_path": context_path,
                "qa_status": status,
                "qa_reason": reason,
                "image_sha256": sha256(crop_bytes),
                "embedding_model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                "embedding_dim": None,
                "embedding_path": "p586_root_v01/embeddings/all_roots_dinov3_vit7b16.npz",
                "qa_confidence": confidence,
                "root_fraction": root_fraction,
                "visibility": str(root_answer.get("visibility") or "") if isinstance(root_answer, dict) else "",
                "metadata": {
                    "plant_bbox": box_json(plant_box),
                    "plant_description": plant_record.get("description"),
                    "plant_has_visible_root": plant_record.get("has_visible_root"),
                    "plant_confidence": numeric(plant_record.get("confidence")),
                    "root_description": root_answer.get("description") if isinstance(root_answer, dict) else None,
                    "root_raw_confidence": numeric(root_answer.get("confidence")) if isinstance(root_answer, dict) else 0.0,
                    "qa_raw": qa_answer,
                    "qa_rule": "explicit-contradiction override; accept>=.80/.30; partial>=.50/.25",
                    "lower_band_fraction": 0.60,
                    "recovered_after_auth_failure": True,
                },
            }
        )
        root_index += 1
    record["metadata"]["plants_unique"] = len(seen)
    record["metadata"]["proposals"] = len(proposals)
    record["metadata"]["accept"] = sum(item["qa_status"] == "accept" for item in proposals)
    record["metadata"]["partial"] = sum(item["qa_status"] == "partial" for item in proposals)
    record["metadata"]["reject"] = sum(item["qa_status"] == "reject" for item in proposals)
    record["included"] = record["metadata"]["accept"] + record["metadata"]["partial"] > 0
    record["inclusion_reason"] = "accepted or sensitivity root present" if record["included"] else "no accepted or partial roots"
    return record, proposals


def main() -> None:
    checkpoint = get(CHECKPOINT_URL).json()
    pages = [item for item in checkpoint.get("pages", []) if int(item.get("canvas_index", -1)) not in RECOVERY_CANVASES]
    proposals = [item for item in checkpoint.get("proposals", []) if int(item.get("canvas_index", -1)) not in RECOVERY_CANVASES]
    manifest = get(MANIFEST_URL).json()
    all_canvases = canvases(manifest)
    log("load_model_start", model=MODEL_ID, recovery_canvases=RECOVERY_CANVASES)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    revision = getattr(model.config, "_commit_hash", None)
    for index in RECOVERY_CANVASES:
        try:
            page_record, page_proposals = process_canvas(index, all_canvases[index], model, processor)
        except Exception as exc:
            page_record = {
                "canvas_index": index,
                "canvas_label": canvas_label(all_canvases[index], index),
                "source_image_url": image_url(all_canvases[index]),
                "included": False,
                "inclusion_reason": "recovery failure",
                "metadata": {"recovery_error": f"{type(exc).__name__}: {exc}"},
            }
            page_proposals = []
        pages.append(page_record)
        proposals.extend(page_proposals)
        log(
            "canvas_recovered",
            canvas=index,
            accept=sum(item["qa_status"] == "accept" for item in page_proposals),
            partial=sum(item["qa_status"] == "partial" for item in page_proposals),
            reject=sum(item["qa_status"] == "reject" for item in page_proposals),
        )
    pages.sort(key=lambda item: int(item.get("canvas_index", -1)))
    proposals.sort(key=lambda item: (int(item.get("canvas_index", -1)), int(item.get("plant_index", -1)), int(item.get("root_index", -1))))
    output = {
        "protocol_id": PROTOCOL,
        "run_id": RUN_ID,
        "recovery_model": MODEL_ID,
        "recovery_model_revision": revision,
        "recovery_canvases": RECOVERY_CANVASES,
        "pages": pages,
        "proposals": proposals,
        "counts": {
            "pages": len(pages),
            "proposals": len(proposals),
            "accept": sum(item.get("qa_status") == "accept" for item in proposals),
            "partial": sum(item.get("qa_status") == "partial" for item in proposals),
            "reject": sum(item.get("qa_status") == "reject" for item in proposals),
        },
    }
    canonical = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    output["result_sha256"] = sha256(canonical)
    encoded = json.dumps(output, indent=2, sort_keys=True).encode()
    upload("p586_root_v01/full_run/checkpoint_recovered.json", "application/json", encoded)
    upload("p586_root_v01/full_run/recovery_report.json", "application/json", encoded)
    print("RESULT_JSON=" + json.dumps({"counts": output["counts"], "sha256": output["result_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
