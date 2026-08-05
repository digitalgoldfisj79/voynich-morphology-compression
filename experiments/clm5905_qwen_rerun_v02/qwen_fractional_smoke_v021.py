#!/usr/bin/env python3
"""Stratified smoke test for corrected Qwen component extraction.

The sole coordinate contract is bbox_frac=[x1,y1,x2,y2] in [0,1].
No similarity calculation occurs in this script.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any

import requests
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROTOCOL = "CLM5905-VMS-QWEN-RF-0.2-20260805"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
BRIDGE = "https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public/bridge/"
FAILED_MANIFEST = BRIDGE + "clm5905_v02/extraction/extraction_manifest_frozen.json"
UPLOAD = "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/clm5905-upload-v02"
PREFIX = "clm5905_v02/corrected_smoke_v021"
CORPUS_ORDER = [
    "clm5905", "voynich", "bnf_lat_6862", "bnf_gr_2179",
    "herb_18f0aa144a2b", "herb_78e2bbc79062", "bsb1784",
]
SAMPLE_N = {"clm5905": 12, "voynich": 12}
S = requests.Session()
S.headers["User-Agent"] = "CLM5905-QWEN-SMOKE/0.2.1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str, tries: int = 5, timeout: int = 180) -> bytes:
    last: Exception | None = None
    for k in range(tries):
        try:
            r = S.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:
            last = exc
            time.sleep(min(15, 1.8**k))
    raise RuntimeError(f"GET failed {url}: {last}")


def upload(path: str, content_type: str, data: bytes) -> None:
    payload = {
        "path": path,
        "content_type": content_type,
        "data_b64": base64.b64encode(data).decode("ascii"),
    }
    r = S.post(UPLOAD, json=payload, timeout=300)
    r.raise_for_status()
    ans = r.json()
    if ans.get("error"):
        raise RuntimeError(str(ans))


def json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object")
        text = text[start : end + 1]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


def contain(im: Image.Image, max_side: int) -> Image.Image:
    im = im.convert("RGB")
    scale = min(1.0, max_side / max(im.size))
    if scale < 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
    return im


def evenly_spaced(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return rows
    idx = sorted({round(i * (len(rows) - 1) / (n - 1)) for i in range(n)})
    return [rows[i] for i in idx]


def qwen_json(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], str]:
    shown = contain(image, 1400)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": shown},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return json_from_text(raw), raw


def localization_prompt(width: int, height: int) -> str:
    return f"""You are examining one medieval herbal plant illustration.
The supplied image has aspect ratio {width}:{height}. Return ONLY one JSON object.

Coordinate contract: every box MUST be bbox_frac=[x1,y1,x2,y2] as DECIMAL FRACTIONS from 0.0 to 1.0 relative to this supplied image. [0,0] is top-left and [1,1] is bottom-right. Do not return pixel coordinates and do not use a 0-1000 grid.

Find homologous botanical components only:
1. root_system: the complete visible root, rhizome, bulb, or rootstock below the stem/root transition. Exclude basal leaves and arbitrary coloured regions. Return null when no visible root system exists.
2. strict flowers: only an open flower, flower_head, or inflorescence. Exclude buds, fruits, berries, seed heads, leaves, and stems. Return each distinct visible structure, maximum 6.

Use tight boxes but include the whole component. JSON schema:
{{"root_system": null OR {{"bbox_frac":[0.0,0.0,0.0,0.0],"confidence":0.0,"description":""}}, "flowers":[{{"bbox_frac":[0.0,0.0,0.0,0.0],"class":"flower|flower_head|inflorescence","confidence":0.0,"description":""}}], "notes":""}}"""


def qa_prompt(channel: str, proposed_class: str) -> str:
    if channel == "root":
        definition = "a visible root system, rhizome, bulb, or rootstock is the main subject; a small stem junction is allowed"
        valid = "root_system"
    else:
        definition = "an open flower, flower head, or inflorescence is the main subject; buds, fruits, berries, seed heads, leaves, and stems are not valid"
        valid = "flower, flower_head, or inflorescence"
    return f"""Independently quality-control this crop from a medieval herbal.
Requested channel: {channel}. Proposed class: {proposed_class}.
Accept only when {definition}.
Return ONLY JSON:
{{"accept":true|false,"class":"root_system|flower|flower_head|inflorescence|bud|fruit|seed_head|leaf|stem|text|blank|ambiguous","confidence":0.0,"component_fraction":0.0,"contamination":["leaf|stem|text|other"],"reason":""}}
The class must describe what is actually visible, not what was proposed."""


def frac_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, value)
    except Exception:
        return None
    if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def padded_crop(im: Image.Image, box: tuple[float, float, float, float], pad: float = 0.06) -> tuple[Image.Image, list[int]]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy)
    px = [round(x1 * im.width), round(y1 * im.height), round(x2 * im.width), round(y2 * im.height)]
    px[2] = max(px[2], px[0] + 2)
    px[3] = max(px[3], px[1] + 2)
    return im.crop(tuple(px)), px


def geometry_ok(channel: str, box: tuple[float, float, float, float]) -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    cy = (y1 + y2) / 2
    if channel == "root":
        if not 0.004 <= area <= 0.45:
            return False, f"root area {area:.4f} outside gate"
        if cy < 0.50:
            return False, f"root centre y {cy:.3f} above lower-half gate"
    else:
        if not 0.0004 <= area <= 0.28:
            return False, f"flower area {area:.4f} outside gate"
    return True, "pass"


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.convert("RGB").save(b, "PNG", optimize=True)
    return b.getvalue()


def contact_sheet(entries: list[dict[str, Any]]) -> bytes:
    thumbs: list[tuple[Image.Image, str]] = []
    for e in entries:
        try:
            im = Image.open(io.BytesIO(get(BRIDGE + e["crop_path"]))).convert("RGB")
            im = ImageOps.contain(im, (300, 240))
            label = f'{e["corpus"]} · {e.get("folio") or e.get("source_id")} · {e["channel"]} · {e["qa"]["class"]} · {e["qa"]["confidence"]:.2f}'
            thumbs.append((im, label))
        except Exception:
            pass
    cols, cell_w, cell_h = 3, 330, 295
    rows = max(1, math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(thumbs):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im, (x + (cell_w - im.width) // 2, y + 5))
        draw.text((x + 7, y + 250), label[:52], fill="black")
    return png(sheet)


def main() -> None:
    failed = json.loads(get(FAILED_MANIFEST))
    by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in failed["records"]:
        by_corpus[r["corpus"]].append(r)
    sources: list[dict[str, Any]] = []
    for corpus in CORPUS_ORDER:
        rows = by_corpus[corpus]
        n = SAMPLE_N.get(corpus, min(6, len(rows)))
        sources.extend(evenly_spaced(rows, n))

    print(json.dumps({"event": "load_model", "sources": len(sources), "per_corpus": dict(Counter(x["corpus"] for x in sources))}), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa", token=os.environ.get("HF_TOKEN"),
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, token=os.environ.get("HF_TOKEN"))

    records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for si, src in enumerate(sources, 1):
        rec: dict[str, Any] = {
            "corpus": src["corpus"], "source_id": src["source_id"],
            "folio": src.get("folio"), "label": src.get("label"),
            "source_path": src["source_path"], "accepted": [], "rejected": [], "errors": [],
        }
        try:
            raw = get(BRIDGE + src["source_path"])
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            rec["source_size"] = list(image.size)
            rec["source_sha256"] = sha(raw)
            loc, loc_raw = qwen_json(model, processor, image, localization_prompt(*image.size), 700)
            rec["localization"] = loc
            rec["localization_raw"] = loc_raw
            proposals: list[tuple[str, str, Any, int, dict[str, Any]]] = []
            root = loc.get("root_system")
            if isinstance(root, dict):
                proposals.append(("root", "root_system", root.get("bbox_frac"), 0, root))
            for j, flower in enumerate(loc.get("flowers", []) if isinstance(loc.get("flowers"), list) else []):
                if isinstance(flower, dict):
                    proposals.append(("flower", str(flower.get("class", "ambiguous")), flower.get("bbox_frac"), j, flower))

            for channel, proposed_class, raw_box, proposal_index, proposal in proposals:
                box = frac_box(raw_box)
                base = {"channel": channel, "proposed_class": proposed_class, "proposal_index": proposal_index, "bbox_frac_raw": raw_box}
                if box is None:
                    rec["rejected"].append({**base, "reason": "bbox_frac contract failure"})
                    continue
                ok, reason = geometry_ok(channel, box)
                if not ok:
                    rec["rejected"].append({**base, "bbox_frac": list(box), "reason": reason})
                    continue
                crop, pixels = padded_crop(image, box)
                try:
                    qa, qa_raw = qwen_json(model, processor, crop, qa_prompt(channel, proposed_class), 350)
                except Exception as exc:
                    rec["rejected"].append({**base, "bbox_frac": list(box), "bbox_pixels": pixels, "reason": f"QA error: {exc}"})
                    continue
                actual = str(qa.get("class", "ambiguous"))
                valid_class = actual == "root_system" if channel == "root" else actual in {"flower", "flower_head", "inflorescence"}
                accept = bool(qa.get("accept")) and valid_class and float(qa.get("confidence", 0)) >= 0.75 and float(qa.get("component_fraction", 0)) >= 0.35
                detail = {**base, "bbox_frac": list(box), "bbox_pixels": pixels, "qa": qa, "qa_raw": qa_raw, "proposal": proposal}
                if not accept:
                    rec["rejected"].append({**detail, "reason": "independent Qwen crop-QA rejection"})
                    continue
                item_id = f'{src["corpus"]}__{src["source_id"]}__{channel}_{proposal_index:02d}'
                data = png(crop)
                path = f"{PREFIX}/crops/{src['corpus']}/{channel}/{item_id}.png"
                upload(path, "image/png", data)
                item = {
                    **detail, "item_id": item_id, "corpus": src["corpus"],
                    "source_id": src["source_id"], "folio": src.get("folio"),
                    "label": src.get("label"), "crop_path": path,
                    "crop_sha256": sha(data), "crop_size": list(crop.size),
                }
                rec["accepted"].append(item)
                accepted.append(item)
        except Exception as exc:
            err = {"corpus": src["corpus"], "source_id": src["source_id"], "error": f"{type(exc).__name__}: {exc}"}
            rec["errors"].append(err)
            errors.append(err)
        records.append(rec)
        print(json.dumps({
            "event": "source_done", "n": si, "total": len(sources),
            "corpus": src["corpus"], "source_id": src["source_id"],
            "accepted_roots": sum(x["channel"] == "root" for x in accepted),
            "accepted_flowers": sum(x["channel"] == "flower" for x in accepted),
            "errors": len(errors),
        }), flush=True)

    counts = {c: {k: sum(x["corpus"] == c and x["channel"] == k for x in accepted) for k in ("root", "flower")} for c in CORPUS_ORDER}
    contract_failures = sum(r.get("reason") == "bbox_frac contract failure" for rec in records for r in rec["rejected"])
    controls_root = sum(counts[c]["root"] > 0 for c in CORPUS_ORDER[2:])
    controls_flower = sum(counts[c]["flower"] > 0 for c in CORPUS_ORDER[2:])
    gate = (
        counts["clm5905"]["root"] > 0 and counts["voynich"]["root"] > 0 and
        counts["clm5905"]["flower"] > 0 and counts["voynich"]["flower"] > 0 and
        controls_root >= 4 and controls_flower >= 4 and contract_failures == 0
    )
    manifest = {
        "protocol_id": PROTOCOL, "stage": "corrected_fractional_smoke_v021",
        "model": MODEL, "revision": REVISION, "source_count": len(sources),
        "accepted_counts": counts, "contract_failures": contract_failures,
        "errors": errors, "records": records, "smoke_gate_automatic": gate,
        "visual_gate_pending": True,
    }
    mdata = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode()
    mpath = f"{PREFIX}/smoke_manifest.json"
    upload(mpath, "application/json", mdata)
    sheet = contact_sheet(accepted)
    spath = f"{PREFIX}/accepted_contact_sheet.png"
    upload(spath, "image/png", sheet)
    result = {
        "protocol_id": PROTOCOL, "stage": manifest["stage"], "accepted_counts": counts,
        "contract_failures": contract_failures, "errors": len(errors),
        "smoke_gate_automatic": gate, "visual_gate_pending": True,
        "manifest_path": mpath, "manifest_sha256": sha(mdata),
        "manifest_url": BRIDGE + mpath, "contact_sheet_path": spath,
        "contact_sheet_sha256": sha(sheet), "contact_sheet_url": BRIDGE + spath,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
