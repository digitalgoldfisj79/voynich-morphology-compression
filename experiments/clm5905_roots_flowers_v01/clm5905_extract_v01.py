#!/usr/bin/env python3
"""Sealed Clm 5905 acquisition and component extraction.

No Voynich image or similarity is opened in this phase. The script freezes the
198 BSB source canvases, calls the fixed component localiser, materialises root,
flower and whole-plant crops, and persists a hash-audited extraction bundle.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps
from pypdf import PdfReader

PROTOCOL = "CLM5905-VMS-RF-0.1-20260804"
MANIFEST_URL = "https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00092488/manifest"
FISCHER_URL = "https://www.zobodat.at/pdf/Berichte-Bayerischen-Bot-Ges-Erforschung-Flora_18_1_0001-0031.pdf"
STRUCTURE_URL = "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/clm5905-structure-v01"
UPLOAD_URL = "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/clm5905-upload-v01"
BRIDGE = "https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public/bridge/"
PREFIX = "clm5905_v01/extraction"
OUT = Path(os.environ.get("CLM_OUT", "/work/clm5905_extract"))
WORKERS = int(os.environ.get("CLM_WORKERS", "4"))
WIDTH = int(os.environ.get("CLM_IIIF_WIDTH", "1600"))
S = requests.Session()
S.headers["User-Agent"] = "CLM5905Morphology/0.1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csha(obj: Any) -> str:
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def get(url: str, tries: int = 6, timeout: int = 240) -> requests.Response:
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


def post_json(url: str, payload: dict[str, Any], tries: int = 6, timeout: int = 300) -> dict[str, Any]:
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
            time.sleep(min(25.0, 1.8**k))
    raise RuntimeError(f"POST failed {url}: {last}")


def upload(path: str, content_type: str, data: bytes) -> None:
    payload = {"path": path, "content_type": content_type, "data_b64": base64.b64encode(data).decode()}
    post_json(UPLOAD_URL, payload, tries=6, timeout=360)


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.convert("RGB").save(b, "PNG", optimize=True)
    return b.getvalue()


def valid_status(value: Any) -> str:
    s = str(value or "uncertain").lower()
    return s if s in {"accept", "partial", "reject", "uncertain"} else "uncertain"


def valid_box(value: Any, w: int, h: int, min_px: int = 12) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        a = [float(x) for x in value]
    except Exception:
        return None
    if max(abs(x) for x in a) <= 1.01:
        x0, y0, x1, y1 = a[0] * w, a[1] * h, a[2] * w, a[3] * h
    elif max(abs(x) for x in a) <= 1000.1:
        x0, y0, x1, y1 = a[0] * w / 1000, a[1] * h / 1000, a[2] * w / 1000, a[3] * h / 1000
    else:
        x0, y0, x1, y1 = a
    x0, x1 = sorted((max(0, min(w, round(x0))), max(0, min(w, round(x1)))))
    y0, y1 = sorted((max(0, min(h, round(y0))), max(0, min(h, round(y1)))))
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return None
    return x0, y0, x1, y1


def expand(box: tuple[int, int, int, int], w: int, h: int, px: float = 0.12, py: float | None = None) -> tuple[int, int, int, int]:
    py = px if py is None else py
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (
        max(0, round(x0 - px * bw)),
        max(0, round(y0 - py * bh)),
        min(w, round(x1 + px * bw)),
        min(h, round(y1 + py * bh)),
    )


def box_dict(box: tuple[int, int, int, int] | None) -> dict[str, int] | None:
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def whiten(im: Image.Image) -> tuple[Image.Image, float, list[float]]:
    """Frozen deterministic parchment-distance foreground sensitivity mask."""
    arr = np.asarray(im.convert("RGB"), dtype=np.int16)
    h, w = arr.shape[:2]
    band = max(2, min(h, w) // 30)
    border = np.concatenate(
        [arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3), arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    dist = np.sqrt(((arr.astype(np.float32) - bg.astype(np.float32)) ** 2).sum(axis=2))
    lum = arr.mean(axis=2)
    fg = (dist >= 18.0) & (lum <= 248)
    out = np.full_like(arr, 255, dtype=np.uint8)
    out[fg] = arr[fg].clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB"), float(fg.mean()), [float(x) for x in bg]


def source_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = get(MANIFEST_URL).content
    manifest = json.loads(raw)
    canvases = manifest["sequences"][0]["canvases"]
    by_label = {str(c.get("label", "")).split()[0]: (i, c) for i, c in enumerate(canvases)}
    rows: list[dict[str, Any]] = []
    for illustration in range(1, 199):
        folio_no = illustration + 94 if illustration <= 83 else illustration + 95
        label = f"{folio_no}r"
        if label not in by_label:
            raise RuntimeError(f"frozen folio missing from IIIF manifest: illustration={illustration} label={label}")
        idx, canvas = by_label[label]
        resource = canvas["images"][0]["resource"]
        service = (resource.get("service") or {}).get("@id")
        if not service or "bsb00092488_" not in service:
            raise RuntimeError(f"unexpected IIIF service for {label}: {service}")
        rows.append(
            {
                "illustration_number": illustration,
                "folio": label,
                "canvas_index": idx,
                "canvas_id": canvas.get("@id"),
                "service_id": service,
                "image_url": f"{service}/full/{WIDTH},/0/default.jpg",
            }
        )
    if len(rows) != 198 or len({r["canvas_index"] for r in rows}) != 198:
        raise RuntimeError("source freeze is not exactly 198 distinct canvases")
    freeze = {
        "protocol_id": PROTOCOL,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": sha(raw),
        "manifest_canvases": len(canvases),
        "image_width_request": WIDTH,
        "selection_rule": "1-83 => 95r-177r; 84-198 => 179r-293r; skip 178r",
        "rows": rows,
    }
    freeze["source_rule_sha256"] = csha(rows)
    return rows, freeze


def fischer_metadata() -> dict[str, Any]:
    raw = get(FISCHER_URL).content
    reader = PdfReader(io.BytesIO(raw))
    page_text = [(p.extract_text() or "").replace("\x00", "") for p in reader.pages]
    text = "\n".join(page_text)
    # Best-effort numbered excerpts. They are annotations only and never gates.
    hits = list(re.finditer(r"(?m)^\s*([1-9]\d{0,2})\s*\.\s*$", text))
    excerpts: dict[str, str] = {}
    for i, m in enumerate(hits):
        n = int(m.group(1))
        if 1 <= n <= 198 and str(n) not in excerpts:
            end = hits[i + 1].start() if i + 1 < len(hits) else min(len(text), m.end() + 1500)
            excerpts[str(n)] = re.sub(r"\s+", " ", text[m.end() : end]).strip()[:1200]
    return {
        "source_url": FISCHER_URL,
        "sha256": sha(raw),
        "bytes": len(raw),
        "pages": len(reader.pages),
        "numbered_excerpts_parsed": len(excerpts),
        "excerpts": excerpts,
        "parse_note": "Best-effort PDF text excerpts; uncertainty and OCR/layout defects are preserved. Not used for image selection or inference.",
        "raw_pdf": raw,
        "raw_text": text,
    }


def choose_plant(plants: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not isinstance(plants, list):
        return None, []
    clean = [p for p in plants if isinstance(p, dict)]
    ranked = sorted(
        clean,
        key=lambda p: (
            valid_status(p.get("whole_status")) in {"accept", "partial"},
            float(p.get("whole_confidence") or 0.0),
        ),
        reverse=True,
    )
    return (ranked[0] if ranked else None), clean


def process(row: dict[str, Any]) -> dict[str, Any]:
    n = int(row["illustration_number"])
    raw = get(row["image_url"]).content
    page = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    answer = post_json(
        STRUCTURE_URL,
        {"imageUrl": row["image_url"], "illustration_number": n, "folio": row["folio"]},
        tries=6,
        timeout=300,
    )
    plant, all_plants = choose_plant(answer.get("plants"))
    rec: dict[str, Any] = {
        **row,
        "source_image_sha256": sha(raw),
        "image_width": page.width,
        "image_height": page.height,
        "source_bytes": len(raw),
        "localiser_model": answer.get("model"),
        "localiser_parse_error": bool(answer.get("parse_error")),
        "localiser_cost_usd": float(answer.get("cost_usd") or 0.0),
        "localiser_usage": answer.get("usage"),
        "raw_plant_count": len(all_plants),
        "raw_response": answer,
        "reproductive": [],
    }
    if plant is None:
        rec.update(whole_status="uncertain", root_status="uncertain", exclusion_reason="no parsed plant")
        return rec

    whole_status = valid_status(plant.get("whole_status"))
    whole = valid_box(plant.get("whole_box"), page.width, page.height, 40)
    rec.update(
        whole_status=whole_status,
        whole_confidence=float(plant.get("whole_confidence") or 0.0),
        label=plant.get("label"),
        whole_box=box_dict(whole),
    )
    if whole is not None:
        whole_pad = expand(whole, page.width, page.height, 0.06, 0.06)
        whole_im = page.crop(whole_pad)
        whole_data = png(whole_im)
        path = f"{PREFIX}/whole/clm5905_{n:03d}.png"
        upload(path, "image/png", whole_data)
        rec.update(whole_crop_path=path, whole_crop_sha256=sha(whole_data), whole_crop_box=box_dict(whole_pad))

    root_status = valid_status(plant.get("root_status"))
    root = valid_box(plant.get("root_box"), page.width, page.height, 14)
    rec.update(
        root_status=root_status,
        root_confidence=float(plant.get("root_confidence") or 0.0),
        root_description=plant.get("root_description"),
        root_box=box_dict(root),
    )
    if root is not None:
        root_pad = expand(root, page.width, page.height, 0.12)
        rim = page.crop(root_pad)
        masked, fg, bg = whiten(rim)
        rdata, mdata = png(rim), png(masked)
        rpath = f"{PREFIX}/roots/ordinary/clm5905_{n:03d}.png"
        mpath = f"{PREFIX}/roots/masked/clm5905_{n:03d}.png"
        upload(rpath, "image/png", rdata)
        upload(mpath, "image/png", mdata)
        rec.update(
            root_crop_path=rpath,
            root_crop_sha256=sha(rdata),
            root_masked_path=mpath,
            root_masked_sha256=sha(mdata),
            root_crop_box=box_dict(root_pad),
            root_foreground_fraction=fg,
            root_background_rgb=bg,
        )

    allowed = {"flower", "flower_head", "inflorescence", "bud", "fruit", "seed_head"}
    reproductive = plant.get("reproductive") if isinstance(plant.get("reproductive"), list) else []
    for j, z in enumerate(reproductive[:5]):
        if not isinstance(z, dict):
            continue
        cls = str(z.get("class") or "").lower()
        if cls not in allowed:
            continue
        rb = valid_box(z.get("box"), page.width, page.height, 10)
        status = valid_status(z.get("status"))
        rr: dict[str, Any] = {
            "repro_id": f"clm5905_{n:03d}_r{j:02d}",
            "class": cls,
            "status": status,
            "confidence": float(z.get("confidence") or 0.0),
            "description": z.get("description"),
            "box": box_dict(rb),
        }
        if rb is not None:
            rpad = expand(rb, page.width, page.height, 0.12)
            im = page.crop(rpad)
            masked, fg, bg = whiten(im)
            data, mdata = png(im), png(masked)
            p = f"{PREFIX}/reproductive/ordinary/{rr['repro_id']}.png"
            mp = f"{PREFIX}/reproductive/masked/{rr['repro_id']}.png"
            upload(p, "image/png", data)
            upload(mp, "image/png", mdata)
            rr.update(
                crop_path=p,
                crop_sha256=sha(data),
                masked_path=mp,
                masked_sha256=sha(mdata),
                crop_box=box_dict(rpad),
                foreground_fraction=fg,
                background_rgb=bg,
            )
        rec["reproductive"].append(rr)
    return rec


def save_checkpoint(report: dict[str, Any]) -> None:
    data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False).encode()
    upload(f"{PREFIX}/checkpoint.json", "application/json", data)


def contact_sheet(records: list[dict[str, Any]], key: str, title: str, cols: int = 6) -> bytes:
    thumbs: list[tuple[dict[str, Any], Image.Image]] = []
    for rec in records:
        path = rec.get(key)
        if not path:
            continue
        try:
            im = Image.open(io.BytesIO(get(BRIDGE + path).content)).convert("RGB")
            thumbs.append((rec, im))
        except Exception:
            continue
    cell_w, cell_h = 220, 250
    rows = max(1, math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * cell_w, 44 + rows * cell_h), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 12), title, fill="black")
    for i, (rec, im) in enumerate(thumbs):
        x, y = (i % cols) * cell_w, 44 + (i // cols) * cell_h
        t = ImageOps.contain(im, (cell_w - 12, cell_h - 40))
        sheet.paste(t, (x + (cell_w - t.width) // 2, y))
        d.text((x + 5, y + cell_h - 34), f"#{rec['illustration_number']} {rec.get('folio','')}", fill="black")
        d.text((x + 5, y + cell_h - 18), str(rec.get("root_status") or rec.get("whole_status") or "")[:24], fill="black")
    return png(sheet)


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for r in records:
        rows.append(
            {
                "illustration_number": r.get("illustration_number"),
                "folio": r.get("folio"),
                "label": r.get("label"),
                "whole_status": r.get("whole_status"),
                "whole_confidence": r.get("whole_confidence"),
                "root_status": r.get("root_status"),
                "root_confidence": r.get("root_confidence"),
                "root_description": r.get("root_description"),
                "root_crop_path": r.get("root_crop_path"),
                "root_masked_path": r.get("root_masked_path"),
                "reproductive_count": len(r.get("reproductive", [])),
                "source_image_url": r.get("image_url"),
                "source_image_sha256": r.get("source_image_sha256"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, freeze = source_rows()
    fischer = fischer_metadata()
    freeze["fischer_source"] = {k: v for k, v in fischer.items() if k not in {"raw_pdf", "raw_text", "excerpts"}}
    upload(f"{PREFIX}/source_freeze.json", "application/json", json.dumps(freeze, indent=2, sort_keys=True).encode())
    upload(f"{PREFIX}/fischer_1925.pdf", "application/pdf", fischer["raw_pdf"])
    ident = {k: v for k, v in fischer.items() if k not in {"raw_pdf", "raw_text"}}
    upload(f"{PREFIX}/fischer_identifications.json", "application/json", json.dumps(ident, indent=2, sort_keys=True, ensure_ascii=False).encode())

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL,
        "target": "BSB Clm 5905",
        "source_freeze_sha256": csha(freeze),
        "source": freeze,
        "localiser_endpoint": STRUCTURE_URL,
        "mask_rule": "border-median RGB distance >=18 and mean luminance <=248; white background",
        "records": [],
        "errors": [],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            n = int(row["illustration_number"])
            try:
                completed[n] = fut.result()
            except Exception as exc:
                completed[n] = {**row, "whole_status": "uncertain", "root_status": "uncertain", "error": f"{type(exc).__name__}: {exc}"}
                report["errors"].append({"illustration_number": n, "error": f"{type(exc).__name__}: {exc}"})
            if i % 10 == 0 or i == len(rows):
                report["records"] = [completed[k] for k in sorted(completed)]
                report["progress"] = {"complete": i, "total": len(rows)}
                save_checkpoint(report)
                print(json.dumps({"event": "extract_progress", "complete": i, "errors": len(report["errors"])}), flush=True)

    records = [completed[k] for k in sorted(completed)]
    if len(records) != 198:
        raise RuntimeError(f"expected 198 extraction records, got {len(records)}")
    report["records"] = records
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["counts"] = {
        "illustrations": len(records),
        "whole_accept": sum(r.get("whole_status") == "accept" for r in records),
        "whole_partial": sum(r.get("whole_status") == "partial" for r in records),
        "root_accept": sum(r.get("root_status") == "accept" and r.get("root_crop_path") for r in records),
        "root_partial": sum(r.get("root_status") == "partial" and r.get("root_crop_path") for r in records),
        "flower_accept": sum(
            z.get("status") == "accept" and z.get("class") in {"flower", "flower_head", "inflorescence"} and z.get("crop_path")
            for r in records
            for z in r.get("reproductive", [])
        ),
        "flower_broad": sum(
            z.get("status") in {"accept", "partial"} and z.get("class") in {"flower", "flower_head", "inflorescence", "bud"} and z.get("crop_path")
            for r in records
            for z in r.get("reproductive", [])
        ),
        "errors": len(report["errors"]),
    }
    report["localiser_cost_usd"] = float(sum(float(r.get("localiser_cost_usd") or 0) for r in records))
    report["extraction_manifest_sha256"] = csha({k: v for k, v in report.items() if k != "extraction_manifest_sha256"})

    manifest_data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False).encode()
    upload(f"{PREFIX}/extraction_manifest_frozen.json", "application/json", manifest_data)
    write_csv(records, OUT / "plants.csv")
    root_sheet = contact_sheet(records, "root_crop_path", "BSB Clm 5905 — frozen root crops")
    upload(f"{PREFIX}/ROOT_CONTACT_SHEET.png", "image/png", root_sheet)

    repro_flat = []
    for r in records:
        for z in r.get("reproductive", []):
            repro_flat.append({**r, **z, "root_status": z.get("status"), "root_crop_path": z.get("crop_path")})
    flower_sheet = contact_sheet(repro_flat, "crop_path", "BSB Clm 5905 — frozen reproductive crops")
    upload(f"{PREFIX}/FLOWER_CONTACT_SHEET.png", "image/png", flower_sheet)

    report_md = f"""# BSB Clm 5905 extraction report\n\nProtocol: `{PROTOCOL}`\n\n- Frozen illustrations: {report['counts']['illustrations']}\n- Whole accept / partial: {report['counts']['whole_accept']} / {report['counts']['whole_partial']}\n- Root strict / broad additions: {report['counts']['root_accept']} / {report['counts']['root_partial']}\n- Strict flowers: {report['counts']['flower_accept']}\n- Broad flower structures: {report['counts']['flower_broad']}\n- Extraction errors: {report['counts']['errors']}\n- Recorded localiser cost: USD {report['localiser_cost_usd']:.4f}\n- Extraction manifest SHA-256: `{report['extraction_manifest_sha256']}`\n\nNo Voynich image or similarity was opened in this phase.\n"""
    upload(f"{PREFIX}/EXTRACTION_REPORT.md", "text/markdown", report_md.encode())

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("00_PROTOCOL/PROTOCOL_ID.txt", PROTOCOL + "\n")
        zf.writestr("01_SOURCE/source_freeze.json", json.dumps(freeze, indent=2, sort_keys=True, ensure_ascii=False))
        zf.writestr("01_SOURCE/fischer_identifications.json", json.dumps(ident, indent=2, sort_keys=True, ensure_ascii=False))
        zf.writestr("02_EXTRACTION/extraction_manifest_frozen.json", manifest_data)
        zf.writestr("02_EXTRACTION/plants.csv", (OUT / "plants.csv").read_bytes())
        zf.writestr("03_QA/ROOT_CONTACT_SHEET.png", root_sheet)
        zf.writestr("03_QA/FLOWER_CONTACT_SHEET.png", flower_sheet)
        zf.writestr("EXTRACTION_REPORT.md", report_md)
    bundle_data = bundle.getvalue()
    upload(f"{PREFIX}/CLM5905_EXTRACTION_CORE.zip", "application/zip", bundle_data)
    print(
        "RESULT_JSON="
        + json.dumps(
            {
                "protocol_id": PROTOCOL,
                "counts": report["counts"],
                "manifest_sha256": report["extraction_manifest_sha256"],
                "bundle_sha256": sha(bundle_data),
                "bundle_bytes": len(bundle_data),
                "bundle_url": BRIDGE + f"{PREFIX}/CLM5905_EXTRACTION_CORE.zip",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
