#!/usr/bin/env python3
from __future__ import annotations

"""Frozen within-manuscript follow-up for BSB Cod.graec. 201 f.176r.

Historical localization is fixed before DINO evaluation: DBBE/Hajdu place the last
two closing epigrams on f.176r in cryptography. BSB image 401 is therefore the
only target page. The deterministic existing word extractor is reused.

Primary cryptographic-region proxy: bottom TWO detected line clusters on image
401. Sensitivity bounds: bottom ONE and bottom THREE clusters. These definitions
are frozen before computing Voynich similarities. Comparison groups are the
remaining lines on the same page, BSB images 399-400, and the same eight ordinary
Greek controls used in specificity-v1.
"""

import itertools, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

import extract_cpu as ec
import run_specificity_cpu_v1 as cpu

ROOT = Path(__file__).parent
SPEC = json.loads((ROOT / "specificity_manifest_v1.json").read_text())
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
BSB_IMAGE = "https://api.digitale-sammlungen.de/iiif/image/v2/bsb00018791_{:05d}/full/1600,/0/default.jpg"
LINE_TOL_FACTOR = 0.70
SEED = 421


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def centroid(E):
    return norm(np.median(np.asarray(E), axis=0))


def cosdist(a, b):
    return float(1.0 - np.dot(norm(a), norm(b)))


def get_bsb_page(image_number: int):
    im = cpu.fetch_img(BSB_IMAGE.format(image_number))
    bw = ec.ink_mask(im)
    boxes = ec.word_boxes(bw)
    crops = [ec.norm_crop(bw, b) for b in boxes]
    return crops, boxes


def line_clusters(boxes):
    if not boxes:
        return []
    medh = float(np.median([b[3] for b in boxes]))
    tol = max(5.0, LINE_TOL_FACTOR * medh)
    lines = []
    for idx, b in enumerate(boxes):
        yc = b[1] + b[3] / 2.0
        cand = [(abs(yc - L["cy"]), j) for j, L in enumerate(lines)
                if abs(yc - L["cy"]) <= tol]
        if not cand:
            lines.append({"cy": yc, "idx": [idx]})
        else:
            _, j = min(cand)
            lines[j]["idx"].append(idx)
            lines[j]["cy"] = float(np.median([
                boxes[k][1] + boxes[k][3] / 2.0 for k in lines[j]["idx"]
            ]))
    return sorted(lines, key=lambda L: L["cy"])


def embed(images, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).to(device).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(images), batch_size):
            batch = images[s:s+batch_size]
            x = proc(images=batch, return_tensors="pt")
            x = {k: v.to(device) for k, v in x.items()}
            y = model(**x).last_hidden_state[:, 0, :].float().cpu().numpy()
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
            out.append(y)
            print(json.dumps({"event":"embed","done":min(s+batch_size,len(images)),"total":len(images)}), flush=True)
    return np.concatenate(out, axis=0)


def line_subset_stat(E_page, lines, vms_c, chosen):
    chosen = set(chosen)
    cidx = [i for j, L in enumerate(lines) if j in chosen for i in L["idx"]]
    pidx = [i for j, L in enumerate(lines) if j not in chosen for i in L["idx"]]
    cc = centroid(E_page[cidx])
    pc = centroid(E_page[pidx])
    dc = cosdist(vms_c, cc)
    dp = cosdist(vms_c, pc)
    return {
        "cipher_line_ids": sorted(chosen),
        "cipher_crop_indices": cidx,
        "plain_crop_indices": pidx,
        "n_cipher_crops": len(cidx),
        "n_plain_crops": len(pidx),
        "d_vms_cipher": dc,
        "d_vms_same_page_plain": dp,
        "plain_minus_cipher": dp - dc,
    }


def main():
    # Exact specificity-v1 Voynich sample.
    vms_items = cpu.even_take(cpu.iiif_object(SPEC["vms_manifest"], "VOYNICH", SPEC["vms_fractions"]), 270)
    vms = [im for im, _ in vms_items]

    # Same eight ordinary Greek manuscripts as specificity-v1.
    greek_sets = []
    greek_names = []
    for r in SPEC["ordinary"]["GREEK"]:
        items = cpu.even_take(cpu.iiif_object(r["manifest"], "GREEK::" + r["shelfmark"], SPEC["control_page_fractions"]), 90)
        greek_sets.append([im for im, _ in items])
        greek_names.append(r["shelfmark"])

    # Same-codex local controls and cryptographic target page.
    adj399, _ = get_bsb_page(399)
    adj400, _ = get_bsb_page(400)
    page401, boxes401 = get_bsb_page(401)
    lines = line_clusters(boxes401)
    if len(page401) != len(boxes401):
        raise RuntimeError("crop/box mismatch")
    if len(lines) < 4:
        raise RuntimeError(f"unexpected line clustering: {len(lines)} lines")

    print(json.dumps({
        "event":"frozen_layout",
        "image":401,
        "n_crops":len(page401),
        "line_tol_factor":LINE_TOL_FACTOR,
        "lines":[{"id":j,"cy":L["cy"],"n":len(L["idx"]),"idx":L["idx"]} for j,L in enumerate(lines)],
        "primary":"bottom_2_lines",
        "sensitivities":["bottom_1_line","bottom_3_lines"]
    }), flush=True)

    # Embed each physical crop once.
    all_images = []
    slices = {}
    def add(name, ims):
        a = len(all_images); all_images.extend(ims); slices[name] = (a, len(all_images))
    add("VMS", vms)
    add("ADJ", adj399 + adj400)
    add("PAGE401", page401)
    for name, ims in zip(greek_names, greek_sets):
        add("GREEK::" + name, ims)

    E = embed(all_images)
    def sl(name):
        a,b = slices[name]; return E[a:b]

    vms_c = centroid(sl("VMS"))
    adj_c = centroid(sl("ADJ"))
    greek_ms_centroids = [centroid(sl("GREEK::" + n)) for n in greek_names]
    greek_c = norm(np.mean(greek_ms_centroids, axis=0))
    E_page = sl("PAGE401")

    base = {
        "d_vms_adjacent_bsb_399_400": cosdist(vms_c, adj_c),
        "d_vms_ordinary_greek_8ms": cosdist(vms_c, greek_c),
        "ordinary_greek_manuscripts": {n: cosdist(vms_c, c) for n,c in zip(greek_names, greek_ms_centroids)},
    }

    results = {}
    L = len(lines)
    for key, k in [("strict_bottom1",1),("primary_bottom2",2),("broad_bottom3",3)]:
        chosen = tuple(range(L-k, L))
        obs = line_subset_stat(E_page, lines, vms_c, chosen)
        null = []
        for comb in itertools.combinations(range(L), k):
            z = line_subset_stat(E_page, lines, vms_c, comb)
            null.append({"line_ids":list(comb),"delta":z["plain_minus_cipher"]})
        vals = np.array([x["delta"] for x in null], float)
        rank = 1 + int(np.sum(vals > obs["plain_minus_cipher"] + 1e-15))
        ge = int(np.sum(vals >= obs["plain_minus_cipher"] - 1e-15))
        obs.update({
            "d_vms_adjacent_bsb_399_400": base["d_vms_adjacent_bsb_399_400"],
            "d_vms_ordinary_greek_8ms": base["d_vms_ordinary_greek_8ms"],
            "line_subset_exact_n": int(len(vals)),
            "line_subset_rank_desc": rank,
            "line_subset_fraction_ge": ge / len(vals),
            "null_delta_min": float(vals.min()),
            "null_delta_median": float(np.median(vals)),
            "null_delta_max": float(vals.max()),
        })
        results[key] = obs

    p = results["primary_bottom2"]
    strict = results["strict_bottom1"]
    broad = results["broad_bottom3"]
    if (p["d_vms_cipher"] < p["d_vms_same_page_plain"] and
        p["d_vms_cipher"] < p["d_vms_ordinary_greek_8ms"] and
        strict["plain_minus_cipher"] > 0 and broad["plain_minus_cipher"] > 0):
        decision = "CIPHER_ENRICHMENT_ROBUST"
    elif p["plain_minus_cipher"] > 0:
        decision = "CIPHER_ENRICHMENT_FRAGILE"
    else:
        decision = "NO_CIPHER_SPECIFIC_ENRICHMENT"

    out = {
        "protocol":"2026-09-01.gr201-cipherline-followup-v1",
        "representation":"dinov3_b_cls_within_gr201",
        "model":MODEL_ID,
        "revision":REVISION,
        "historical_target":"BSB Cod.graec.201 f.176r / BSB image 401; last two closing epigrams documented as cryptographic",
        "line_definition":{"factor":LINE_TOL_FACTOR,"n_lines":L,"line_sizes":[len(x["idx"]) for x in lines]},
        "base_controls":base,
        "tests":results,
        "decision":decision,
    }
    print("GR201_CIPHERLINE_RESULT=" + json.dumps(out, separators=(",",":")), flush=True)


if __name__ == "__main__":
    main()
