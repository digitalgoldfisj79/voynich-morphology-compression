#!/usr/bin/env python3
"""Koen G r->a uncertain-space physical-gap test.

Uses the existing Voynich DINO baseline measurement primitives (adaptive ink mask
and low-ink vertical cuts) on the Voynichese.com legacy raster/coordinate layer.
ZL 3b boundary labels are parsed separately and joined only after physical gap
measurement. Primary inference is certain r.a vs uncertain r,a, stratified by
folio and by independent VT locator topology (same-word vs cross-word).
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import numpy as np
import requests

SEED = 6037
N_PERM = 9999
LINE_SIM_MIN = 0.45
ZL_URL = "https://www.voynich.nu/data/ZL3b-n.txt"
V_BASE = "https://www.voynichese.com/1/data/folio"
USER_AGENT = "voynich-koen-ra-gap/0.1 (+research; DINO-baseline reuse)"
OUT = Path("results/koen_ra_gap")

PAGE_RE = re.compile(r"^<(?P<folio>f[^>]+)>\s+<!\s*(?P<meta>.*?)>")
LOCUS_RE = re.compile(r"^<(?P<locus>f[^,>]+\.[^,>]+),(?P<kind>[^>]+)>\s*(?P<text>.*)$")
COMMENT_RE = re.compile(r"<!.*?>")
META_RE = re.compile(r"\$(?P<key>[A-Z])=(?P<val>[^\s>]+)")


@dataclass
class ZLLine:
    folio: str
    locus: str
    hand: str
    quire: str
    text: str
    letters: str
    events: list[dict]


@dataclass
class VBox:
    idx: int
    eva: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class VLine:
    boxes: list[VBox]
    letters: str
    char_map: list[tuple[int, int]]


def get(session: requests.Session, url: str, *, binary=False, tries=4):
    last = None
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=60 if not binary else 90)
            r.raise_for_status()
            return r.content if binary else r.text
        except Exception as exc:
            last = exc
            time.sleep(1.25 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def remove_comments(s: str) -> str:
    old = None
    while old != s:
        old = s
        s = COMMENT_RE.sub("", s)
    return s.replace("<%>", "").replace("<$>", "")


def letters_only(s: str) -> str:
    return "".join(ch for ch in s.lower() if "a" <= ch <= "z")


def parse_line_events(text: str) -> tuple[str, list[dict]]:
    t = remove_comments(text).replace("<->", "#")
    pieces = re.split(r"([.,#])", t)
    token_records = []
    offset = 0
    events: list[dict] = []

    for i in range(0, len(pieces), 2):
        raw = pieces[i]
        norm = letters_only(raw)
        plain = bool(re.fullmatch(r"[a-z]+", raw.strip()))
        rec = {"raw": raw, "norm": norm, "plain": plain, "offset": offset}
        token_records.append(rec)
        if plain:
            for m in re.finditer("ra", raw.strip()):
                events.append({
                    "class": "joined",
                    "r_index": offset + m.start(),
                    "a_index": offset + m.start() + 1,
                    "raw_context": raw.strip(),
                })
        offset += len(norm)

    for k in range(len(token_records) - 1):
        sep_idx = 2 * k + 1
        if sep_idx >= len(pieces):
            break
        sep = pieces[sep_idx]
        left = token_records[k]
        right = token_records[k + 1]
        if sep not in (".", ","):
            continue
        if not (left["plain"] and right["plain"]):
            continue
        if left["raw"].strip().endswith("r") and right["raw"].strip().startswith("a"):
            events.append({
                "class": "certain" if sep == "." else "uncertain",
                "r_index": left["offset"] + len(left["norm"]) - 1,
                "a_index": right["offset"],
                "raw_context": left["raw"].strip() + sep + right["raw"].strip(),
            })

    return "".join(r["norm"] for r in token_records), events


def parse_zl(text: str):
    page_meta: dict[str, dict[str, str]] = {}
    by_folio: dict[str, list[ZLLine]] = defaultdict(list)
    current_folio = None
    for raw_line in text.splitlines():
        pm = PAGE_RE.match(raw_line)
        if pm:
            current_folio = pm.group("folio").lower()
            page_meta[current_folio] = {
                m.group("key"): m.group("val") for m in META_RE.finditer(pm.group("meta"))
            }
            continue
        lm = LOCUS_RE.match(raw_line)
        if not lm or current_folio is None or "P" not in lm.group("kind"):
            continue
        letters, events = parse_line_events(lm.group("text").strip())
        if not letters:
            continue
        meta = page_meta.get(current_folio, {})
        by_folio[current_folio].append(ZLLine(
            current_folio, lm.group("locus"), meta.get("H", ""), meta.get("Q", ""),
            lm.group("text").strip(), letters, events,
        ))
    return by_folio, page_meta


def fetch_v_boxes(folio: str) -> list[VBox]:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    payload = json.loads(get(s, f"{V_BASE}/script/{folio}.js"))
    words = [row[0] for row in payload[0]]
    boxes = []
    for idx, row in enumerate(payload[1]):
        word_id, x, y, w, h = row
        boxes.append(VBox(idx, str(words[word_id]).lower(), float(x), float(y), float(w), float(h)))
    return boxes


def build_v_lines(boxes: list[VBox], page_width=636.0, page_height=900.0) -> list[VLine]:
    positive_h = [b.h for b in boxes if b.h > 0]
    med_h = statistics.median(positive_h) if positive_h else 10.0
    baseline_jump = max(4.0 * med_h, 0.06 * page_height)
    lines: list[list[VBox]] = []
    cur: list[VBox] = []
    prev = None
    for b in boxes:
        new = False
        if prev is not None:
            if b.x < prev.x - 0.075 * page_width:
                new = True
            elif abs(b.y - prev.y) > baseline_jump:
                new = True
        if new and cur:
            lines.append(cur)
            cur = []
        cur.append(b)
        prev = b
    if cur:
        lines.append(cur)

    out = []
    for line in lines:
        letters, cmap = [], []
        for bi, b in enumerate(line):
            clean = letters_only(b.eva)
            for ci, ch in enumerate(clean):
                letters.append(ch)
                cmap.append((bi, ci))
        out.append(VLine(line, "".join(letters), cmap))
    return out


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def align_lines(zlines: list[ZLLine], vlines: list[VLine]):
    n, m = len(zlines), len(vlines)
    dp = np.full((n + 1, m + 1), -1e18, dtype=float)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            base = dp[i, j]
            if base < -1e17:
                continue
            if i < n and base - 0.30 > dp[i + 1, j]:
                dp[i + 1, j] = base - 0.30
                bt[i + 1, j] = 1
            if j < m and base - 0.18 > dp[i, j + 1]:
                dp[i, j + 1] = base - 0.18
                bt[i, j + 1] = 2
            if i < n and j < m:
                ss = sim(zlines[i].letters, vlines[j].letters)
                val = base + (2.0 * ss - 0.55)
                if val > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = val
                    bt[i + 1, j + 1] = 3
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 3:
            ss = sim(zlines[i - 1].letters, vlines[j - 1].letters)
            pairs.append((i - 1, j - 1, ss))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        else:
            if i > 0:
                i -= 1
            elif j > 0:
                j -= 1
    return list(reversed(pairs))


def exact_char_mapping(a: str, b: str) -> dict[int, int]:
    sm = SequenceMatcher(None, a, b, autojunk=False)
    mp = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                mp[i1 + off] = j1 + off
    return mp


# Exact DINO-baseline image primitives.
def ink_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def best_vertical_cuts(mask: np.ndarray, segments: int) -> list[int]:
    width = mask.shape[1]
    if segments <= 1 or width < segments * 2:
        return []
    projection = mask.astype(np.float32).sum(axis=0) / 255.0
    target = width / segments
    candidates = list(range(2, width - 2))
    inf = float("inf")
    dp = [[inf] * (width + 1) for _ in range(segments)]
    parent = [[-1] * (width + 1) for _ in range(segments)]
    for x in candidates:
        dp[1][x] = projection[x] + 0.03 * abs(x - target)
    for k in range(2, segments):
        expected = k * target
        for x in candidates:
            if x < 2 * k:
                continue
            best, best_prev = inf, -1
            for px in range(max(2, x - int(2.2 * target)), x - 1):
                if dp[k - 1][px] == inf:
                    continue
                segw = x - px
                score = (
                    dp[k - 1][px] + projection[x]
                    + 0.04 * abs(segw - target) + 0.02 * abs(x - expected)
                )
                if score < best:
                    best, best_prev = score, px
            dp[k][x], parent[k][x] = best, best_prev
    if not candidates:
        return []
    end = min(
        candidates,
        key=lambda x: dp[segments - 1][x] + 0.02 * abs(x - (segments - 1) * target),
    )
    if not math.isfinite(dp[segments - 1][end]):
        return []
    cuts = [end]
    for k in range(segments - 1, 1, -1):
        end = parent[k][end]
        if end < 0:
            return []
        cuts.append(end)
    return sorted(cuts)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2 image decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def word_slot_extent(img: np.ndarray, box: VBox, char_index: int):
    clean = letters_only(box.eva)
    if not clean or not (0 <= char_index < len(clean)):
        return None
    x0 = max(0, int(math.floor(box.x)))
    y0 = max(0, int(math.floor(box.y)))
    x1 = min(img.shape[1], int(math.ceil(box.x + box.w)))
    y1 = min(img.shape[0], int(math.ceil(box.y + box.h)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    rgb = img[y0:y1, x0:x1]
    mask = ink_mask(rgb)
    cuts = best_vertical_cuts(mask, len(clean))
    bounds = [0] + cuts + [mask.shape[1]]
    if len(bounds) != len(clean) + 1:
        return None
    a, b = bounds[char_index], bounds[char_index + 1]
    if b <= a:
        return None
    sub = mask[:, a:b]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return None
    return {
        "left": x0 + a + int(xs.min()),
        "right": x0 + a + int(xs.max()),
        "box_height": float(box.h),
    }


def measure_event(img: np.ndarray, vline: VLine, vr: int, va: int):
    if va != vr + 1:
        return None
    bri, cri = vline.char_map[vr]
    bai, cai = vline.char_map[va]
    br, ba = vline.boxes[bri], vline.boxes[bai]
    er = word_slot_extent(img, br, cri)
    ea = word_slot_extent(img, ba, cai)
    if er is None or ea is None or ea["left"] < er["left"]:
        return None
    gap = max(0, int(ea["left"] - er["right"] - 1))
    scale = max(1e-9, (er["box_height"] + ea["box_height"]) / 2.0)
    topology = "same_word" if bri == bai else "cross_word"
    if topology == "cross_word" and bai != bri + 1:
        return None
    return {
        "gap_px": gap,
        "gap_norm": float(gap / scale),
        "vt_topology": topology,
        "r_word": br.eva,
        "a_word": ba.eva,
        "r_word_index": br.idx,
        "a_word_index": ba.idx,
        "r_slot": cri,
        "a_slot": cai,
        "scale_height": scale,
    }


def group_summary(vals):
    vals = [float(x) for x in vals]
    if not vals:
        return {"n": 0}
    a = np.asarray(vals)
    return {
        "n": len(vals),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "q10": float(np.quantile(a, .1)),
        "q25": float(np.quantile(a, .25)),
        "q75": float(np.quantile(a, .75)),
        "q90": float(np.quantile(a, .9)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def weighted_stratified_effect(rows, value_key, positive="certain", negative="uncertain", topology=True):
    groups = defaultdict(lambda: {positive: [], negative: []})
    for r in rows:
        if r["class"] not in (positive, negative):
            continue
        key = (r["folio"], r["vt_topology"] if topology else "*")
        groups[key][r["class"]].append(float(r[value_key]))
    effects = []
    for key, g in groups.items():
        if not g[positive] or not g[negative]:
            continue
        np_, nn = len(g[positive]), len(g[negative])
        w = np_ * nn / (np_ + nn)
        d = statistics.mean(g[positive]) - statistics.mean(g[negative])
        effects.append((key, d, w, np_, nn))
    if not effects:
        return float("nan"), effects
    return (
        sum(d * w for _, d, w, _, _ in effects) / sum(w for _, _, w, _, _ in effects),
        effects,
    )


def stratified_permutation(rows, value_key, *, positive="certain", negative="uncertain", topology=True, n_perm=N_PERM, seed=SEED):
    keys = sorted({
        (r["folio"], r["vt_topology"] if topology else "*")
        for r in rows if r["class"] in (positive, negative)
    })
    strata = []
    for key in keys:
        rr = [
            r for r in rows
            if r["class"] in (positive, negative)
            and (r["folio"], r["vt_topology"] if topology else "*") == key
        ]
        np_ = sum(r["class"] == positive for r in rr)
        nn = len(rr) - np_
        if np_ == 0 or nn == 0:
            continue
        vals = np.array([float(r[value_key]) for r in rr], dtype=float)
        strata.append((vals, np_, nn, np_ * nn / (np_ + nn), key))
    if not strata:
        return {
            "actual": float("nan"), "null_mean": float("nan"), "null_sd": float("nan"),
            "z": float("nan"), "p_plus1": float("nan"), "ge": 0,
            "n_perm": n_perm, "n_strata": 0,
        }
    actual, _ = weighted_stratified_effect(rows, value_key, positive, negative, topology)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    totalw = sum(s[3] for s in strata)
    for b in range(n_perm):
        num = 0.0
        for vals, np_, nn, w, _ in strata:
            idx = rng.permutation(len(vals))
            num += w * float(vals[idx[:np_]].mean() - vals[idx[np_:]].mean())
        null[b] = num / totalw
    nm = float(null.mean())
    ns = float(null.std(ddof=1))
    ge = int(np.sum(null >= actual))
    return {
        "actual": float(actual), "null_mean": nm, "null_sd": ns,
        "z": float((actual - nm) / ns) if ns > 0 else float("nan"),
        "p_plus1": float((1 + ge) / (1 + n_perm)), "ge": ge,
        "n_perm": n_perm, "n_strata": len(strata),
        "null_q025": float(np.quantile(null, .025)),
        "null_q975": float(np.quantile(null, .975)),
    }


def stratified_auc(rows, value_key, positive="certain", negative="uncertain", topology=True):
    wins = ties = total = nstr = 0
    groups = defaultdict(lambda: {positive: [], negative: []})
    for r in rows:
        if r["class"] in (positive, negative):
            key = (r["folio"], r["vt_topology"] if topology else "*")
            groups[key][r["class"]].append(float(r[value_key]))
    for g in groups.values():
        if not g[positive] or not g[negative]:
            continue
        nstr += 1
        for a in g[positive]:
            for b in g[negative]:
                total += 1
                if a > b:
                    wins += 1
                elif a == b:
                    ties += 1
    return {
        "auc": float((wins + .5 * ties) / total) if total else float("nan"),
        "pairs": total, "n_strata": nstr,
    }


def leave_one_hand_out(rows, value_key):
    out = []
    for h in sorted({r["hand"] for r in rows if r["hand"]}):
        sub = [r for r in rows if r["hand"] != h]
        eff, cells = weighted_stratified_effect(sub, value_key)
        out.append({"excluded_hand": h, "effect": eff, "n": len(sub), "n_strata": len(cells)})
    return out


def hand_effects(rows, value_key):
    out = []
    for h in sorted({r["hand"] for r in rows if r["hand"]}):
        sub = [r for r in rows if r["hand"] == h]
        eff, cells = weighted_stratified_effect(sub, value_key)
        out.append({
            "hand": h,
            "n_certain": sum(r["class"] == "certain" for r in sub),
            "n_uncertain": sum(r["class"] == "uncertain" for r in sub),
            "effect": eff, "n_strata": len(cells),
        })
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    zl_text = get(s, ZL_URL)
    by_folio, _ = parse_zl(zl_text)
    pre_counts = Counter(
        e["class"] for lines in by_folio.values() for ln in lines for e in ln.events
    )
    candidate_folios = sorted(
        f for f, lines in by_folio.items() if any(ln.events for ln in lines)
    )
    print("CHECKPOINT candidate_counts", json.dumps({"counts": dict(pre_counts), "folios": len(candidate_folios)}, sort_keys=True), flush=True)

    box_data, fetch_errors = {}, {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_v_boxes, f): f for f in candidate_folios}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                box_data[f] = fut.result()
            except Exception as exc:
                fetch_errors[f] = str(exc)
    print("CHECKPOINT coordinate_fetch", json.dumps({"ok": len(box_data), "failed": len(fetch_errors)}, sort_keys=True), flush=True)

    mapped_events, line_audit = [], []
    for folio in candidate_folios:
        if folio not in box_data:
            continue
        zlines = by_folio[folio]
        vlines = build_v_lines(box_data[folio])
        pairs = align_lines(zlines, vlines)
        pair_by_z = {zi: (vi, ss) for zi, vi, ss in pairs if ss >= LINE_SIM_MIN}
        for zi, zl in enumerate(zlines):
            if not zl.events:
                continue
            if zi not in pair_by_z:
                line_audit.append({"folio": folio, "locus": zl.locus, "status": "unmapped", "sim": None, "events": len(zl.events)})
                continue
            vi, ss = pair_by_z[zi]
            vl = vlines[vi]
            cmap = exact_char_mapping(zl.letters, vl.letters)
            kept = 0
            for e in zl.events:
                zr, za = e["r_index"], e["a_index"]
                if zr not in cmap or za not in cmap:
                    continue
                vr, va = cmap[zr], cmap[za]
                if vr < 0 or va >= len(vl.letters):
                    continue
                if vl.letters[vr] != "r" or vl.letters[va] != "a" or va != vr + 1:
                    continue
                mapped_events.append({
                    "folio": folio, "locus": zl.locus, "hand": zl.hand,
                    "quire": zl.quire, "class": e["class"],
                    "raw_context": e["raw_context"], "line_sim": ss,
                    "vline_index": vi, "vr": vr, "va": va, "vline": vl,
                })
                kept += 1
            line_audit.append({"folio": folio, "locus": zl.locus, "status": "mapped", "sim": ss, "events": len(zl.events), "mapped_events": kept})
    print("CHECKPOINT char_mapping", json.dumps({"candidate_events": sum(pre_counts.values()), "mapped_events": len(mapped_events), "folios": len({e['folio'] for e in mapped_events})}, sort_keys=True), flush=True)

    needed = sorted({e["folio"] for e in mapped_events})
    images, image_errors = {}, {}
    def fetch_img(f):
        ss = requests.Session()
        ss.headers["User-Agent"] = USER_AGENT
        return decode_image(get(ss, f"{V_BASE}/image/glance/color/large/{f}.jpg", binary=True))
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(fetch_img, f): f for f in needed}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                images[f] = fut.result()
            except Exception as exc:
                image_errors[f] = str(exc)
    print("CHECKPOINT image_fetch", json.dumps({"ok": len(images), "failed": len(image_errors)}, sort_keys=True), flush=True)

    rows = []
    failures = Counter()
    for e in mapped_events:
        img = images.get(e["folio"])
        if img is None:
            failures["missing_image"] += 1
            continue
        meas = measure_event(img, e["vline"], e["vr"], e["va"])
        if meas is None:
            failures["measurement"] += 1
            continue
        row = {k: v for k, v in e.items() if k not in ("vline", "vr", "va")}
        row.update(meas)
        rows.append(row)
    print("CHECKPOINT measured", json.dumps({"n": len(rows), "counts": dict(Counter(r['class'] for r in rows)), "failures": dict(failures)}, sort_keys=True), flush=True)

    groups = {c: group_summary([r["gap_norm"] for r in rows if r["class"] == c]) for c in ("joined", "uncertain", "certain")}
    groups_px = {c: group_summary([r["gap_px"] for r in rows if r["class"] == c]) for c in ("joined", "uncertain", "certain")}
    topology_counts = {
        c: {t: sum(r["class"] == c and r["vt_topology"] == t for r in rows) for t in ("same_word", "cross_word")}
        for c in ("joined", "uncertain", "certain")
    }

    primary = stratified_permutation(rows, "gap_norm", topology=True, seed=SEED)
    primary_px = stratified_permutation(rows, "gap_px", topology=True, seed=SEED + 1)
    no_topology = stratified_permutation(rows, "gap_norm", topology=False, seed=SEED + 2)
    auc = stratified_auc(rows, "gap_norm", topology=True)
    strict = [r for r in rows if r["line_sim"] >= 0.70]
    strict_test = stratified_permutation(strict, "gap_norm", topology=True, seed=SEED + 3)
    cross = [r for r in rows if r["vt_topology"] == "cross_word"]
    cross_test = stratified_permutation(cross, "gap_norm", topology=True, seed=SEED + 4)
    uj = stratified_permutation(rows, "gap_norm", positive="uncertain", negative="joined", topology=True, seed=SEED + 5)
    cj = stratified_permutation(rows, "gap_norm", positive="certain", negative="joined", topology=True, seed=SEED + 6)

    hand = hand_effects(rows, "gap_norm")
    loho = leave_one_hand_out(rows, "gap_norm")

    actual, cells = weighted_stratified_effect(rows, "gap_norm")
    rng = np.random.default_rng(SEED + 10)
    folios = sorted({key[0] for key, _, _, _, _ in cells})
    byf = defaultdict(list)
    for cell in cells:
        byf[cell[0][0]].append(cell)
    boots = []
    if folios:
        for _ in range(5000):
            picks = rng.choice(folios, size=len(folios), replace=True)
            num = den = 0.0
            for f in picks:
                for _, d, w, _, _ in byf[f]:
                    num += d * w
                    den += w
            if den:
                boots.append(num / den)
    boot_ci = [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))] if boots else [float("nan"), float("nan")]

    summary = {
        "protocol": "KOEN_RA_GAP_V01_20260822",
        "source": {
            "zl": ZL_URL, "zl_version": "3b", "voynichese_base": V_BASE,
            "dino_primitives": "adaptiveThreshold Gaussian C / opening 2x2 / low-ink DP cuts",
        },
        "pre_measurement_counts": dict(pre_counts),
        "candidate_folios": len(candidate_folios),
        "coordinate_fetch_errors": fetch_errors,
        "image_fetch_errors": image_errors,
        "mapped_events": len(mapped_events),
        "measured_events": len(rows),
        "measurement_failures": dict(failures),
        "measured_counts": dict(Counter(r["class"] for r in rows)),
        "topology_counts": topology_counts,
        "gap_norm_groups": groups,
        "gap_px_groups": groups_px,
        "primary_certain_minus_uncertain_stratified_folio_vt_topology": primary,
        "primary_raw_px_sensitivity": primary_px,
        "folio_only_no_vt_topology_sensitivity": no_topology,
        "strict_line_similarity_070": strict_test,
        "cross_word_only_vt_agrees_boundary": cross_test,
        "stratified_auc_certain_gt_uncertain": auc,
        "secondary_uncertain_minus_joined": uj,
        "secondary_certain_minus_joined": cj,
        "folio_bootstrap_95ci_primary": boot_ci,
        "hand_effects": hand,
        "leave_one_hand_out": loho,
        "line_alignment": {
            "threshold": LINE_SIM_MIN, "audit_rows": len(line_audit),
            "mapped_lines": sum(a["status"] == "mapped" for a in line_audit),
        },
        "seed": SEED, "permutations": N_PERM,
    }

    loho_positive = all(x["effect"] > 0 for x in loho if math.isfinite(x["effect"])) if loho else False
    ci_positive = boot_ci[0] > 0
    primary_resolved = bool(primary["p_plus1"] <= .05 and primary["z"] >= 2 and ci_positive and loho_positive)
    continuum_resolved = bool(primary_resolved and uj["p_plus1"] <= .05 and uj["z"] >= 2)
    summary["adjudication"] = {
        "primary_physical_distinction": "RESOLVED" if primary_resolved else "NOT_RESOLVED",
        "three_way_continuum": "RESOLVED" if continuum_resolved else "NOT_RESOLVED",
        "criteria": {
            "primary": "p<=.05, z>=2, folio-bootstrap 95% CI >0, all finite leave-one-hand-out effects >0",
            "continuum": "primary + uncertain-minus-joined p<=.05 and z>=2",
        },
    }

    with (OUT / "events.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "folio", "locus", "hand", "quire", "class", "raw_context", "line_sim",
            "vline_index", "gap_px", "gap_norm", "vt_topology", "r_word", "a_word",
            "r_word_index", "a_word_index", "r_slot", "a_slot", "scale_height",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with (OUT / "line_alignment_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for a in line_audit for k in a})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(line_audit)
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")

    md = []
    md.append("# Koen r→a physical-gap test — results\n")
    md.append(f"Measured {len(rows)} events from {len(set(r['folio'] for r in rows))} folios.\n")
    md.append("## Primary: ZL certain `r.a` minus uncertain `r,a`\n")
    md.append(
        f"Normalized stratified effect = **{primary['actual']:.6f}**, null mean {primary['null_mean']:.6f}, "
        f"null SD **{primary['null_sd']:.6f}**, z={primary['z']:.3f}, p={primary['p_plus1']:.5f}; "
        f"{primary['n_strata']} mixed folio×VT-topology strata.\n"
    )
    md.append(
        f"Folio bootstrap 95% CI: [{boot_ci[0]:.6f}, {boot_ci[1]:.6f}]. "
        f"Stratified AUC={auc['auc']:.4f} across {auc['pairs']} within-stratum pairs.\n"
    )
    md.append("## Group distributions (normalized gap)\n")
    md.append("| class | n | mean | median | q25 | q75 |\n|---|---:|---:|---:|---:|---:|")
    for c in ("joined", "uncertain", "certain"):
        g = groups[c]
        md.append(
            f"| {c} | {g.get('n', 0)} | {g.get('mean', float('nan')):.5f} | "
            f"{g.get('median', float('nan')):.5f} | {g.get('q25', float('nan')):.5f} | "
            f"{g.get('q75', float('nan')):.5f} |"
        )
    md.append("\n## Sensitivities\n")
    for name, res in [
        ("raw pixels", primary_px),
        ("folio-only (no VT-topology stratification)", no_topology),
        ("line similarity ≥ .70", strict_test),
        ("VT cross-word only", cross_test),
        ("uncertain − joined", uj),
    ]:
        md.append(
            f"- {name}: effect {res['actual']:.6f}; null SD {res['null_sd']:.6f}; "
            f"z {res['z']:.3f}; p {res['p_plus1']:.5f}; strata {res['n_strata']}."
        )
    md.append("\n## Hands\n")
    for x in hand:
        md.append(
            f"- Hand {x['hand']}: certain n={x['n_certain']}, uncertain n={x['n_uncertain']}, "
            f"stratified effect={x['effect']:.6f}, strata={x['n_strata']}."
        )
    md.append("\n## Adjudication\n")
    md.append(f"- Primary physical distinction: **{summary['adjudication']['primary_physical_distinction']}**")
    md.append(f"- Three-way continuum: **{summary['adjudication']['three_way_continuum']}**")
    md.append(
        "\nBoundary labels were not used by the ink measurement. VT word topology is treated as a nuisance "
        "stratum in the primary test, so the comparison is not allowed to win merely because the independent "
        "Voynichese.com locator made the same split decision as ZL."
    )
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("FINAL_SUMMARY", json.dumps(summary, sort_keys=True, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print("FATAL", repr(exc), file=sys.stderr, flush=True)
        raise
