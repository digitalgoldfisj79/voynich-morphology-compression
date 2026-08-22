#!/usr/bin/env python3
"""KOEN_RA_GAP_V02_BOUNDARYBLIND_20260822.

CPU-only, deterministic physical-gap test for EVA r->a.

The critical design constraint is that VT/Takahashi internal word boundaries are never
used by the character splitter or physical gap measurement. VT is used only to locate
and match a complete physical line after all separators are deleted. ZL boundary labels
are sealed before image measurement and joined back only after measurements_blind.csv
has been written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import requests

SEED = 6037
N_PERM = 9999
N_BOOT = 5000
LINE_SIM_MIN = 0.45
REG_WIDTH = 1800
CANONICAL_SCALE = 2.0
ZL_URL = "https://www.voynich.nu/data/ZL3b-n.txt"
MANIFEST_URL = "https://collections.library.yale.edu/manifests/2002046"
V_BASE = "https://www.voynichese.com/1/data/folio"
USER_AGENT = "voynich-koen-ra-gap-boundaryblind/0.2 (+research)"
OUT = Path("results/koen_ra_gap_blind")
WORK = OUT / "work"
PROTOCOL = Path("experiments/koen_ra_gap_blind/PROTOCOL.md")

PAGE_RE = re.compile(r"^<(?P<folio>f\d+[rv]\d*)>\s+<!\s*(?P<meta>.*?)>", re.I)
LOCUS_RE = re.compile(r"^<(?P<locus>f[^,>]+\.[^,>]+),(?P<kind>[^>]+)>\s*(?P<text>.*)$")
COMMENT_RE = re.compile(r"<!.*?>")
META_RE = re.compile(r"\$(?P<key>[A-Z])=(?P<val>[^\s>]+)")
INLINE_HAND_RE = re.compile(r"^<@H=(?P<hand>[^>]+)>", re.I)


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


@dataclass
class Canvas:
    index: int
    label: str
    canvas_id: str
    width: int
    height: int
    body_id: str
    derivative_url: str


def log(event: str, **kw):
    print(json.dumps({"event": event, **kw}, sort_keys=True), flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(session: requests.Session, url: str, *, binary=False, tries=5, timeout=120):
    last = None
    for k in range(tries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content if binary else r.text
        except Exception as exc:
            last = exc
            time.sleep(min(10.0, 1.25 * (k + 1)))
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
    recs = []
    offset = 0
    events = []
    for i in range(0, len(pieces), 2):
        raw = pieces[i]
        stripped = raw.strip()
        norm = letters_only(raw)
        plain = bool(re.fullmatch(r"[a-z]+", stripped))
        recs.append({"raw": raw, "norm": norm, "plain": plain, "offset": offset})
        if plain:
            for m in re.finditer("ra", stripped):
                events.append({
                    "class": "joined",
                    "r_index": offset + m.start(),
                    "a_index": offset + m.start() + 1,
                    "raw_context": stripped,
                })
        offset += len(norm)
    for k in range(len(recs) - 1):
        si = 2 * k + 1
        if si >= len(pieces):
            break
        sep = pieces[si]
        left, right = recs[k], recs[k + 1]
        if sep not in (".", ",") or not (left["plain"] and right["plain"]):
            continue
        if left["raw"].strip().endswith("r") and right["raw"].strip().startswith("a"):
            events.append({
                "class": "certain" if sep == "." else "uncertain",
                "r_index": left["offset"] + len(left["norm"]) - 1,
                "a_index": right["offset"],
                "raw_context": left["raw"].strip() + sep + right["raw"].strip(),
            })
    return "".join(r["norm"] for r in recs), events


def parse_zl(text: str):
    by_folio = defaultdict(list)
    page_meta = {}
    current_folio = None
    current_hand = ""
    current_quire = ""
    for raw in text.splitlines():
        pm = PAGE_RE.match(raw)
        if pm:
            current_folio = pm.group("folio").lower()
            meta = {m.group("key"): m.group("val") for m in META_RE.finditer(pm.group("meta"))}
            page_meta[current_folio] = meta
            current_hand = meta.get("H", "")
            current_quire = meta.get("Q", "")
            continue
        hm = INLINE_HAND_RE.match(raw)
        if hm:
            current_hand = hm.group("hand")
            continue
        lm = LOCUS_RE.match(raw)
        if not lm or current_folio is None or "P" not in lm.group("kind"):
            continue
        letters, events = parse_line_events(lm.group("text").strip())
        if not letters:
            continue
        by_folio[current_folio].append(ZLLine(
            current_folio, lm.group("locus"), current_hand, current_quire,
            lm.group("text").strip(), letters, events,
        ))
    return by_folio, page_meta


def make_event_id(folio: str, locus: str, r_index: int, a_index: int, ordinal: int) -> str:
    s = f"{folio}|{locus}|{r_index}|{a_index}|{ordinal}".encode()
    return hashlib.sha256(s).hexdigest()[:20]


def write_blinded_targets(by_folio):
    blinded = []
    labels = []
    counts = Counter()
    for folio in sorted(by_folio):
        for z in by_folio[folio]:
            for ordinal, ev in enumerate(sorted(z.events, key=lambda e: (e["r_index"], e["a_index"], e["class"]))):
                eid = make_event_id(folio, z.locus, ev["r_index"], ev["a_index"], ordinal)
                blinded.append({
                    "event_id": eid,
                    "folio": folio,
                    "locus": z.locus,
                    "hand": z.hand,
                    "quire": z.quire,
                    "r_index": int(ev["r_index"]),
                    "a_index": int(ev["a_index"]),
                    "line_letters": z.letters,
                })
                labels.append({"event_id": eid, "class": ev["class"], "raw_context": ev["raw_context"]})
                counts[ev["class"]] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    blind_path = OUT / "targets_blinded.json"
    label_path = OUT / "labels_sealed.json"
    blind_path.write_text(json.dumps(blinded, indent=2, sort_keys=True))
    label_path.write_text(json.dumps(labels, indent=2, sort_keys=True))
    return blinded, labels, counts, sha256_file(label_path)


def body_to_derivative(body_id: str, width: int) -> str:
    if re.search(r"/full/full/0/default\.jpg(?:\?.*)?$", body_id):
        return re.sub(r"/full/full/0/default\.jpg(?:\?.*)?$", f"/full/{width},/0/default.jpg", body_id)
    if re.search(r"/full/[^/]+/0/default\.jpg(?:\?.*)?$", body_id):
        return re.sub(r"/full/[^/]+/0/default\.jpg(?:\?.*)?$", f"/full/{width},/0/default.jpg", body_id)
    return body_id


def parse_manifest(obj: dict) -> list[Canvas]:
    out = []
    for i, c in enumerate(obj.get("items", [])):
        vals = []
        for v in c.get("label", {}).values():
            vals.extend(v)
        label = " ".join(str(x) for x in vals)
        body = (((c.get("items") or [{}])[0].get("items") or [{}])[0].get("body") or {})
        body_id = body.get("id")
        if not body_id:
            continue
        out.append(Canvas(
            i, label, c.get("id", ""), int(c.get("width") or 0), int(c.get("height") or 0),
            body_id, body_to_derivative(body_id, REG_WIDTH),
        ))
    return out


def folio_base(key: str):
    m = re.fullmatch(r"f(\d+)([rv])(\d+)?", key.lower())
    if not m:
        raise ValueError(key)
    return int(m.group(1)), m.group(2), f"{int(m.group(1))}{m.group(2)}"


def candidate_canvases(folio: str, canvases: list[Canvas]):
    n, side, base = folio_base(folio)
    token = re.compile(rf"(?<!\d){re.escape(base)}(?!\d)", re.I)
    exact = [c for c in canvases if token.search(c.label)]
    if exact:
        return exact
    nearby = []
    for c in canvases:
        lab = c.label.lower()
        if any(re.search(rf"(?<!\d){nn}[rv](?!\d)", lab) for nn in (n - 1, n, n + 1)):
            nearby.append(c)
    return nearby


def decode_image(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("image decode failed")
    return im


def prep_gray(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def register_page(legacy, target):
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02, edgeThreshold=12)
    k1, d1 = sift.detectAndCompute(prep_gray(legacy), None)
    k2, d2 = sift.detectAndCompute(prep_gray(target), None)
    if d1 is None or d2 is None:
        raise RuntimeError("no SIFT descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in pairs if a.distance < 0.77 * b.distance]
    if len(good) < 4:
        raise RuntimeError(f"only {len(good)} ratio-test matches")
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, keep = cv2.findHomography(p1, p2, method, 3.0, maxIters=20000, confidence=0.999)
    if H is None or keep is None:
        raise RuntimeError("homography failure")
    keep = keep.ravel().astype(bool)
    proj = cv2.perspectiveTransform(p1[:, None, :], H)[:, 0, :]
    err = np.linalg.norm(proj - p2, axis=1)[keep]
    sh, sw = legacy.shape[:2]
    th, tw = target.shape[:2]
    quad = cv2.perspectiveTransform(np.float32([[[0, 0], [sw, 0], [sw, sh], [0, sh]]]), H)[0]
    bounds = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    inter, _ = cv2.intersectConvexConvex(quad.astype(np.float32), bounds)
    intersection_ratio = float(inter / max(area, 1e-9))
    area_ratio = float(area / max(tw * th, 1))
    stats = {
        "matches": len(good),
        "inliers": int(keep.sum()),
        "inlier_ratio": float(keep.mean()),
        "median_reprojection_px": float(np.median(err)) if len(err) else float("inf"),
        "p95_reprojection_px": float(np.quantile(err, .95)) if len(err) else float("inf"),
        "intersection_ratio": intersection_ratio,
        "target_area_ratio": area_ratio,
    }
    stats["accepted"] = bool(
        stats["inliers"] >= 50 and stats["inlier_ratio"] >= .55
        and stats["median_reprojection_px"] <= 3.0
        and intersection_ratio >= .80 and .005 <= area_ratio <= 1.5
        and np.isfinite(H).all()
    )
    return H, stats


def fetch_v_boxes(session: requests.Session, folio: str):
    payload = json.loads(get(session, f"{V_BASE}/script/{folio}.js"))
    words = [str(row[0]).lower() for row in payload[0]]
    boxes = []
    for idx, row in enumerate(payload[1]):
        wid, x, y, w, h = row
        boxes.append(VBox(idx, words[wid], float(x), float(y), float(w), float(h)))
    return boxes


def build_v_lines(boxes: list[VBox], page_width=636.0, page_height=900.0):
    hs = [b.h for b in boxes if b.h > 0]
    med_h = statistics.median(hs) if hs else 10.0
    baseline_jump = max(4.0 * med_h, .06 * page_height)
    groups, cur = [], []
    prev = None
    for b in boxes:
        new = False
        if prev is not None:
            if b.x < prev.x - .075 * page_width:
                new = True
            elif abs(b.y - prev.y) > baseline_jump:
                new = True
        if new and cur:
            groups.append(cur)
            cur = []
        cur.append(b)
        prev = b
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        letters = "".join(letters_only(b.eva) for b in g)
        out.append(VLine(g, letters))
    return out


def sim(a: str, b: str):
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


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
            if i < n and base - .30 > dp[i + 1, j]:
                dp[i + 1, j] = base - .30; bt[i + 1, j] = 1
            if j < m and base - .18 > dp[i, j + 1]:
                dp[i, j + 1] = base - .18; bt[i, j + 1] = 2
            if i < n and j < m:
                ss = sim(zlines[i].letters, vlines[j].letters)
                val = base + (2.0 * ss - .55)
                if val > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = val; bt[i + 1, j + 1] = 3
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 3:
            pairs.append((i - 1, j - 1, sim(zlines[i - 1].letters, vlines[j - 1].letters)))
            i -= 1; j -= 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def exact_char_mapping(a: str, b: str):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    mp = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                mp[i1 + off] = j1 + off
    return mp


def ink_mask_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def boundary_blind_cuts(mask: np.ndarray, segments: int):
    """Low-ink DP cuts; receives only pixels + number of boundary-stripped glyphs."""
    h, w = mask.shape[:2]
    if segments <= 1 or w < segments * 2:
        return []
    proj = mask.astype(np.float32).sum(axis=0) / 255.0
    if w >= 5:
        proj = np.convolve(proj, np.array([1, 2, 3, 2, 1], dtype=float) / 9.0, mode="same")
    target = w / float(segments)
    all_cands = []
    for k in range(1, segments):
        center = k * target
        lo = max(2, int(math.floor(center - 1.40 * target)))
        hi = min(w - 3, int(math.ceil(center + 1.40 * target)))
        if hi < lo:
            return []
        xs = np.arange(lo, hi + 1, dtype=int)
        score = proj[xs] + .025 * np.abs(xs - center)
        order = np.argsort(score)
        keep = xs[order[:min(60, len(xs))]]
        all_cands.append(np.sort(keep))
    prev_cost = {int(x): float(proj[x] + .025 * abs(x - target)) for x in all_cands[0]}
    parents = []
    minw = max(2.0, .20 * target)
    maxw = 3.2 * target
    for k in range(1, len(all_cands)):
        center = (k + 1) * target
        cur_cost = {}
        cur_parent = {}
        for x0 in all_cands[k]:
            x = int(x0)
            best = float("inf"); bestp = None
            local = float(proj[x] + .025 * abs(x - center))
            for p, pc in prev_cost.items():
                segw = x - p
                if segw < minw or segw > maxw:
                    continue
                c = pc + local + .045 * abs(segw - target)
                if c < best:
                    best = c; bestp = p
            if bestp is not None:
                cur_cost[x] = best; cur_parent[x] = bestp
        if not cur_cost:
            return []
        parents.append(cur_parent)
        prev_cost = cur_cost
    end = min(prev_cost, key=prev_cost.get)
    cuts = [end]
    for parent in reversed(parents):
        end = parent[end]
        cuts.append(end)
    cuts = sorted(cuts)
    if len(cuts) != segments - 1 or any(b <= a for a, b in zip(cuts, cuts[1:])):
        return []
    return cuts


def line_envelope(vline: VLine, shape):
    h, w = shape[:2]
    bs = vline.boxes
    if not bs:
        return None
    medh = statistics.median([b.h for b in bs if b.h > 0] or [10.0])
    # Internal word boundaries are deliberately collapsed. Only the outer line envelope remains.
    x0 = min(b.x for b in bs) - .65 * medh
    x1 = max(b.x + b.w for b in bs) + .65 * medh
    y0 = min(b.y for b in bs) - .45 * medh
    y1 = max(b.y + b.h for b in bs) + .45 * medh
    s = CANONICAL_SCALE
    return (
        max(0, int(math.floor(x0 * s))),
        max(0, int(math.floor(y0 * s))),
        min(w, int(math.ceil(x1 * s))),
        min(h, int(math.ceil(y1 * s))),
        medh,
    )


def warp_yale_to_canonical(target, H, legacy_shape):
    lh, lw = legacy_shape[:2]
    S = np.array([[CANONICAL_SCALE, 0, 0], [0, CANONICAL_SCALE, 0], [0, 0, 1]], dtype=float)
    M = S @ np.linalg.inv(H)
    return cv2.warpPerspective(target, M, (int(round(lw * CANONICAL_SCALE)), int(round(lh * CANONICAL_SCALE))),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))


def measure_line(canonical, vline: VLine, zletters: str, targets: list[dict]):
    env = line_envelope(vline, canonical.shape)
    if env is None:
        return [], "no_line_envelope"
    x0, y0, x1, y1, _ = env
    if x1 - x0 < 20 or y1 - y0 < 8:
        return [], "tiny_line_envelope"
    crop = canonical[y0:y1, x0:x1]
    mask = ink_mask_bgr(crop)
    n = len(zletters)
    cuts = boundary_blind_cuts(mask, n)
    if len(cuts) != n - 1:
        return [], "cut_failure"
    bounds = [0] + cuts + [mask.shape[1]]
    ys_all, xs_all = np.where(mask > 0)
    if len(ys_all) < 10:
        return [], "no_line_ink"
    q05, q95 = np.quantile(ys_all, [.05, .95])
    line_ink_height = max(1.0, float(q95 - q05 + 1.0))
    out = []
    for t in targets:
        ri, ai = int(t["r_index"]), int(t["a_index"])
        if ai != ri + 1 or ri < 0 or ai >= n:
            continue
        la, lb = bounds[ri], bounds[ri + 1]
        ra, rb = bounds[ai], bounds[ai + 1]
        left = mask[:, la:lb]
        right = mask[:, ra:rb]
        yl, xl = np.where(left > 0)
        yr, xr = np.where(right > 0)
        if len(xl) == 0 or len(xr) == 0:
            continue
        r_right = la + int(xl.max())
        a_left = ra + int(xr.min())
        gap = max(0, a_left - r_right - 1)
        boundary_x = bounds[ai]
        proj = mask.astype(np.float32).sum(axis=0) / 255.0
        out.append({
            "event_id": t["event_id"],
            "folio": t["folio"],
            "locus": t["locus"],
            "hand": t["hand"],
            "quire": t["quire"],
            "r_index": ri,
            "a_index": ai,
            "gap_px_registered_yale": int(gap),
            "gap_px_legacy_equiv": float(gap / CANONICAL_SCALE),
            "gap_norm": float(gap / line_ink_height),
            "line_ink_height": line_ink_height,
            "line_crop_width": int(mask.shape[1]),
            "line_crop_height": int(mask.shape[0]),
            "boundary_x_in_crop": int(boundary_x),
            "boundary_column_ink": float(proj[boundary_x]) if 0 <= boundary_x < len(proj) else float("nan"),
        })
    return out, None


def group_summary(vals):
    vals = np.asarray([float(x) for x in vals], dtype=float)
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": int(len(vals)), "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "median": float(np.median(vals)), "q10": float(np.quantile(vals, .10)),
        "q25": float(np.quantile(vals, .25)), "q75": float(np.quantile(vals, .75)),
        "q90": float(np.quantile(vals, .90)), "min": float(vals.min()), "max": float(vals.max()),
    }


def weighted_effect(rows, value_key, positive="certain", negative="uncertain"):
    groups = defaultdict(lambda: {positive: [], negative: []})
    for r in rows:
        if r["class"] in (positive, negative):
            groups[r["folio"]][r["class"]].append(float(r[value_key]))
    cells = []
    for f, g in groups.items():
        if not g[positive] or not g[negative]:
            continue
        np_, nn = len(g[positive]), len(g[negative])
        w = np_ * nn / (np_ + nn)
        d = statistics.mean(g[positive]) - statistics.mean(g[negative])
        cells.append((f, d, w, np_, nn))
    if not cells:
        return float("nan"), cells
    return sum(d * w for _, d, w, _, _ in cells) / sum(w for _, _, w, _, _ in cells), cells


def permutation_test(rows, value_key, positive="certain", negative="uncertain", n_perm=N_PERM, seed=SEED):
    strata = []
    for f in sorted({r["folio"] for r in rows}):
        rr = [r for r in rows if r["folio"] == f and r["class"] in (positive, negative)]
        np_ = sum(r["class"] == positive for r in rr)
        nn = len(rr) - np_
        if not np_ or not nn:
            continue
        vals = np.asarray([float(r[value_key]) for r in rr], dtype=float)
        strata.append((f, vals, np_, nn, np_ * nn / (np_ + nn)))
    actual, _ = weighted_effect(rows, value_key, positive, negative)
    if not strata:
        return {"actual": actual, "null_mean": float("nan"), "null_sd": float("nan"), "z": float("nan"),
                "p_plus1": float("nan"), "ge": 0, "n_perm": n_perm, "n_strata": 0}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    totalw = sum(x[4] for x in strata)
    for b in range(n_perm):
        num = 0.0
        for _, vals, np_, nn, w in strata:
            idx = rng.permutation(len(vals))
            num += w * float(vals[idx[:np_]].mean() - vals[idx[np_:]].mean())
        null[b] = num / totalw
    nm = float(null.mean()); ns = float(null.std(ddof=1))
    ge = int(np.sum(null >= actual))
    return {
        "actual": float(actual), "null_mean": nm, "null_sd": ns,
        "z": float((actual - nm) / ns) if ns > 0 else float("nan"),
        "p_plus1": float((1 + ge) / (1 + n_perm)), "ge": ge,
        "n_perm": n_perm, "n_strata": len(strata),
        "null_q025": float(np.quantile(null, .025)), "null_q975": float(np.quantile(null, .975)),
    }


def stratified_auc(rows, value_key, positive="certain", negative="uncertain"):
    wins = ties = total = nstr = 0
    groups = defaultdict(lambda: {positive: [], negative: []})
    for r in rows:
        if r["class"] in (positive, negative):
            groups[r["folio"]][r["class"]].append(float(r[value_key]))
    for g in groups.values():
        if not g[positive] or not g[negative]:
            continue
        nstr += 1
        for a in g[positive]:
            for b in g[negative]:
                total += 1
                if a > b: wins += 1
                elif a == b: ties += 1
    return {"auc": float((wins + .5 * ties) / total) if total else float("nan"), "pairs": total, "n_strata": nstr}


def bootstrap_ci(rows, value_key, positive="certain", negative="uncertain", n_boot=N_BOOT, seed=SEED + 1):
    _, cells = weighted_effect(rows, value_key, positive, negative)
    if not cells:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = [cells[i] for i in rng.integers(0, len(cells), len(cells))]
        vals[b] = sum(d * w for _, d, w, _, _ in sampled) / sum(w for _, _, w, _, _ in sampled)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def hand_effects(rows, value_key):
    out = []
    for h in sorted({r["hand"] for r in rows if r.get("hand")}):
        sub = [r for r in rows if r.get("hand") == h]
        eff, cells = weighted_effect(sub, value_key)
        out.append({"hand": h, "effect": eff, "n": len(sub), "n_strata": len(cells),
                    "counts": dict(Counter(r["class"] for r in sub))})
    return out


def leave_one_hand_out(rows, value_key):
    out = []
    for h in sorted({r["hand"] for r in rows if r.get("hand")}):
        sub = [r for r in rows if r.get("hand") != h]
        eff, cells = weighted_effect(sub, value_key)
        out.append({"excluded_hand": h, "effect": eff, "n": len(sub), "n_strata": len(cells)})
    return out


def leave_one_folio_out(rows, value_key):
    out = []
    for f in sorted({r["folio"] for r in rows}):
        sub = [r for r in rows if r["folio"] != f]
        eff, cells = weighted_effect(sub, value_key)
        if math.isfinite(eff):
            out.append({"excluded_folio": f, "effect": eff, "n": len(sub), "n_strata": len(cells)})
    return out


def analyze(rows, value_key, positive="certain", negative="uncertain", seed=SEED):
    p = permutation_test(rows, value_key, positive, negative, seed=seed)
    p["auc"] = stratified_auc(rows, value_key, positive, negative)
    p["bootstrap95"] = bootstrap_ci(rows, value_key, positive, negative, seed=seed + 100)
    return p


def csv_write(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.headers["User-Agent"] = USER_AGENT

    protocol_hash = sha256_file(PROTOCOL)
    zl_text = get(session, ZL_URL)
    zl_hash = sha256_bytes(zl_text.encode())
    by_folio, _ = parse_zl(zl_text)
    blinded, labels, candidate_counts, labels_hash = write_blinded_targets(by_folio)
    target_folios = sorted({t["folio"] for t in blinded})
    log("frozen_targets", counts=dict(candidate_counts), folios=len(target_folios), labels_sha256=labels_hash)

    manifest_text = get(session, MANIFEST_URL)
    manifest_hash = sha256_bytes(manifest_text.encode())
    canvases = parse_manifest(json.loads(manifest_text))
    log("manifest", canvases=len(canvases), sha256=manifest_hash)

    targets_by_locus = defaultdict(list)
    for t in blinded:
        targets_by_locus[(t["folio"], t["locus"])].append(t)

    measured = []
    registrations = []
    failures = Counter()
    line_audit = []

    for fi, folio in enumerate(target_folios, 1):
        try:
            legacy = decode_image(get(session, f"{V_BASE}/image/glance/color/large/{folio}.jpg", binary=True))
            boxes = fetch_v_boxes(session, folio)
            vlines = build_v_lines(boxes)
            zlines = by_folio[folio]
            pairs = align_lines(zlines, vlines)
            pair_by_zi = {zi: (vj, ss) for zi, vj, ss in pairs}

            best = None
            best_target = None
            best_H = None
            cands = candidate_canvases(folio, canvases)
            if not cands:
                raise RuntimeError("no Yale canvas candidates")
            for c in cands:
                try:
                    target = decode_image(get(session, c.derivative_url, binary=True, timeout=180))
                    H, rs = register_page(legacy, target)
                    row = {"folio": folio, "canvas_index": c.index, "canvas_label": c.label,
                           "canvas_id": c.canvas_id, "derivative_url": c.derivative_url, **rs}
                    registrations.append(row)
                    key = (int(rs["accepted"]), rs["inliers"], rs["inlier_ratio"], -rs["median_reprojection_px"])
                    if best is None or key > best[0]:
                        best = (key, row); best_target = target; best_H = H
                except Exception as exc:
                    registrations.append({"folio": folio, "canvas_index": c.index, "canvas_label": c.label,
                                          "accepted": False, "error": f"{type(exc).__name__}: {exc}"})
            if best is None or not best[1].get("accepted"):
                failures["registration_rejected"] += 1
                continue
            canonical = warp_yale_to_canonical(best_target, best_H, legacy.shape)

            for zi, z in enumerate(zlines):
                tlist = targets_by_locus.get((folio, z.locus), [])
                if not tlist:
                    continue
                if zi not in pair_by_zi:
                    failures["unmatched_line"] += len(tlist)
                    continue
                vj, ss = pair_by_zi[zi]
                v = vlines[vj]
                cmap = exact_char_mapping(z.letters, v.letters)
                eligible = []
                for t in tlist:
                    ri, ai = t["r_index"], t["a_index"]
                    if ss < LINE_SIM_MIN:
                        failures["line_similarity"] += 1
                        continue
                    if cmap.get(ri) is None or cmap.get(ai) is None or cmap[ai] != cmap[ri] + 1:
                        failures["target_char_alignment"] += 1
                        continue
                    if z.letters[ri:ai + 1] != "ra" or v.letters[cmap[ri]:cmap[ai] + 1] != "ra":
                        failures["target_not_exact_ra"] += 1
                        continue
                    tt = dict(t)
                    tt["line_similarity"] = float(ss)
                    tt["exact_line"] = bool(z.letters == v.letters)
                    eligible.append(tt)
                if not eligible:
                    continue
                rows, err = measure_line(canonical, v, z.letters, eligible)
                if err:
                    failures[err] += len(eligible)
                    continue
                meta = {t["event_id"]: t for t in eligible}
                for r in rows:
                    r["line_similarity"] = meta[r["event_id"]]["line_similarity"]
                    r["exact_line"] = meta[r["event_id"]]["exact_line"]
                    r["canvas_label"] = best[1]["canvas_label"]
                    r["registration_inliers"] = best[1]["inliers"]
                    r["registration_ratio"] = best[1]["inlier_ratio"]
                    r["registration_median_reprojection_px"] = best[1]["median_reprojection_px"]
                measured.extend(rows)
                failures["measurement_slot_missing"] += max(0, len(eligible) - len(rows))
                line_audit.append({"folio": folio, "locus": z.locus, "line_similarity": ss,
                                   "exact_line": z.letters == v.letters, "z_chars": len(z.letters),
                                   "vt_chars_boundary_stripped": len(v.letters), "eligible_targets": len(eligible),
                                   "measured_targets": len(rows), "line_words_used_only_for_outer_envelope": len(v.boxes)})
        except Exception as exc:
            failures[f"folio_exception:{type(exc).__name__}"] += 1
            log("folio_error", folio=folio, error=f"{type(exc).__name__}: {exc}")
        if fi % 10 == 0 or fi == len(target_folios):
            log("progress", folios_done=fi, folios_total=len(target_folios), measured=len(measured), failures=dict(failures))

    # Freeze physical outcomes before labels are opened.
    csv_write(OUT / "measurements_blind.csv", measured)
    blind_measurement_hash = sha256_file(OUT / "measurements_blind.csv")
    csv_write(OUT / "registrations.csv", registrations)
    csv_write(OUT / "line_audit.csv", line_audit)

    label_map = {r["event_id"]: r for r in labels}
    events = []
    for r in measured:
        if r["event_id"] not in label_map:
            continue
        rr = dict(r)
        rr.update(label_map[r["event_id"]])
        events.append(rr)
    csv_write(OUT / "events_unblinded.csv", events)

    group_norm = {c: group_summary([r["gap_norm"] for r in events if r["class"] == c])
                  for c in ("joined", "uncertain", "certain")}
    group_px = {c: group_summary([r["gap_px_registered_yale"] for r in events if r["class"] == c])
                for c in ("joined", "uncertain", "certain")}
    group_legacy = {c: group_summary([r["gap_px_legacy_equiv"] for r in events if r["class"] == c])
                    for c in ("joined", "uncertain", "certain")}

    primary = analyze(events, "gap_norm")
    raw_px = analyze(events, "gap_px_registered_yale")
    secondary = analyze(events, "gap_norm", "uncertain", "joined", seed=SEED + 17)
    certain_joined = analyze(events, "gap_norm", "certain", "joined", seed=SEED + 23)

    sim70_rows = [r for r in events if float(r["line_similarity"]) >= .70]
    exact_rows = [r for r in events if str(r["exact_line"]).lower() in ("true", "1") or r["exact_line"] is True]
    sim70 = analyze(sim70_rows, "gap_norm")
    exact = analyze(exact_rows, "gap_norm")
    loho = leave_one_hand_out(events, "gap_norm")
    hands = hand_effects(events, "gap_norm")
    lofo = leave_one_folio_out(events, "gap_norm")

    measured_counts = Counter(r["class"] for r in events)
    retention = {c: {
        "candidates": int(candidate_counts[c]), "measured": int(measured_counts[c]),
        "fraction": float(measured_counts[c] / candidate_counts[c]) if candidate_counts[c] else float("nan")
    } for c in ("joined", "uncertain", "certain")}

    finite_loho = [x["effect"] for x in loho if math.isfinite(x["effect"])]
    ci = primary["bootstrap95"]
    primary_resolved = bool(
        primary["p_plus1"] <= .05 and primary["z"] >= 2.0 and ci[0] > 0
        and finite_loho and all(x > 0 for x in finite_loho)
    )
    continuum_resolved = bool(primary_resolved and secondary["p_plus1"] <= .05 and secondary["z"] >= 2.0)

    result = {
        "protocol_id": "KOEN_RA_GAP_V02_BOUNDARYBLIND_20260822",
        "protocol_sha256": protocol_hash,
        "zl_sha256": zl_hash,
        "yale_manifest_sha256": manifest_hash,
        "labels_sealed_sha256": labels_hash,
        "measurements_blind_sha256": blind_measurement_hash,
        "seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT,
        "instrument": {
            "reg_width": REG_WIDTH, "canonical_scale": CANONICAL_SCALE,
            "vt_internal_word_boundaries_used_for_measurement": False,
            "vt_word_topology_available_to_inference": False,
            "vt_text_role": "boundary-stripped line locator only",
            "dino_embedding_inference": False, "gpu": False,
        },
        "candidate_counts": dict(candidate_counts), "measured_counts": dict(measured_counts),
        "retention": retention, "n_events": len(events), "n_folios_measured": len({r["folio"] for r in events}),
        "failure_counts": dict(failures),
        "groups_gap_norm": group_norm, "groups_registered_yale_px": group_px,
        "groups_legacy_equiv_px": group_legacy,
        "primary_certain_minus_uncertain": primary,
        "raw_registered_yale_px_certain_minus_uncertain": raw_px,
        "secondary_uncertain_minus_joined": secondary,
        "certain_minus_joined": certain_joined,
        "sensitivity_similarity_ge_070": sim70,
        "sensitivity_exact_boundary_stripped_line": exact,
        "hand_effects": hands, "leave_one_hand_out": loho,
        "leave_one_folio_out": lofo,
        "decision": {"primary_physical_distinction": "RESOLVED" if primary_resolved else "NOT_RESOLVED",
                     "three_way_continuum": "RESOLVED" if continuum_resolved else "NOT_RESOLVED"},
        "retraction": "V01 wording that VT-cross-word was an independent-transcriber control is retracted; V01 numerical outputs are not used as V02 evidence."
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))

    def fmt(x, nd=6):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    lines = [
        "# KOEN_RA_GAP_V02_BOUNDARYBLIND_20260822 — result",
        "",
        "## RETRACTED FINDINGS",
        "The V01 description of the VT/Takahashi cross-word sensitivity as an **independent-transcriber control** is retracted. V02 does not use VT internal word-boundary x coordinates or VT topology in the physical outcome or inferential null.",
        "",
        "## Blind physical instrument",
        f"Sealed-label SHA-256: `{labels_hash}`",
        f"Blind-measurement SHA-256 before label join: `{blind_measurement_hash}`",
        f"Protocol SHA-256: `{protocol_hash}`",
        "",
        "The image stage saw boundary-stripped glyph indices only. It measured one continuous registered-Yale line raster; internal VT/Takahashi word rectangles were collapsed before character cutting. No DINO embedding inference, Hugging Face Job or GPU was used.",
        "",
        "## Counts",
        f"Candidates: joined {candidate_counts['joined']}, uncertain {candidate_counts['uncertain']}, certain {candidate_counts['certain']}.",
        f"Measured: joined {measured_counts['joined']}, uncertain {measured_counts['uncertain']}, certain {measured_counts['certain']} = {len(events)} total across {len({r['folio'] for r in events})} folios.",
        "",
        "## Physical gaps",
        "| class | n | mean norm | median norm | mean registered-Yale px | median px | legacy-equivalent mean px |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ("joined", "uncertain", "certain"):
        gn, gp, gl = group_norm[c], group_px[c], group_legacy[c]
        lines.append(f"| {c} | {gn.get('n',0)} | {fmt(gn.get('mean'))} | {fmt(gn.get('median'))} | {fmt(gp.get('mean'),3)} | {fmt(gp.get('median'),3)} | {fmt(gl.get('mean'),3)} |")
    lines += [
        "",
        "## Primary: certain − uncertain (within-folio permutation)",
        f"Normalized effect **{fmt(primary['actual'])}**; null mean {fmt(primary['null_mean'])}; null SD **{fmt(primary['null_sd'])}**; z **{fmt(primary['z'],3)}**; p **{fmt(primary['p_plus1'],4)}**; {primary['n_strata']} mixed folio strata.",
        f"Folio bootstrap 95% CI: [{fmt(primary['bootstrap95'][0])}, {fmt(primary['bootstrap95'][1])}]. Stratified AUC {fmt(primary['auc']['auc'],3)} over {primary['auc']['pairs']} within-folio pairs.",
        f"Raw registered-Yale pixels: effect **{fmt(raw_px['actual'],3)} px**; null SD **{fmt(raw_px['null_sd'],3)} px**; z **{fmt(raw_px['z'],3)}**; p **{fmt(raw_px['p_plus1'],4)}**.",
        "",
        "## Secondary: uncertain − joined",
        f"Effect **{fmt(secondary['actual'])}**; null SD **{fmt(secondary['null_sd'])}**; z **{fmt(secondary['z'],3)}**; p **{fmt(secondary['p_plus1'],4)}**.",
        "",
        "## Fixed sensitivities",
        f"Similarity >=0.70: effect {fmt(sim70['actual'])}; null SD {fmt(sim70['null_sd'])}; z {fmt(sim70['z'],3)}; p {fmt(sim70['p_plus1'],4)}; n={len(sim70_rows)}.",
        f"Exact boundary-stripped ZL=VT lines: effect {fmt(exact['actual'])}; null SD {fmt(exact['null_sd'])}; z {fmt(exact['z'],3)}; p {fmt(exact['p_plus1'],4)}; n={len(exact_rows)}.",
        "",
        "## Decision",
        f"Primary physical distinction: **{result['decision']['primary_physical_distinction']}**.",
        f"Three-way continuum: **{result['decision']['three_way_continuum']}**.",
        "",
        "Interpretation is limited to boundary strength in the physical manuscript. This does not establish linguistic wordhood, semantics, decipherment, or three discrete scribal categories.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": protocol_hash, "zl_sha256": zl_hash, "manifest_sha256": manifest_hash,
        "labels_sealed_sha256": labels_hash, "measurements_blind_sha256": blind_measurement_hash,
        "files": {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file()}
    }, indent=2, sort_keys=True))
    print((OUT / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
