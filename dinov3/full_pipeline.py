#!/usr/bin/env python3
"""Full-corpus Voynichese -> Yale -> DINOv3 image-derived grapheme programme.

Standalone Hugging Face Jobs entry point. It deliberately treats EVA as audit metadata,
not training truth, and never calls an image cluster a glyph unless all admission tests pass.
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import faiss
import numpy as np
import requests
import scipy.sparse as sp
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


MANIFEST_URL = "https://collections.library.yale.edu/manifests/2002046"
HOME_URL = "https://www.voynichese.com/"
SCRIPT_URL = "https://www.voynichese.com/1/data/folio/script/{folio}.js"
LEGACY_URL = "https://www.voynichese.com/1/data/folio/image/glance/color/large/{folio}.jpg"
MODEL_B = "facebook/dinov3-vitb16-pretrain-lvd1689m"
MODEL_L = "facebook/dinov3-vitl16-pretrain-lvd1689m"
PROTOCOL_VERSION = "2026-07-13.full-corpus.v1"
SEED = 408


@dataclass
class Thresholds:
    min_inliers: int = 50
    min_inlier_ratio: float = 0.55
    max_median_reprojection_px: float = 3.0


@dataclass
class Canvas:
    index: int
    label: str
    canvas_id: str
    width: int
    height: int
    body_id: str
    derivative_url: str


@dataclass
class RegistrationResult:
    folio: str
    candidate_index: int
    candidate_label: str
    canvas_id: str
    body_id: str
    canvas_width: int
    canvas_height: int
    target_width: int
    target_height: int
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    median_reprojection_px: float = math.inf
    p95_reprojection_px: float = math.inf
    plausible_intersection: bool = False
    intersection_ratio: float = 0.0
    target_area_ratio: float = 0.0
    homography_derivative: list[list[float]] | None = None
    homography_full: list[list[float]] | None = None
    target_quad_derivative: list[list[float]] | None = None
    target_quad_full: list[list[float]] | None = None
    accepted: bool = False
    error: str | None = None


class Timer:
    def __init__(self): self.t0 = time.time()
    def elapsed(self): return time.time() - self.t0


def log(event: str, **kw: Any) -> None:
    payload = {"event": event, "t": round(time.time(), 3), **kw}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def retry_get(session: requests.Session, url: str, timeout: int = 90, attempts: int = 5) -> requests.Response:
    last = None
    for k in range(attempts):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(min(20, 1.5 ** k + random.random()))
    raise RuntimeError(f"GET failed after {attempts}: {url}: {last!r}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def body_to_derivative(body_id: str, width: int) -> str:
    return re.sub(r"/full/full/0/default\.jpg(?:\?.*)?$", f"/full/{width},/0/default.jpg", body_id)


def parse_manifest(obj: dict[str, Any], reg_width: int) -> list[Canvas]:
    out: list[Canvas] = []
    for i, c in enumerate(obj.get("items", [])):
        vals: list[str] = []
        for v in c.get("label", {}).values(): vals.extend(v)
        label = " ".join(vals)
        body = (((c.get("items") or [{}])[0].get("items") or [{}])[0].get("body") or {})
        body_id = body.get("id")
        if not body_id: continue
        out.append(Canvas(i, label, c.get("id", ""), int(c.get("width") or 0), int(c.get("height") or 0), body_id, body_to_derivative(body_id, reg_width)))
    return out


def parse_home(html: str) -> tuple[list[str], dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    ids: list[str] = []
    classes: dict[str, str] = {}
    for d in soup.select("div.r-y[id]"):
        f = d.get("id")
        if f:
            ids.append(f)
            classes[f] = " ".join(d.get("class") or [])
    return ids, classes


def folio_base(key: str) -> tuple[int, str, str]:
    m = re.fullmatch(r"f(\d+)([rv])(\d+)?", key.lower())
    if not m: raise ValueError(key)
    return int(m.group(1)), m.group(2), f"{int(m.group(1))}{m.group(2)}"


def candidate_canvases(folio: str, canvases: list[Canvas], broad: bool = False) -> list[int]:
    n, side, base = folio_base(folio)
    token = re.compile(rf"(?<!\d){re.escape(base)}(?!\d)", re.I)
    exact = [c.index for c in canvases if token.search(c.label)]
    if exact and not broad:
        return exact
    candidates = set(exact)
    for c in canvases:
        lab = c.label.lower()
        for nn in (n - 1, n, n + 1):
            if re.search(rf"(?<!\d){nn}[rv](?!\d)", lab):
                candidates.add(c.index)
    return sorted(candidates)


def download_inputs(work: Path, folios: list[str], canvases: list[Canvas]) -> None:
    s = requests.Session(); s.headers["User-Agent"] = "Voynich-DINOv3-research/1.0"
    (work / "legacy").mkdir(parents=True, exist_ok=True)
    (work / "scripts").mkdir(parents=True, exist_ok=True)
    (work / "targets").mkdir(parents=True, exist_ok=True)
    needed = sorted({i for f in folios for i in candidate_canvases(f, canvases, broad=False)})
    for k, f in enumerate(folios, 1):
        for typ, url, ext in (("legacy", LEGACY_URL.format(folio=f), ".jpg"), ("scripts", SCRIPT_URL.format(folio=f), ".js")):
            p = work / typ / (f + ext)
            if not p.exists():
                data = retry_get(s, url).content
                p.write_bytes(data)
        if k % 25 == 0 or k == len(folios): log("download_folios", done=k, total=len(folios))
    for k, idx in enumerate(needed, 1):
        p = work / "targets" / f"{idx:03d}.jpg"
        if not p.exists(): p.write_bytes(retry_get(s, canvases[idx].derivative_url, timeout=180).content)
        if k % 20 == 0 or k == len(needed): log("download_targets", done=k, total=len(needed))


def _prep_gray(im: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _register_task(args: tuple[str, int, dict[str, Any], str, str, Thresholds]) -> dict[str, Any]:
    folio, idx, cd, source_path, target_path, thresholds = args
    c = Canvas(**cd)
    try:
        src = cv2.imread(source_path, cv2.IMREAD_COLOR)
        tgt = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if src is None or tgt is None: raise ValueError("image decode failure")
        sift = cv2.SIFT_create(nfeatures=14000, contrastThreshold=0.02, edgeThreshold=12)
        k1, d1 = sift.detectAndCompute(_prep_gray(src), None)
        k2, d2 = sift.detectAndCompute(_prep_gray(tgt), None)
        if d1 is None or d2 is None: raise ValueError("no descriptors")
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
        good = [a for a, b in pairs if a.distance < 0.77 * b.distance]
        if len(good) < 4: raise ValueError(f"only {len(good)} ratio-test matches")
        p1 = np.float32([k1[m.queryIdx].pt for m in good])
        p2 = np.float32([k2[m.trainIdx].pt for m in good])
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, mask = cv2.findHomography(p1, p2, method, 3.0, maxIters=20000, confidence=0.999)
        if H is None or mask is None: raise ValueError("homography failure")
        keep = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(p1[:, None, :], H)[:, 0, :]
        err = np.linalg.norm(proj - p2, axis=1)[keep]
        sh, sw = src.shape[:2]; th, tw = tgt.shape[:2]
        sq = np.float32([[[0, 0], [sw, 0], [sw, sh], [0, sh]]])
        tq = cv2.perspectiveTransform(sq, H)[0]
        poly = tq.astype(np.float32)
        bounds = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        area = abs(float(cv2.contourArea(poly)))
        inter_area, _ = cv2.intersectConvexConvex(poly, bounds)
        intersection_ratio = float(inter_area / max(area, 1e-9))
        target_area_ratio = float(area / max(tw * th, 1))
        plausible = bool(intersection_ratio >= 0.80 and 0.005 <= target_area_ratio <= 1.5 and np.isfinite(H).all())
        inliers = int(keep.sum()); ratio = float(keep.mean()); med = float(np.median(err)); p95 = float(np.quantile(err, 0.95))
        accepted = plausible and inliers >= thresholds.min_inliers and ratio >= thresholds.min_inlier_ratio and med <= thresholds.max_median_reprojection_px
        sx, sy = c.width / tw, c.height / th
        S = np.diag([sx, sy, 1.0])
        Hfull = S @ H
        tq_full = cv2.perspectiveTransform(sq, Hfull)[0]
        r = RegistrationResult(folio, idx, c.label, c.canvas_id, c.body_id, c.width, c.height, tw, th,
            len(good), inliers, ratio, med, p95, plausible, intersection_ratio, target_area_ratio,
            H.tolist(), Hfull.tolist(), tq.tolist(), tq_full.tolist(), accepted, None)
        return asdict(r)
    except Exception as e:
        r = RegistrationResult(folio, idx, c.label, c.canvas_id, c.body_id, c.width, c.height, 0, 0, error=f"{type(e).__name__}: {e}")
        return asdict(r)


def run_registrations(work: Path, folios: list[str], canvases: list[Canvas], workers: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    thresholds = Thresholds()
    tasks = []
    for f in folios:
        for idx in candidate_canvases(f, canvases, broad=False):
            tasks.append((f, idx, asdict(canvases[idx]), str(work / "legacy" / f"{f}.jpg"), str(work / "targets" / f"{idx:03d}.jpg"), thresholds))
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_register_task, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if k % 25 == 0 or k == len(futures): log("registration_progress", done=k, total=len(futures))
    by_f: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results: by_f[r["folio"]].append(r)
    selected: dict[str, dict[str, Any]] = {}
    for f, rr in by_f.items():
        good = [r for r in rr if r["accepted"]]
        pool = good or [r for r in rr if not r.get("error")]
        if pool:
            selected[f] = max(pool, key=lambda r: (bool(r["accepted"]), r["inliers"], r["inlier_ratio"], -r["median_reprojection_px"]))
    failed = [f for f in folios if f not in selected or not selected[f]["accepted"]]
    if failed:
        log("registration_fallback_start", failures=len(failed), folios=failed)
        existing = {(r["folio"], r["candidate_index"]) for r in results}
        extra = []
        s = requests.Session(); s.headers["User-Agent"] = "Voynich-DINOv3-research/1.0"
        for f in failed:
            for idx in candidate_canvases(f, canvases, broad=True):
                if (f, idx) in existing: continue
                p = work / "targets" / f"{idx:03d}.jpg"
                if not p.exists(): p.write_bytes(retry_get(s, canvases[idx].derivative_url, timeout=180).content)
                extra.append((f, idx, asdict(canvases[idx]), str(work / "legacy" / f"{f}.jpg"), str(p), thresholds))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fs = [ex.submit(_register_task, t) for t in extra]
            for k, fut in enumerate(as_completed(fs), 1):
                r = fut.result(); results.append(r); by_f[r["folio"]].append(r)
                if k % 20 == 0 or k == len(fs): log("registration_fallback_progress", done=k, total=len(fs))
        for f in failed:
            rr = by_f[f]; good = [r for r in rr if r["accepted"]]; pool = good or [r for r in rr if not r.get("error")]
            if pool: selected[f] = max(pool, key=lambda r: (bool(r["accepted"]), r["inliers"], r["inlier_ratio"], -r["median_reprojection_px"]))
    return sorted(results, key=lambda r: (r["folio"], r["candidate_index"])), selected


def parse_runtime(path: Path) -> tuple[list[list[Any]], list[list[Any]]]:
    x = json.loads(path.read_text())
    if not isinstance(x, list) or len(x) != 2: raise ValueError(f"unexpected runtime structure {path}")
    return x[0], x[1]


def infer_lines(boxes: list[list[Any]]) -> dict[int, dict[str, Any]]:
    rows = []
    for src_i, b in enumerate(boxes):
        wi, x, y, w, h = b
        rows.append((src_i, float(x), float(y), float(w), float(h), float(y) + float(h) / 2))
    rows.sort(key=lambda z: (z[5], z[1]))
    medh = statistics.median([r[4] for r in rows]) if rows else 20
    lines: list[list[tuple]] = []
    for r in rows:
        best = None; bestd = 1e9
        for i, line in enumerate(lines):
            cy = statistics.median([q[5] for q in line]); d = abs(r[5] - cy)
            if d < bestd and d <= max(8.0, 0.65 * medh): best, bestd = i, d
        if best is None: lines.append([r])
        else: lines[best].append(r)
    lines.sort(key=lambda line: statistics.median([q[5] for q in line]))
    out: dict[int, dict[str, Any]] = {}
    for li, line in enumerate(lines):
        line.sort(key=lambda z: z[1])
        for ti, r in enumerate(line):
            out[r[0]] = {"line_index": li, "token_index": ti, "line_length": len(line), "line_start": ti == 0, "line_end": ti == len(line)-1}
    return out


def perspective_rect(rect: tuple[float, float, float, float], H: np.ndarray) -> tuple[np.ndarray, list[int]]:
    x, y, w, h = rect
    p = np.float32([[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]])
    q = cv2.perspectiveTransform(p, H)[0]
    mn = np.floor(q.min(0)).astype(int); mx = np.ceil(q.max(0)).astype(int)
    return q, [int(mn[0]), int(mn[1]), int(mx[0]-mn[0]), int(mx[1]-mn[1])]


def clamp_xywh(b: list[int], W: int, H: int, pad: int = 0) -> list[int]:
    x, y, w, h = b; x0 = max(0, x-pad); y0 = max(0, y-pad); x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
    return [x0, y0, max(0, x1-x0), max(0, y1-y0)]


def normalize_ink(im: Image.Image) -> Image.Image:
    g = np.array(ImageOps.grayscale(im))
    if min(g.shape) < 3: return im.convert("RGB")
    bg = cv2.GaussianBlur(g, (0, 0), max(3, min(g.shape)/8))
    n = cv2.divide(g, bg, scale=255)
    n = cv2.createCLAHE(2.0, (8, 8)).apply(n)
    return Image.fromarray(n).convert("RGB")


def ink_fraction(im: Image.Image) -> float:
    g = np.array(im.convert("L"))
    if g.size == 0: return 0.0
    thr = min(220, int(np.quantile(g, 0.35)))
    return float(np.mean(g < thr))


def make_proposals(im: Image.Image) -> list[dict[str, Any]]:
    g = np.array(im.convert("L"))
    H, W = g.shape
    if W < 3 or H < 3: return []
    block = max(7, (min(31, max(7, (min(W, H)//2)*2+1)) // 2)*2+1)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = map(int, stats[i])
        if a < 5 or w < 2 or h < 2 or w*h > 0.95*W*H: continue
        comps.append([x, y, w, h, a])
    comps.sort(key=lambda z: (z[0], z[1]))
    out = []
    for c in comps[:16]: out.append({"proposal_type":"component", "xywh":c[:4], "ink_area":c[4]})
    if comps:
        medh = statistics.median([c[3] for c in comps]); gap_lim = max(2, int(0.18*medh))
        cur = comps[0][:4]
        merged = []
        for c in comps[1:]:
            x,y,w,h = c[:4]; cx,cy,cw,ch = cur
            gap = x-(cx+cw); ov = max(0, min(cy+ch,y+h)-max(cy,y)); ovf = ov/max(1,min(ch,h))
            if gap <= gap_lim and ovf >= 0.20:
                x0=min(cx,x); y0=min(cy,y); x1=max(cx+cw,x+w); y1=max(cy+ch,y+h); cur=[x0,y0,x1-x0,y1-y0]
            else:
                if cur not in [q[:4] for q in comps]: merged.append(cur)
                cur=c[:4]
        if cur not in [q[:4] for q in comps]: merged.append(cur)
        for m in merged[:8]: out.append({"proposal_type":"merged_component", "xywh":m})
    for frac in (0.38, 0.62):
        ww = max(6, min(W, int(round(W*frac)))); step = max(3, ww//2)
        xs = list(range(0, max(1, W-ww+1), step))
        if xs and xs[-1] != W-ww: xs.append(max(0, W-ww))
        for x in sorted(set(xs))[:8]: out.append({"proposal_type":"window", "xywh":[x,0,ww,H]})
    seen=set(); ded=[]
    for p in out:
        key=(p["proposal_type"],tuple(p["xywh"]))
        if key not in seen:
            seen.add(key); ded.append(p)
    for i,p in enumerate(ded): p["proposal_index"] = i
    return ded


def make_variant(im: Image.Image, variant: int) -> Image.Image:
    if variant == 0: return im
    if variant == 1: return ImageEnhance.Contrast(im).enhance(1.18)
    if variant == 2:
        a=np.array(im.convert("L")); k=np.ones((2,2),np.uint8); a=cv2.erode(a,k,iterations=1); return Image.fromarray(a).convert("RGB")
    if variant == 3:
        a=np.array(im.convert("L")); k=np.ones((2,2),np.uint8); a=cv2.dilate(a,k,iterations=1); return Image.fromarray(a).convert("RGB")
    if variant == 4:
        w,h=im.size; dx=max(1,int(.03*w)); dy=max(1,int(.05*h)); canvas=Image.new("RGB",(w,h),"white"); crop=im.crop((dx,dy,w,h)); canvas.paste(crop,(0,0)); return canvas
    return im


class DinoEmbedder:
    def __init__(self, model_id: str, token: str | None, batch_size: int):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch=torch; self.batch_size=batch_size; self.model_id=model_id
        self.proc=AutoImageProcessor.from_pretrained(model_id, token=token)
        self.model=AutoModel.from_pretrained(model_id, token=token, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval().cuda()
        self.revision=getattr(self.model.config,"_commit_hash",None) or "unknown"
    def embed(self, images: list[Image.Image], dense: bool=False) -> np.ndarray:
        torch=self.torch; outs=[]
        with torch.inference_mode():
            for i in range(0,len(images),self.batch_size):
                b=images[i:i+self.batch_size]
                x=self.proc(images=b,return_tensors="pt").to("cuda")
                with torch.autocast(device_type="cuda",dtype=torch.float16): y=self.model(**x).last_hidden_state
                if dense:
                    z=y[:,5:].float(); z=torch.nn.functional.normalize(z,dim=-1)
                else:
                    z=torch.nn.functional.normalize(y[:,0].float(),dim=1)
                outs.append(z.cpu().numpy())
        return np.concatenate(outs,axis=0) if outs else np.empty((0,768),np.float32)
    def close(self):
        del self.model; del self.proc; gc.collect(); self.torch.cuda.empty_cache()


def faiss_knn(X: np.ndarray, k: int=20, hnsw: bool=True) -> tuple[np.ndarray,np.ndarray]:
    X=np.asarray(X,np.float32); faiss.normalize_L2(X)
    if hnsw and len(X)>5000:
        index=faiss.IndexHNSWFlat(X.shape[1],32,faiss.METRIC_INNER_PRODUCT); index.hnsw.efConstruction=120; index.hnsw.efSearch=max(80,k*4)
    else: index=faiss.IndexFlatIP(X.shape[1])
    index.add(X); D,I=index.search(X,min(k,len(X)))
    return D,I


def sampled_auc(X: np.ndarray, labels: list[str | None], rng: np.random.Generator, n_pairs: int=100000) -> float | None:
    groups=defaultdict(list)
    for i,l in enumerate(labels):
        if l: groups[l].append(i)
    eligible=[v for v in groups.values() if len(v)>=2]
    if not eligible: return None
    sims=[]; ys=[]
    for _ in range(n_pairs//2):
        g=eligible[int(rng.integers(len(eligible)))]; a,b=rng.choice(g,2,replace=False); sims.append(float(X[a]@X[b])); ys.append(1)
        a=int(rng.integers(len(X))); b=int(rng.integers(len(X)))
        tries=0
        while labels[a] and labels[a]==labels[b] and tries<20: b=int(rng.integers(len(X))); tries+=1
        sims.append(float(X[a]@X[b])); ys.append(0)
    return float(roc_auc_score(ys,sims))


def word_audit(X: np.ndarray, meta: list[dict[str,Any]], rng: np.random.Generator) -> dict[str,Any]:
    if len(X)<2: return {}
    D,I=faiss_knn(X,k=min(100,len(X)),hnsw=True)
    labels=[m.get("eva") for m in meta]; folios=[m["folio"] for m in meta]
    nn=I[:,1] if I.shape[1]>1 else I[:,0]
    exact=float(np.mean([labels[i] is not None and labels[i]==labels[nn[i]] for i in range(len(X))]))
    same_folio=float(np.mean([folios[i]==folios[nn[i]] for i in range(len(X))]))
    eligible=[]; hits=[]; rr=[]
    folio_by_label=defaultdict(set)
    for l,f in zip(labels,folios):
        if l: folio_by_label[l].add(f)
    for i,l in enumerate(labels):
        if not l or len(folio_by_label[l])<2: continue
        eligible.append(i); rank=None
        for r,j in enumerate(I[i,1:],1):
            if folios[j]!=folios[i] and labels[j]==l: rank=r; break
        hits.append(rank==1); rr.append(0.0 if rank is None else 1.0/rank)
    return {"n":len(X),"exact_eva_nn":exact,"nearest_same_folio_rate":same_folio,
        "eligible_cross_folio_queries":len(eligible),
        "eligible_cross_folio_exact_eva_at_1":float(np.mean(hits)) if hits else None,
        "eligible_cross_folio_mrr":float(np.mean(rr)) if rr else None,
        "same_vs_different_eva_cosine_auc":sampled_auc(X,labels,rng,min(100000,max(2000,len(X)*4)))}


def discover_clusters(X: np.ndarray, meta: list[dict[str,Any]], max_n: int, rng: np.random.Generator) -> dict[str,Any]:
    n=len(X)
    if n==0: return {"assignments":np.empty(0,int),"sample_indices":np.empty(0,int),"clusters":[],"threshold":None}
    if n>max_n:
        by=defaultdict(list)
        for i,m in enumerate(meta): by[m["folio"]].append(i)
        per=max(20,max_n//max(1,len(by))); idx=[]
        for f,arr in sorted(by.items()):
            if len(arr)>per: arr=list(rng.choice(arr,per,replace=False))
            idx.extend(arr)
        if len(idx)>max_n: idx=list(rng.choice(idx,max_n,replace=False))
        idx=np.array(sorted(idx),int)
    else: idx=np.arange(n)
    Z=np.asarray(X[idx],np.float32); faiss.normalize_L2(Z)
    D,I=faiss_knn(Z,k=min(12,len(Z)),hnsw=True)
    nonself=D[:,1:].ravel() if D.shape[1]>1 else D.ravel()
    threshold=float(max(0.82,np.quantile(nonself,0.94)))
    rows=[]; cols=[]
    for i in range(len(Z)):
        for d,j in zip(D[i,1:],I[i,1:]):
            if d>=threshold and j>=0 and j!=i: rows.append(i); cols.append(int(j))
    A=sp.csr_matrix((np.ones(len(rows),np.uint8),(rows,cols)),shape=(len(Z),len(Z)))
    M=A.multiply(A.T); M.eliminate_zeros()
    nc,lab=connected_components(M,directed=False)
    sizes=np.bincount(lab,minlength=nc)
    valid=[c for c,s in enumerate(sizes) if 5<=s<=2000]
    valid=sorted(valid,key=lambda c:int(sizes[c]),reverse=True)
    remap={c:i for i,c in enumerate(valid)}
    assign=np.array([remap.get(int(c),-1) for c in lab],int)
    clusters=[]
    for old,new in remap.items():
        members=np.where(lab==old)[0]; V=Z[members]; cent=V.mean(0); cent/=max(np.linalg.norm(cent),1e-9)
        sims=V@cent; med_local=int(members[int(np.argmax(sims))])
        fs=Counter(meta[int(idx[m])]["folio"] for m in members)
        types=Counter(meta[int(idx[m])]["proposal_type"] for m in members)
        clusters.append({"cluster_id":new,"size":int(len(members)),"distinct_folios":len(fs),"top_folios":fs.most_common(5),"proposal_types":types,"mean_centroid_cosine":float(sims.mean()),"p05_centroid_cosine":float(np.quantile(sims,.05)),"medoid_sample_index":med_local,"centroid":cent})
    return {"assignments":assign,"sample_indices":idx,"clusters":clusters,"threshold":threshold,"mutual_edges":int(M.nnz//2),"sample_n":len(idx),"unclustered_rate":float(np.mean(assign<0))}


def safe_bifolios() -> list[tuple[int,int]]:
    out=[]
    for start in (1,9,17,25,33,41,49):
        for k in range(4):
            a=start+k; b=start+7-k
            if a==12 or b==12: continue
            out.append((a,b))
    out += [(75,84),(76,83),(77,82),(78,81),(79,80)]
    out += [(93,96),(94,95),(99,102),(100,101)]
    return out


def base_number(folio: str) -> int: return folio_base(folio)[0]


def heldout_bifolio_test(X: np.ndarray, meta: list[dict[str,Any]], rng: np.random.Generator) -> dict[str,Any]:
    pairs=safe_bifolios(); pair_by_num={n:i for i,p in enumerate(pairs) for n in p}
    eligible=[i for i,m in enumerate(meta) if base_number(m["folio"]) in pair_by_num]
    if len(eligible)<100: return {"eligible":len(eligible),"status":"insufficient"}
    pair_ids=np.array([pair_by_num[base_number(meta[i]["folio"])] for i in eligible])
    held_pairs={i for i in range(len(pairs)) if hashlib.sha256(f"{SEED}:{i}".encode()).digest()[0] < 52}
    train=np.array([eligible[j] for j,p in enumerate(pair_ids) if int(p) not in held_pairs],int)
    test=np.array([eligible[j] for j,p in enumerate(pair_ids) if int(p) in held_pairs],int)
    if len(train)<100 or len(test)<30: return {"eligible":len(eligible),"status":"split_too_small"}
    k=min(128,max(16,int(math.sqrt(len(train)/2))))
    km=MiniBatchKMeans(n_clusters=k,random_state=SEED,batch_size=2048,n_init=5,max_iter=200).fit(X[train])
    tr_lab=km.labels_; tr_cent=km.cluster_centers_; tr_cent/=np.linalg.norm(tr_cent,axis=1,keepdims=True)+1e-9
    tr_sim=np.sum(X[train]*tr_cent[tr_lab],axis=1)
    thresholds=np.array([np.quantile(tr_sim[tr_lab==c],.05) if np.any(tr_lab==c) else 1.0 for c in range(k)])
    sims=X[test]@tr_cent.T; te_lab=sims.argmax(1); te_sim=sims[np.arange(len(test)),te_lab]
    accepted=te_sim>=thresholds[te_lab]
    rec=[]
    for c in range(k):
        tr_f={meta[i]["folio"] for i in train[tr_lab==c]}; te_f={meta[i]["folio"] for i,a,l in zip(test,accepted,te_lab) if a and l==c}
        rec.append({"cluster":c,"train_n":int(np.sum(tr_lab==c)),"train_folios":len(tr_f),"heldout_n":int(np.sum((te_lab==c)&accepted)),"heldout_folios":len(te_f)})
    recurrent=[r for r in rec if r["train_n"]>=10 and r["train_folios"]>=3 and r["heldout_n"]>=3 and r["heldout_folios"]>=2]
    eligible_clusters=[r for r in rec if r["train_n"]>=10 and r["train_folios"]>=3]
    return {"status":"ok","pairs":pairs,"heldout_pair_indices":sorted(held_pairs),"train_n":len(train),"heldout_n":len(test),"k":k,"heldout_assignment_accept_rate":float(np.mean(accepted)),"eligible_train_clusters":len(eligible_clusters),"recurrent_clusters":len(recurrent),"recurrent_fraction":len(recurrent)/max(1,len(eligible_clusters))}


def visual_variant_function_test(X: np.ndarray, meta: list[dict[str,Any]], rng: np.random.Generator) -> dict[str,Any]:
    by=defaultdict(list)
    for i,m in enumerate(meta):
        if m.get("eva"): by[m["eva"]].append(i)
    types=[(e,idx) for e,idx in by.items() if len(idx)>=20 and len({meta[i]["folio"] for i in idx})>=3]
    results=[]
    for eva,idx in sorted(types,key=lambda z:len(z[1]),reverse=True)[:150]:
        idx=np.array(idx,int); km=MiniBatchKMeans(n_clusters=2,random_state=SEED,n_init=10,batch_size=256).fit(X[idx]); lab=km.labels_
        if min(np.bincount(lab))<5: continue
        obs=abs(np.mean([meta[i]["line_start"] for i,l in zip(idx,lab) if l==0])-np.mean([meta[i]["line_start"] for i,l in zip(idx,lab) if l==1]))
        null=[]; fol=defaultdict(list)
        for pos,i in enumerate(idx): fol[meta[i]["folio"]].append(pos)
        for _ in range(200):
            pl=lab.copy()
            for arr in fol.values(): pl[arr]=rng.permutation(pl[arr])
            d=abs(np.mean([meta[i]["line_start"] for i,l in zip(idx,pl) if l==0])-np.mean([meta[i]["line_start"] for i,l in zip(idx,pl) if l==1]))
            null.append(d)
        p=(1+sum(x>=obs for x in null))/(len(null)+1)
        results.append({"eva":eva,"n":len(idx),"observed_line_start_gap":float(obs),"p":float(p)})
    ps=np.array([r["p"] for r in results]) if results else np.array([]); sig=[]
    if len(ps):
        order=np.argsort(ps); cutoff=-1
        for rank,j in enumerate(order,1):
            if ps[j] <= .05*rank/len(ps): cutoff=rank
        if cutoff>0: sig=[results[int(j)] for j in order[:cutoff]]
    return {"tested_eva_types":len(results),"fdr05_significant_types":len(sig),"significant":sig[:30],"median_gap":float(np.median([r["observed_line_start_gap"] for r in results])) if results else None}


def page_section_leakage(X: np.ndarray, meta: list[dict[str,Any]]) -> dict[str,Any]:
    labels=[m.get("site_class","") for m in meta]; counts=Counter(labels); keep_classes={c for c,n in counts.items() if c and n>=50}
    idx=np.array([i for i,l in enumerate(labels) if l in keep_classes],int)
    if len(idx)<200 or len(keep_classes)<2: return {"status":"insufficient","classes":counts}
    if len(idx)>12000: idx=np.random.default_rng(SEED).choice(idx,12000,replace=False)
    le=LabelEncoder(); y=le.fit_transform([labels[i] for i in idx]); groups=np.array([meta[i]["folio"] for i in idx])
    clf=LogisticRegression(max_iter=400,n_jobs=1,C=.5); cv=GroupKFold(n_splits=min(5,len(set(groups))))
    scores=cross_val_score(clf,X[idx],y,groups=groups,cv=cv,scoring="accuracy")
    return {"status":"ok","classes":le.classes_.tolist(),"n":len(idx),"grouped_cv_accuracy_mean":float(scores.mean()),"grouped_cv_accuracy_sd":float(scores.std()),"majority_baseline":float(np.bincount(y).max()/len(y))}


def make_contact_sheet(discovery: dict[str,Any], prop_meta: list[dict[str,Any]], work: Path, canvases: list[Canvas], limit_clusters: int=24, members_per: int=6) -> tuple[bytes,list[dict[str,Any]]]:
    idx=discovery["sample_indices"]; ass=discovery["assignments"]; clusters=discovery["clusters"]
    chosen=[c for c in clusters if c["distinct_folios"]>=3][:limit_clusters]
    cellw,cellh=170,105; margin=24; titleh=25
    sheet=Image.new("RGB",(margin*2+members_per*cellw,margin*2+len(chosen)*(cellh+titleh)),"white")
    draw=ImageDraw.Draw(sheet); summaries=[]; target_cache={}
    for row,c in enumerate(chosen):
        cid=c["cluster_id"]; loc=np.where(ass==cid)[0]; picks=[c["medoid_sample_index"]]; seen_f=set()
        for q in loc:
            pm=prop_meta[int(idx[q])]
            if pm["folio"] not in seen_f: picks.append(int(q)); seen_f.add(pm["folio"])
            if len(picks)>=members_per: break
        picks=picks[:members_per]; y0=margin+row*(cellh+titleh)
        draw.text((margin,y0),f"C{cid} n={c['size']} folios={c['distinct_folios']} cos={c['mean_centroid_cosine']:.3f}",fill="black")
        sm={k:v for k,v in c.items() if k!="centroid"}; sm["members"]=[]
        for col,q in enumerate(picks):
            pm=prop_meta[int(idx[q])]; ci=pm["canvas_index"]
            if ci not in target_cache: target_cache[ci]=Image.open(work/"targets"/f"{ci:03d}.jpg").convert("RGB")
            im=target_cache[ci]; x,y,w,h=pm["derivative_xywh"]; crop=im.crop((x,y,x+w,y+h)); crop.thumbnail((cellw-8,cellh-20))
            x0=margin+col*cellw; sheet.paste(crop,(x0+(cellw-crop.width)//2,y0+titleh)); draw.text((x0,y0+titleh+cellh-16),pm["folio"],fill="black")
            sm["members"].append({"folio":pm["folio"],"source_index":pm["source_index"],"proposal_index":pm["proposal_index"],"proposal_type":pm["proposal_type"],"yale_xywh":pm["yale_xywh"]})
        summaries.append(sm)
    out=io.BytesIO(); sheet.save(out,"JPEG",quality=86,optimize=True); return out.getvalue(),summaries


def emit_b64(name: str, data: bytes, chunk: int=70000) -> None:
    s=base64.b64encode(data).decode(); print(f"{name}_CHUNKS={math.ceil(len(s)/chunk)}",flush=True)
    for i in range(0,len(s),chunk): print(f"{name}_{i//chunk:04d}={s[i:i+chunk]}",flush=True)


def sanitize_json(x: Any) -> Any:
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict): return {k:sanitize_json(v) for k,v in x.items()}
    if isinstance(x,list): return [sanitize_json(v) for v in x]
    if isinstance(x,tuple): return [sanitize_json(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.integer,np.floating)): return x.item()
    return x


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["smoke","full"],default=os.getenv("MODE","full")); ap.add_argument("--work",default="/tmp/voynich_dinov3"); ap.add_argument("--registration-width",type=int,default=2500); ap.add_argument("--batch-size",type=int,default=int(os.getenv("BATCH_SIZE","64"))); ap.add_argument("--max-discovery",type=int,default=int(os.getenv("MAX_DISCOVERY","120000"))); args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); rng=np.random.default_rng(SEED); work=Path(args.work); work.mkdir(parents=True,exist_ok=True); timer=Timer()
    session=requests.Session(); session.headers["User-Agent"]="Voynich-DINOv3-research/1.0"
    home=retry_get(session,HOME_URL).text; all_folios,classes=parse_home(home)
    folios=[f for f in ["f1r","f45r","f57v","f67r2","f68v1"] if f in all_folios] if args.mode=="smoke" else all_folios
    canvases=parse_manifest(retry_get(session,MANIFEST_URL).json(),args.registration_width); log("inventory",mode=args.mode,folios=len(folios),canvases=len(canvases),protocol=PROTOCOL_VERSION)
    download_inputs(work,folios,canvases); workers=max(1,min(8,(os.cpu_count() or 4)-1)); regs,selected=run_registrations(work,folios,canvases,workers)
    accepted=[f for f in folios if f in selected and selected[f]["accepted"]]; rejected=[f for f in folios if f not in selected or not selected[f]["accepted"]]; log("registration_complete",accepted=len(accepted),rejected=len(rejected),rejected_folios=rejected)
    token=os.getenv("HF_TOKEN"); emb=DinoEmbedder(MODEL_B,token,args.batch_size)
    word_meta=[]; word_raw_chunks=[]; word_norm_chunks=[]; prop_meta=[]; prop_chunks=[]; audit_images=[]; audit_base=[]; audit_meta=[]; dense_word_images=[]; dense_word_meta=[]; total_words=0; total_props=0
    for fi,f in enumerate(accepted,1):
        r=selected[f]; ci=r["candidate_index"]; target=Image.open(work/"targets"/f"{ci:03d}.jpg").convert("RGB"); tw,th=target.size; unique,boxes=parse_runtime(work/"scripts"/f"{f}.js"); lineinfo=infer_lines(boxes); Hder=np.array(r["homography_derivative"],np.float64); Hfull=np.array(r["homography_full"],np.float64)
        raw_imgs=[]; norm_imgs=[]; fol_word_meta=[]; fol_prop_imgs=[]; fol_prop_meta=[]; temp=[]
        for si,b in enumerate(boxes):
            wi,x,y,w,h=b; eva=unique[int(wi)][0] if 0<=int(wi)<len(unique) else None; qd,bd=perspective_rect((float(x),float(y),float(w),float(h)),Hder); qf,bf=perspective_rect((float(x),float(y),float(w),float(h)),Hfull); bd=clamp_xywh(bd,tw,th,pad=5); bf=clamp_xywh(bf,r["canvas_width"],r["canvas_height"],pad=max(2,int(5*r["canvas_width"]/tw)))
            if bd[2]<2 or bd[3]<2 or bf[2]<1 or bf[3]<1: continue
            crop=target.crop((bd[0],bd[1],bd[0]+bd[2],bd[1]+bd[3])); norm=normalize_ink(crop); info=lineinfo.get(si,{})
            m={"folio":f,"canvas_index":ci,"canvas_label":r["candidate_label"],"source_index":si,"word_index":int(wi),"eva":eva,"source_rect":[float(x),float(y),float(w),float(h)],"derivative_xywh":bd,"yale_xywh":bf,"yale_polygon":qf.tolist(),"site_class":classes.get(f,""),"ink_fraction":ink_fraction(norm),**info}
            temp.append((m,crop,norm)); raw_imgs.append(crop); norm_imgs.append(norm); fol_word_meta.append(m)
        byline=defaultdict(list)
        for j,m in enumerate(fol_word_meta): byline[m.get("line_index",-1)].append((m.get("token_index",0),j))
        for arr in byline.values():
            arr.sort()
            for pos,(_,j) in enumerate(arr): fol_word_meta[j]["prev_eva"]=fol_word_meta[arr[pos-1][1]]["eva"] if pos else None; fol_word_meta[j]["next_eva"]=fol_word_meta[arr[pos+1][1]]["eva"] if pos+1<len(arr) else None
        R=emb.embed(raw_imgs); N=emb.embed(norm_imgs); word_raw_chunks.append(R.astype(np.float16)); word_norm_chunks.append(N.astype(np.float16)); word_meta.extend(fol_word_meta)
        for j,(m,crop,norm) in enumerate(temp):
            ps=make_proposals(norm)
            for p in ps:
                x,y,w,h=p["xywh"]
                if w<2 or h<2: continue
                pc=norm.crop((x,y,x+w,y+h)); inf=ink_fraction(pc)
                if inf<0.005 or inf>0.92: continue
                full=[m["yale_xywh"][0]+int(round(x*m["yale_xywh"][2]/max(1,m["derivative_xywh"][2]))),m["yale_xywh"][1]+int(round(y*m["yale_xywh"][3]/max(1,m["derivative_xywh"][3]))),max(1,int(round(w*m["yale_xywh"][2]/max(1,m["derivative_xywh"][2])))),max(1,int(round(h*m["yale_xywh"][3]/max(1,m["derivative_xywh"][3]))))]; dm=[m["derivative_xywh"][0]+x,m["derivative_xywh"][1]+y,w,h]
                pm={"folio":f,"canvas_index":ci,"source_index":m["source_index"],"eva":m["eva"],"line_index":m.get("line_index"),"proposal_index":p["proposal_index"],"proposal_type":p["proposal_type"],"word_relative_xywh":[x,y,w,h],"derivative_xywh":dm,"yale_xywh":full,"ink_fraction":inf,"site_class":m.get("site_class","")}; fol_prop_imgs.append(pc); fol_prop_meta.append(pm)
                hsh=hashlib.sha256(f"{SEED}:{f}:{m['source_index']}:{p['proposal_index']}".encode()).digest()
                if hsh[0]<3 and len(audit_images)<1600: audit_images.append(pc.copy()); audit_meta.append({"entity":"proposal",**pm})
            hsh=hashlib.sha256(f"dense:{SEED}:{f}:{m['source_index']}".encode()).digest()
            if hsh[0]<2 and len(dense_word_images)<700: dense_word_images.append(norm.copy()); dense_word_meta.append(m)
            hsh=hashlib.sha256(f"auditword:{SEED}:{f}:{m['source_index']}".encode()).digest()
            if hsh[0]<2 and len(audit_images)<1600: audit_images.append(norm.copy()); audit_meta.append({"entity":"word",**m})
        P=emb.embed(fol_prop_imgs); prop_chunks.append(P.astype(np.float16)); prop_meta.extend(fol_prop_meta); total_words+=len(fol_word_meta); total_props+=len(fol_prop_meta); log("folio_embedded",done=fi,total=len(accepted),folio=f,words=len(fol_word_meta),proposals=len(fol_prop_meta),cum_words=total_words,cum_proposals=total_props,elapsed_s=round(timer.elapsed(),1)); del target,raw_imgs,norm_imgs,fol_prop_imgs,R,N,P; gc.collect()
    WR=np.concatenate(word_raw_chunks).astype(np.float32) if word_raw_chunks else np.empty((0,768),np.float32); WN=np.concatenate(word_norm_chunks).astype(np.float32) if word_norm_chunks else np.empty((0,768),np.float32); PP=np.concatenate(prop_chunks).astype(np.float32) if prop_chunks else np.empty((0,768),np.float32)
    for X in (WR,WN,PP):
        if len(X): X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-9
    WF=WR+WN
    if len(WF): WF/=np.linalg.norm(WF,axis=1,keepdims=True)+1e-9
    log("base_embeddings_complete",words=len(word_meta),proposals=len(prop_meta),revision=emb.revision)
    dense_result={"n_images":len(dense_word_images)}
    if len(dense_word_images)>=20:
        DT=emb.embed(dense_word_images,dense=True); same=[]; diff=[]; by=defaultdict(list)
        for i,m in enumerate(dense_word_meta):
            if m.get("eva"): by[m["eva"]].append(i)
        eligible=[v for v in by.values() if len({dense_word_meta[i]["folio"] for i in v})>=2]
        for _ in range(min(300,len(dense_word_images)*2)):
            if eligible:
                g=eligible[int(rng.integers(len(eligible)))]; a,b=rng.choice(g,2,replace=False)
                if dense_word_meta[a]["folio"]!=dense_word_meta[b]["folio"]: same.append(float((DT[a]@DT[b].T).max(axis=1).mean()))
            a,b=rng.choice(len(DT),2,replace=False)
            if dense_word_meta[a].get("eva")!=dense_word_meta[b].get("eva"): diff.append(float((DT[a]@DT[b].T).max(axis=1).mean()))
        if same and diff: dense_result.update({"same_mean":float(np.mean(same)),"different_mean":float(np.mean(diff)),"auc":float(roc_auc_score([1]*len(same)+[0]*len(diff),same+diff)),"same_n":len(same),"different_n":len(diff)})
        del DT
    audits={"raw":word_audit(WR,word_meta,rng),"ink_normalized":word_audit(WN,word_meta,rng),"fused":word_audit(WF,word_meta,rng)}; discovery=discover_clusters(PP,prop_meta,args.max_discovery,rng); held=heldout_bifolio_test(PP,prop_meta,rng); functional=visual_variant_function_test(WN,word_meta,rng); leakage=page_section_leakage(WN,word_meta)
    stability={"status":"insufficient"}; clusters=discovery["clusters"]; stable_clusters=[c for c in clusters if c["size"]>=10 and c["distinct_folios"]>=3][:80]
    if stable_clusters:
        idx=discovery["sample_indices"]; ass=discovery["assignments"]; cents=np.stack([c["centroid"] for c in stable_clusters]); ids=[c["cluster_id"] for c in stable_clusters]; aug_imgs=[]; original_ids=[]
        for c in stable_clusters:
            members=np.where(ass==c["cluster_id"])[0][:12]
            for q in members:
                pm=prop_meta[int(idx[q])]; im=Image.open(work/"targets"/f"{pm['canvas_index']:03d}.jpg").convert("RGB"); x,y,w,h=pm["derivative_xywh"]; crop=normalize_ink(im.crop((x,y,x+w,y+h)))
                for v in (1,2,3,4): aug_imgs.append(make_variant(crop,v)); original_ids.append(c["cluster_id"])
        AV=emb.embed(aug_imgs); sims=AV@cents.T; pred=np.array([ids[j] for j in sims.argmax(1)]); stability={"status":"ok","n":len(pred),"same_cluster_rate":float(np.mean(pred==np.array(original_ids))),"mean_best_cosine":float(np.max(sims,axis=1).mean())}; del AV,aug_imgs
    if audit_images: audit_base=emb.embed(audit_images)
    model_b_revision=emb.revision; emb.close(); del emb; gc.collect(); large_audit={"status":"not_run"}
    if audit_images:
        try:
            embL=DinoEmbedder(MODEL_L,token,max(8,args.batch_size//2)); AL=embL.embed(audit_images); AB=np.asarray(audit_base,np.float32); Db,Ib=faiss_knn(AB,k=min(11,len(AB)),hnsw=False); Dl,Il=faiss_knn(AL,k=min(11,len(AL)),hnsw=False); overlaps=[]
            for i in range(len(AB)): overlaps.append(len(set(Ib[i,1:])&set(Il[i,1:]))/max(1,len(set(Ib[i,1:])|set(Il[i,1:]))))
            large_audit={"status":"ok","n":len(AB),"model":MODEL_L,"revision":embL.revision,"mean_knn10_jaccard":float(np.mean(overlaps))}; embL.close(); del embL,AL,AB
        except Exception as e: large_audit={"status":"failed","error":f"{type(e).__name__}: {e}"}; traceback.print_exc()
    contact,medoids=make_contact_sheet(discovery,prop_meta,work,canvases); reg_comp=zlib.compress(json.dumps(sanitize_json({"selected":selected,"candidates":regs}),separators=(",",":"),allow_nan=False).encode(),9); med_comp=zlib.compress(json.dumps(sanitize_json(medoids),separators=(",",":"),allow_nan=False).encode(),9)
    reg_stats={"candidate_records":len(regs),"accepted_folios":len(accepted),"rejected_folios":rejected,"median_inliers":float(np.median([selected[f]["inliers"] for f in accepted])) if accepted else None,"median_inlier_ratio":float(np.median([selected[f]["inlier_ratio"] for f in accepted])) if accepted else None,"median_reprojection_px":float(np.median([selected[f]["median_reprojection_px"] for f in accepted])) if accepted else None,"p95_reprojection_px_across_selected":float(np.quantile([selected[f]["p95_reprojection_px"] for f in accepted],.95)) if accepted else None}
    cluster_public=[{k:v for k,v in c.items() if k!="centroid"} for c in clusters[:200]]; admission={"augmentation_stability":stability,"heldout_bifolio":held,"hand_or_page_confound_control":{"page_or_section_leakage":leakage,"true_hand_labels_available":False},"functional_beyond_eva":functional,"blinded_visual_audit":{"status":"contact_sheet_generated_not_yet_blinded","clusters_shown":len(medoids)}}; criteria={"augmentation_stable":stability.get("same_cluster_rate",0)>=.80,"heldout_bifolia_recur":held.get("recurrent_fraction",0)>=.50,"not_hand_classifier":False,"independent_functional_prediction":functional.get("fdr05_significant_types",0)>0,"blinded_visual_audit_passed":False}; admitted=all(v is True for v in criteria.values())
    summary={"status":"GLYPH_ONTOLOGY_ADMITTED" if admitted else "PIPELINE_COMPLETE_NO_GLYPH_ONTOLOGY_ADMITTED","run":{"protocol":PROTOCOL_VERSION,"mode":args.mode,"seed":SEED,"elapsed_s":timer.elapsed(),"model_b":MODEL_B,"model_b_revision":model_b_revision},"registration":reg_stats,"counts":{"voynichese_panels":len(folios),"registered_panels":len(accepted),"mapped_words":len(word_meta),"image_derived_proposals":len(prop_meta),"discovery_sample":discovery.get("sample_n")},"word_embedding_audit":audits,"dense_patch_audit":dense_result,"proposal_discovery":{"threshold":discovery.get("threshold"),"mutual_edges":discovery.get("mutual_edges"),"clusters":len(clusters),"unclustered_rate":discovery.get("unclustered_rate"),"top_clusters":cluster_public[:50]},"dinov3_large_audit":large_audit,"admission_tests":admission,"admission_criteria":criteria,"glyph_ontology_admitted":admitted,"limitations":["No independent Davis-scribe/Currier-hand labels were available in the connected catalogue; the hand-confound criterion therefore cannot pass.","The visual audit artifact is generated but has not yet been judged blind by independent readers.","The safe bifolio map deliberately excludes uncertain foldout gatherings; its recurrence result is conservative and partial.","Raw corpus embeddings were computed in-job but are not emitted through logs; compact derived artifacts and exact rerun code are emitted."]}
    print("FINAL_SUMMARY_JSON="+json.dumps(sanitize_json(summary),separators=(",",":"),ensure_ascii=False,allow_nan=False),flush=True); emit_b64("REGISTRATIONS_ZLIB_B64",reg_comp); emit_b64("MEDOIDS_ZLIB_B64",med_comp); emit_b64("CONTACT_SHEET_JPEG_B64",contact); log("pipeline_finished",status=summary["status"],elapsed_s=round(timer.elapsed(),1))


if __name__ == "__main__": main()
