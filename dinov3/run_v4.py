#!/usr/bin/env python3
"""Patch the full pipeline with deterministic multiscale registration and bounded proposals."""
from pathlib import Path
import subprocess
import sys

p = Path(__file__).with_name("full_pipeline.py")
s = p.read_text()

# Canonical Yale derivative used for all downstream crops. Registration is
# performed on local downscales of this image and transformed back to canonical.
s = s.replace('ap.add_argument("--registration-width",type=int,default=2500)',
              'ap.add_argument("--registration-width",type=int,default=1800)')

reg_start = s.index("def _register_task(args: tuple[str, int, dict[str, Any], str, str, Thresholds]) -> dict[str, Any]:")
reg_end = s.index("\n\ndef run_registrations", reg_start)
reg_replacement = r'''def _register_task(args: tuple[str, int, dict[str, Any], str, str, Thresholds]) -> dict[str, Any]:
    """Register at several target scales, returning a canonical-target homography.

    Voynichese glance images are about 636 px wide. A 2500 px Yale target creates
    a severe feature-scale mismatch on some folios. We therefore search fixed,
    predeclared local target widths, evaluate the same numerical gates at a
    canonical 1800 px scale, and return a homography into the canonical image.
    """
    folio, idx, cd, source_path, target_path, thresholds = args
    c = Canvas(**cd)
    try:
        src = cv2.imread(source_path, cv2.IMREAD_COLOR)
        canonical = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if src is None or canonical is None:
            raise ValueError("image decode failure")
        cth, ctw = canonical.shape[:2]
        widths = sorted({w for w in (1000, 1200, 1400, 1600, 1800, ctw) if 600 <= w <= ctw})
        best = None
        scale_errors = []
        for regw in widths:
            try:
                regh = max(1, int(round(cth * regw / ctw)))
                tgt = canonical if regw == ctw else cv2.resize(canonical, (regw, regh), interpolation=cv2.INTER_AREA)
                seed = int(hashlib.sha256(f"{SEED}:{folio}:{idx}:{regw}".encode()).hexdigest()[:8], 16) & 0x7fffffff
                cv2.setRNGSeed(seed)
                sift = cv2.SIFT_create(nfeatures=14000, contrastThreshold=0.02, edgeThreshold=12)
                k1, d1 = sift.detectAndCompute(_prep_gray(src), None)
                k2, d2 = sift.detectAndCompute(_prep_gray(tgt), None)
                if d1 is None or d2 is None:
                    raise ValueError("no descriptors")
                pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
                good = [a for a, b in pairs if a.distance < 0.77 * b.distance]
                if len(good) < 4:
                    raise ValueError(f"only {len(good)} ratio-test matches")
                p1 = np.float32([k1[m.queryIdx].pt for m in good])
                p2 = np.float32([k2[m.trainIdx].pt for m in good])
                method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
                Hs, mask = cv2.findHomography(p1, p2, method, 3.0, maxIters=20000, confidence=0.999)
                if Hs is None or mask is None:
                    raise ValueError("homography failure")
                keep = mask.ravel().astype(bool)
                proj = cv2.perspectiveTransform(p1[:, None, :], Hs)[:, 0, :]
                native_err = np.linalg.norm(proj - p2, axis=1)[keep]
                if not len(native_err):
                    raise ValueError("zero inlier errors")

                # Convert the scale-specific homography and error into the
                # canonical 1800px target coordinate system.
                R = np.diag([ctw / regw, cth / regh, 1.0])
                Hcanon = R @ Hs
                error_scale = 0.5 * ((ctw / regw) + (cth / regh))
                err = native_err * error_scale
                sh, sw = src.shape[:2]
                sq = np.float32([[[0, 0], [sw, 0], [sw, sh], [0, sh]]])
                tq = cv2.perspectiveTransform(sq, Hcanon)[0]
                poly = tq.astype(np.float32)
                bounds = np.float32([[0, 0], [ctw, 0], [ctw, cth], [0, cth]])
                area = abs(float(cv2.contourArea(poly)))
                inter_area, _ = cv2.intersectConvexConvex(poly, bounds)
                intersection_ratio = float(inter_area / max(area, 1e-9))
                target_area_ratio = float(area / max(ctw * cth, 1))
                plausible = bool(intersection_ratio >= 0.80 and 0.005 <= target_area_ratio <= 1.5 and np.isfinite(Hcanon).all())
                inliers = int(keep.sum())
                ratio = float(keep.mean())
                med = float(np.median(err))
                p95 = float(np.quantile(err, 0.95))
                accepted = plausible and inliers >= thresholds.min_inliers and ratio >= thresholds.min_inlier_ratio and med <= thresholds.max_median_reprojection_px
                Sfull = np.diag([c.width / ctw, c.height / cth, 1.0])
                Hfull = Sfull @ Hcanon
                tq_full = cv2.perspectiveTransform(sq, Hfull)[0]
                r = RegistrationResult(
                    folio, idx, c.label, c.canvas_id, c.body_id, c.width, c.height,
                    ctw, cth, len(good), inliers, ratio, med, p95, plausible,
                    intersection_ratio, target_area_ratio, Hcanon.tolist(), Hfull.tolist(),
                    tq.tolist(), tq_full.tolist(), accepted, None
                )
                d = asdict(r)
                d.update({
                    "registration_scale_width": int(regw),
                    "registration_scale_height": int(regh),
                    "native_median_reprojection_px": float(np.median(native_err)),
                    "native_p95_reprojection_px": float(np.quantile(native_err, 0.95)),
                    "canonical_target_width": int(ctw),
                    "canonical_target_height": int(cth),
                })
                score = (bool(accepted), inliers, ratio, -med, regw)
                if best is None or score > best[0]:
                    best = (score, d)
            except Exception as e:
                scale_errors.append(f"{regw}:{type(e).__name__}:{e}")
        if best is None:
            raise ValueError("all scales failed: " + " | ".join(scale_errors))
        best[1]["scale_errors"] = scale_errors
        return best[1]
    except Exception as e:
        r = RegistrationResult(
            folio, idx, c.label, c.canvas_id, c.body_id, c.width, c.height,
            0, 0, error=f"{type(e).__name__}: {e}"
        )
        return asdict(r)
'''
s = s[:reg_start] + reg_replacement + s[reg_end:]

# Bounded, visually defensible proposals discovered during the smoke audit.
prop_start = s.index("def make_proposals(im: Image.Image) -> list[dict[str, Any]]:")
prop_end = s.index("\n\ndef make_variant", prop_start)
prop_replacement = r'''def make_proposals(im: Image.Image) -> list[dict[str, Any]]:
    """Bound proposals to components with defensible visual support."""
    g = np.array(im.convert("L"))
    H, W = g.shape
    if W < 8 or H < 8:
        return []
    block = max(7, (min(31, max(7, (min(W, H)//2)*2+1)) // 2)*2+1)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    min_area = max(8, int(round(0.002 * W * H)))
    comps = []
    for i in range(1, n):
        x, y, w, h, a = map(int, stats[i])
        if a < min_area or w*h < 24 or min(w, h) < 2 or max(w, h) < 5 or w*h > 0.95*W*H:
            continue
        comps.append([x, y, w, h, a])
    if len(comps) > 10:
        comps = sorted(comps, key=lambda z: z[4], reverse=True)[:10]
    comps.sort(key=lambda z: (z[0], z[1]))
    out = [{"proposal_type":"component", "xywh":c[:4], "ink_area":c[4]} for c in comps]
    if comps:
        medh = statistics.median([c[3] for c in comps]); gap_lim = max(2, int(0.20*medh))
        cur = comps[0][:4]; merged = []
        for c in comps[1:]:
            x,y,w,h = c[:4]; cx,cy,cw,ch = cur
            gap = x-(cx+cw); ov = max(0, min(cy+ch,y+h)-max(cy,y)); ovf = ov/max(1,min(ch,h))
            if gap <= gap_lim and ovf >= 0.20:
                x0=min(cx,x); y0=min(cy,y); x1=max(cx+cw,x+w); y1=max(cy+ch,y+h); cur=[x0,y0,x1-x0,y1-y0]
            else:
                if cur not in [q[:4] for q in comps] and cur[2]*cur[3] >= 32: merged.append(cur)
                cur=c[:4]
        if cur not in [q[:4] for q in comps] and cur[2]*cur[3] >= 32: merged.append(cur)
        for m in merged[:6]: out.append({"proposal_type":"merged_component", "xywh":m})
    for frac in (0.38, 0.62):
        ww = max(8, min(W, int(round(W*frac)))); step = max(4, int(round(ww*0.60)))
        xs = list(range(0, max(1, W-ww+1), step))
        if xs and xs[-1] != W-ww: xs.append(max(0, W-ww))
        for x in sorted(set(xs))[:4]: out.append({"proposal_type":"window", "xywh":[x,0,ww,H]})
    seen=set(); ded=[]
    for proposal in out:
        key=(proposal["proposal_type"],tuple(proposal["xywh"]))
        if key not in seen:
            seen.add(key); ded.append(proposal)
    ded = ded[:18]
    for i,proposal in enumerate(ded): proposal["proposal_index"] = i
    return ded


def prepare_model_image(im: Image.Image, min_side: int = 16) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    if w >= min_side and h >= min_side:
        return im
    canvas = Image.new("RGB", (max(min_side, w), max(min_side, h)), "white")
    canvas.paste(im, ((canvas.width-w)//2, (canvas.height-h)//2))
    return canvas
'''
s = s[:prop_start] + prop_replacement + s[prop_end:]

old = '''                b=images[i:i+self.batch_size]
                x=self.proc(images=b,return_tensors="pt").to("cuda")'''
new = '''                b=[prepare_model_image(im) for im in images[i:i+self.batch_size]]
                x=self.proc(images=b,return_tensors="pt").to("cuda")'''
assert old in s
s = s.replace(old, new)

old = '''            if bd[2]<2 or bd[3]<2 or bf[2]<1 or bf[3]<1: continue'''
new = '''            if bd[2] < 8 or bd[3] < 8 or bd[2]*bd[3] < 80 or bf[2] < 1 or bf[3] < 1: continue'''
assert old in s
s = s.replace(old, new)

old = '''                if w<2 or h<2: continue'''
new = '''                if w < 4 or h < 4 or w*h < 24: continue'''
assert old in s
s = s.replace(old, new)

# Permit a cheap, full-corpus registration audit before releasing the GPU stage.
needle = '''    accepted=[f for f in folios if f in selected and selected[f]["accepted"]]; rejected=[f for f in folios if f not in selected or not selected[f]["accepted"]]; log("registration_complete",accepted=len(accepted),rejected=len(rejected),rejected_folios=rejected)
    token=os.getenv("HF_TOKEN")'''
replacement = '''    accepted=[f for f in folios if f in selected and selected[f]["accepted"]]; rejected=[f for f in folios if f not in selected or not selected[f]["accepted"]]; log("registration_complete",accepted=len(accepted),rejected=len(rejected),rejected_folios=rejected)
    if os.getenv("REGISTRATION_ONLY") == "1":
        details={f:{"accepted":bool(selected.get(f,{}).get("accepted",False)),"candidate_index":selected.get(f,{}).get("candidate_index"),"candidate_label":selected.get(f,{}).get("candidate_label"),"registration_scale_width":selected.get(f,{}).get("registration_scale_width"),"inliers":selected.get(f,{}).get("inliers"),"inlier_ratio":selected.get(f,{}).get("inlier_ratio"),"median_reprojection_px":selected.get(f,{}).get("median_reprojection_px"),"error":selected.get(f,{}).get("error")} for f in folios}
        print("REGISTRATION_ONLY_JSON="+json.dumps({"protocol":PROTOCOL_VERSION+".multiscale","accepted":len(accepted),"rejected":rejected,"candidate_records":len(regs),"scale_counts":dict(Counter(selected[f].get("registration_scale_width") for f in accepted)),"details":details},separators=(",",":"),allow_nan=False),flush=True)
        return
    token=os.getenv("HF_TOKEN")'''
assert needle in s
s = s.replace(needle, replacement)

patched = Path('/tmp/voynich_full_pipeline_v4.py')
patched.write_text(s)
raise SystemExit(subprocess.call([sys.executable, str(patched), *sys.argv[1:]]))
