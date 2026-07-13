#!/usr/bin/env python3
"""Patch full_pipeline.py to bound tiny proposals, then execute a real module file."""
from pathlib import Path
import subprocess
import sys

p = Path(__file__).with_name("full_pipeline.py")
s = p.read_text()
start = s.index("def make_proposals(im: Image.Image) -> list[dict[str, Any]]:")
end = s.index("\n\ndef make_variant", start)
replacement = '''def make_proposals(im: Image.Image) -> list[dict[str, Any]]:
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
s = s[:start] + replacement + s[end:]
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
patched = Path('/tmp/voynich_full_pipeline_v3.py')
patched.write_text(s)
raise SystemExit(subprocess.call([sys.executable, str(patched), *sys.argv[1:]]))
