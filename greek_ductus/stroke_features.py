#!/usr/bin/env python3
"""Fixed CPU shape/skeleton descriptor for preregistered Greek-ductus test.

This module operates on normalized black-ink-on-white crop images. It does not
infer true pen order from static images. Directional quantities are local
skeleton-geometry proxies only.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import euler_number

FEATURE_VERSION="2026-08-31.stroke.v1"
ORI_BINS=12
GRID=(4,8)

def _mask(im:Image.Image)->np.ndarray:
    a=np.array(im.convert('L'))
    return a<180

def _skeleton(m:np.ndarray)->np.ndarray:
    return skeletonize(m).astype(np.uint8)

def _degree(s:np.ndarray)->np.ndarray:
    k=np.ones((3,3),np.uint8); k[1,1]=0
    return cv2.filter2D(s,-1,k,borderType=cv2.BORDER_CONSTANT)

def _orientation_hist(s:np.ndarray)->np.ndarray:
    # Count undirected local skeleton links once. Bins cover [0,pi).
    ys,xs=np.where(s>0); h=np.zeros(ORI_BINS,np.float64)
    dirs=((0,1),(1,-1),(1,0),(1,1))
    H,W=s.shape
    for y,x in zip(ys,xs):
        for dy,dx in dirs:
            yy,xx=y+dy,x+dx
            if 0<=yy<H and 0<=xx<W and s[yy,xx]:
                ang=math.atan2(dy,dx)%math.pi
                b=min(ORI_BINS-1,int(ang/math.pi*ORI_BINS)); h[b]+=1
    if h.sum(): h/=h.sum()
    return h

def _grid_density(m:np.ndarray)->np.ndarray:
    H,W=m.shape; gy,gx=GRID; vals=[]
    for iy in range(gy):
        y0=round(iy*H/gy); y1=round((iy+1)*H/gy)
        for ix in range(gx):
            x0=round(ix*W/gx); x1=round((ix+1)*W/gx)
            p=m[y0:y1,x0:x1]; vals.append(float(p.mean()) if p.size else 0.)
    return np.asarray(vals,np.float64)

def _projections(m:np.ndarray)->np.ndarray:
    # coarse normalized horizontal/vertical projection profiles
    H,W=m.shape
    v=m.mean(0); h=m.mean(1)
    def pool(x,n=16):
        return np.array([float(x[round(i*len(x)/n):round((i+1)*len(x)/n)].mean()) if round((i+1)*len(x)/n)>round(i*len(x)/n) else 0. for i in range(n)])
    return np.r_[pool(v),pool(h)]

def descriptor(im:Image.Image)->np.ndarray:
    m=_mask(im); H,W=m.shape
    if not m.any(): return np.zeros(8+ORI_BINS+GRID[0]*GRID[1]+32+7,dtype=np.float32)
    s=_skeleton(m); deg=_degree(s)
    ys,xs=np.where(m); yss,xss=np.where(s)
    ink=float(m.mean()); sk=float(s.mean());
    endpoints=float(np.logical_and(s>0,deg==1).sum()); junctions=float(np.logical_and(s>0,deg>=3).sum())
    comps=cv2.connectedComponents(m.astype(np.uint8),8)[0]-1
    holes=max(0,int(comps-euler_number(m,connectivity=2)))
    ar=float((xs.max()-xs.min()+1)/max(1,ys.max()-ys.min()+1))
    # Hu moments on binary ink; signed log transform for dynamic-range control
    mom=cv2.moments(m.astype(np.uint8)); hu=cv2.HuMoments(mom).ravel()
    hu=np.sign(hu)*np.log10(np.abs(hu)+1e-30)
    base=np.array([
        math.log1p(W/H), ink, sk,
        endpoints/max(1,s.sum()), junctions/max(1,s.sum()),
        math.log1p(max(0,comps)), math.log1p(holes), math.log1p(ar)
    ],dtype=np.float64)
    out=np.r_[base,_orientation_hist(s),_grid_density(m),_projections(m),hu]
    return out.astype(np.float32)

def feature_names()->list[str]:
    n=['log_canvas_aspect','ink_density','skeleton_density','endpoint_rate','junction_rate','log_components','log_holes','log_ink_aspect']
    n += [f'ori_{i}' for i in range(ORI_BINS)]
    n += [f'grid_{y}_{x}' for y in range(GRID[0]) for x in range(GRID[1])]
    n += [f'vproj_{i}' for i in range(16)]+[f'hproj_{i}' for i in range(16)]
    n += [f'hu_{i}' for i in range(7)]
    return n

if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument('image');args=ap.parse_args()
    x=descriptor(Image.open(args.image));print(json.dumps({'version':FEATURE_VERSION,'dim':len(x),'names':feature_names(),'features':x.tolist()}))
