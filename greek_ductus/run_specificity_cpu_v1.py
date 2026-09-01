#!/usr/bin/env python3
from __future__ import annotations
import io,json,math,re,time,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np, requests, cv2
from PIL import Image
from skimage.morphology import skeletonize
import extract_cpu as ec

ROOT=Path(__file__).parent
SPEC=json.loads((ROOT/'specificity_manifest_v1.json').read_text())
UA={'User-Agent':'voynich-greek-specificity/2026-09-01'}
SEED=412
rng=np.random.default_rng(SEED)


def retry_get(url, binary=False, tries=5):
    last=None
    for k in range(tries):
        try:
            r=requests.get(url,headers=UA,timeout=90); r.raise_for_status()
            return r.content if binary else r
        except Exception as e:
            last=e; print(json.dumps({'event':'retry','url':url,'attempt':k+1,'error':repr(e)}),flush=True); time.sleep(2*(k+1))
    raise last

def get_json(url): return retry_get(url).json()
def fetch_img(url): return Image.open(io.BytesIO(retry_get(url,True))).convert('RGB')

def frac_indices(n,fracs): return sorted(set(max(0,min(n-1,int(round(f*(n-1))))) for f in fracs))
def even_take(items,n):
    if len(items)<=n:return items
    idx=np.linspace(0,len(items)-1,n).round().astype(int)
    return [items[int(i)] for i in idx]

def openn_urls(html_url):
    t=retry_get(html_url).text
    xs=re.findall(r'href="([^"]+_web\.jpg)"',t)
    seen=[]
    for x in xs:
        if x.startswith('/'): x='https://openn.library.upenn.edu'+x
        elif not x.startswith('http'): x=html_url.rsplit('/',1)[0]+'/'+x
        if x not in seen: seen.append(x)
    return seen

def page_crops_from_image(im, object_id, page_id):
    bw=ec.ink_mask(im); boxes=ec.word_boxes(bw); out=[]
    for j,b in enumerate(boxes):
        out.append((ec.norm_crop(bw,b),{'object':object_id,'page':int(page_id),'crop':j}))
    return out

def iiif_object(manifest, object_id, fracs=None, last6=False):
    cs=ec.canvases(get_json(manifest)); inds=list(range(max(0,len(cs)-6),len(cs))) if last6 else frac_indices(len(cs),fracs)
    out=[]
    for pi in inds:
        im=fetch_img(ec.image_url(cs[pi])); cc=page_crops_from_image(im,object_id,pi); out.extend(cc)
        print(json.dumps({'event':'page','object':object_id,'page':pi,'crops':len(cc)}),flush=True)
    return out

def openn_object(html, object_id, fracs):
    us=openn_urls(html); inds=frac_indices(len(us),fracs); out=[]
    for pi in inds:
        cc=page_crops_from_image(fetch_img(us[pi]),object_id,pi); out.extend(cc)
        print(json.dumps({'event':'page','object':object_id,'page':pi,'crops':len(cc)}),flush=True)
    return out

def descriptor(im):
    g=np.array(im.convert('L'),np.uint8); m=g<200
    ys,xs=np.where(m)
    if len(xs)<5:return None
    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max(); z=m[y0:y1+1,x0:x1+1]
    h,w=z.shape; area=float(z.sum()); box=float(max(1,h*w))
    ncc=cv2.connectedComponents(z.astype(np.uint8),8)[0]-1
    sk=skeletonize(z); sl=float(sk.sum())
    nb=cv2.filter2D(sk.astype(np.uint8),-1,np.ones((3,3),np.uint8))-sk.astype(np.uint8)
    end=float(np.sum(sk & (nb==1))); jun=float(np.sum(sk & (nb>=3)))
    cnt,hier=cv2.findContours((z*255).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes=0
    if hier is not None: holes=sum(1 for q in hier[0] if q[3]>=0)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(gx,gy); ang=(np.arctan2(gy,gx)+np.pi)%(np.pi)
    oh,_=np.histogram(ang,bins=12,range=(0,np.pi),weights=mag); oh=oh/(oh.sum()+1e-9)
    px=z.mean(0); py=z.mean(1)
    def bins(v,n):
        parts=np.array_split(v,n); return np.array([float(p.mean()) if len(p) else 0 for p in parts])
    bx=bins(px,16); by=bins(py,8)
    mom=cv2.moments((z*255).astype(np.uint8)); hu=cv2.HuMoments(mom).ravel(); hu=-np.sign(hu)*np.log10(np.abs(hu)+1e-30); hu=np.clip(hu,-20,20)
    base=np.array([w/max(1,h),area/box,ncc,sl/max(area,1),end/max(sl,1),jun/max(sl,1),holes,w/256,h/80],float)
    return np.concatenate([base,oh,bx,by,hu])

def robust_scale(X):
    med=np.median(X,0); q1=np.quantile(X,.25,axis=0); q3=np.quantile(X,.75,axis=0); s=q3-q1; s[s<1e-6]=1.0
    return med,s

def vec(crops):
    ds=[descriptor(im) for im,_ in crops]; ds=np.stack([d for d in ds if d is not None]); return ds

def main():
    objects={}; meta={}
    # ordinary controls
    for fam,rows in SPEC['ordinary'].items():
        for r in rows:
            oid=fam+'::'+r['shelfmark']; items=iiif_object(r['manifest'],oid,SPEC['control_page_fractions']); items=even_take(items,90)
            objects[oid]=vec(items); meta[oid]={'family':fam,'type':'ordinary','n':len(items)}
            print(json.dumps({'event':'object','object':oid,'selected':len(items)}),flush=True)
    # special comparators
    for s in SPEC['special']:
        oid=s['object']; st=s['source_type']
        if st=='iiif': items=iiif_object(s['manifest'],oid,SPEC['special_page_fractions'])
        elif st=='iiif_last6': items=iiif_object(s['manifest'],oid,last6=True)
        elif st=='openn': items=openn_object(s['html'],oid,SPEC['special_page_fractions'])
        else: raise ValueError(st)
        items=even_take(items,180); objects[oid]=vec(items); meta[oid]={'family':s['family'],'type':'special','n':len(items),'note':s.get('note')}
        print(json.dumps({'event':'object','object':oid,'selected':len(items)}),flush=True)
    # Voynich target
    items=iiif_object(SPEC['vms_manifest'],'VOYNICH',SPEC['vms_fractions']); items=even_take(items,270); objects['VOYNICH']=vec(items); meta['VOYNICH']={'family':'VOYNICH','type':'target','n':len(items)}

    ctrl=np.concatenate([X for k,X in objects.items() if k!='VOYNICH'],axis=0); med,scale=robust_scale(ctrl)
    Z={k:(X-med)/scale for k,X in objects.items()}; ov={k:np.median(X,0) for k,X in Z.items()}
    greek=[ov[k] for k in ov if meta[k]['family']=='GREEK']; latin=[ov[k] for k in ov if meta[k]['family']=='LATIN']
    cg=np.mean(greek,0); cl=np.mean(latin,0); vv=ov['VOYNICH']
    dist=lambda a,b:float(np.linalg.norm(a-b))
    dG,dL=dist(vv,cg),dist(vv,cl)
    special={k:dist(vv,ov[k]) for k in ov if meta[k]['type']=='special'}
    nearest_special=min(special,key=special.get); sm=special[nearest_special]-dG
    # family centroids for special families
    sf={}
    for fam in sorted({meta[k]['family'] for k in ov if meta[k]['type']=='special'}):
        ks=[k for k in ov if meta[k]['family']==fam]; c=np.mean([ov[k] for k in ks],0); sf[fam]=dist(vv,c)
    # ordinary LOO
    ordkeys=[k for k in ov if meta[k]['type']=='ordinary']; conf=Counter(); hits=0
    for k in ordkeys:
        fam=meta[k]['family']; gg=[ov[j] for j in ordkeys if j!=k and meta[j]['family']=='GREEK']; ll=[ov[j] for j in ordkeys if j!=k and meta[j]['family']=='LATIN']
        pred='GREEK' if dist(ov[k],np.mean(gg,0))<dist(ov[k],np.mean(ll,0)) else 'LATIN'; conf[(fam,pred)]+=1; hits+=pred==fam
    # reproduce Greek-vs-Latin manuscript-label null
    Xord=np.stack([ov[k] for k in ordkeys]); labs=np.array([meta[k]['family'] for k in ordkeys]); obs=dL-dG; null=[]
    for _ in range(20000):
        p=rng.permutation(labs); a=Xord[p=='GREEK'].mean(0); b=Xord[p=='LATIN'].mean(0); null.append(dist(vv,b)-dist(vv,a))
    null=np.array(null); z=float((obs-null.mean())/(null.std(ddof=1)+1e-12)); p=float((1+np.sum(null>=obs))/(len(null)+1))
    result={'protocol':SPEC['protocol'],'representation':'cpu_stroke_geometry_v1','descriptor_dim':int(vv.size),'objects':{k:{**meta[k],'distance_to_voynich':(0 if k=='VOYNICH' else dist(vv,ov[k]))} for k in ov},'distances':{'GREEK':dG,'LATIN':dL,'special_objects':special,'special_families':sf},'nearest_special':nearest_special,'specificity_margin_special_minus_greek':sm,'nearest_overall':min({'GREEK':dG,'LATIN':dL,**special},key={'GREEK':dG,'LATIN':dL,**special}.get),'ordinary_loo_accuracy':hits/len(ordkeys),'ordinary_loo_confusion':{f'{a}->{b}':n for (a,b),n in conf.items()},'greek_vs_latin_reproduction':{'advantage':obs,'perm_mean':float(null.mean()),'perm_sd':float(null.std(ddof=1)),'z':z,'empirical_p_one_sided':p,'permutations':20000},'decision_cpu':('GREEK_SURVIVES_SPECIALS' if sm>0 else 'SPECIAL_ABSORBS_GREEK_SIGNAL')}
    print('SPECIFICITY_CPU_RESULT='+json.dumps(result,separators=(',',':')),flush=True)
if __name__=='__main__':main()
