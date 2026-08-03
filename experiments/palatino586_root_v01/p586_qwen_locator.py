#!/usr/bin/env python3
import os, io, re, json, time, base64, hashlib, math, random, warnings
from pathlib import Path
import numpy as np
import requests
import torch
from PIL import Image, ImageOps, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

warnings.filterwarnings('ignore')
SEED=20260803
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RUN_ID=os.environ.get('RUN_ID','unknown')
MODEL_ID='Qwen/Qwen2.5-VL-3B-Instruct'
MANIFEST='https://iiif.archive.org/iiif/bncf-pal.-586-images/manifest.json'
SUPABASE='https://ymaqlcfjmdwncdbjprmw.supabase.co'
UPLOAD_ENDPOINT=SUPABASE+'/functions/v1/p586-root-upload-v01'
PREFIX='p586_root_v01/qwen_locator_'+RUN_ID
DINO_RUN=os.environ.get('DINO_RUN_ID','77e97abe-931d-41b5-939a-7f3f8b68ca97')
DINO_PROPOSALS=SUPABASE+'/storage/v1/object/public/bridge/p586_root_v01/dinov3_scan_'+DINO_RUN+'/proposals.json'
PAGES=list(range(22,80))
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
S=requests.Session(); S.headers.update({'User-Agent':'VoynichRootResearch/0.1'})


def log(event,**kw): print(json.dumps({'event':event,'time':time.time(),**kw},sort_keys=True),flush=True)
def get(url,tries=5,timeout=120):
    last=None
    for k in range(tries):
        try:
            r=S.get(url,timeout=timeout); r.raise_for_status(); return r
        except Exception as e: last=e; time.sleep(min(15,1.5**k+random.random()))
    raise RuntimeError(f'GET failed {url}: {last}')
def upload(relpath,ctype,data):
    payload={'path':f'{PREFIX}/{relpath}','content_type':ctype,'data_b64':base64.b64encode(data).decode()}
    r=S.post(UPLOAD_ENDPOINT,headers={'x-upload-token':os.environ['UPLOAD_TOKEN'],'content-type':'application/json'},json=payload,timeout=180)
    r.raise_for_status(); out=r.json(); log('upload',path=out.get('path'),bytes=out.get('bytes')); return out

def parse_manifest(m): return m.get('items') or (m.get('sequences') or [{}])[0].get('canvases') or []
def image_url(c,w=1200):
    try:
        b=c['items'][0]['items'][0]['body']; sv=b.get('service')
        if isinstance(sv,list): sv=sv[0] if sv else None
        if isinstance(sv,dict):
            base=sv.get('id') or sv.get('@id')
            if base:return base.rstrip('/')+f'/full/{w},/0/default.jpg'
        return b.get('id') or b.get('@id')
    except Exception: pass
    try:
        b=c['images'][0]['resource']; sv=b.get('service')
        if isinstance(sv,list): sv=sv[0] if sv else None
        if isinstance(sv,dict):
            base=sv.get('@id') or sv.get('id')
            if base:return base.rstrip('/')+f'/full/{w},/0/default.jpg'
        return b.get('@id') or b.get('id')
    except Exception:return None

def load_image(url): return ImageOps.exif_transpose(Image.open(io.BytesIO(get(url).content))).convert('RGB')
def parse_json(text):
    t=text.strip().replace('```json','').replace('```','').strip()
    try:return json.loads(t)
    except Exception: pass
    a=t.find('{'); b=t.rfind('}')
    if a>=0 and b>a:
        try:return json.loads(t[a:b+1])
        except Exception: pass
    return None

def run_vlm(model,processor,image,prompt,max_new=180):
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[image],padding=True,return_tensors='pt').to(DEVICE)
    with torch.inference_mode():
        out=model.generate(**inputs,max_new_tokens=max_new,do_sample=False,use_cache=True)
    trimmed=[o[len(i):] for i,o in zip(inputs.input_ids,out)]
    return processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

def norm_box_to_px(box,W,H,pad=.04):
    if not isinstance(box,list) or len(box)!=4:return None
    try:x0,y0,x1,y1=[float(v) for v in box]
    except:return None
    x0=max(0,min(1000,x0)); y0=max(0,min(1000,y0)); x1=max(0,min(1000,x1)); y1=max(0,min(1000,y1))
    if x1<x0:x0,x1=x1,x0
    if y1<y0:y0,y1=y1,y0
    px=[x0/1000*W,y0/1000*H,x1/1000*W,y1/1000*H]
    pw=(px[2]-px[0])*pad; ph=(px[3]-px[1])*pad
    return [max(0,int(px[0]-pw)),max(0,int(px[1]-ph)),min(W,int(px[2]+pw)),min(H,int(px[3]+ph))]
def iou(a,b):
    x0=max(a[0],b[0]); y0=max(a[1],b[1]); x1=min(a[2],b[2]); y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0); aa=max(1,(a[2]-a[0])*(a[3]-a[1])); bb=max(1,(b[2]-b[0])*(b[3]-b[1]))
    return inter/(aa+bb-inter)
def contact_sheet(rows,start,count=12):
    batch=rows[start:start+count]; cols=4; nr=math.ceil(len(batch)/cols); cw=300; ch=280
    sheet=Image.new('RGB',(cols*cw,nr*ch),'white'); d=ImageDraw.Draw(sheet)
    colors={'CLEAR_ROOT':'green','PARTIAL_ROOT':'orange','NOT_ROOT':'red','UNCERTAIN':'purple','UNPARSED':'gray'}
    for j,r in enumerate(batch):
        x=(j%cols)*cw; y=(j//cols)*ch; im=r['crop'].copy(); im.thumbnail((280,205))
        sheet.paste(im,(x+(cw-im.width)//2,y+45)); col=colors.get(r['qa_label'],'gray')
        d.rectangle((x,y,x+cw-1,y+ch-1),outline=col,width=4)
        d.text((x+6,y+6),f"canvas {r['page']:03d} root {r['root_index']}",fill='black')
        d.text((x+6,y+21),f"{r['qa_label']} {r['qa_confidence']:.2f} overlap {r['dino_overlap']:.2f}",fill='black')
    bio=io.BytesIO(); sheet.save(bio,'JPEG',quality=90,optimize=True); return bio.getvalue()

def page_sheet(page_rows,start,count=6):
    batch=page_rows[start:start+count]; cols=3; nr=math.ceil(len(batch)/cols); cw=400; ch=560
    sheet=Image.new('RGB',(cols*cw,nr*ch),'white'); d=ImageDraw.Draw(sheet)
    for j,r in enumerate(batch):
        x=(j%cols)*cw; y=(j//cols)*ch; im=r['image'].copy(); dr=ImageDraw.Draw(im)
        for k,b in enumerate(r['boxes']): dr.rectangle(b,outline='green',width=max(3,im.width//300))
        im.thumbnail((370,500)); sheet.paste(im,(x+(cw-im.width)//2,y+40)); d.rectangle((x,y,x+cw-1,y+ch-1),outline='black',width=1)
        d.text((x+8,y+8),f"canvas {r['page']:03d} roots {len(r['boxes'])}",fill='black')
        d.text((x+8,y+23),f"locator confidence {r['locator_confidence']:.2f}",fill='black')
    bio=io.BytesIO(); sheet.save(bio,'JPEG',quality=88,optimize=True); return bio.getvalue()

protocol={'protocol_id':'P586-ROOT-QWEN-LOCATOR-V0.1-20260803','run_id':RUN_ID,'seed':SEED,'model_id':MODEL_ID,'pages':PAGES,
          'independence':'full-page locator receives no Voynich image, embedding, similarity, or DINO proposal','coordinates':'normalized 0-1000','acceptance':'CLEAR_ROOT or PARTIAL_ROOT after separate crop QA','purpose':'first-pass crop construction'}
upload('protocol.json','application/json',json.dumps(protocol,indent=2,sort_keys=True).encode())
manifest=get(MANIFEST).json(); canvases=parse_manifest(manifest)
try:dino=get(DINO_PROPOSALS,tries=2,timeout=30).json()
except Exception:dino=[]
dino_by_page={}
for p in dino:dino_by_page.setdefault(int(p['page']),[]).append(p)

processor=AutoProcessor.from_pretrained(MODEL_ID,min_pixels=256*28*28,max_pixels=768*28*28)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID,torch_dtype=torch.float16,device_map='auto').eval()
locator_prompt='''You are locating botanical roots in a digitized medieval manuscript page. Examine only the image. Return exactly one JSON object with this schema: {"page_has_botanical_illustration": true, "roots": [{"bbox": [x0,y0,x1,y1], "confidence": 0.0, "visibility": "clear|partial"}], "note": "at most 12 words"}. Coordinates are normalized from 0 to 1000 over the full image. Each bbox must tightly contain the drawn root system and its immediate stem junction, excluding leaves, text, borders, and unrelated decoration. Roots may be branched, bulbous, tuberous, claw-like, or stylized. If no visible root system exists, return roots: []. If several separate plants have roots, return each root system separately, maximum three. Do not infer from filenames or manuscript identity.'''
qa_prompt='''This crop was proposed as the root system of a botanical illustration. Judge only what is visibly inside the crop. Return exactly JSON: {"label":"CLEAR_ROOT|PARTIAL_ROOT|NOT_ROOT|UNCERTAIN","confidence":0.0,"reason":"at most 12 words"}. CLEAR_ROOT means most of a recognizable root system is present. PARTIAL_ROOT means roots are present but truncated or mixed with substantial stem/text. NOT_ROOT means no visible roots. Do not infer from filename or context.'''

page_rows=[]; crop_rows=[]
for pi in PAGES:
    u=image_url(canvases[pi],1200)
    try:im=load_image(u)
    except Exception as e: log('page_fail',page=pi,error=str(e)); continue
    raw=run_vlm(model,processor,im,locator_prompt,200); obj=parse_json(raw)
    roots=[]; loc_conf=0.0
    if isinstance(obj,dict):
        rr=obj.get('roots') if isinstance(obj.get('roots'),list) else []
        for r in rr[:3]:
            if not isinstance(r,dict):continue
            b=norm_box_to_px(r.get('bbox'),*im.size)
            if not b:continue
            area=(b[2]-b[0])*(b[3]-b[1])/(im.width*im.height)
            if area<.002 or area>.55:continue
            roots.append((b,float(r.get('confidence') or 0),str(r.get('visibility') or '')))
        if roots:loc_conf=float(np.mean([r[1] for r in roots]))
    page_row={'page':pi,'page_url':u,'image':im,'boxes':[r[0] for r in roots],'locator_confidence':loc_conf,'locator_raw':raw,'locator_parsed':obj}
    page_rows.append(page_row)
    log('located',page=pi,n_roots=len(roots),confidence=loc_conf)
    for ri,(box,conf,vis) in enumerate(roots):
        crop=im.crop(box); qa_raw=run_vlm(model,processor,crop,qa_prompt,100); qa=parse_json(qa_raw) or {}
        label=str(qa.get('label') or 'UNPARSED').upper(); qconf=float(qa.get('confidence') or 0)
        overlaps=[]
        for dp in dino_by_page.get(pi,[]):
            try:overlaps.append(iou(box,[int(v) for v in dp['box']]))
            except:pass
        overlap=max(overlaps) if overlaps else 0.0
        bio=io.BytesIO(); crop.save(bio,'PNG',optimize=True)
        up=upload(f'crops/canvas_{pi:03d}_root{ri}.png','image/png',bio.getvalue())
        accepted=label in ('CLEAR_ROOT','PARTIAL_ROOT')
        crop_rows.append({'page':pi,'root_index':ri,'box':box,'locator_confidence':conf,'locator_visibility':vis,'qa_label':label,'qa_confidence':qconf,
                          'qa_reason':str(qa.get('reason') or ''),'qa_raw':qa_raw,'accepted':accepted,'dino_overlap':float(overlap),
                          'crop_path':up['path'],'crop_url':up['public_url'],'page_url':u,'crop':crop})

for s in range(0,len(crop_rows),12):upload(f'contact_sheets/crops_{s:03d}-{min(s+11,len(crop_rows)-1):03d}.jpg','image/jpeg',contact_sheet(crop_rows,s,12))
for s in range(0,len(page_rows),6):upload(f'contact_sheets/pages_{s:03d}-{min(s+5,len(page_rows)-1):03d}.jpg','image/jpeg',page_sheet(page_rows,s,6))
serial=[]
for r in crop_rows:serial.append({k:v for k,v in r.items() if k!='crop'})
accepted=[r for r in serial if r['accepted']]
report={'protocol':protocol,'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),'pages_processed':len(page_rows),
        'root_boxes':len(serial),'accepted':len(accepted),'labels':{k:sum(r['qa_label']==k for r in serial) for k in ['CLEAR_ROOT','PARTIAL_ROOT','NOT_ROOT','UNCERTAIN','UNPARSED']},
        'mean_dino_overlap':float(np.mean([r['dino_overlap'] for r in serial])) if serial else None,'rows':serial}
blob=json.dumps(report,sort_keys=True,separators=(',',':')).encode(); report['result_sha256']=hashlib.sha256(blob).hexdigest()
upload('locator_results.json','application/json',json.dumps(serial,indent=2,sort_keys=True).encode())
upload('accepted_manifest.json','application/json',json.dumps(accepted,indent=2,sort_keys=True).encode())
upload('run_report.json','application/json',json.dumps(report,indent=2,sort_keys=True).encode())
print('RESULT_JSON='+json.dumps({'run_id':RUN_ID,'prefix':PREFIX,'pages':len(page_rows),'boxes':len(serial),'accepted':len(accepted),'labels':report['labels'],'sha256':report['result_sha256']},sort_keys=True),flush=True)
