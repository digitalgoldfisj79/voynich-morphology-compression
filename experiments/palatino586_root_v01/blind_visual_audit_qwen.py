#!/usr/bin/env python3
import io,json,re,base64,requests,torch
from PIL import Image,ImageOps
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
IMG='https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public/manuscripts/p586_root_v01/final/corrected_blind_triptychs.jpg'
KEY='https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public/bridge/p586_root_v01/final/corrected_blind_key.json'
UP='https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/p586-root-resume-upload-v01'
MODEL='Qwen/Qwen2.5-VL-7B-Instruct'
S=requests.Session();S.headers['User-Agent']='VoynichRootResearch/0.1-blind-audit'
def get(u):
 r=S.get(u,timeout=120);r.raise_for_status();return r
def parse(t):
 t=t.replace('```json','').replace('```','').strip();m=re.search(r'\{.*\}',t,re.S)
 return json.loads(m.group(0)) if m else {'choice':'tie','confidence':0,'reason':'parse failure','raw':t}
sheet=ImageOps.exif_transpose(Image.open(io.BytesIO(get(IMG).content))).convert('RGB');keys=get(KEY).json()
proc=AutoProcessor.from_pretrained(MODEL,min_pixels=256*28*28,max_pixels=1024*28*28)
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa').eval()
prompt='''This is a blinded root-morphology trial from medieval botanical manuscripts. The left image is QUERY. The middle is A. The right is B. Decide which option is visibly more morphologically similar to QUERY as a root/rhizome/bulb drawing. Ignore parchment colour, crop size, scan quality, text, and general artistic style. Compare branching topology, number and arrangement of lobes or tendrils, central axis, bulbous versus fibrous structure, symmetry, and attachment geometry. Return JSON only: {"choice":"A|B|tie","confidence":0.0,"query_is_valid_root":true,"a_is_valid_root":true,"b_is_valid_root":true,"shared_features":["..."],"reason":"brief"}. Use tie when neither is meaningfully closer or crops are invalid.'''
out=[];row_h=sheet.height//len(keys)
for i,k in enumerate(keys):
 row=sheet.crop((0,i*row_h,sheet.width,(i+1)*row_h));msg=[{'role':'user','content':[{'type':'image','image':row},{'type':'text','text':prompt}]}]
 txt=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);ii,vi=process_vision_info(msg);inp=proc(text=[txt],images=ii,videos=vi,padding=True,return_tensors='pt').to(model.device)
 with torch.inference_mode():gen=model.generate(**inp,max_new_tokens=300,do_sample=False)
 raw=proc.batch_decode([gen[0][len(inp.input_ids[0]):]],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
 ans=parse(raw);truth='A' if k['A']=='p586' else 'B';ans.update(trial=i+1,truth=truth,correct=(ans.get('choice')==truth),palatino_similarity=k['p586_similarity'],bsb_similarity=k['bsb1784_similarity'],voynich_key=k['voynich_key'],p586_key=k['p586_key'],bsb1784_key=k['bsb1784_key']);out.append(ans);print('TRIAL='+json.dumps(ans,sort_keys=True),flush=True)
valid=[x for x in out if x.get('choice') in ('A','B')]
summary={'protocol':'P586-VMS-ROOT-0.1-20260803','audit':'blind visual adjudication of corrected sensitivity top pairs','model':MODEL,'n_trials':len(out),'n_decisions':len(valid),'n_ties':sum(x.get('choice')=='tie' for x in out),'palatino_choices':sum(x.get('choice')==x.get('truth') for x in valid),'accuracy_among_decisions':sum(x.get('correct') for x in valid)/len(valid) if valid else None,'valid_root_counts':{'query':sum(bool(x.get('query_is_valid_root')) for x in out),'A':sum(bool(x.get('a_is_valid_root')) for x in out),'B':sum(bool(x.get('b_is_valid_root')) for x in out)},'trials':out,'limitations':['Single VLM adjudicator, not human expert review.','Trials were selected as top embedding pairs and are not a random sample.','Accuracy tests whether the visible Palatino option was preferred over a similarity-matched BSB decoy, not manuscript-level affinity.']}
data=json.dumps(summary,indent=2).encode();r=S.post(UP,json={'path':'p586_root_v01/final/blind_visual_adjudication_qwen.json','content_type':'application/json','data_b64':base64.b64encode(data).decode()},timeout=180);r.raise_for_status();print('UPLOAD='+r.text,flush=True);print('RESULT_JSON='+json.dumps(summary,sort_keys=True),flush=True)
