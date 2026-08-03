#!/usr/bin/env python3
"""Amendment-07 fixed-page secondary control extraction launcher."""
import requests,time
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/066502810b3fe3ad86e3a07afa6348a45afa54e7/experiments/palatino586_plant_v01/p586_control_extract.py"
source=requests.get(URL,timeout=120).text
helper='''\ndef fast_image(url):\n    last=None\n    for k in range(3):\n        try:\n            r=requests.get(url,timeout=30,headers={"User-Agent":"P586PlantMorphology/0.1"});r.raise_for_status();return r.content\n        except Exception as e:\n            last=e;time.sleep(1.5**k)\n    raise RuntimeError(f"bounded image fetch failed: {url}: {last}")\n\ndef fixed_secondary_rows(all_rows):\n    if not all_rows:return []\n    n=len(all_rows);qs=(.05,.13,.21,.29,.37,.45,.55,.63,.71,.79,.87,.95)\n    idx=set(range(min(10,n)))\n    idx.update(max(0,min(n-1,round(q*(n-1)))) for q in qs)\n    return [all_rows[i] for i in sorted(idx)]\n\n'''
needle='def resize(im,maxside):\n'
if source.count(needle)!=1:raise RuntimeError('resize insertion point mismatch')
source=source.replace(needle,helper+needle,1)
old='source=get(CORPUS).json(); rows=source["rows"]'
new='source=get(CORPUS).json(); rows=fixed_secondary_rows(source["rows"])'
if source.count(old)!=1:raise RuntimeError('row-selection patch mismatch')
source=source.replace(old,new,1)
source=source.replace('raw=get(row["image_url"]).content; page=', 'raw=fast_image(row["image_url"]); page=',1)
source=source.replace('"amendment":"PREREGISTRATION_AMENDMENT_01"','"amendment":"PREREGISTRATION_AMENDMENT_07"',1)
source=source.replace('"broad_cap":20','"broad_cap":8',1)
source=source.replace('>=20','>=8')
if 'fixed_secondary_rows' not in source or '"broad_cap":8' not in source or '>=20' in source:raise RuntimeError('secondary-panel patches incomplete')
exec(compile(source,URL,"exec"),{"__name__":"__main__","requests":requests,"time":time})
