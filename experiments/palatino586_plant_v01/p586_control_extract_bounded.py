#!/usr/bin/env python3
"""Audited launcher adding bounded per-page image fetch retries to control extraction."""
import requests,time
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/066502810b3fe3ad86e3a07afa6348a45afa54e7/experiments/palatino586_plant_v01/p586_control_extract.py"
source=requests.get(URL,timeout=120).text
helper='''\ndef fast_image(url):\n    last=None\n    for k in range(3):\n        try:\n            r=requests.get(url,timeout=30,headers={"User-Agent":"P586PlantMorphology/0.1"}); r.raise_for_status(); return r.content\n        except Exception as e:\n            last=e; time.sleep(1.5**k)\n    raise RuntimeError(f"bounded image fetch failed: {url}: {last}")\n\n'''
needle='def resize(im,maxside):\n'
source=source.replace(needle,helper+needle,1)
source=source.replace('raw=get(row["image_url"]).content; page=', 'raw=fast_image(row["image_url"]); page=',1)
if 'def fast_image' not in source or 'raw=fast_image(row["image_url"]); page=' not in source:
    raise RuntimeError('bounded-fetch patch did not apply exactly')
exec(compile(source,URL,"exec"),{"__name__":"__main__","requests":requests,"time":time})
