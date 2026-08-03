#!/usr/bin/env python3
"""Audited DINO launcher with endpoint fix and Amendments 02/07 control panel."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/26915172edf7fa04c633ed2126ff6b265d28fe11/experiments/palatino586_plant_v01/p586_embed_analyse.py"
source=requests.get(URL,timeout=120).text
source=source.replace(';UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01"',';UPLOAD_EP=SUPA+"/functions/v1/p586-plant-upload-v01"')
source=source.replace('S.post(UPLOAD,headers=', 'S.post(UPLOAD_EP,headers=')
old='MAIN=["bnf_lat_6862","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784"]'
new='MAIN=["bnf_lat_6862","bnf_gr_2179","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784","bcufL0309","cpg_214","cpg_539","pal_germ_432","herb_02d0c6eab3e2","herb_147e30c9f1b5","herb_2d835cd9812e","herb_445bcd207a09","herb_5131dbf077c7","herb_5ed380084126","herb_69b9c0aff8fb","herb_6bfd65319273","herb_746acb516c71","herb_75adf9f1c0d6","herb_794d5cc06ca5","herb_822ccef50005"]'
if source.count(old)!=1:raise RuntimeError(f"MAIN patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
if 'UPLOAD_EP=SUPA+' not in source or 'S.post(UPLOAD_EP,headers=' not in source or '"bcufL0309"' not in source:raise RuntimeError('launcher patches incomplete')
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
