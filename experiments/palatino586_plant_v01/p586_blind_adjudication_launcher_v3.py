#!/usr/bin/env python3
"""Audited blind-review launcher using the full frozen DINO manuscript panel."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/08da1b8c643379c7ce5786f61faaf1c2911c446d/experiments/palatino586_plant_v01/p586_blind_adjudication.py"
source=requests.get(URL,timeout=120).text
old='MAIN=["bnf_lat_6862","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784"]'
new='MAIN=["bnf_lat_6862","bnf_gr_2179","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784","bcufL0309","cpg_214","cpg_539","pal_germ_432","herb_02d0c6eab3e2","herb_147e30c9f1b5","herb_2d835cd9812e","herb_445bcd207a09","herb_5131dbf077c7","herb_5ed380084126","herb_69b9c0aff8fb","herb_6bfd65319273","herb_746acb516c71","herb_75adf9f1c0d6","herb_794d5cc06ca5","herb_822ccef50005"]'
if source.count(old)!=1:raise RuntimeError(f"MAIN patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
old='ALL=[TARGET,VOYNICH]+MAIN+["herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc"]'
new='ALL=[TARGET,VOYNICH]+MAIN+["herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc"]'
if source.count(old)!=1:raise RuntimeError("ALL patch point missing")
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
