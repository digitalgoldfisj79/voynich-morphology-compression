#!/usr/bin/env python3
"""Amendment-04 launcher for frozen colour-mask audit and reproductive extraction.

Allows terminal implementation failures only when every broad object has a record and
at least eight successful masks remain. Failed objects stay in raw whole-plant analyses
and are excluded from masked/above/reproductive channels. No mask is repaired or replaced.
"""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/3f12241f9174e434828eed5594b97ce8343d23cc/experiments/palatino586_plant_v01/p586_color_channel_qwen.py"
source=requests.get(URL,timeout=120).text
old='''    if not colors.get("complete"):raise RuntimeError("colour-mask phase is not complete")'''
new='''    broad_expected=[x for x in whole["plants"] if x.get("qa_status") in {"accept","partial"}]
    terminal_records=colors.get("records",[])
    terminal_ids={x.get("plant_id") for x in terminal_records}
    successful=sum(x.get("status")=="success" for x in terminal_records)
    if not colors.get("complete"):
        if len(terminal_records)!=len(broad_expected) or terminal_ids!={x.get("plant_id") for x in broad_expected} or successful<8:
            raise RuntimeError("colour-mask phase lacks a complete terminal ledger or eight successful masks")
        colors["amendment_04_terminal_failure_mode"]=True'''
if source.count(old)!=1:
    raise RuntimeError(f"completion-gate patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
