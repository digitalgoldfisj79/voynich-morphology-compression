#!/usr/bin/env python3
"""Amendment-09 isolated reproductive-repair shard launcher."""
import os,requests
START=int(os.environ["SHARD_START"]);END=int(os.environ["SHARD_END"]);SHARD=f"{START}_{END}"
if not (0<=START<END):raise RuntimeError("invalid shard range")
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/bce66f0271fadad6d5b51534aa0fe720f31a1dc7/experiments/palatino586_plant_v01/p586_reproductive_repair.py"
source=requests.get(URL,timeout=120).text
needle='PREFIX="p586_plant_v01/target" if CID==TARGET else f"p586_plant_v01/controls/{CID}"'
source=source.replace(needle,needle+'\nSTART=int(os.environ["SHARD_START"]);END=int(os.environ["SHARD_END"]);SHARD=f"{START}_{END}"',1)
source=source.replace('PREFIX+"/reproductive_repair_checkpoint.json"','PREFIX+f"/reproductive_shards/checkpoint_{SHARD}.json"')
old='base_sha=base.get("channel_manifest_sha256")\n    cp=load_checkpoint(base_sha)'
new='base_sha=base.get("channel_manifest_sha256")\n    base["plants"]=base.get("plants",[])[START:END]\n    cp=load_checkpoint(base_sha)'
if source.count(old)!=1:raise RuntimeError("slice patch point mismatch")
source=source.replace(old,new,1)
source=source.replace('upload(PREFIX+"/channels_manifest_frozen.json"','upload(PREFIX+f"/reproductive_shards/result_{SHARD}.json"',1)
source=source.replace('upload(PREFIX+"/channels_report.json"','upload(PREFIX+f"/reproductive_shards/report_{SHARD}.json"',1)
if 'reproductive_shards/result_' not in source or 'base.get("plants",[])[START:END]' not in source:raise RuntimeError("shard patches incomplete")
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
