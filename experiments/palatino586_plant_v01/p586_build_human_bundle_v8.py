#!/usr/bin/env python3
"""Build final bundle into an isolated path in the existing private HF dataset."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/12e54ae4b830e85083df8330c17f03bfb4dba285/experiments/palatino586_plant_v01/p586_build_human_bundle_v7.py"
source=requests.get(URL,timeout=120).text
source=source.replace('repo_id="Digitalgoldfish79/p586-plant-morphology-artifact-v01"','repo_id="Digitalgoldfish79/voynich-dinov3-pipeline"',1)
source=source.replace('path_in_repo="P586_PLANT_MORPHOLOGY_COMPLETE.zip"','path_in_repo="artifacts/p586_plant_v01/P586_PLANT_MORPHOLOGY_COMPLETE.zip"',1)
old='api.create_repo(repo_id=repo_id,repo_type="dataset",private=True,exist_ok=True)\n        '
if source.count(old)!=1:raise RuntimeError(f"create-repo patch mismatch: {source.count(old)}")
source=source.replace(old,'',1)
source=source.replace('path_in_repo="artifact_manifest.json"','path_in_repo="artifacts/p586_plant_v01/artifact_manifest.json"',1)
if 'voynich-dinov3-pipeline' not in source or 'create_repo(' in source:raise RuntimeError('existing-repo handoff patch incomplete')
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
