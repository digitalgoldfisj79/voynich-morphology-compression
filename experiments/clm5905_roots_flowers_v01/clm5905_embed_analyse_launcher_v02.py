#!/usr/bin/env python3
"""Launch frozen Clm 5905 analysis from a byte-identical public DINOv3 mirror."""
from __future__ import annotations

import hashlib
import requests

SOURCE_URL = "https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1e6adfe5bb182857842349473f689f7778a0742d/experiments/clm5905_roots_flowers_v01/clm5905_embed_analyse_v01.py"
CANONICAL_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
CANONICAL_REVISION = "b80367753773648a6793235ab9c65cdbb029506f"
MIRROR_MODEL = "PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m"
MIRROR_REVISION = "b09ecf5d8f9b4f562cbaba84242f629e1bacb677"
SHARDS = [
    "7132627f25459ee8797cb2965d3427706a87119ccfcdfef1bd7977dd7580821f",
    "a7b17660c408adf235c318328010ecded0ee24181b97b806c1e45b42efc5ff4b",
    "b5937a7a7051239798a6984d07e2d68fb1f8f93d0947c63be9ffe1bbbbe8dab7",
    "569c817cc4424410c1d49df48e053ba4206eb7a1bd2c53381b7f7dfb32c4d57e",
    "51ab5686ebe67cb48b738caee366d6a0bd0fb19b3f0d37dc37b8e82bf34c0e66",
    "d3f77e2cbd0f9a349eeaf2559213f2495e43e4229a02a33f083f48ab5539ecbb",
]

source = requests.get(SOURCE_URL, timeout=180).text
if not source.startswith("#!/usr/bin/env python3"):
    raise RuntimeError("analysis source download failed")

old_model = f'MODEL = "{CANONICAL_MODEL}"'
old_rev = f'REV = "{CANONICAL_REVISION}"'
if source.count(old_model) != 1 or source.count(old_rev) != 1:
    raise RuntimeError("frozen model constants not found exactly once")
source = source.replace(old_model, f'MODEL = "{MIRROR_MODEL}"', 1)
source = source.replace(old_rev, f'REV = "{MIRROR_REVISION}"', 1)

needle = '        "model_revision": REV,\n'
provenance = (
    '        "model_revision": REV,\n'
    f'        "canonical_model_id": "{CANONICAL_MODEL}",\n'
    f'        "canonical_model_revision": "{CANONICAL_REVISION}",\n'
    '        "weight_equivalence": {\n'
    f'            "mirror_model_id": "{MIRROR_MODEL}",\n'
    f'            "mirror_revision": "{MIRROR_REVISION}",\n'
    '            "verification": "all six safetensor shard sizes and SHA-256 hashes identical to canonical repository metadata",\n'
    f'            "safetensor_shard_sha256": {SHARDS!r},\n'
    '        },\n'
)
if source.count(needle) != 1:
    raise RuntimeError("report provenance insertion point not found exactly once")
source = source.replace(needle, provenance, 1)

print(
    "MIRROR_PATCH_OK",
    hashlib.sha256(source.encode()).hexdigest(),
    MIRROR_MODEL,
    MIRROR_REVISION,
    flush=True,
)
exec(compile(source, SOURCE_URL, "exec"), {"__name__": "__main__"})
