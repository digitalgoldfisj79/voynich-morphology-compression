# Preregistration Amendment 02 — byte-identical DINOv3 mirror

Date: 2026-08-04
Protocol: `CLM5905-VMS-RF-0.1-20260804`

## Trigger

The first analysis job (`6a72106ea00abefd4b29214a`) reached the model-loading stage but Hugging Face returned HTTP 401 for the gated canonical repository `facebook/dinov3-vit7b16-pretrain-lvd1689m`, despite `HF_TOKEN` being attached to the job. No target embedding or similarity was computed.

## Recovery

The analysis may load the model from the public mirror:

- repository: `PIA-SPACE-LAB/dinov3-vit7b16-pretrain-lvd1689m`
- pinned revision: `b09ecf5d8f9b4f562cbaba84242f629e1bacb677`

This is not a model substitution. Hugging Face repository metadata was checked before relaunch and all six safetensor shards have identical byte sizes and SHA-256 hashes in the canonical Meta repository and the mirror:

1. `7132627f25459ee8797cb2965d3427706a87119ccfcdfef1bd7977dd7580821f`
2. `a7b17660c408adf235c318328010ecded0ee24181b97b806c1e45b42efc5ff4b`
3. `b5937a7a7051239798a6984d07e2d68fb1f8f93d0947c63be9ffe1bbbbe8dab7`
4. `569c817cc4424410c1d49df48e053ba4206eb7a1bd2c53381b7f7dfb32c4d57e`
5. `51ab5686ebe67cb48b738caee366d6a0bd0fb19b3f0d37dc37b8e82bf34c0e66`
6. `d3f77e2cbd0f9a349eeaf2559213f2495e43e4229a02a33f083f48ab5539ecbb`

The canonical frozen model identity remains:

- `facebook/dinov3-vit7b16-pretrain-lvd1689m`
- revision `b80367753773648a6793235ab9c65cdbb029506f`

Only the download location changes. Preprocessing, CLS extraction, normalization, statistical tests, seeds, reference embeddings and decision rules are unchanged.
