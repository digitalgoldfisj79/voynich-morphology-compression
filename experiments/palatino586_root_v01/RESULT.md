# Palatino 586 ↔ Voynich root morphology v0.1 — corrected closeout

**Protocol:** `P586-VMS-ROOT-0.1-20260803`  
**Run ID:** `ef6a0302-8269-4166-8c1c-63b14abb9c47`  
**Status:** `CLOSED — STRONG NEGATIVE VS BSB 1784 CONTROL`  
**Final result SHA-256:** `588e0aa0a759f84fd8468c043f3fb3eb17b173f0c2fc4af3d186a351148eb206`

## What was built

The 139-canvas BNCF Palatino 586 facsimile was screened. The illustrated herbal block was processed with Qwen 2.5-VL 7B for plant/root localisation and crop QA. The common comparison pool was then embedded with `facebook/dinov3-vit7b16-pretrain-lvd1689m` and compared against existing Voynich and BSB Cgm 728 root crops.

Final frozen localisation counts:

- pages processed: 66;
- proposed crops: 178;
- strict accepts: 30;
- partial roots: 107;
- rejects: 41;
- retained sensitivity pool: 137.

Control/query counts:

- Voynich roots: 128;
- BSB Cgm 728 roots: 21.

## Interruption and correction

The original A100 job `6a709f9c6b79c09949c20b4e` completed localisation and persisted every crop, but failed at the control fetch because the Supabase REST request omitted the `apikey` header.

A provisional recovery replayed canvas 67. Qwen's nondeterministic replay produced two additional partial crops and one additional reject, inflating the provisional totals to 181 proposals and 139 retained roots. That replay is not used in the final sensitivity analysis.

The final result uses:

1. the original frozen checkpoint through canvas 79;
2. the nine terminal decisions recovered from the failed-job logs and focused late-page QA;
3. the common DINOv3 embedding bundle, filtered back to the exact 137 frozen retained paths.

Correction script commit: `3227cc4103ccca59b50d5152004b97c5f3411e27`.

## Corrected results

### Primary: 30 strict Palatino roots

| Metric | Palatino 586 | BSB 1784 / comparison |
|---|---:|---:|
| Mean best Voynich similarity | 0.339293 | 0.489247 |
| Mean Palatino − BSB difference | -0.149954 | |
| Group-bootstrap 95% CI | [-0.173955, -0.113831] | |
| Palatino share of top-1 neighbours | 0.0625 | null mean 0.5863 |
| Label-permutation lower-tail p | 0.00009999 | |
| Equal-size 21-v-21 audit difference | -0.157818 | interval [-0.168114, -0.151361] |
| Reciprocal Voynich links | 7 | BSB: 7 |

### Sensitivity: 30 accepts + 107 partials

| Metric | Palatino 586 | BSB 1784 / comparison |
|---|---:|---:|
| Mean best Voynich similarity | 0.377655 | 0.489247 |
| Mean Palatino − BSB difference | -0.111592 | |
| Group-bootstrap 95% CI | [-0.139301, -0.077627] | |
| Palatino share of top-1 neighbours | 0.109375 | null mean 0.8655 |
| Label-permutation lower-tail p | 0.00009999 | |
| Equal-size 21-v-21 audit difference | -0.162733 | interval [-0.194183, -0.137704] |
| Reciprocal Voynich links | 12 | BSB: 7 |

The larger Palatino pool does not rescue it. Even after reference-pool sizes are equalised, BSB remains decisively closer.

## Blinded visual audit

The twelve strongest corrected Palatino–Voynich embedding pairs were each placed beside a similarity-matched BSB decoy. A/B identity was hidden from `Qwen/Qwen2.5-VL-7B-Instruct`.

- trials: 12;
- valid A/B decisions: 6;
- abstentions or malformed tie outputs: 6;
- BSB selected: 6/6 valid decisions;
- Palatino selected: 0/6;
- several Palatino crops were judged not to depict a valid root.

This is a single-VLM, top-pair-selected sanity check, not human specialist adjudication. It nevertheless fails to rescue the Palatino hypothesis visually.

## Conclusion

Under this common DINOv3 image pipeline, Voynich roots are substantially closer to the BSB Cgm 728 control roots than to Palatino 586 roots. The strict analysis, broad sensitivity analysis, equal-size control audit and blinded visual check all point in the same direction.

This is a **strong negative for Palatino 586 as a close root-morphology comparator**. It does not establish anything broader about source, lineage, provenance, botanical identity or the manuscript as a whole.

## Run ledger

- interrupted localisation job: `6a709f9c6b79c09949c20b4e`;
- late-page QA: `6a70aa776b79c09949c20bbb`;
- completed common-embedding comparison: `6a70aba86b79c09949c20bca`;
- frozen-set correction: `6a70ae74a00abefd4b28fa14`;
- blinded visual audit: `6a70aff3a00abefd4b28fa39`.

All Hugging Face jobs are stopped. The temporary P586 upload, ingest, chunk, control, finalize and resume endpoints have been replaced by authenticated `410 Gone` stubs.
