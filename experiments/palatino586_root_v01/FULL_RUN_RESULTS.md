# Palatino 586 root comparison — recovered full run

Date: 2026-08-03

## Executive result

The recovered Qwen 7B + DINOv3-7B run gives a strong negative result for Palatino 586 as a close root-morphology comparator to the Voynich Manuscript.

Under the frozen model representation, Voynich roots are substantially closer to the BSB Cgm 728 control roots than to Palatino 586 roots. This holds in both:

- the strict primary analysis using only `accept` Palatino crops; and
- the broad sensitivity analysis using `accept + partial` crops.

This is not evidence about exemplar, source, lineage, provenance, textual relationship, or botanical identity. It is a controlled image-morphology result against one external herbal control.

## Interruption and recovery

Original full-run job:

- Hugging Face job `6a709f9c6b79c09949c20b4e`.
- Qwen localisation completed before failure.
- Failure point: the REST request for existing Voynich/BSB root records omitted the required `apikey` header and returned HTTP 401.
- No DINOv3 comparison result was produced by the errored job.

Recovery:

- Recovery script commit: `e48655cc4ee22810260f3e5d3f3486a24e0c5d59`.
- Recovery job: `6a70aaaca00abefd4b28f9b9`.
- Only canvases 67 and 80–85 were rerun; the earlier localisation checkpoint was retained.
- Corrected comparison script commit: `225a8d4967135a24ebf010e3dfd9424f811c1d91`.
- Comparison job: `6a70aba86b79c09949c20bca`.
- Recovered checkpoint SHA-256: `a667613f3e8c1847103b7583b4b7f4057ae2a2ae127a21822b8408dedd1092ee`.
- Comparison result SHA-256: `af2ca32f569a043ca9b373f9870aef80cf73290e1f7444d0f433509f5cb6bb2f`.

## Localisation and QA

The model-localised Palatino corpus contains:

- 66 processed canvases;
- 181 root proposals;
- 30 `accept` crops;
- 109 `partial` crops;
- 42 `reject` crops.

The QA distribution is genuinely discriminating, unlike the earlier Qwen 3B locator whose crop QA collapsed to one class.

The localisation and QA are still model-generated rather than human-verified. The strict `accept` set is therefore the primary analysis; `accept + partial` is a sensitivity analysis.

## Comparison design

Representation:

- `facebook/dinov3-vit7b16-pretrain-lvd1689m`;
- 4,096-dimensional normalized CLS embeddings.

Reference and control sets:

- 128 usable Voynich roots;
- 21 usable roots from BSB Cgm 728;
- 30 strict Palatino roots;
- 139 Palatino roots in the broad sensitivity set.

Tests:

- top-1, top-5 and top-10 target-corpus shares;
- mean best-match difference;
- 10,000 target-label permutations preserving corpus counts;
- reciprocal nearest-neighbour counts;
- 3,000 two-way group bootstraps;
- equal-pool-size secondary audit;
- blinded triptych review pack generated but not yet adjudicated.

## Primary result — accept only

Counts:

- Voynich queries: 128;
- Palatino targets: 30;
- BSB targets: 21.

Best-match similarities:

- mean best Palatino similarity: `0.33935`;
- mean best BSB similarity: `0.48925`;
- Palatino-minus-BSB difference: `−0.14989`;
- group-bootstrap 95% interval: `−0.17379` to `−0.11373`.

Permutation diagnostics:

- lower-tail p-value for the mean-best difference: `0.00010`;
- Palatino share of top-1 matches: `0.0625`;
- top-1 null mean: `0.5863`;
- lower-tail p-value for top-1 share: `0.00030`.

Equal-pool secondary audit:

- 21 Palatino roots versus 21 BSB roots per repetition;
- mean best-match difference: `−0.15768`;
- reference-subsample interval: `−0.16778` to `−0.15124`;
- mean Voynich-query Palatino win fraction: `0.0551`.

The strict result strongly favours the BSB control.

## Sensitivity result — accept plus partial

Counts:

- Voynich queries: 128;
- Palatino targets: 139;
- BSB targets: 21.

Best-match similarities:

- mean best Palatino similarity: `0.37789`;
- mean best BSB similarity: `0.48925`;
- Palatino-minus-BSB difference: `−0.11136`;
- group-bootstrap 95% interval: `−0.13937` to `−0.07642`.

Permutation diagnostics:

- lower-tail p-value for the mean-best difference: `0.00010`;
- Palatino share of top-1 matches: `0.1094`;
- top-1 null mean: `0.8698`;
- lower-tail p-value for top-1 share: `0.00010`.

Equal-pool secondary audit:

- 21 Palatino roots versus 21 BSB roots per repetition;
- mean best-match difference: `−0.16285`;
- reference-subsample interval: `−0.19301` to `−0.13798`;
- mean Voynich-query Palatino win fraction: `0.04165`.

The broad sensitivity result therefore confirms rather than weakens the strict negative result.

## Relationship to the smaller first pass

The earlier DINOv3-L first pass found no Palatino-specific enrichment after controlling reference-pool size. The larger DINOv3-7B run is directionally stronger: it places the BSB control substantially closer to Voynich than Palatino.

The two runs are therefore consistent:

1. raw nearest-neighbour browsing can be distorted by target-pool size;
2. controlled comparison does not support Palatino 586 as a close Voynich root comparator;
3. the stronger model and stricter QA make the negative result more pronounced.

## Interpretation limits

The accepted conclusion is narrow:

> In this model-localised and DINOv3-7B representation, Palatino 586 root crops are not Voynich-enriched relative to BSB Cgm 728; they are substantially less similar to Voynich roots than the BSB control roots.

The result does not show that:

- no individual Palatino motif resembles a Voynich motif;
- Palatino is irrelevant to the wider herbal tradition;
- BSB Cgm 728 is a source or close historical relative of Voynich;
- DINO similarity establishes botanical identity or transmission.

## Persisted outputs

- `p586_root_v01/full_run/checkpoint_recovered.json`
- `p586_root_v01/full_run/recovery_report.json`
- `p586_root_v01/full_run/comparison_result_recovered.json`
- `p586_root_v01/full_run/root_proposals_recovered.csv`
- `p586_root_v01/embeddings/all_roots_dinov3_vit7b16_recovered.npz`
- `p586_root_v01/full_run/audit/blind_triptychs_recovered.jpg`
- `p586_root_v01/full_run/audit/blind_key_recovered.json`

The blind triptych pack remains unadjudicated and must not be described as supporting either corpus.
