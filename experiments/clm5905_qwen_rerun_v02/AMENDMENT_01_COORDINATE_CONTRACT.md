# Amendment 01 — coordinate-contract failure and corrected smoke gate

Protocol: `CLM5905-VMS-QWEN-RF-0.2-20260805`

Date: 2026-08-05

## Status of the first v0.2 extraction attempt

Job `6a72b6e76b79c09949c22c51` completed technically but its extraction output is invalid and must not be analysed. It accepted only 3 roots and 0 flowers across 425 source images.

The failure is now identified. Qwen returned bounding boxes in source-image pixel coordinates. The runner interpreted those values as coordinates on a 0–1000 normalized grid. This produced incorrect crops (leaves, text, blank margins, or clipped slivers), which the independent crop-QA stage then correctly rejected.

The frozen manifest from this failed attempt is retained as an audit artifact only:

- path: `clm5905_v02/extraction/extraction_manifest_frozen.json`
- SHA-256: `9aebc4020e1eb13f4af3e5ae8ae5a9c9315e82e1dbf158953d97d4f10bb24aea`

No DINO embeddings or similarity analysis may use that manifest.

## Corrected coordinate contract

The replacement localizer must return `bbox_frac=[x1,y1,x2,y2]`, where every coordinate is a decimal fraction in `[0,1]` relative to the exact image supplied to Qwen. The parser rejects any candidate outside that range. Fractions are converted directly to source-image pixels. There is no adaptive interpretation and no 0–1000 convention.

The independent crop-QA stage remains mandatory and unchanged in principle. The exact ordinary crop accepted by QA is the only crop eligible for embedding and display.

## Mandatory smoke gate before full rerun

A stratified smoke panel is run across Clm 5905, Voynich, and every control corpus before another full-corpus job. The full rerun is permitted only if:

1. accepted roots and accepted strict flowers are both nonzero in Clm 5905 and Voynich;
2. at least four of five control corpora yield an accepted component in each primary channel, unless the source panel visibly lacks that component;
3. no accepted crop is a whole plant, blank margin, text-only crop, leaf-only crop, or wrong reproductive class;
4. a contact sheet made from the exact accepted bytes passes visual inspection;
5. the coordinate parser reports zero out-of-range accepted boxes.

This amendment changes only the coordinate representation and introduces a smoke gate. The source panel, component ontology, independent-QA requirement, deduplication rules, DINO representation, and statistical estimand remain frozen.