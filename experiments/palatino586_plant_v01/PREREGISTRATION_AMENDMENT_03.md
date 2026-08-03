# Preregistration Amendment 03 — Manuscript-specific mask pipeline

Date: 2026-08-03

Protocol: `P586-VMS-PLANT-0.1-20260803`

Status at amendment: all target whole-plant proposals were frozen, but no valid target or control mask set, no DINOv3 embeddings, no Voynich similarities, no statistical comparisons and no blind adjudications had been produced.

## Triggering implementation audit

The preregistered SAM2.1 box-prompt pipeline failed on medieval line-art crops. A non-inferential diagnostic on the first eight frozen Voynich crops showed that the masks with the highest predicted IoU were almost entirely parchment/background, usually covering 0.95–1.00 of the crop. The small plant-like ink masks received predicted IoU values near zero. Selecting another SAM candidate by a post hoc area rule would therefore introduce an unvalidated crop-dependent heuristic and would not reliably recover filled leaves, stems or flowers.

This is a modality failure detected before any target comparison, not a result-dependent threshold failure. The failed SAM2 outputs are retained as diagnostic evidence and excluded from analysis.

## Replacement primary mask pipeline

Use the existing Plant Mask Painter manuscript-specific automatic segmentation policy identically for Palatino, Voynich and every control:

- model: `google/gemini-3.1-flash-image` through the project's existing OpenRouter integration;
- prompt: the exact frozen `autoMaskFolio` prompt already present in Plant Mask Painter;
- output colours: pure green `#00FF00` for above-ground plant parts, pure red `#FF0000` for roots/bulbs/rhizomes, and pure white for background;
- input: the frozen ordinary whole-plant crop only, without manuscript identity, DINO vectors or similarity outcomes;
- no handwritten hints, edge-map amendments or per-corpus prompt changes;
- deterministic post-processing: resize to the source crop dimensions with nearest-neighbour only if required, assign every output pixel to the nearest of green/red/white in RGB distance, and preserve the exact ordinary source pixels under the resulting foreground mask.

The foreground mask is green plus red. The strict above-ground mask is green only. The context above-ground crop adds a fixed five per cent of crop height below the lowest green row, bounded by the crop. Root/above-ground boundaries from the earlier Qwen extraction remain an independent audit field and are not used to alter the colour mask.

The original preregistered validity gate remains unchanged: foreground area fraction must be between 0.01 and 0.90. Colour masks outside this gate are retained and marked invalid. Mask model, prompt SHA-256, OpenRouter response metadata where available, source/output hashes, colour fractions, resize status and failure reason are recorded.

A fixed visual validation pass examines every invalid mask and a deterministic 10% sample of valid masks before embedding. No mask is manually edited. A failed crop remains excluded under the frozen validity rule; it is not regenerated with a changed prompt.

## Reproductive structures

Dedicated reproductive-structure proposals continue to be generated from the masked whole crop under the original fixed Qwen prompt and QA rules. Flower, flower-head, inflorescence, bud, fruit and seed-head labels remain distinct.

## Interpretation

Masked-channel claims now refer to the Plant Mask Painter / Gemini colour-mask pipeline. SAM2 remains a documented negative known-method check for medieval line-art segmentation and is not silently represented as a successful mask source.
