# Colour-mask implementation freeze

Protocol: `P586-VMS-PLANT-0.1-20260803`

Date: 2026-08-03

Status at freeze: the temporary Plant Mask Painter endpoint passed one live source-hash and PNG-schema smoke test. No complete colour-mask corpus, DINOv3 embedding, target similarity, statistical result or blind adjudication existed.

## Endpoint and model

- Endpoint: `POST https://plant-mask-painter.lovable.app/api/p586-mask-v01`
- Model: `google/gemini-3.1-flash-image`
- Canonical prompt SHA-256: `bc834fb40c1afb10dc11ca7e9f1a979836e3bc3146642b84c71a078552c7a38e`
- No hint, edge map, manuscript label or similarity information is supplied.
- The source crop URL is the public immutable experiment-storage copy identified in the frozen whole-plant manifest.
- A call is retried only on transport, 429 or transient 5xx failure, with the same source and prompt. A successful image is never regenerated.

## Deterministic image conversion

1. Decode the returned PNG as RGB.
2. If its dimensions differ from the source crop, resize the returned mask to the exact source dimensions using nearest-neighbour resampling and record `mask_resized=true`.
3. For each output pixel, compute squared RGB distance to exactly three colours: green `(0,255,0)`, red `(255,0,0)`, white `(255,255,255)`.
4. Assign the pixel to the nearest colour; ties resolve in the fixed order green, red, white.
5. Preserve the quantised three-colour mask and its SHA-256.
6. Construct the masked whole crop by copying ordinary source pixels where the quantised class is green or red and setting every other pixel to white.
7. Construct strict above-ground by copying source pixels where the class is green and setting every other pixel to white.
8. Construct context above-ground by copying source pixels where the class is green or red and the row is no lower than five per cent of source height below the lowest green row. Every other pixel is white. If no green pixel exists, context above-ground is invalid.

No morphology, connected-component or colour-distance threshold is tuned per manuscript.

## Frozen validity rules

- Whole foreground area is `(green + red) / all pixels`.
- A whole mask is area-valid exactly when `0.01 <= foreground_fraction <= 0.90`.
- An above-ground channel object is valid exactly when the whole mask is final-valid and `green_fraction >= 0.005`.
- Returned-source SHA-256 must equal the locally fetched source SHA-256. A mismatch is an implementation failure.
- Decode, source-hash or endpoint-schema failures are retained as invalid records.

## Visual mask audit

Before embedding:

- every area-invalid or implementation-invalid mask is audited;
- a deterministic ten per cent of area-valid masks is audited, selected when the first unsigned 32 bits of `SHA256(protocol_id + "|" + corpus_id + "|" + plant_id)` are below `0.10 * 2^32`;
- the fixed audit view contains the ordinary crop, quantised three-colour mask and resulting masked crop;
- the fixed Qwen audit labels are `pass`, `partial`, `fail`, `uncertain`.

A sampled area-valid crop labelled `fail` is excluded from masked, above-ground and reproductive channels. `partial` and `uncertain` remain in the broad sensitivity set but not the strict masked sensitivity set. Unaudited area-valid masks remain eligible under the objective gate.

If more than 20% of audited area-valid masks are labelled `fail` in either Palatino or Voynich, all masked/above-ground/reproductive results are declared methodologically unreliable and are reported only as exploratory; unmasked whole-plant analysis remains valid.

## Strict and broad mask sets

- strict masked set: whole-plant QA `accept`, objective mask valid, and audit status not `partial`, `uncertain` or `fail` when audited;
- broad masked set: whole-plant QA `accept` or `partial`, objective mask valid, and audit status not `fail` when audited.

Reproductive proposals are generated only from broad final-valid masked plants. The preregistered Qwen proposal cap, classes and separate sub-object QA rules remain unchanged.
