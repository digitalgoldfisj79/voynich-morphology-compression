# Preregistration Amendment 05 — reproductive response parser correction

Protocol: `P586-VMS-PLANT-0.1-20260803`

This amendment was made before DINOv3 embedding or any target similarity was computed.

## Trigger

The frozen reproductive detector produced non-empty, classed proposals in its raw responses, but the channel runner recorded zero proposals. Inspection showed a deterministic interface mismatch:

- expected: `{ "reproductive_structures": [...] }` with `bbox_1000`;
- returned by the frozen Qwen model: a top-level JSON array with `bbox_2d`.

The returned coordinates are in the same normalized 0–1000 coordinate convention: values exceed the pixel dimensions of the small whole-plant crops and map coherently under the existing normalized-box helper.

## Correction

The correction:

1. reuses every already frozen raw detector response;
2. does not rerun or reprompt reproductive detection;
3. accepts either a top-level list or the preregistered wrapped object;
4. accepts `bbox_2d`, `bbox_1000`, or `bbox`, in that order only as interface aliases;
5. retains the first five proposals in original response order;
6. applies the unchanged allowed-class list, crop padding, and visual QA prompt;
7. records parser revision and all malformed/rejected proposals.

No proposal, class, confidence, box, crop, threshold, or manuscript is selected using DINO similarity outcomes.
