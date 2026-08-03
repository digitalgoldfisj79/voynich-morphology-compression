# Preregistration Amendment 04 — bounded mask implementation failures

Protocol: `P586-VMS-PLANT-0.1-20260803`

This amendment was made before DINOv3 embedding or similarity analysis.

## Trigger

The frozen colour-mask endpoint failed twice for one BnF Latin 6862 object (`s0059_p00`) while the remaining 14 broad objects in that manuscript produced valid masks under the unchanged model, prompt, quantisation and area rules.

## Rule

A corpus may proceed to channel freezing when:

1. every broad whole-plant object has a terminal colour-mask record;
2. successful masks remain unchanged;
3. failed objects remain available in the ordinary whole-plant channel;
4. failed objects are excluded from masked whole-plant, above-ground and reproductive channels;
5. at least eight successful masks remain for manuscript-level masked-channel eligibility;
6. failures, error text and counts remain explicit in the frozen manifest.

No substitute mask, manual repair, threshold change or similarity-informed exclusion is permitted.

## Consequence

BnF Latin 6862 remains eligible with 14 successful masks. `s0059_p00` contributes only to unmasked whole-plant analyses. The same rule applies prospectively to any later corpus before target similarities are opened.
