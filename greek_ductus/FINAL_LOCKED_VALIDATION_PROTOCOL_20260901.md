# Final locked validation protocol — segmentation-independent Greek graphical affinity

Protocol ID: `2026-09-01.greek-affinity.final-fixedpatch.v1`

Status: **FROZEN BEFORE ANY NEW FIXED-PATCH FEATURE OR EMBEDDING VALUES ARE COMPUTED**

This protocol is a confirmatory stress test of the previously observed Voynich–Greek graphical affinity. It does not test language, plaintext, provenance, ethnicity, or literal pen-stroke order. The measured construct is static graphical morphology.

## 1. Motivation and retraction boundary

The previous CPU and DINO analyses used a shared `word_boxes()` extraction stage. Therefore their agreement cannot by itself exclude a token/word-envelope confound. This final stage removes word grouping completely.

No result from this stage may be used to retroactively relabel the original preregistered CPU result, which remains formally unresolved at its own gate. Prior positive replication/DINO results remain prior evidence, not part of the new null distribution.

## 2. Frozen source cohort

Primary control cohort is exactly `replication1_manifest.json`, unchanged: 8 Greek and 8 Latin manuscripts, six fixed page fractions per control manuscript. The target is the same Beinecke 408 manifest and the same 18 fixed Voynich page fractions in that manifest.

This deliberately reuses the independently selected image cohort while changing the representation. No manuscript or page can be added/dropped based on fixed-patch output. Acquisition failure is reported; a failed manuscript is not silently replaced.

## 3. Segmentation-independent fixed-patch extraction

For every sampled page:

1. Fetch the same IIIF page image using the existing deterministic acquisition adapter.
2. Apply the existing `ink_mask()` preprocessing unchanged (grayscale, 3.5% page-edge suppression, 3x3 Gaussian blur, adaptive threshold, 2x2 morphological opening).
3. **Do not call `word_boxes()` and do not infer words.**
4. Detect text-line bands only from the horizontal ink projection. A row is active when its smoothed ink count exceeds `max(3, 0.08 * percentile95(row_ink))`. Merge active runs separated by <= 3 pixels. Retain line bands with height between 8 px and 8% of page height and horizontal ink coverage >= 8% of page width.
5. Pad each retained line band vertically by 20% of its height, clipped to the page.
6. Slide a fixed-width window along the entire line band. Window width = `4 * line_height`, clipped to [96, 256] pixels in the 1600px page representation; stride = 50% of window width. Patch height is the padded line-band height.
7. Retain a patch only if ink fraction is in [0.02, 0.35] and connected-component count is in [2, 60]. These gates are fixed before output inspection.
8. Resize retained binary patches to 224x224 with preserved foreground/background polarity. No content-dependent aspect-ratio normalization is used.
9. Deterministic ordering is page, line-y, window-x. If a manuscript supplies more than 360 accepted patches, take 360 evenly spaced patches. If fewer than 120 survive, retain all but flag the manuscript as low-yield. The target Voynich bank is capped at 720 evenly spaced patches overall.

This representation preserves local stroke/line morphology while eliminating token length and word-envelope geometry as an explicit segmentation choice.

## 4. CPU morphology representation

For each retained binary patch compute a fixed vector without learned weights:

- ink fraction;
- skeleton length per patch area;
- endpoint count / skeleton length;
- junction count / skeleton length;
- Euler-hole count / component count;
- 12-bin undirected skeleton orientation histogram;
- 12-bin local turning-angle histogram along skeleton paths;
- horizontal and vertical projection entropy;
- median and IQR connected-component area normalized by patch area.

The **headline morphology vector excludes the gross-statistics-only fields** ink fraction, component count, projection entropy, and component area statistics. Those fields form the confound-null representation in Section 8. Headline orientation/turning/topology features are aggregated per manuscript by median and IQR.

## 5. Primary statistic and exact null

For the 16 control manuscript summaries:

- fit dimension-wise robust scaling on controls only: median center and `1.4826 * MAD`; tiny-MAD dimensions use scale 1 and are flagged;
- family centroid = dimension-wise median of the eight scaled manuscript summaries;
- target distance = Euclidean distance / sqrt(p);
- observed Greek advantage: `A = D(VMS,LATIN) - D(VMS,GREEK)`.

Because there are exactly C(16,8)=12,870 balanced 8-v-8 assignments, **enumerate every balanced label assignment exactly**. The observed assignment is excluded from the null. Report null mean, null SD, standardized effect `Z`, and exact one-sided permutation p.

Primary CPU decision:

- `Z >= 2.0` and `A > 0`: fixed-patch morphology supports persistence of Greek affinity beyond word segmentation;
- `1.0 <= Z < 2.0`: unresolved;
- `Z < 1.0` or `A <= 0`: segmentation-independent morphology does not support the affinity.

No secondary metric may rescue this decision.

## 6. DINO confirmation gate

DINOv3-B is run only if the CPU fixed-patch result has `Z >= 2.0` and the gross-statistics null in Section 8 has `Z < 2.0`.

Model and revision are unchanged from prior confirmation:

- model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- revision: `5931719e67bbdb9737e363e781fb0c67687896bc`
- no fine-tuning;
- cosine distance on L2-normalized manuscript centroids;
- exact 12,870 balanced-label null, not Monte Carlo.

DINO decision: `Z >= 2.0` is required for the claim that the result survives an independent learned representation on segmentation-independent inputs.

## 7. Voynich hand-stratified sensitivity

Using the Lisa Fagin Davis five-hand partition as a **sensitivity partition, not as proof that five people necessarily wrote the codex**, compute one fixed-patch centroid per hand using all eligible pages assigned to that hand that are present in the IIIF manifest.

To avoid unequal-page-count dominance, cap each hand at 360 evenly spaced patches; Hand 5 may use all available accepted patches if below cap. Compare each hand separately to the frozen Greek and Latin centroids.

Report `A_h = D(hand_h,LATIN) - D(hand_h,GREEK)` for Hands 1–5. This does not create five independent manuscripts and does not increase headline N.

Interpretation rule:

- common-script-substrate language is permitted only if at least 4/5 hand partitions are Greekward (`A_h > 0`) and no hand is strongly Latinward by more than the median absolute hand advantage;
- otherwise report heterogeneity and do not generalize the manuscript-level signal to all Davis partitions.

## 8. Gross-statistics confound null

Construct a deliberately trivial representation from only:

- ink fraction;
- connected-component count per fixed patch;
- median connected-component area / patch area;
- horizontal projection entropy;
- vertical projection entropy;
- retained-line height / page height;
- accepted-patch yield per page.

Aggregate and analyze with the identical manuscript-level statistic and exact permutation null.

If this gross-statistics representation itself gives `Z >= 2.0` in the Greek direction, the headline morphology result is considered confounded/ambiguous and the Greek-substrate interpretation is not promoted, regardless of DINO.

## 9. Mandatory diagnostics

Report without changing the headline decision:

1. all Greek/Latin/Voynich distances;
2. exact permutation distribution mean and SD;
3. leave-one-manuscript-out Greek/Latin control classification;
4. per-manuscript distances;
5. accepted patch counts per manuscript and page;
6. per-Voynich-page Greek advantage;
7. five Davis-partition advantages;
8. gross-statistics null result;
9. contact sheets of fixed patches for Greek, Latin and Voynich generated by deterministic even sampling;
10. sensitivity to one predeclared stricter ink-fraction gate [0.03, 0.30], reported only as sensitivity.

## 10. Positive-control calibration

A separate calibration asks whether artificial-script regions can move a known manuscript away from its ordinary-script baseline while retaining same-codex morphology. Primary calibration candidates are the previously frozen exact loci in Fontana Cod.icon.242 and Pal.germ.597. This calibration is descriptive unless a same-hand attribution is independently verified from scholarship. It cannot validate the Greek historical interpretation by itself.

## 11. Chronology/multi-library extension

A second, separately frozen cohort will test authority-verified 1400–1450 Greek, northern-Italian/Veneto Latin, and German/Alemannic material from multiple libraries. Cohort membership must be fixed from catalogue metadata before any feature extraction. Broad 15th-century dates are not silently promoted to 1400–1450. This extension cannot alter the result of Sections 3–9.

## 12. Claims permitted

If CPU and DINO both pass and gross statistics do not:

> Voynich writing shows a reproducible Greek-minuscule graphical affinity that survives removal of word-level segmentation in the tested cohort.

Not permitted from this experiment alone:

- Greek language;
- Greek author or ethnicity;
- Constantinopolitan/Venetian provenance;
- literal ductus/stroke-order reconstruction;
- proof that a Greek-trained person invented the script.

Those remain historical or mechanistic hypotheses requiring separate evidence.
