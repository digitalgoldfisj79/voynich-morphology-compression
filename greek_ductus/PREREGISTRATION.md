# Greek-minuscule motor-grammar test — preregistration

Protocol ID: `2026-08-31.greek-ductus.v1`
Frozen before any Greek-vs-control DINO result is inspected.

## Hypothesis
Voynichese is an artificial but fluent script whose graphic inventory preserves the motor/shape grammar of an existing scribal tradition. The primary candidate tested here is late-Byzantine Greek minuscule.

This is **not** a Greek decipherment test and **not** a Constantinople provenance test.

## Primary question
Does a representation derived from Voynich glyph/word-image geometry place Voynich closer to a preregistered late-Byzantine Greek-minuscule corpus than to matched contemporary Latin and German manuscript controls?

## Families
1. `VMS`: Voynich, stratified by folio/section and, where independently available, Davis scribe.
2. `GREEK`: Greek minuscule, 1300–1450; selection by date/provenance/institution, never by prior resemblance to Voynich.
3. `ITALIAN_LATIN`: Italian Latin documentary/book hands, 1300–1450.
4. `GERMAN`: Upper-German/Alemannic Latin/German hands, 1300–1450.
5. `ARTIFICIAL_WEST`: Western artificial/cipher alphabets, 1300–1450 (including Fontana where usable).
6. `BYZ_CIPHER`: Byzantine secret/fantasy alphabets, pre-1450 where usable.
7. `ARABIC_SYRIAN_CIPHER`: Arabic/Syrian artificial alphabets, pre-1450 where usable (including LJS 51 only as a control family, not as ground truth for Voynich).

## Inclusion rules
- Manuscript-level date overlaps 1300–1450, except older Byzantine cipher examples retained only in `BYZ_CIPHER` and analysed separately.
- Images must be sufficiently resolved for ink segmentation.
- For ordinary-script controls, pages are selected without looking at Voynich resemblance.
- No manuscript may contribute to both training/reference and held-out evaluation in the same fold.
- Synthetic typefaces, modern transcriptions and printed facsimile redrawing are excluded from the primary analysis.

## Primary representation
CPU-derived ink/skeleton geometry, independent of EVA labels:
- normalized binary ink mask;
- skeleton graph;
- endpoint/junction counts;
- loop/cycle count;
- curvature histogram;
- orientation-transition histogram;
- aspect/ink-density features;
- connected-component and ligature proxies;
- multi-scale shape-context descriptors.

Static images do not uniquely determine true pen order. No inferred stroke order will be treated as observed fact. Where order-sensitive proxies are used they must be explicitly labelled as such.

## Independent representation
Frozen DINOv3 ViT-B/16 embeddings from the existing repo implementation. DINO is an independent visual-shape check, not evidence of pen motion.

The GPU pass is performed once. Every embedding is persisted with:
- image SHA256;
- source manuscript/page/crop ID;
- blinded family code;
- crop coordinates;
- preprocessing version;
- model ID and exact model revision;
- float16 embedding.

No image is re-embedded during hypothesis testing.

## Experimental unit
The manuscript (or independently established scribal hand where available), **not the crop**. Bootstrap/permutation resampling is blocked by manuscript/hand.

## Primary statistic
For each representation, compute a manuscript-blocked Voynich-to-family distance under held-out manuscripts. Let `d_G` be distance to Greek and `d_null` the distribution of corresponding distances/advantages against matched non-Greek ordinary-script controls.

Headline effect:

`Z = (advantage_G - mean(advantage_null)) / sd(advantage_null)`

where positive values favour Greek.

Decision rule:
- `Z >= 2.0` in the CPU shape representation **and** `Z >= 2.0` in DINOv3-B: Greek motor/shape affinity survives the preregistered test.
- Otherwise: **the experiment does not resolve Greek affinity**.
- A result significant in only one representation is reported as representation-dependent, not confirmed.

No p-value or crop-level sample size may override the manuscript-blocked decision rule.

## Conditional large-model audit
DINOv3-L/large-capacity audit is forbidden unless both primary representations satisfy `Z >= 2.0`.

If released, it runs only on a frozen audit subset (medoids, nearest competitors and hard negatives; target <=10% of the full feature-bank crop count). It cannot reverse a failed base-model result; it is robustness-only.

## Confound checks
Before interpreting family affinity:
1. Test whether background/parchment-only crops predict family. If yes, repeat on synthetic black-on-white ink renderings and treat raw-image DINO as confounded.
2. Test whether manuscript identity is trivially recoverable from normalized crops. Report it.
3. Repeat distances after size/aspect normalization and after rotation/scale perturbation.
4. Repeat with alternative ink-threshold/skeleton parameters. Direction reversal => fragile result.
5. Exclude generic low-information primitives (single vertical/minim/near-circle) in a sensitivity analysis.

## Multiple-scribe secondary test
Compare within-Voynich between-scribe variation with between-scribe variation in ordinary controls, asking whether Voynich shows stable shared glyph structure plus individual ductus variation. This is secondary and cannot rescue a failed primary test.

## Stop rules
- No fine-tuning.
- No parameter sweeps chosen by Greek-vs-Voynich outcome.
- No post-hoc addition of 'nice-looking' Greek manuscripts.
- No full-corpus DINO large/7B run.
- If the CPU representation fails badly (`Z < 1`) after QC, GPU extraction for newly acquired control images is optional and may be cancelled to conserve credits.

## Interpretation limits
A positive result would support affinity to a Greek-minuscule graphic/motor repertoire. It would not prove Greek plaintext, Byzantine authorship, Constantinopolitan production, or a particular cipher mechanism.
