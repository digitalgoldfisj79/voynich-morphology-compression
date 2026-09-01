# VMS Script-Origin & Production Programme v1.0
## Parallel-safe arms: frozen preregistration

Frozen: 2026-09-01, before new target outcomes were inspected.

## RETRACTED / DOWNGRADED FINDINGS

1. **Greek affinity survives removal of word-level segmentation — RETRACTED / NOT SUPPORTED.** The locked fixed-patch primary was Latinward and unresolved (`A=-0.1362726172`, null SD `0.1332979902`, `Z=-1.0223950583`, one-sided exact `p=0.8394716395`).
2. **The word-level Greekward result demonstrates Greek scribal motor habits — NOT SUPPORTED.** The replicated effect is representation-dependent.
3. **Northern Italy wins the historical localisation — RETRACTED / NOT ESTABLISHED.** Prior D1–D6 synthesis gave Padua–Venice and Pavia–Milan 5/6, Bologna, Vienna–Klosterneuburg and Upper Rhine–Alsace 4/6; removing D1 produced a five-way tie.
4. **Four independent Siena ManuComp signals — RETRACTED.** Tradition correction reduces the four apparent hits to one because the Taccola witnesses are correlated.
5. **1404–1438 is a writing terminus — RETRACTED.** It is the radiocarbon interval for parchment manufacture/animal death; 1439–1445 remains a separately labelled near-window.

## Firewall

- Gate 0 remains unresolved until the reconstruction result is compared with the frozen DINO word-object result.
- No W0–W4, Davis-hand W0 omnibus, known-answer palaeographic calibration, final multi-script validation, or specificity result may be run or inspected early.
- Historical/iconographic findings cannot select palaeographic manuscripts or metrics.
- Palaeographic findings cannot alter historical corpus inclusion.
- Target crop/page counts never become inferential `n`.

## Arm E1 — Copying/error archaeology

### Target census

Enumerate without section/hand labels visible:

- erasure;
- overwriting;
- supralinear insertion;
- marginal insertion;
- false start;
- abandoned glyph group;
- overwritten character;
- line restart;
- unusually squeezed line ending.

Each candidate receives an anonymous page ID, pixel coordinates, mark class, observable cues, confidence (`clear`, `probable`, `ambiguous`), coder ID, and adjudication state. `Ambiguous` remains in the ledger but is excluded from the locked primary and included only in a predeclared upper-bound sensitivity.

### Sequence detector features

- maximal exact repeats at token and glyph-string levels;
- near repeats at frozen normalized edit-distance thresholds `0`, `<=0.20`, and `<=0.34`;
- immediate duplication;
- interrupted repetition;
- candidate omission between flanking similar sequences;
- line-end/line-start recurrence;
- transitions within a fixed window around independently coded correction sites.

The transcription file and byte hash must be frozen before target execution. EVA systems may not be mixed. Tokenization ambiguities are reported as a separate sensitivity, not silently reconciled.

### Calibration and primary question

Detector thresholds are calibrated on manuscripts/texts with independently identified copying errors. The primary test compares a multivariate copying-signature vector against matched copied-text and non-copy/composition controls at the manuscript level. Matching fields: date band, script type, text layout, manuscript length, image/transcription resolution, and correction visibility. If fewer than three independent usable calibration manuscripts per control class survive, no inferential copying claim is allowed.

Primary effect: standardized distance from the matched non-copy null toward the copied-text centroid. Report effect size and null SD in the same sentence. A ratio below 2 is reported as: **the metric does not resolve this**.

## Arm E2 — Production workflow across Davis hands

### Frozen variables

- baseline and interline-spacing median/IQR;
- left/right/top/bottom margin behaviour;
- paragraph-marker placement;
- gallows placement relative to line and paragraph;
- line-fill residual space and squeeze rate;
- inter-token spacing median/IQR;
- illustration-avoidance distance and overlap;
- page-start and page-end conventions.

### Units and confounds

Measurements are crop-level, summaries page-level, inference clustered by bifolium/page. Hand effects are not estimated without controlling section, quire, page geometry, illustration load, text density, and recto/verso. Mixed-hand `f115r` is excluded from hand contrasts but retained in the audit ledger.

Primary output is a variance decomposition into hand, section, quire/bifolium, and residual components. The system-level-invariant claim requires low between-hand variance relative to a matched within-hand page null while section/quire effects are held fixed. With only five hands, inferential precision and decision-rule fragility are mandatory reporting items.

## Arm E3 — Sacred-city/cosmological iconography

### Inclusion

- securely dated or bounded before 1450; uncertain dates retained only in sensitivity;
- Heavenly Jerusalem, Apocalypse city, ideal church ensemble, cosmological architecture, or sacred-city diagram;
- digitized image sufficient to code the frozen variables;
- catalogue-driven inclusion, never resemblance-driven.

Geographic strata: Byzantine/Constantinopolitan; Venetian mainland; Venetian Crete/colonial sphere; central Italy; German lands; other Latin West.

### Frozen visual variables

- radial/concentric geometry;
- monumental-unit count;
- domed/pillar-like unit count;
- central celestial feature;
- enclosing wall;
- routes/channels;
- gates;
- mixed plan/elevation;
- repeated vertical architectural elements.

Corpus rows are assigned blind IDs. Provenance is stored in a sealed lookup and joined only after coding. Two genuinely independent coders are preferred; sequential recoding by one model is not counted as independent. Inter-rater agreement is Cohen's kappa for binary variables and ICC/agreement interval for counts. Without an independent second coder, results remain descriptive.

Primary comparison uses the complete nine-variable configuration, not an isolated tower motif. Enrichment is assessed against geography-balanced sacred-city controls. The Rosettes page is one target object, not nine independent observations.

## Arm E4 — Veneto–Siena–southern Germany comparison

### Environments

Primary: Veneto/Padua–Venice; Siena; southern Germany/Alemannic sphere. Pavia–Milan, Bologna, Vienna–Klosterneuburg, and Upper Rhine–Alsace remain audit comparators because prior work found them competitive.

### Evidence families

1. northern Italian herbal imagery;
2. German/Alemannic visual material;
3. Greek manuscript access;
4. medical/pharmaceutical culture;
5. balneology;
6. cosmography/geography;
7. technical drawing;
8. artificial/cipher writing;
9. mobile people;
10. book-production environment.

### Cell states

- `Local/Documented`: dated evidence physically produced/held/practised in the environment during 1400–1450;
- `Directly accessible`: a documented institutional/person network path of one edge during the window;
- `Requires import`: tradition is external and no local instance is documented, but a specific transfer route is documented;
- `Unsupported`: no qualifying evidence found after the balanced search protocol.

`Not found` is not converted to `Unsupported` until the same repositories/source classes and search effort have been applied to all environments. Each cell requires at least one citable primary or scholarly source and records source date, evidence date, and whether the evidence is correlated with another cell.

Primary robustness: leave-one-evidence-family-out ranking plus a correlation-collapse analysis. No numeric scoring weights may be tuned after inspection. Ordinal display weights, if required, are frozen as `3,2,1,0` in the state order above and are sensitivity-only; the primary is the full cell matrix.

## Arm E5 — Prosopography

Window: 1400–1450; wider context allowed only when explicitly flagged.

A person qualifies only when documentary evidence connects them to at least two of: Greek manuscripts; medicine; pharmacy; book production; German travel/students; technical arts; astronomy/cosmography.

Allowed edges: person–institution, person–manuscript, person–city, teacher–student, person–patron, documented meeting. Every edge requires a source and date/range. Co-presence without a documented interaction is not a meeting edge. Famous visitors with only one qualifying domain are excluded.

Primary network output reports connected components, degree, betweenness, edge-source diversity, and temporal-overlap validity. A cluster is not interpreted unless every connecting path is temporally possible and survives removal of any single undated/inferred edge.

## Arm E6 — Professional milieu

Competing environments: university physician; apothecary/pharmacy; hospital/xenon; court technical household; humanist/private library; commercial manuscript workshop.

Primary unit: one surviving inventory, holding list, account series, or independently documented institutional collection. Each unit is coded for herbals, materia medica, pharmacy/receptaria, balneology, astronomy/astrology, cosmography, Greek books, technical/mechanical books, and cipher/artificial alphabets.

Co-occurrence means categories occur within the same documentary unit. Combining unrelated holdings from the same city is prohibited. Bartolo di Tura's private library, Santa Maria della Scala's hospital/apothecary evidence, and Carrara holdings remain distinct units.

Report raw category co-occurrence, source survival/coverage, and Wilson intervals where proportions are compared. Absence from an incomplete inventory is `not attested`, not `absent`. Professional-milieu rankings must survive leave-one-documentary-unit-out analysis or remain unresolved.

## Audit sequence

Before interpretation, check in this order: circularity; leakage; confounds; matched nulls; control fairness; measurement degeneracy; representation dependence; decision-rule fragility; audit completeness.

