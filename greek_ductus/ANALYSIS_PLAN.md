# Primary analysis plan — frozen before feature values

Protocol: `2026-08-31.greek-ductus.v1`

This file operationalizes the preregistered headline statistic before any Greek-vs-Voynich feature result is computed.

## Crop-to-manuscript aggregation
For each crop, compute the fixed `stroke_features.py` vector. For each manuscript independently, summarize every feature dimension by:
- median
- interquartile range (Q75-Q25)

Concatenate median and IQR. The manuscript, not the crop, is the inferential unit.

Voynich is represented by one robust manuscript summary over all accepted crops from the frozen 18-page sampling rule. Page-block summaries are retained only for sensitivity analysis and do not increase headline N.

## Scaling
Fit a robust dimension-wise scaler on the **18 ordinary-script control manuscript summaries only**:
- center = median across control manuscripts
- scale = MAD × 1.4826
- dimensions with zero/tiny MAD are assigned scale 1 and flagged.

Apply that frozen scaler to controls and VMS. VMS does not influence scaling.

## Primary family distance
For family F, its centroid is the dimension-wise median of its six scaled manuscript summaries.

Primary distance:
`D(q,F) = EuclideanDistance(q, centroid_F) / sqrt(p)`
where p is the summary-vector dimension.

No covariance fitting or feature weighting is used in the primary result.

## Greek advantage
For target q = VMS:
`A_obs = 0.5 * (D(q, ITALIAN_LATIN) + D(q, GERMAN)) - D(q, GREEK)`

Positive A means Greek is closer than the average of the two ordinary-script matched controls.

## Balanced manuscript-label permutation null
Keep all 18 control manuscript summaries fixed. Randomly permute the family labels while preserving exactly 6 manuscripts per family. For each permutation recompute the three family centroids and `A_perm` for VMS.

- seed = 408
- permutations = 20,000
- observed family assignment is excluded from the null sample if encountered.

Headline standardized effect:
`Z = (A_obs - mean(A_perm)) / sd(A_perm)`

Empirical one-sided permutation p is also reported, but **does not override the preregistered Z decision rule**.

Decision:
- Z >= 2.0: CPU shape representation passes its half of the preregistered criterion.
- Z < 2.0: CPU shape representation does not resolve Greek affinity.
- Z < 1.0 after QC: preregistered GPU conservation stop may be invoked; new DINO control extraction is optional and should normally be cancelled.

## Mandatory diagnostics
Report without changing the primary result:
1. all three observed distances;
2. leave-one-manuscript-out within-control family classification accuracy and confusion matrix (diagnostic only);
3. family centroid pairwise distances;
4. per-manuscript distance table;
5. page-block VMS distance spread;
6. result after excluding low-information crops defined *a priori* as normalized ink crops with skeleton endpoint count <=2 AND junction count 0 AND aspect ratio <0.55, or near-circular crops with hole count >=1 and aspect ratio 0.75–1.33;
7. alternate crop-threshold sensitivity using one predeclared stricter binarization, with no retuning based on outcome.

## Secondary metrics only
Cosine distance, shrinkage-Mahalanobis, PCA projections, classifiers, nearest-neighbour galleries and individual feature contributions are descriptive/sensitivity analyses. None can replace or rescue the primary Euclidean/permutation result.

The identical family-level analysis is used later for persisted DINOv3-B manuscript summaries. A DINO-only positive is insufficient; both independent representations must satisfy Z >= 2.0 for the preregistered 'survives' conclusion.
