import base64
import collections
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260713
rng = np.random.default_rng(SEED)
root = Path('/data')
out = Path('/tmp/out')
out.mkdir(exist_ok=True)


def checkpoint(name: str, **data: object) -> None:
    print('CHECKPOINT', name, json.dumps(data, sort_keys=True), flush=True)


meta_url = (
    'https://raw.githubusercontent.com/digitalgoldfisj79/'
    'voynich-morphology-compression/feat/nomic-atlas/'
    'tools/dinov3_program/folio_metadata.tsv.b64'
)
meta = pd.read_csv(
    io.BytesIO(base64.b64decode(urllib.request.urlopen(meta_url).read())),
    sep='\t',
    dtype=str,
).fillna('')
mm = meta.set_index('folio')

words: list[dict] = []
props: list[dict] = []
manifest = root / 'results/corpus_crop_manifest.jsonl'
for idx, line in enumerate(manifest.open()):
    record = json.loads(line)
    if record.get('view') == 'norm' and record.get('kind') == 'word':
        record['_i'] = idx
        words.append(record)
    elif (
        record.get('view') == 'norm'
        and record.get('kind') in ('cc', 'ccmerge')
        and not record.get('low_conf', False)
    ):
        record['_i'] = idx
        props.append(record)

W = pd.DataFrame(words)
Pall = pd.DataFrame(props)
for frame in (W, Pall):
    frame['hand'] = frame.folio.map(mm.hand).fillna('')
    frame['currier'] = frame.folio.map(mm.language_flag).fillna('')
    frame['section'] = frame.folio.map(mm.section).fillna('Unassigned')
checkpoint(
    'manifest_loaded',
    words=len(W),
    word_types=int(W.word.nunique()),
    eligible_proposals=len(Pall),
)

Pall['_stratum'] = (
    Pall.kind.astype(str) + '|' + Pall.eva_aligned.fillna('').astype(str)
)
selected: list[int] = []
for _, group in Pall.groupby('_stratum'):
    quota = min(
        len(group),
        max(20, round(100000 * len(group) / len(Pall))),
    )
    selected.extend(rng.choice(group.index.to_numpy(), quota, replace=False))
if len(selected) > 100000:
    selected = list(rng.choice(np.array(selected), 100000, replace=False))
P = Pall.loc[sorted(selected)].reset_index(drop=True)
del Pall

npz = np.load(root / 'results/corpus_embeddings_full.npz')
vectors = npz['vectors']
Xw = np.asarray(vectors[W._i.to_numpy()], dtype=np.float32)
Xp = np.asarray(vectors[P._i.to_numpy()], dtype=np.float32)
Xw /= np.maximum(np.linalg.norm(Xw, axis=1, keepdims=True), 1e-9)
Xp /= np.maximum(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-9)
checkpoint(
    'vectors_loaded',
    word_shape=list(Xw.shape),
    proposal_shape=list(Xp.shape),
)

# Exact word neighbours, including a stricter cross-folio view.
raw_nn = NearestNeighbors(
    n_neighbors=129,
    metric='cosine',
    n_jobs=-1,
).fit(Xw)
dist129, ind129 = raw_nn.kneighbors(Xw)
y = W.word.to_numpy()
folios = W.folio.to_numpy()
ind20 = ind129[:, 1:21]
dist20 = dist129[:, 1:21]
purity = (y[ind20] == y[:, None]).mean(1)

cross_ind = np.empty((len(W), 20), dtype=np.int32)
cross_dist = np.empty((len(W), 20), dtype=np.float32)
for i in range(len(W)):
    candidates = ind129[i, 1:]
    candidate_distances = dist129[i, 1:]
    different_folio = folios[candidates] != folios[i]
    kept = candidates[different_folio][:20]
    kept_distances = candidate_distances[different_folio][:20]
    if len(kept) < 20:
        raise RuntimeError(
            f'Insufficient cross-folio neighbours for row {i}: {len(kept)}'
        )
    cross_ind[i] = kept
    cross_dist[i] = kept_distances
cross_purity = (y[cross_ind] == y[:, None]).mean(1)

W['knn_purity20'] = purity
W['cross_folio_knn_purity20'] = cross_purity
W['nn_cosine'] = 1 - dist20[:, 0]
W['cross_folio_nn_cosine'] = 1 - cross_dist[:, 0]
W['len'] = W.word.str.len()

null: list[float] = []
null_cross: list[float] = []
for _ in range(200):
    permuted = y.copy()
    for _, group_indices in W.groupby(['len', 'section']).groups.items():
        group_indices = np.asarray(list(group_indices))
        permuted[group_indices] = rng.permutation(permuted[group_indices])
    null.append(float((permuted[ind20] == permuted[:, None]).mean()))
    null_cross.append(
        float((permuted[cross_ind] == permuted[:, None]).mean())
    )
obs = float(purity.mean())
obs_cross = float(cross_purity.mean())
checkpoint(
    'word_knn_done',
    observed=obs,
    cross_folio=obs_cross,
    null=float(np.mean(null)),
    cross_null=float(np.mean(null_cross)),
)

pairs: collections.Counter[tuple[str, str]] = collections.Counter()
for i in range(len(W)):
    for j in cross_ind[i, :5]:
        if y[i] != y[j]:
            pairs[tuple(sorted((y[i], y[j])))] += 1
alias = pd.DataFrame(
    [(a, b, count) for (a, b), count in pairs.most_common(500)],
    columns=['word_a', 'word_b', 'cross_folio_knn_count'],
)

# Within-label multimodality and association with manuscript metadata.
mult_rows: list[tuple] = []
for word, group in W.groupby('word'):
    if len(group) < 20:
        continue
    ids = group.index.to_numpy()
    xx = Xw[ids]
    labels = MiniBatchKMeans(
        2,
        random_state=SEED,
        n_init=10,
        batch_size=256,
    ).fit_predict(xx)
    sil = float(silhouette_score(xx, labels, metric='cosine'))

    def cramers(column: str) -> float:
        table = pd.crosstab(labels, group[column].fillna('')).to_numpy()
        n = table.sum()
        if min(table.shape) < 2:
            return 0.0
        expected = np.outer(table.sum(1), table.sum(0)) / n
        chi = ((table - expected) ** 2 / np.maximum(expected, 1e-9)).sum()
        return float(
            math.sqrt(chi / (n * max(1, min(table.shape) - 1)))
        )

    mult_rows.append(
        (
            word,
            len(group),
            sil,
            cramers('hand'),
            cramers('currier'),
            cramers('section'),
        )
    )
mult = pd.DataFrame(
    mult_rows,
    columns=[
        'word',
        'n',
        'silhouette2',
        'cramers_hand',
        'cramers_currier',
        'cramers_section',
    ],
).sort_values('silhouette2', ascending=False)
checkpoint(
    'multimodality_done',
    labels=len(mult),
    silhouette_ge_0_1=int((mult.silhouette2 >= 0.1).sum()),
)

# Remove lexical centroids, then classify metadata with folio-grouped folds.
centroids = {
    word: Xw[group.index].mean(0)
    for word, group in W.groupby('word')
}
Xr = np.stack([Xw[i] - centroids[y[i]] for i in range(len(W))]).astype(
    np.float32
)
Xr /= np.maximum(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-9)
class_rows: list[tuple] = []
for target in ['hand', 'currier', 'section']:
    mask = W[target].astype(str).str.len() > 0
    target_values = W.loc[mask, target].to_numpy()
    groups = W.loc[mask, 'folio'].to_numpy()
    counts = pd.Series(target_values).value_counts()
    keep = np.array([counts[value] >= 20 for value in target_values])
    ids = np.where(mask)[0][keep]
    target_values = target_values[keep]
    groups = groups[keep]
    if len(set(target_values)) < 2:
        continue
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    for name, xx in [('raw', Xw[ids]), ('word_residual', Xr[ids])]:
        classifier = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeClassifier(alpha=10.0, class_weight='balanced'),
        )
        scores = cross_val_score(
            classifier,
            xx,
            target_values,
            groups=groups,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
        )
        class_rows.append(
            (
                target,
                name,
                float(scores.mean()),
                float(scores.std()),
                len(ids),
                len(set(target_values)),
                float(1 / len(set(target_values))),
            )
        )
classdf = pd.DataFrame(
    class_rows,
    columns=[
        'target',
        'features',
        'balanced_accuracy',
        'sd',
        'n',
        'classes',
        'chance',
    ],
)
checkpoint('classification_done', rows=len(classdf))

# Unsupervised proposal-family discovery.
pca_train = rng.choice(len(Xp), min(30000, len(Xp)), replace=False)
pca = PCA(n_components=64, random_state=SEED).fit(Xp[pca_train])
Z = pca.transform(Xp).astype(np.float32)
sub = rng.choice(len(Z), min(15000, len(Z)), replace=False)
k_rows: list[tuple[int, float, float]] = []
for k in [32, 64, 96, 128, 192, 256]:
    kmeans = MiniBatchKMeans(
        k,
        random_state=SEED,
        n_init=3,
        batch_size=4096,
        max_iter=150,
    ).fit(Z)
    sil = float(
        silhouette_score(
            Z[sub],
            kmeans.predict(Z[sub]),
            sample_size=min(5000, len(sub)),
            random_state=SEED,
        )
    )
    k_rows.append((k, float(kmeans.inertia_ / len(Z)), sil))
kdf = pd.DataFrame(
    k_rows,
    columns=['k', 'inertia_per_row', 'silhouette'],
)
best = int(
    kdf.sort_values(['silhouette', 'k'], ascending=[False, True]).iloc[0].k
)
kmeans = MiniBatchKMeans(
    best,
    random_state=SEED,
    n_init=10,
    batch_size=4096,
    max_iter=300,
).fit(Z)
clusters = kmeans.labels_
P['cluster'] = clusters
aris: list[float] = []
for offset in [1, 2, 3]:
    alternate = MiniBatchKMeans(
        best,
        random_state=SEED + offset,
        n_init=3,
        batch_size=4096,
        max_iter=200,
    ).fit_predict(Z)
    aris.append(adjusted_rand_score(clusters, alternate))

eva = P.eva_aligned.fillna('').astype(str).to_numpy()
folds = np.array(
    [int(hashlib.sha1(folio.encode()).hexdigest(), 16) % 5 for folio in P.folio]
)
transfer_acc: list[float] = []
transfer_baseline: list[float] = []
for fold in range(5):
    train = folds != fold
    test = ~train
    mapping = {
        cluster: group.eva_aligned.fillna('').value_counts().index[0]
        for cluster, group in P[train].groupby('cluster')
    }
    prediction = np.array([mapping.get(cluster, '') for cluster in clusters[test]])
    transfer_acc.append(float((prediction == eva[test]).mean()))
    majority = pd.Series(eva[train]).value_counts().index[0]
    transfer_baseline.append(float((eva[test] == majority).mean()))

proposal_ind = NearestNeighbors(
    n_neighbors=11,
    metric='cosine',
    n_jobs=-1,
).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
proposal_purity = float((eva[proposal_ind] == eva[:, None]).mean())
proposal_freq_baseline = float(
    np.sum(pd.Series(eva).value_counts(normalize=True).to_numpy() ** 2)
)
checkpoint(
    'proposal_done',
    selected_k=best,
    stability=float(np.mean(aris)),
    transfer=float(np.mean(transfer_acc)),
)

# Ordered component reconstruction and length/section-matched null.
sequences = {
    (folio, int(word_index)): '-'.join(
        map(str, group.sort_values('slot').cluster.tolist())
    )
    for (folio, word_index), group in P.groupby(['folio', 'word_index'])
}
W['cluster_sequence'] = [
    sequences.get((row.folio, int(row.word_index)), '')
    for row in W.itertuples()
]
valid = W.cluster_sequence != ''
valid_words = W.loc[valid].copy().reset_index(drop=True)
sequence_groups = [
    group.index.to_numpy()
    for _, group in valid_words.groupby('cluster_sequence')
    if len(group) >= 2
]
if sequence_groups:
    weighted_obs = float(
        sum(
            valid_words.loc[group, 'word'].value_counts().iloc[0]
            for group in sequence_groups
        )
        / sum(len(group) for group in sequence_groups)
    )
    mean_obs = float(
        np.mean(
            [
                valid_words.loc[group, 'word'].value_counts().iloc[0]
                / len(group)
                for group in sequence_groups
            ]
        )
    )
else:
    weighted_obs = float('nan')
    mean_obs = float('nan')
sequence_null: list[float] = []
for _ in range(100):
    shuffled = valid_words.word.to_numpy().copy()
    grouping = valid_words.groupby(
        [valid_words.word.str.len(), 'section']
    ).groups
    for _, indices in grouping.items():
        indices = np.asarray(list(indices))
        shuffled[indices] = rng.permutation(shuffled[indices])
    sequence_null.append(
        float(
            sum(
                pd.Series(shuffled[group]).value_counts().iloc[0]
                for group in sequence_groups
            )
            / sum(len(group) for group in sequence_groups)
        )
    )
checkpoint(
    'composition_done',
    valid_words=int(valid.sum()),
    groups=len(sequence_groups),
    weighted=weighted_obs,
    null=float(np.mean(sequence_null)),
)

W.to_parquet(out / 'word_results.parquet', index=False)
P.to_parquet(out / 'proposal_results.parquet', index=False)
alias.to_csv(out / 'alias_pairs.csv', index=False)
mult.to_csv(out / 'multimodality.csv', index=False)
classdf.to_csv(out / 'classification.csv', index=False)
kdf.to_csv(out / 'cluster_selection.csv', index=False)

summary = {
    'schema': 'vdino3-glyph-program/2',
    'seed': SEED,
    'counts': {
        'words': len(W),
        'word_types': int(W.word.nunique()),
        'eligible_proposals': 747672,
        'proposal_sample': len(P),
    },
    'word_knn': {
        'mean_purity20': obs,
        'null_mean': float(np.mean(null)),
        'null_sd': float(np.std(null)),
        'z': float((obs - np.mean(null)) / np.std(null)),
        'cross_folio_mean_purity20': obs_cross,
        'cross_folio_null_mean': float(np.mean(null_cross)),
        'cross_folio_null_sd': float(np.std(null_cross)),
        'cross_folio_z': float(
            (obs_cross - np.mean(null_cross)) / np.std(null_cross)
        ),
    },
    'multimodality': {
        'tested_labels': len(mult),
        'silhouette_ge_0_1': int((mult.silhouette2 >= 0.1).sum()),
    },
    'classification': classdf.to_dict('records'),
    'proposal': {
        'pca_variance64': float(pca.explained_variance_ratio_.sum()),
        'selected_k': best,
        'stability_ari_mean': float(np.mean(aris)),
        'eva_knn_purity10': proposal_purity,
        'eva_frequency_baseline': proposal_freq_baseline,
        'heldout_cluster_to_eva_accuracy': float(np.mean(transfer_acc)),
        'heldout_majority_baseline': float(np.mean(transfer_baseline)),
    },
    'composition': {
        'words_with_sequences': int(valid.sum()),
        'repeated_sequence_groups': len(sequence_groups),
        'mean_group_label_purity': mean_obs,
        'weighted_sequence_label_purity': weighted_obs,
        'matched_null_mean': float(np.mean(sequence_null)),
        'matched_null_sd': float(np.std(sequence_null)),
        'matched_null_z': float(
            (weighted_obs - np.mean(sequence_null)) / np.std(sequence_null)
        ),
    },
    'notes': [
        'All quantitative tests use original 768-dimensional word vectors or a fixed 64-component PCA of proposal vectors.',
        'Word-neighbour nulls preserve word length and manuscript section.',
        'Classifier estimates use StratifiedGroupKFold by folio and RidgeClassifier to avoid the convergence failures in v1.',
        'Proposal clustering is unsupervised and evaluated on held-out folios.',
    ],
}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'REPORT.md').write_text(
    '# DINOv3 Glyph Baseline Programme v2\n\n```json\n'
    + json.dumps(summary, indent=2)
    + '\n```\n'
)

files: list[dict] = []
for path in sorted(out.iterdir()):
    if path.is_file():
        files.append(
            {
                'file': path.name,
                'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
(out / 'MANIFEST.json').write_text(
    json.dumps(
        {
            'source': 'Digitalgoldfish79/vdino3-crops',
            'source_commit': 'b86d96600dc49c5298e07dcecb1ebdbd44970483',
            'metadata_commit': '85473f247f1221ca6e82f7103666b9d04de4f0f7',
            'files': files,
        },
        indent=2,
    )
)

archive = Path('/tmp/vdino3-glyph-program-results-v2.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(__file__, 'run_v2.py')
    for path in sorted(out.iterdir()):
        if path.is_file():
            zf.write(path, f'out/{path.name}')
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
print('SUMMARY', json.dumps(summary), flush=True)
print('ARCHIVE', archive, archive.stat().st_size, sha, flush=True)
