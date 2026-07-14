from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

SEED = 20260713
RNG = np.random.default_rng(SEED)
ROOT = Path('/data')
OUT = Path('/tmp/voynich_atlas_v2')
WORD_DIR = OUT / 'word_bundle'
PROP_DIR = OUT / 'proposal_bundle'
REVIEW_DIR = OUT / 'review'
IMG_DIR = REVIEW_DIR / 'images'
for path in (WORD_DIR, PROP_DIR, REVIEW_DIR, IMG_DIR):
    path.mkdir(parents=True, exist_ok=True)

SOURCE_COMMIT = 'b86d96600dc49c5298e07dcecb1ebdbd44970483'
METADATA_COMMIT = '85473f247f1221ca6e82f7103666b9d04de4f0f7'
SCHEMA = 'voynich-atlas-enriched/2'


def checkpoint(name: str, **values: Any) -> None:
    print('CHECKPOINT', name, json.dumps(values, sort_keys=True, default=str), flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if value is None or isinstance(value, (str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    return x


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open('w', encoding='utf-8') as fh:
        for row in frame.to_dict('records'):
            fh.write(json.dumps({k: scalar(v) for k, v in row.items()}, ensure_ascii=False, separators=(',', ':')) + '\n')


def make_bundle(path: Path, frame: pd.DataFrame, vectors: np.ndarray, level: str) -> dict[str, Any]:
    meta_path = path / 'atlas_metadata.jsonl'
    vec_path = path / 'atlas_embeddings.npy'
    idx_path = path / 'source_indices.npy'
    write_jsonl(meta_path, frame)
    np.save(vec_path, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    np.save(idx_path, frame['source_row'].to_numpy(dtype=np.int64), allow_pickle=False)
    manifest = {
        'schema_version': 'voynich-nomic-atlas-manifest/2',
        'run_id': 'vdino3-atlas-v2-20260713',
        'level': level,
        'source': {
            'dataset': 'Digitalgoldfish79/vdino3-crops',
            'commit': SOURCE_COMMIT,
            'embedding_model': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dimensions': int(vectors.shape[1]),
        },
        'prepared': {
            'rows': len(frame),
            'dimensions': int(vectors.shape[1]),
            'normalization': 'l2',
            'seed': SEED,
            'metadata_file': meta_path.name,
            'embeddings_file': vec_path.name,
            'source_indices_file': idx_path.name,
        },
        'integrity': {
            'metadata_sha256': sha256(meta_path),
            'embeddings_sha256': sha256(vec_path),
            'source_indices_sha256': sha256(idx_path),
        },
    }
    (path / 'atlas_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def copy_crop(crop_path: str, crop_id: str) -> str:
    if not crop_path:
        return ''
    source = ROOT / 'crops' / 'crop_shard_000' / crop_path
    if not source.exists():
        return ''
    target = IMG_DIR / f'{crop_id}.png'
    if not target.exists():
        shutil.copyfile(source, target)
    return f'images/{target.name}'


def select_unique_pairs(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda r: (-int(bool(r.get('both_images'))), -float(r['cosine_similarity']), str(r['left_id']), str(r['right_id'])))
    for row in ordered:
        key = tuple(sorted((str(row['left_id']), str(row['right_id']))))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
        if len(output) >= limit:
            break
    return output


def build_review_html(pair_frame: pd.DataFrame, proposal_frame: pd.DataFrame) -> None:
    pair_records = pair_frame.fillna('').to_dict('records')
    proposal_records = proposal_frame.fillna('').to_dict('records')
    payload_pairs = json.dumps(pair_records, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    payload_props = json.dumps(proposal_records, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voynich DINOv3 Atlas v2 Review</title><style>body{{font-family:system-ui,sans-serif;margin:0;background:#f5f5f5;color:#111}}header{{position:sticky;top:0;background:white;border-bottom:1px solid #ccc;padding:12px;z-index:5}}main{{max-width:1200px;margin:auto;padding:16px}}button{{margin:3px;padding:7px 10px}}.card{{background:white;border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.imgbox{{min-height:100px;background:#eee;display:flex;align-items:center;justify-content:center}}img{{max-width:100%;max-height:180px}}.meta{{font-size:13px}}.hidden{{display:none}}</style></head><body><header><b>Voynich DINOv3 Atlas v2 Review</b><button onclick="showTab('pairs')">Pair benchmark</button><button onclick="showTab('props')">Proposal clusters</button><button onclick="exportCSV()">Export decisions CSV</button></header><main><section id="pairs"><h2>Pair benchmark candidates</h2><label>Category <select id="cat" onchange="renderPairs()"><option value="">all</option><option>exact_positive</option><option>allographic_positive</option><option>hard_negative</option></select></label><label> Unreviewed only <input id="unreviewed" type="checkbox" checked onchange="renderPairs()"></label><div id="pairlist"></div></section><section id="props" class="hidden"><h2>Proposal cluster queue</h2><div id="proplist"></div></section></main><script>const pairs={payload_pairs};const props={payload_props};const decisions=JSON.parse(localStorage.getItem('voynichReviewV2')||'{{}}');function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}function img(p){{return p?`<img src="${{esc(p)}}">`:'<span>No stored crop</span>'}}function showTab(t){{document.getElementById('pairs').classList.toggle('hidden',t!=='pairs');document.getElementById('props').classList.toggle('hidden',t!=='props')}}function decide(id,v){{decisions[id]=v;localStorage.setItem('voynichReviewV2',JSON.stringify(decisions));renderPairs()}}function renderPairs(){{let c=document.getElementById('cat').value,u=document.getElementById('unreviewed').checked;let rows=pairs.filter(r=>(!c||r.category===c)&&(!u||!decisions[r.pair_id])).slice(0,200);document.getElementById('pairlist').innerHTML=rows.map(r=>`<div class="card"><b>${{esc(r.pair_id)}}</b> · ${{esc(r.category)}} · cosine ${{Number(r.cosine_similarity).toFixed(4)}} · edit ${{esc(r.edit_distance)}} · <b>${{esc(decisions[r.pair_id]||'unreviewed')}}</b><div class="pair"><div><div class="imgbox">${{img(r.left_image)}}</div><div class="meta">${{esc(r.left_word)}} · ${{esc(r.left_folio)}} · ${{esc(r.left_id)}}</div></div><div><div class="imgbox">${{img(r.right_image)}}</div><div class="meta">${{esc(r.right_word)}} · ${{esc(r.right_folio)}} · ${{esc(r.right_id)}}</div></div></div><div><button onclick="decide('${{r.pair_id}}','same_form')">Same form</button><button onclick="decide('${{r.pair_id}}','allograph')">Allograph</button><button onclick="decide('${{r.pair_id}}','different')">Different</button><button onclick="decide('${{r.pair_id}}','bad_crop')">Bad crop</button><button onclick="decide('${{r.pair_id}}','skip')">Skip</button></div></div>`).join('')}}function renderProps(){{document.getElementById('proplist').innerHTML=props.map(r=>`<div class="card"><b>Cluster ${{esc(r.cluster)}} · ${{esc(r.review_type)}}</b><div class="imgbox">${{img(r.image)}}</div><div class="meta">id=${{esc(r.id)}} eva=${{esc(r.eva_aligned)}} kind=${{esc(r.kind)}} folio=${{esc(r.folio)}} distance=${{esc(r.cluster_distance)}}</div></div>`).join('')}}function exportCSV(){{let lines=['pair_id,decision'];Object.entries(decisions).forEach(([k,v])=>lines.push(`"${{k}}","${{v}}"`));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')],{{type:'text/csv'}}));a.download='voynich_pair_decisions.csv';a.click()}}renderPairs();renderProps();</script></body></html>'''
    (REVIEW_DIR / 'index.html').write_text(page, encoding='utf-8')


meta_url = 'https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/feat/nomic-atlas/tools/dinov3_program/folio_metadata.tsv.b64'
meta = pd.read_csv(io.BytesIO(base64.b64decode(urllib.request.urlopen(meta_url).read())), sep='\t', dtype=str).fillna('')
folio_meta = meta.set_index('folio')

crop_lookup: dict[tuple[str, str], str] = {}
crop_manifest = ROOT / 'crops/crop_shard_000/crop_manifest.jsonl'
if crop_manifest.exists():
    with crop_manifest.open() as fh:
        for line in fh:
            row = json.loads(line)
            crop_lookup[(str(row.get('id', '')), str(row.get('view', '')))] = str(row.get('path', ''))
checkpoint('crop_lookup', records=len(crop_lookup))

words: list[dict[str, Any]] = []
proposals: list[dict[str, Any]] = []
with (ROOT / 'results/corpus_crop_manifest.jsonl').open() as fh:
    for source_row, line in enumerate(fh):
        row = json.loads(line)
        if row.get('view') == 'norm' and row.get('kind') == 'word':
            row['source_row'] = source_row; words.append(row)
        elif row.get('view') == 'norm' and row.get('kind') in ('cc', 'ccmerge') and not row.get('low_conf', False):
            row['source_row'] = source_row; proposals.append(row)
W = pd.DataFrame(words); Pall = pd.DataFrame(proposals)
for frame in (W, Pall):
    frame['hand'] = frame.folio.map(folio_meta.hand).fillna('')
    frame['currier'] = frame.folio.map(folio_meta.language_flag).fillna('')
    frame['section'] = frame.folio.map(folio_meta.section).fillna('Unassigned')
    frame['crop_path_private'] = [crop_lookup.get((str(i), 'norm'), '') for i in frame.id]
    frame['crop_available'] = frame.crop_path_private.str.len().gt(0).astype(int)
checkpoint('manifest_loaded', words=len(W), word_types=int(W.word.nunique()), proposals=len(Pall), word_crops=int(W.crop_available.sum()), proposal_crops=int(Pall.crop_available.sum()))

Pall['_stratum'] = Pall.kind.astype(str) + '|' + Pall.eva_aligned.fillna('').astype(str)
selected: list[int] = []
for _, group in Pall.groupby('_stratum'):
    quota = min(len(group), max(20, round(100000 * len(group) / len(Pall))))
    selected.extend(RNG.choice(group.index.to_numpy(), quota, replace=False).tolist())
if len(selected) > 100000:
    selected = RNG.choice(np.asarray(selected), 100000, replace=False).tolist()
P = Pall.loc[sorted(selected)].reset_index(drop=True); del Pall

source_npz = ROOT / 'results/corpus_embeddings_full.npz'; local_npz = Path('/tmp/corpus_embeddings_full.npz')
checkpoint('copy_npz_start', bytes=source_npz.stat().st_size); shutil.copyfile(source_npz, local_npz); checkpoint('copy_npz_done', bytes=local_npz.stat().st_size)
with np.load(local_npz) as archive:
    vectors = archive['vectors']; Xw = norm_rows(vectors[W.source_row.to_numpy()]); Xp = norm_rows(vectors[P.source_row.to_numpy()])
checkpoint('vectors_loaded', words=list(Xw.shape), proposals=list(Xp.shape))

nn = NearestNeighbors(n_neighbors=129, metric='cosine', n_jobs=-1).fit(Xw); dist, ind = nn.kneighbors(Xw)
y = W.word.astype(str).to_numpy(); folios = W.folio.astype(str).to_numpy(); word_ids = W.id.astype(str).to_numpy()
cross_ind = np.empty((len(W), 20), dtype=np.int32); cross_dist = np.empty((len(W), 20), dtype=np.float32)
nearest_same = np.full(len(W), -1, dtype=np.int32); nearest_same_sim = np.full(len(W), np.nan, dtype=np.float32)
nearest_other = np.full(len(W), -1, dtype=np.int32); nearest_other_sim = np.full(len(W), np.nan, dtype=np.float32)
for i in range(len(W)):
    candidates = ind[i, 1:]; distances = dist[i, 1:]; mask = folios[candidates] != folios[i]; candidates = candidates[mask]; distances = distances[mask]
    cross_ind[i] = candidates[:20]; cross_dist[i] = distances[:20]
    for j, d in zip(candidates, distances, strict=False):
        if nearest_same[i] < 0 and y[j] == y[i]: nearest_same[i] = int(j); nearest_same_sim[i] = float(1 - d)
        if nearest_other[i] < 0 and y[j] != y[i]: nearest_other[i] = int(j); nearest_other_sim[i] = float(1 - d)
        if nearest_same[i] >= 0 and nearest_other[i] >= 0: break
W['word_frequency'] = W.word.map(W.word.value_counts()).astype(int)
W['knn_purity20'] = (y[ind[:, 1:21]] == y[:, None]).mean(1); W['cross_folio_knn_purity20'] = (y[cross_ind] == y[:, None]).mean(1)
W['nearest_neighbor_cosine'] = 1 - dist[:, 1]; W['cross_folio_nearest_cosine'] = 1 - cross_dist[:, 0]
W['nearest_same_word_id'] = [word_ids[j] if j >= 0 else '' for j in nearest_same]; W['nearest_same_word_folio'] = [folios[j] if j >= 0 else '' for j in nearest_same]; W['nearest_same_word_cosine'] = nearest_same_sim
W['nearest_other_word_id'] = [word_ids[j] if j >= 0 else '' for j in nearest_other]; W['nearest_other_word'] = [y[j] if j >= 0 else '' for j in nearest_other]; W['nearest_other_word_folio'] = [folios[j] if j >= 0 else '' for j in nearest_other]; W['nearest_other_word_cosine'] = nearest_other_sim
W['word_outlier_score'] = 1 - W.nearest_same_word_cosine.fillna(0)

alias_counter: collections.Counter[tuple[str, str]] = collections.Counter()
for i in range(len(W)):
    for j in cross_ind[i, :5]:
        if y[i] != y[j]: alias_counter[tuple(sorted((y[i], y[j])))] += 1
alias_rows = [(a, b, c) for (a, b), c in alias_counter.most_common(1000)]; alias_df = pd.DataFrame(alias_rows, columns=['word_a', 'word_b', 'cross_folio_knn_count'])
alias_max: dict[str, int] = collections.defaultdict(int)
for a, b, c in alias_rows: alias_max[a] = max(alias_max[a], c); alias_max[b] = max(alias_max[b], c)
W['alias_max_count'] = W.word.map(alias_max).fillna(0).astype(int)

W['visual_mode'] = -1; W['mode_silhouette'] = np.nan; mode_summary: list[dict[str, Any]] = []
for word, group in W.groupby('word'):
    if len(group) < 20: continue
    ids = group.index.to_numpy(); labels = MiniBatchKMeans(2, random_state=SEED, n_init=10, batch_size=256).fit_predict(Xw[ids]); sil = float(silhouette_score(Xw[ids], labels, metric='cosine'))
    W.loc[ids, 'visual_mode'] = labels; W.loc[ids, 'mode_silhouette'] = sil
    def cramers(column: str) -> float:
        table = pd.crosstab(labels, group[column].fillna('')).to_numpy(); n = table.sum()
        if min(table.shape) < 2: return 0.0
        expected = np.outer(table.sum(1), table.sum(0)) / n; chi = ((table - expected) ** 2 / np.maximum(expected, 1e-9)).sum(); return float(math.sqrt(chi / (n * max(1, min(table.shape) - 1))))
    mode_summary.append({'word': word, 'n': len(group), 'silhouette2': sil, 'cramers_hand': cramers('hand'), 'cramers_currier': cramers('currier'), 'cramers_section': cramers('section')})
mode_df = pd.DataFrame(mode_summary).sort_values('silhouette2', ascending=False); checkpoint('word_analysis_done', aliases=len(alias_df), multimodal_labels=len(mode_df))

pair_rows: list[dict[str, Any]] = []
for i in range(len(W)):
    j = nearest_same[i]
    if j >= 0: pair_rows.append({'category': 'exact_positive', 'left': i, 'right': int(j), 'cosine_similarity': float(nearest_same_sim[i])})
    j = nearest_other[i]
    if j >= 0: pair_rows.append({'category': 'hard_negative', 'left': i, 'right': int(j), 'cosine_similarity': float(nearest_other_sim[i])})
    if int(W.at[i, 'visual_mode']) >= 0:
        for j, d in zip(ind[i, 1:], dist[i, 1:], strict=False):
            if folios[j] == folios[i] or y[j] != y[i]: continue
            if int(W.at[j, 'visual_mode']) >= 0 and int(W.at[j, 'visual_mode']) != int(W.at[i, 'visual_mode']):
                pair_rows.append({'category': 'allographic_positive', 'left': i, 'right': int(j), 'cosine_similarity': float(1 - d)}); break
expanded: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
for row in pair_rows:
    i, j = int(row['left']), int(row['right']); left_path, right_path = str(W.at[i, 'crop_path_private']), str(W.at[j, 'crop_path_private'])
    expanded[row['category']].append({'category': row['category'], 'left_id': word_ids[i], 'right_id': word_ids[j], 'left_word': y[i], 'right_word': y[j], 'left_folio': folios[i], 'right_folio': folios[j], 'cosine_similarity': float(row['cosine_similarity']), 'edit_distance': levenshtein(y[i], y[j]), 'left_crop_path_private': left_path, 'right_crop_path_private': right_path, 'both_images': int(bool(left_path and right_path)), 'review_status': 'unreviewed'})
selected_pairs = select_unique_pairs(expanded['exact_positive'], 400) + select_unique_pairs(expanded['allographic_positive'], 300) + select_unique_pairs(expanded['hard_negative'], 300)
for number, row in enumerate(selected_pairs, 1):
    row['pair_id'] = f"P{number:04d}-{row['category']}"; row['left_image'] = copy_crop(row['left_crop_path_private'], row['left_id']); row['right_image'] = copy_crop(row['right_crop_path_private'], row['right_id'])
pairs_df = pd.DataFrame(selected_pairs); pairs_df.to_csv(REVIEW_DIR / 'benchmark_candidates.csv', index=False); alias_df.to_csv(REVIEW_DIR / 'alias_pairs.csv', index=False); mode_df.to_csv(REVIEW_DIR / 'multimodality.csv', index=False)
word_queue: dict[str, set[str]] = collections.defaultdict(set)
for row in selected_pairs: word_queue[row['left_id']].add(row['category']); word_queue[row['right_id']].add(row['category'])
W['review_queue'] = ['|'.join(sorted(word_queue.get(i, set()))) for i in W.id.astype(str)]
W['review_priority'] = W.alias_max_count.astype(float) + 10 * W.mode_silhouette.fillna(0) + 5 * W.word_outlier_score.fillna(0) + 2 * (W.review_queue.str.len() > 0).astype(float)

pca_train = RNG.choice(len(Xp), min(30000, len(Xp)), replace=False); pca = PCA(n_components=64, random_state=SEED).fit(Xp[pca_train]); Z = pca.transform(Xp).astype(np.float32)
kmeans = MiniBatchKMeans(32, random_state=SEED, n_init=10, batch_size=4096, max_iter=300).fit(Z); clusters = kmeans.labels_; P['cluster'] = clusters; centers = kmeans.cluster_centers_; P['cluster_distance'] = np.linalg.norm(Z - centers[clusters], axis=1); P['cluster_size'] = P.cluster.map(P.cluster.value_counts()).astype(int)
cluster_stats: list[dict[str, Any]] = []
for cluster, group in P.groupby('cluster'):
    eva_counts = group.eva_aligned.fillna('').astype(str).value_counts(normalize=True); kind_counts = group.kind.astype(str).value_counts(normalize=True)
    cluster_stats.append({'cluster': int(cluster), 'cluster_eva_purity': float(eva_counts.iloc[0]), 'cluster_dominant_eva': str(eva_counts.index[0]), 'cluster_kind_purity': float(kind_counts.iloc[0]), 'cluster_dominant_kind': str(kind_counts.index[0]), 'cluster_size': len(group)})
cluster_df = pd.DataFrame(cluster_stats); P = P.merge(cluster_df, on=['cluster', 'cluster_size'], how='left')
pn = NearestNeighbors(n_neighbors=11, metric='cosine', n_jobs=-1).fit(Z); pdist, pind = pn.kneighbors(Z); eva = P.eva_aligned.fillna('').astype(str).to_numpy(); P['eva_knn_purity10'] = (eva[pind[:, 1:]] == eva[:, None]).mean(1); P['proposal_outlier_score'] = pdist[:, 1:].mean(1); P['slot_fraction'] = P.slot.fillna(-1).astype(float) / np.maximum(P.n_slots.fillna(1).astype(float) - 1, 1)
folds = np.array([int(hashlib.sha1(str(f).encode()).hexdigest(), 16) % 5 for f in P.folio]); predicted = np.full(len(P), '', dtype=object)
for fold in range(5):
    train = folds != fold; test = ~train; mapping = {int(cluster): str(group.eva_aligned.fillna('').astype(str).value_counts().index[0]) for cluster, group in P.loc[train].groupby('cluster')}; predicted[test] = [mapping.get(int(c), '') for c in clusters[test]]
P['heldout_eva_prediction'] = predicted; P['heldout_eva_correct'] = (predicted == eva).astype(int)

review_props: list[dict[str, Any]] = []
for cluster, group in P.groupby('cluster'):
    for review_type, subset in [('medoid', group.nsmallest(5, 'cluster_distance')), ('boundary', group.nlargest(5, 'cluster_distance'))]:
        for row in subset.itertuples():
            review_props.append({'cluster': int(cluster), 'review_type': review_type, 'id': str(row.id), 'folio': str(row.folio), 'kind': str(row.kind), 'eva_aligned': str(row.eva_aligned), 'cluster_distance': float(row.cluster_distance), 'cluster_eva_purity': float(row.cluster_eva_purity), 'image': copy_crop(str(row.crop_path_private), str(row.id))})
proposal_review_df = pd.DataFrame(review_props); proposal_review_df.to_csv(REVIEW_DIR / 'proposal_review_queue.csv', index=False); cluster_df.to_csv(REVIEW_DIR / 'proposal_cluster_summary.csv', index=False)
build_review_html(pairs_df, proposal_review_df)
(REVIEW_DIR / 'README.md').write_text('# Voynich DINOv3 Atlas v2 review\n\nOpen `index.html` locally. Decisions are stored in browser localStorage and can be exported as CSV.\n\nThe 1,000 pair records are candidates, not adjudicated ground truth.\n', encoding='utf-8')
checkpoint('review_outputs_done', pairs=len(pairs_df), pair_images=len(list(IMG_DIR.glob('*.png'))), proposal_review=len(proposal_review_df))

word_columns = ['id','folio','word','word_index','hand','currier','section','word_frequency','crop_available','crop_path_private','knn_purity20','cross_folio_knn_purity20','nearest_neighbor_cosine','cross_folio_nearest_cosine','nearest_same_word_id','nearest_same_word_folio','nearest_same_word_cosine','nearest_other_word_id','nearest_other_word','nearest_other_word_folio','nearest_other_word_cosine','word_outlier_score','alias_max_count','visual_mode','mode_silhouette','review_queue','review_priority','source_row']
word_atlas = W[word_columns].copy(); word_atlas['atlas_id'] = 'word-v2-' + word_atlas.id.astype(str); word_atlas['atlas_level'] = 'word'; word_atlas['display_text'] = word_atlas.word.astype(str) + ' | ' + word_atlas.folio.astype(str) + ' | H' + word_atlas.hand.astype(str) + ' | C' + word_atlas.currier.astype(str); word_atlas['crop_path_private'] = word_atlas.crop_path_private.replace('', 'unavailable')
prop_columns = ['id','folio','kind','word','word_index','slot','n_slots','word_len','eva_aligned','hand','currier','section','crop_available','crop_path_private','cluster','cluster_distance','cluster_size','cluster_eva_purity','cluster_dominant_eva','cluster_kind_purity','cluster_dominant_kind','eva_knn_purity10','proposal_outlier_score','slot_fraction','heldout_eva_prediction','heldout_eva_correct','source_row']
prop_atlas = P[prop_columns].copy(); prop_atlas['atlas_id'] = 'proposal-v2-' + prop_atlas.id.astype(str); prop_atlas['atlas_level'] = 'proposal'; prop_atlas['display_text'] = prop_atlas.kind.astype(str) + ' | ' + prop_atlas.eva_aligned.fillna('').astype(str) + ' | ' + prop_atlas.folio.astype(str) + ' | cluster ' + prop_atlas.cluster.astype(str); prop_atlas['crop_path_private'] = prop_atlas.crop_path_private.replace('', 'unavailable')
word_manifest = make_bundle(WORD_DIR, word_atlas, Xw, 'word'); prop_manifest = make_bundle(PROP_DIR, prop_atlas, Xp, 'proposal')
summary = {'schema': SCHEMA, 'seed': SEED, 'source_commit': SOURCE_COMMIT, 'metadata_commit': METADATA_COMMIT, 'word_rows': len(word_atlas), 'proposal_rows': len(prop_atlas), 'benchmark_pairs': pairs_df.category.value_counts().to_dict(), 'review_images': len(list(IMG_DIR.glob('*.png'))), 'proposal_clusters': int(P.cluster.nunique()), 'word_crop_coverage': float(W.crop_available.mean()), 'proposal_crop_coverage': float(P.crop_available.mean()), 'datasets': {'word': 'edwardbozzard/voynich-dinov3-words-enriched-20260713-v2', 'proposal': 'edwardbozzard/voynich-dinov3-proposals-100k-enriched-20260713-v2'}, 'bundle_manifests': {'word': word_manifest, 'proposal': prop_manifest}}
(OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
files = []
for path in sorted(OUT.rglob('*')):
    if path.is_file(): files.append({'path': str(path.relative_to(OUT)), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
(OUT / 'MANIFEST.json').write_text(json.dumps({'schema': SCHEMA, 'files': files}, indent=2) + '\n', encoding='utf-8')
archive = Path('/tmp/voynich-atlas-enriched-v2.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
    for path in sorted(OUT.rglob('*')):
        if path.is_file(): zf.write(path, path.relative_to(OUT.parent))
print('ARCHIVE', archive, archive.stat().st_size, sha256(archive), flush=True); print('SUMMARY', json.dumps(summary, sort_keys=True), flush=True)
