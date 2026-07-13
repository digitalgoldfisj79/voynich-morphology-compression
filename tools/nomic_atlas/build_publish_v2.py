#!/usr/bin/env python3
from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import math
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from nomic import AtlasDataset, login
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

SEED = 20260713
RNG = np.random.default_rng(SEED)
OUT = Path("atlas_v2_output")
REVIEW = OUT / "review"
IMAGES = REVIEW / "images"
for directory in (OUT, REVIEW, IMAGES):
    directory.mkdir(parents=True, exist_ok=True)

WORD_SOURCE = "edwardbozzard/voynich-dinov3-words-20260713-v1"
PROPOSAL_SOURCE = "edwardbozzard/voynich-dinov3-proposals-50k-20260713-v1"
WORD_TARGET = "edwardbozzard/voynich-dinov3-words-enriched-20260713-v2"
PROPOSAL_TARGET = "edwardbozzard/voynich-dinov3-proposals-50k-enriched-20260713-v2"
SOURCE_COMMIT = "b86d96600dc49c5298e07dcecb1ebdbd44970483"
HF_DATASET = "Digitalgoldfish79/vdino3-crops"


def checkpoint(name: str, **values: Any) -> None:
    print("CHECKPOINT", name, json.dumps(values, sort_keys=True, default=str), flush=True)


def normalise(vectors: np.ndarray) -> np.ndarray:
    output = np.asarray(vectors, dtype=np.float32)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-9)
    return output


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype("int8")
        elif pd.api.types.is_object_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else str(value)
            )
        elif pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
    return frame


def load_map(identifier: str) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = AtlasDataset(identifier)
    dataset._latest_dataset_state()
    maps = list(dataset.maps)
    if not maps:
        raise RuntimeError(f"{identifier}: no map")
    projection = maps[-1]
    metadata = projection.data.df
    projected = projection.embeddings.projected
    vectors = np.asarray(projection.embeddings.latent, dtype=np.float32)
    if len(metadata) != len(projected) or len(projected) != len(vectors):
        raise RuntimeError(f"{identifier}: metadata/projected/latent row mismatch")
    if "atlas_id" not in metadata or "atlas_id" not in projected:
        raise RuntimeError(f"{identifier}: atlas_id unavailable")
    if metadata.atlas_id.duplicated().any() or projected.atlas_id.duplicated().any():
        raise RuntimeError(f"{identifier}: duplicate atlas_id")
    aligned = metadata.set_index(metadata.atlas_id.astype(str), drop=False).loc[
        projected.atlas_id.astype(str)
    ].reset_index(drop=True)
    if not np.array_equal(
        aligned.atlas_id.astype(str).to_numpy(),
        projected.atlas_id.astype(str).to_numpy(),
    ):
        raise RuntimeError(f"{identifier}: alignment failed")
    checkpoint("map_loaded", identifier=identifier, rows=len(aligned), dimensions=vectors.shape[1])
    return aligned, normalise(vectors)


def get_folio_metadata() -> pd.DataFrame:
    url = (
        "https://raw.githubusercontent.com/digitalgoldfisj79/"
        "voynich-morphology-compression/feat/nomic-atlas/"
        "tools/dinov3_program/folio_metadata.tsv.b64"
    )
    raw = base64.b64decode(urllib.request.urlopen(url, timeout=60).read())
    return pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str).fillna("").set_index("folio")


def add_folio_metadata(frame: pd.DataFrame, lookup: pd.DataFrame) -> None:
    frame["hand"] = frame.folio.astype(str).map(lookup.hand).fillna("")
    frame["currier"] = frame.folio.astype(str).map(lookup.language_flag).fillna("")
    frame["section"] = frame.folio.astype(str).map(lookup.section).fillna("Unassigned")


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def pair_record(words: pd.DataFrame, left: int, right: int, similarity: float, category: str) -> dict[str, Any]:
    left_word = str(words.at[left, "word"])
    right_word = str(words.at[right, "word"])
    left_crop = str(words.at[left, "crop_path"]) if "crop_path" in words and pd.notna(words.at[left, "crop_path"]) else ""
    right_crop = str(words.at[right, "crop_path"]) if "crop_path" in words and pd.notna(words.at[right, "crop_path"]) else ""
    return {
        "category": category,
        "left_id": str(words.at[left, "id"]),
        "right_id": str(words.at[right, "id"]),
        "left_word": left_word,
        "right_word": right_word,
        "left_folio": str(words.at[left, "folio"]),
        "right_folio": str(words.at[right, "folio"]),
        "cosine_similarity": similarity,
        "edit_distance": levenshtein(left_word, right_word),
        "left_crop_path": left_crop,
        "right_crop_path": right_crop,
        "both_images": int(bool(left_crop and right_crop)),
        "review_status": "unreviewed",
    }


def pick_unique(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (
            -int(row["both_images"]),
            -float(row["cosine_similarity"]),
            str(row["left_id"]),
            str(row["right_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = tuple(sorted((str(row["left_id"]), str(row["right_id"]))))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def analyse_words(words: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    words = words.copy().reset_index(drop=True)
    labels = words.word.fillna("").astype(str).to_numpy()
    folios = words.folio.fillna("").astype(str).to_numpy()
    ids = words.id.astype(str).to_numpy()
    model = NearestNeighbors(n_neighbors=129, metric="cosine", n_jobs=-1).fit(vectors)
    distances, indices = model.kneighbors(vectors)
    words["word_frequency"] = words.word.map(words.word.value_counts()).astype(int)
    words["knn_purity20"] = (labels[indices[:, 1:21]] == labels[:, None]).mean(1)
    words["nearest_neighbor_cosine"] = 1 - distances[:, 1]

    cross_indices = np.empty((len(words), 20), dtype=np.int32)
    cross_distances = np.empty((len(words), 20), dtype=np.float32)
    nearest_same = np.full(len(words), -1, dtype=np.int32)
    nearest_other = np.full(len(words), -1, dtype=np.int32)
    same_similarity = np.full(len(words), -1.0, dtype=np.float32)
    other_similarity = np.full(len(words), -1.0, dtype=np.float32)
    for row in range(len(words)):
        candidates = indices[row, 1:]
        candidate_distances = distances[row, 1:]
        mask = folios[candidates] != folios[row]
        candidates = candidates[mask]
        candidate_distances = candidate_distances[mask]
        if len(candidates) < 20:
            raise RuntimeError(f"word row {row}: insufficient cross-folio neighbours")
        cross_indices[row] = candidates[:20]
        cross_distances[row] = candidate_distances[:20]
        for candidate, distance in zip(candidates, candidate_distances, strict=False):
            if nearest_same[row] < 0 and labels[candidate] == labels[row]:
                nearest_same[row] = int(candidate)
                same_similarity[row] = 1 - distance
            if nearest_other[row] < 0 and labels[candidate] != labels[row]:
                nearest_other[row] = int(candidate)
                other_similarity[row] = 1 - distance
            if nearest_same[row] >= 0 and nearest_other[row] >= 0:
                break

    words["cross_folio_knn_purity20"] = (labels[cross_indices] == labels[:, None]).mean(1)
    words["cross_folio_nearest_cosine"] = 1 - cross_distances[:, 0]
    words["nearest_same_word_id"] = [ids[index] if index >= 0 else "" for index in nearest_same]
    words["nearest_same_word_folio"] = [folios[index] if index >= 0 else "" for index in nearest_same]
    words["nearest_same_word_cosine"] = same_similarity
    words["nearest_other_word_id"] = [ids[index] if index >= 0 else "" for index in nearest_other]
    words["nearest_other_word"] = [labels[index] if index >= 0 else "" for index in nearest_other]
    words["nearest_other_word_folio"] = [folios[index] if index >= 0 else "" for index in nearest_other]
    words["nearest_other_word_cosine"] = other_similarity
    words["word_outlier_score"] = 1 - np.maximum(same_similarity, 0)

    alias_counter: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in range(len(words)):
        for candidate in cross_indices[row, :5]:
            if labels[row] != labels[candidate]:
                alias_counter[tuple(sorted((labels[row], labels[candidate])))] += 1
    aliases = pd.DataFrame(
        [(a, b, count) for (a, b), count in alias_counter.most_common(1000)],
        columns=["word_a", "word_b", "cross_folio_knn_count"],
    )
    maximum_alias: dict[str, int] = collections.defaultdict(int)
    for record in aliases.itertuples():
        maximum_alias[record.word_a] = max(maximum_alias[record.word_a], record.cross_folio_knn_count)
        maximum_alias[record.word_b] = max(maximum_alias[record.word_b], record.cross_folio_knn_count)
    words["alias_max_count"] = words.word.map(maximum_alias).fillna(0).astype(int)

    words["visual_mode"] = -1
    words["mode_silhouette"] = -1.0
    multimodal_rows: list[dict[str, Any]] = []
    for word, group in words.groupby("word"):
        if len(group) < 20:
            continue
        row_ids = group.index.to_numpy()
        modes = MiniBatchKMeans(
            2, random_state=SEED, n_init=10, batch_size=256
        ).fit_predict(vectors[row_ids])
        silhouette = float(silhouette_score(vectors[row_ids], modes, metric="cosine"))
        words.loc[row_ids, "visual_mode"] = modes
        words.loc[row_ids, "mode_silhouette"] = silhouette

        def cramers(column: str) -> float:
            table = pd.crosstab(modes, group[column].fillna("")).to_numpy()
            total = table.sum()
            if min(table.shape) < 2:
                return 0.0
            expected = np.outer(table.sum(1), table.sum(0)) / total
            chi = ((table - expected) ** 2 / np.maximum(expected, 1e-9)).sum()
            return float(math.sqrt(chi / (total * max(1, min(table.shape) - 1))))

        multimodal_rows.append({
            "word": word,
            "n": len(group),
            "silhouette2": silhouette,
            "cramers_hand": cramers("hand"),
            "cramers_currier": cramers("currier"),
            "cramers_section": cramers("section"),
        })
    multimodality = pd.DataFrame(multimodal_rows)
    if not multimodality.empty:
        multimodality = multimodality.sort_values("silhouette2", ascending=False)

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in range(len(words)):
        same = nearest_same[row]
        if same >= 0:
            groups["exact_positive"].append(
                pair_record(words, row, int(same), float(same_similarity[row]), "exact_positive")
            )
        other = nearest_other[row]
        if other >= 0:
            groups["hard_negative"].append(
                pair_record(words, row, int(other), float(other_similarity[row]), "hard_negative")
            )
        if words.at[row, "visual_mode"] >= 0:
            for candidate, distance in zip(indices[row, 1:], distances[row, 1:], strict=False):
                if folios[candidate] == folios[row] or labels[candidate] != labels[row]:
                    continue
                if (
                    words.at[candidate, "visual_mode"] >= 0
                    and words.at[candidate, "visual_mode"] != words.at[row, "visual_mode"]
                ):
                    groups["allographic_positive"].append(
                        pair_record(words, row, int(candidate), float(1 - distance), "allographic_positive")
                    )
                    break
    pairs = (
        pick_unique(groups["exact_positive"], 400)
        + pick_unique(groups["allographic_positive"], 300)
        + pick_unique(groups["hard_negative"], 300)
    )
    for number, pair in enumerate(pairs, 1):
        pair["pair_id"] = f"P{number:04d}-{pair['category']}"
    pair_frame = pd.DataFrame(pairs)
    queue: dict[str, set[str]] = collections.defaultdict(set)
    for pair in pairs:
        queue[pair["left_id"]].add(pair["category"])
        queue[pair["right_id"]].add(pair["category"])
    words["review_queue"] = ["|".join(sorted(queue.get(identifier, set()))) for identifier in words.id.astype(str)]
    words["review_priority"] = (
        words.alias_max_count.astype(float)
        + 10 * np.maximum(words.mode_silhouette, 0)
        + 5 * words.word_outlier_score
        + 2 * (words.review_queue.str.len() > 0).astype(float)
    )
    checkpoint(
        "word_analysis",
        rows=len(words),
        aliases=len(aliases),
        multimodal_labels=len(multimodality),
        pairs=pair_frame.category.value_counts().to_dict(),
    )
    return words, aliases, multimodality, pair_frame


def analyse_proposals(proposals: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proposals = proposals.copy().reset_index(drop=True)
    training = RNG.choice(len(vectors), min(30000, len(vectors)), replace=False)
    pca = PCA(n_components=64, random_state=SEED).fit(vectors[training])
    reduced = pca.transform(vectors).astype(np.float32)
    model = MiniBatchKMeans(
        32, random_state=SEED, n_init=10, batch_size=4096, max_iter=300
    ).fit(reduced)
    clusters = model.labels_
    proposals["cluster"] = clusters
    proposals["cluster_distance"] = np.linalg.norm(
        reduced - model.cluster_centers_[clusters], axis=1
    )
    proposals["cluster_size"] = proposals.cluster.map(proposals.cluster.value_counts()).astype(int)

    cluster_rows: list[dict[str, Any]] = []
    for cluster, group in proposals.groupby("cluster"):
        eva_counts = group.eva_aligned.fillna("").astype(str).value_counts(normalize=True)
        kind_counts = group.kind.fillna("").astype(str).value_counts(normalize=True)
        cluster_rows.append({
            "cluster": int(cluster),
            "cluster_eva_purity": float(eva_counts.iloc[0]),
            "cluster_dominant_eva": str(eva_counts.index[0]),
            "cluster_kind_purity": float(kind_counts.iloc[0]),
            "cluster_dominant_kind": str(kind_counts.index[0]),
        })
    cluster_summary = pd.DataFrame(cluster_rows)
    proposals = proposals.merge(cluster_summary, on="cluster", how="left")

    distances, indices = NearestNeighbors(
        n_neighbors=11, metric="cosine", n_jobs=-1
    ).fit(reduced).kneighbors(reduced)
    eva = proposals.eva_aligned.fillna("").astype(str).to_numpy()
    proposals["eva_knn_purity10"] = (eva[indices[:, 1:]] == eva[:, None]).mean(1)
    proposals["proposal_outlier_score"] = distances[:, 1:].mean(1)
    slots = pd.to_numeric(proposals.get("slot", -1), errors="coerce").fillna(-1)
    slot_counts = pd.to_numeric(proposals.get("n_slots", 1), errors="coerce").fillna(1)
    proposals["slot_fraction"] = slots / np.maximum(slot_counts - 1, 1)

    folds = np.array([
        int(hashlib.sha1(str(folio).encode()).hexdigest(), 16) % 5
        for folio in proposals.folio
    ])
    prediction = np.full(len(proposals), "", dtype=object)
    for fold in range(5):
        train = folds != fold
        test = ~train
        mapping = {
            int(cluster): str(group.eva_aligned.fillna("").astype(str).value_counts().index[0])
            for cluster, group in proposals.loc[train].groupby("cluster")
        }
        prediction[test] = [mapping.get(int(cluster), "") for cluster in clusters[test]]
    proposals["heldout_eva_prediction"] = prediction
    proposals["heldout_eva_correct"] = (prediction == eva).astype(int)

    review_rows: list[dict[str, Any]] = []
    for cluster, group in proposals.groupby("cluster"):
        for review_type, subset in (
            ("medoid", group.nsmallest(5, "cluster_distance")),
            ("boundary", group.nlargest(5, "cluster_distance")),
        ):
            for row in subset.itertuples():
                review_rows.append({
                    "cluster": int(cluster),
                    "review_type": review_type,
                    "id": str(row.id),
                    "folio": str(row.folio),
                    "kind": str(row.kind),
                    "eva_aligned": str(row.eva_aligned),
                    "cluster_distance": float(row.cluster_distance),
                    "cluster_eva_purity": float(row.cluster_eva_purity),
                    "crop_path": str(getattr(row, "crop_path", "")),
                })
    review = pd.DataFrame(review_rows)
    checkpoint(
        "proposal_analysis",
        rows=len(proposals),
        clusters=int(proposals.cluster.nunique()),
        review_rows=len(review),
        pca_variance=float(pca.explained_variance_ratio_.sum()),
    )
    return proposals, cluster_summary, review


def download_crop(path: str, crop_id: str, token: str) -> str:
    if not path or not token:
        return ""
    target = IMAGES / f"{crop_id}.png"
    if target.exists():
        return f"images/{target.name}"
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{SOURCE_COMMIT}/{path}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    if response.status_code != 200:
        return ""
    target.write_bytes(response.content)
    return f"images/{target.name}"


def build_review(pairs: pd.DataFrame, proposal_review: pd.DataFrame) -> None:
    token = os.environ.get("HF_TOKEN", "")
    if token:
        left_images = []
        right_images = []
        for row in pairs.itertuples():
            left_images.append(download_crop(str(row.left_crop_path), str(row.left_id), token))
            right_images.append(download_crop(str(row.right_crop_path), str(row.right_id), token))
        pairs["left_image"] = left_images
        pairs["right_image"] = right_images
        proposal_review["image"] = [
            download_crop(str(row.crop_path), str(row.id), token)
            for row in proposal_review.itertuples()
        ]
    else:
        pairs["left_image"] = ""
        pairs["right_image"] = ""
        proposal_review["image"] = ""

    pairs.to_csv(REVIEW / "benchmark_candidates.csv", index=False)
    proposal_review.to_csv(REVIEW / "proposal_review_queue.csv", index=False)
    payload_pairs = json.dumps(pairs.fillna("").to_dict("records"), ensure_ascii=False).replace("</", "<\\/")
    payload_props = json.dumps(proposal_review.fillna("").to_dict("records"), ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html><meta charset="utf-8"><title>Voynich Atlas v2 review</title>
<style>body{{font-family:system-ui;margin:20px;background:#f3f3f3}}.card{{background:white;padding:12px;margin:10px 0;border:1px solid #ccc}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{max-width:100%;max-height:180px}}button{{margin:3px}}</style>
<h1>Voynich DINOv3 Atlas v2 review</h1>
<button onclick="tab('pairs')">Pair benchmark</button><button onclick="tab('props')">Proposal clusters</button><button onclick="exportCSV()">Export decisions</button>
<section id="pairs"><h2>1,000 benchmark candidates</h2><div id="pairlist"></div></section>
<section id="props" hidden><h2>Cluster medoids and boundaries</h2><div id="proplist"></div></section>
<script>
const pairs={payload_pairs}, props={payload_props};
const decisions=JSON.parse(localStorage.getItem('voynichReviewV2')||'{{}}');
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const image=p=>p?`<img src="${{esc(p)}}">`:'<i>Image unavailable; use crop_path metadata</i>';
function tab(x){{document.querySelector('#pairs').hidden=x!=='pairs';document.querySelector('#props').hidden=x!=='props'}}
function decide(id,v){{decisions[id]=v;localStorage.setItem('voynichReviewV2',JSON.stringify(decisions));render()}}
function render(){{
 document.querySelector('#pairlist').innerHTML=pairs.slice(0,1000).map(r=>`<div class=card><b>${{esc(r.pair_id)}}</b> · ${{esc(r.category)}} · cosine ${{Number(r.cosine_similarity).toFixed(4)}} · decision <b>${{esc(decisions[r.pair_id]||'unreviewed')}}</b><div class=pair><div>${{image(r.left_image)}}<br>${{esc(r.left_word)}} · ${{esc(r.left_folio)}} · ${{esc(r.left_id)}}</div><div>${{image(r.right_image)}}<br>${{esc(r.right_word)}} · ${{esc(r.right_folio)}} · ${{esc(r.right_id)}}</div></div><button onclick="decide('${{r.pair_id}}','same_form')">Same form</button><button onclick="decide('${{r.pair_id}}','allograph')">Allograph</button><button onclick="decide('${{r.pair_id}}','different')">Different</button><button onclick="decide('${{r.pair_id}}','bad_crop')">Bad crop</button></div>`).join('');
 document.querySelector('#proplist').innerHTML=props.map(r=>`<div class=card><b>Cluster ${{esc(r.cluster)}} · ${{esc(r.review_type)}}</b><br>${{image(r.image)}}<br>${{esc(r.id)}} · EVA ${{esc(r.eva_aligned)}} · ${{esc(r.kind)}} · ${{esc(r.folio)}}</div>`).join('');
}}
function exportCSV(){{let rows=['pair_id,decision',...Object.entries(decisions).map(([k,v])=>`"${{k}}","${{v}}"`)];let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='voynich_pair_decisions.csv';a.click()}}
render();
</script>"""
    (REVIEW / "index.html").write_text(html, encoding="utf-8")
    (REVIEW / "README.md").write_text(
        "# Voynich DINOv3 Atlas v2 review\n\nOpen `index.html` locally. Decisions are stored in browser localStorage and can be exported as CSV. Candidates are not adjudicated ground truth.\n",
        encoding="utf-8",
    )
    checkpoint("review_built", pair_rows=len(pairs), proposal_rows=len(proposal_review), images=len(list(IMAGES.glob("*.png"))))


def select_columns(frame: pd.DataFrame, columns: list[str], level: str) -> pd.DataFrame:
    selected = frame[[column for column in columns if column in frame.columns]].copy()
    selected["atlas_id"] = level + "-v2-" + selected["id"].astype(str)
    selected["atlas_level"] = level
    if level == "word":
        selected["display_text"] = (
            selected["word"].astype(str)
            + " | "
            + selected["folio"].astype(str)
            + " | hand "
            + selected["hand"].astype(str)
            + " | Currier "
            + selected["currier"].astype(str)
        )
    else:
        selected["display_text"] = (
            selected["kind"].astype(str)
            + " | EVA "
            + selected["eva_aligned"].astype(str)
            + " | "
            + selected["folio"].astype(str)
            + " | cluster "
            + selected["cluster"].astype(str)
        )
    return clean_frame(selected)


def projection_stage(projection: Any) -> str:
    try:
        return str(projection._status.get("index_build_stage"))
    except Exception:
        return ""


def publish(identifier: str, frame: pd.DataFrame, vectors: np.ndarray, index_name: str, description: str) -> dict[str, Any]:
    dataset = AtlasDataset(
        identifier,
        description=description,
        unique_id_field="atlas_id",
        is_public=True,
    )
    dataset._latest_dataset_state()
    existing = int(dataset.total_datums)
    if existing == 0:
        dataset.add_data(data=frame, embeddings=np.asarray(vectors, dtype=np.float32))
    elif existing != len(frame):
        raise RuntimeError(f"{identifier}: existing rows {existing}, expected {len(frame)}; refusing to append")
    dataset._latest_dataset_state()
    maps = list(dataset.maps)
    if not maps:
        projection = dataset.create_index(
            name=index_name,
            modality="embedding",
            topic_model=False,
            duplicate_detection=False,
        )
    else:
        projection = maps[-1]
    deadline = time.monotonic() + 3600
    while projection is not None and time.monotonic() < deadline:
        stage = projection_stage(projection)
        if stage == "Completed":
            break
        if stage in {"Failed", "Error", "Cancelled", "Canceled"}:
            raise RuntimeError(f"{identifier}: index stage {stage}")
        time.sleep(15)
        dataset._latest_dataset_state()
        projection = list(dataset.maps)[-1] if dataset.maps else projection
    dataset._latest_dataset_state()
    if int(dataset.total_datums) != len(frame):
        raise RuntimeError(f"{identifier}: final row count mismatch")
    result = {
        "identifier": dataset.identifier,
        "rows": int(dataset.total_datums),
        "stage": projection_stage(projection),
        "dataset_link": f"https://atlas.nomic.ai/data/{dataset.meta['organization_slug']}/{dataset.meta['slug']}",
        "map_link": f"https://atlas.nomic.ai/data/{dataset.meta['organization_slug']}/{dataset.meta['slug']}/map",
    }
    checkpoint("published", **result)
    return result


def main() -> None:
    token = os.environ.get("NOMIC_API_KEY", "")
    if not token:
        raise RuntimeError("NOMIC_API_KEY is missing")
    login(token)
    words, word_vectors = load_map(WORD_SOURCE)
    proposals, proposal_vectors = load_map(PROPOSAL_SOURCE)
    metadata = get_folio_metadata()
    add_folio_metadata(words, metadata)
    add_folio_metadata(proposals, metadata)

    words, aliases, multimodality, pairs = analyse_words(words, word_vectors)
    proposals, cluster_summary, proposal_review = analyse_proposals(proposals, proposal_vectors)
    aliases.to_csv(REVIEW / "alias_pairs.csv", index=False)
    multimodality.to_csv(REVIEW / "multimodality.csv", index=False)
    cluster_summary.to_csv(REVIEW / "proposal_cluster_summary.csv", index=False)
    build_review(pairs, proposal_review)

    word_columns = [
        "id", "folio", "word", "word_index", "hand", "currier", "section",
        "word_frequency", "crop_path", "has_crop_image", "knn_purity20",
        "cross_folio_knn_purity20", "nearest_neighbor_cosine",
        "cross_folio_nearest_cosine", "nearest_same_word_id",
        "nearest_same_word_folio", "nearest_same_word_cosine",
        "nearest_other_word_id", "nearest_other_word",
        "nearest_other_word_folio", "nearest_other_word_cosine",
        "word_outlier_score", "alias_max_count", "visual_mode",
        "mode_silhouette", "review_queue", "review_priority", "source_row",
        "source_dataset", "source_commit",
    ]
    proposal_columns = [
        "id", "folio", "kind", "word", "word_index", "slot", "n_slots",
        "word_len", "eva_aligned", "hand", "currier", "section", "crop_path",
        "has_crop_image", "cluster", "cluster_distance", "cluster_size",
        "cluster_eva_purity", "cluster_dominant_eva",
        "cluster_kind_purity", "cluster_dominant_kind",
        "eva_knn_purity10", "proposal_outlier_score", "slot_fraction",
        "heldout_eva_prediction", "heldout_eva_correct", "source_row",
        "source_dataset", "source_commit",
    ]
    word_atlas = select_columns(words, word_columns, "word")
    proposal_atlas = select_columns(proposals, proposal_columns, "proposal")

    results = [
        publish(
            WORD_TARGET,
            word_atlas,
            word_vectors,
            "voynich-dinov3-word-map-enriched-v2",
            "Enriched Voynich DINOv3 word map with cross-folio neighbours, outliers, multimodality, hand, Currier and section metadata.",
        ),
        publish(
            PROPOSAL_TARGET,
            proposal_atlas,
            proposal_vectors,
            "voynich-dinov3-proposal-map-enriched-v2",
            "Enriched Voynich DINOv3 proposal map with 32-cluster diagnostics, medoid distances, held-out EVA transfer and manuscript metadata.",
        ),
    ]
    summary = {
        "schema_version": "voynich-atlas-v2/1",
        "seed": SEED,
        "sources": [WORD_SOURCE, PROPOSAL_SOURCE],
        "targets": results,
        "word_rows": len(word_atlas),
        "proposal_rows": len(proposal_atlas),
        "pair_counts": pairs.category.value_counts().to_dict(),
        "proposal_clusters": int(proposals.cluster.nunique()),
        "review_images": len(list(IMAGES.glob("*.png"))),
        "hf_images_enabled": bool(os.environ.get("HF_TOKEN")),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    archive = Path("voynich-atlas-enriched-v2.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(OUT.parent))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    checkpoint("complete", archive=str(archive), sha256=digest, summary=summary)


if __name__ == "__main__":
    main()
