#!/usr/bin/env python3
"""Audit the existing 100k enriched Voynich proposal Atlas in place.

This script does not create, mutate, append to, or delete any Nomic dataset. It
materialises the existing map metadata and latent vectors, computes cluster and
projection diagnostics, and writes review queues and a reproducible report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from nomic import AtlasDataset, login
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

SEED = 20260714
DEFAULT_DATASET = "edwardbozzard/voynich-dinov3-proposals-100k-enriched-20260713-v2"
MAP_LINK = "https://atlas.nomic.ai/data/edwardbozzard/voynich-dinov3-proposals-100k-enriched-20260713-v2/map"


def checkpoint(name: str, **payload: object) -> None:
    print("CHECKPOINT", name, json.dumps(payload, sort_keys=True, default=str), flush=True)


def entropy(values: pd.Series) -> float:
    counts = values.fillna("").astype(str).value_counts().to_numpy(dtype=float)
    if counts.sum() == 0 or len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def dominance(values: pd.Series, *, ignore_blank: bool = False) -> tuple[str, float, int]:
    s = values.fillna("").astype(str)
    if ignore_blank:
        s = s[s != ""]
    counts = s.value_counts()
    if counts.empty:
        return "", 0.0, 0
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum()), int(len(counts))


def normalized_entropy(values: pd.Series, *, ignore_blank: bool = False) -> float:
    s = values.fillna("").astype(str)
    if ignore_blank:
        s = s[s != ""]
    n = s.nunique()
    if n <= 1:
        return 0.0
    return float(entropy(s) / math.log2(n))


def stable_hash(*parts: object) -> str:
    text = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.blake2b(text, digest_size=10).hexdigest()


def choose_review_class(row: dict[str, object]) -> tuple[str, float, str]:
    eva = float(row["eva_purity"])
    kind = float(row["kind_purity"])
    folios = int(row["folio_count"])
    init = float(row["initial_fraction"])
    final = float(row["final_fraction"])
    page = float(row["max_folio_share"])
    margin = float(row["centroid_margin_median"])

    if eva >= 0.55 and folios >= 20 and page <= 0.15:
        return "candidate_eva_unit", eva + 0.3 * margin, "high EVA concentration across many folios"
    if eva >= 0.32 and max(init, final) >= 0.70 and folios >= 15:
        return "candidate_positional_allograph", eva + max(init, final), "EVA signal concentrated at a word edge"
    if kind >= 0.90 and eva < 0.22:
        return "segmentation_geometry", kind - eva, "proposal type dominates while EVA alignment is diffuse"
    if page >= 0.25:
        return "folio_or_crop_artifact", page, "a large share of the cluster comes from one folio"
    if eva < 0.18 and margin < 0.04:
        return "mixed_boundary_region", 1.0 - eva + 1.0 - margin, "low EVA purity and weak centroid assignment margin"
    return "mixed_recurrent_form", eva + 0.25 * margin, "recurrent visual family without a clean EVA interpretation"


def dedupe_pairs(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, object]] = []
    for row in rows:
        a, b = sorted((str(row["atlas_id_a"]), str(row["atlas_id_b"])))
        key = (str(row["pair_class"]), a, b)
        if a == b or key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("nomic_100k_audit"))
    args = parser.parse_args()

    token = os.environ.get("NOMIC_API_KEY")
    if not token:
        raise SystemExit("NOMIC_API_KEY is required for authenticated map export")

    np.random.seed(SEED)
    random.seed(SEED)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    login(token)
    ds = AtlasDataset(args.dataset)
    ds._latest_dataset_state()
    if ds.total_datums != 100_000:
        raise RuntimeError(f"Expected 100,000 rows, found {ds.total_datums:,}")
    if len(ds.maps) != 1:
        raise RuntimeError(f"Expected one map, found {len(ds.maps)}")
    projection = ds.maps[0]

    df = projection.data.df.reset_index(drop=True)
    X = np.asarray(projection.embeddings.latent, dtype=np.float32)
    projected = projection.embeddings.projected.reset_index(drop=True)
    if len(df) != len(X) or len(projected) != len(X):
        raise RuntimeError(f"Map alignment failure: metadata={len(df)}, latent={len(X)}, projected={len(projected)}")
    if X.ndim != 2:
        raise RuntimeError(f"Latent array is not 2-D: {X.shape}")
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

    # Nomic guarantees map.data and map.embeddings orders are aligned. Preserve
    # both identifiers where available and attach the projected coordinates.
    id_col = "atlas_id" if "atlas_id" in df.columns else df.columns[0]
    if df[id_col].astype(str).duplicated().any():
        raise RuntimeError(f"Unique ID field {id_col!r} contains duplicates")
    if {"x", "y"}.issubset(projected.columns):
        df["map_x"] = projected["x"].to_numpy()
        df["map_y"] = projected["y"].to_numpy()
    else:
        raise RuntimeError(f"Projected coordinates missing x/y: {list(projected.columns)}")

    required = {
        "cluster", "cluster_distance", "eva_aligned", "kind", "folio", "slot_fraction",
        "eva_knn_purity10", "proposal_outlier_score", "hand", "currier", "section",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"Missing enriched fields: {missing}")

    df["cluster"] = pd.to_numeric(df["cluster"], errors="raise").astype(int)
    df["cluster_distance"] = pd.to_numeric(df["cluster_distance"], errors="coerce")
    df["slot_fraction"] = pd.to_numeric(df["slot_fraction"], errors="coerce").fillna(0.5)
    df["eva_knn_purity10"] = pd.to_numeric(df["eva_knn_purity10"], errors="coerce").fillna(0.0)
    df["proposal_outlier_score"] = pd.to_numeric(df["proposal_outlier_score"], errors="coerce").fillna(0.0)
    df["position_class"] = np.where(
        df["slot_fraction"] <= 0.15,
        "initial",
        np.where(df["slot_fraction"] >= 0.85, "final", "medial"),
    )

    checkpoint(
        "loaded",
        dataset=ds.identifier,
        rows=len(df),
        dimensions=X.shape[1],
        clusters=int(df.cluster.nunique()),
        fields=len(df.columns),
    )

    # Original-space cluster centroids and assignment margins.
    cluster_ids = sorted(df.cluster.unique().tolist())
    centroids = []
    for cluster in cluster_ids:
        centroid = X[df.cluster.to_numpy() == cluster].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        centroids.append(centroid)
    C = np.stack(centroids).astype(np.float32)
    sim = X @ C.T
    order = np.argsort(sim, axis=1)
    best = order[:, -1]
    second = order[:, -2]
    df["assigned_centroid_similarity"] = sim[np.arange(len(df)), df.cluster.map({c: i for i, c in enumerate(cluster_ids)}).to_numpy()]
    df["best_centroid_cluster"] = np.array(cluster_ids)[best]
    df["centroid_margin"] = sim[np.arange(len(df)), best] - sim[np.arange(len(df)), second]
    df["centroid_assignment_agrees"] = (df.best_centroid_cluster.to_numpy() == df.cluster.to_numpy()).astype(int)
    df["latent_outlier"] = 1.0 - df["assigned_centroid_similarity"]

    cluster_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    for cluster in cluster_ids:
        g = df[df.cluster == cluster]
        ids = g.index.to_numpy()
        dominant_eva, eva_purity, eva_labels = dominance(g.eva_aligned, ignore_blank=True)
        dominant_kind, kind_purity, kind_labels = dominance(g.kind)
        dominant_hand, hand_purity, hand_labels = dominance(g.hand, ignore_blank=True)
        dominant_currier, currier_purity, currier_labels = dominance(g.currier, ignore_blank=True)
        dominant_section, section_purity, section_labels = dominance(g.section, ignore_blank=True)
        folio_counts = g.folio.astype(str).value_counts()
        max_folio_share = float(folio_counts.iloc[0] / len(g))
        position = g.position_class.value_counts(normalize=True)

        local_sim = g.assigned_centroid_similarity.to_numpy()
        reps = ids[np.argsort(-local_sim)[:12]]
        outliers = ids[np.argsort(local_sim)[:10]]
        boundaries = ids[np.argsort(g.centroid_margin.to_numpy())[:10]]

        row: dict[str, object] = {
            "cluster": int(cluster),
            "size": int(len(g)),
            "folio_count": int(g.folio.nunique()),
            "max_folio": str(folio_counts.index[0]),
            "max_folio_share": max_folio_share,
            "dominant_eva": dominant_eva,
            "eva_purity": eva_purity,
            "eva_label_count": eva_labels,
            "eva_entropy_norm": normalized_entropy(g.eva_aligned, ignore_blank=True),
            "dominant_kind": dominant_kind,
            "kind_purity": kind_purity,
            "kind_label_count": kind_labels,
            "kind_entropy_norm": normalized_entropy(g.kind),
            "dominant_hand": dominant_hand,
            "hand_purity": hand_purity,
            "hand_label_count": hand_labels,
            "dominant_currier": dominant_currier,
            "currier_purity": currier_purity,
            "currier_label_count": currier_labels,
            "dominant_section": dominant_section,
            "section_purity": section_purity,
            "section_label_count": section_labels,
            "initial_fraction": float(position.get("initial", 0.0)),
            "medial_fraction": float(position.get("medial", 0.0)),
            "final_fraction": float(position.get("final", 0.0)),
            "slot_fraction_mean": float(g.slot_fraction.mean()),
            "slot_fraction_sd": float(g.slot_fraction.std(ddof=0)),
            "cluster_distance_median": float(g.cluster_distance.median()),
            "latent_outlier_median": float(g.latent_outlier.median()),
            "latent_outlier_p95": float(g.latent_outlier.quantile(0.95)),
            "centroid_margin_median": float(g.centroid_margin.median()),
            "centroid_assignment_agreement": float(g.centroid_assignment_agrees.mean()),
            "eva_knn_purity10_mean": float(g.eva_knn_purity10.mean()),
            "map_x_sd": float(g.map_x.std(ddof=0)),
            "map_y_sd": float(g.map_y.std(ddof=0)),
            "medoid_atlas_id": str(df.loc[reps[0], id_col]),
            "medoid_crop_id": str(df.loc[reps[0], "id"]),
            "representative_atlas_ids": json.dumps(df.loc[reps, id_col].astype(str).tolist()),
            "outlier_atlas_ids": json.dumps(df.loc[outliers, id_col].astype(str).tolist()),
            "boundary_atlas_ids": json.dumps(df.loc[boundaries, id_col].astype(str).tolist()),
        }
        review_class, review_score, reason = choose_review_class(row)
        row["review_class"] = review_class
        row["review_score"] = float(review_score)
        row["review_reason"] = reason
        cluster_rows.append(row)

        for role, selected in (("representative", reps), ("outlier", outliers), ("boundary", boundaries)):
            for rank, index in enumerate(selected, 1):
                item = df.loc[index]
                representative_rows.append({
                    "cluster": int(cluster),
                    "role": role,
                    "rank": rank,
                    "atlas_id": str(item[id_col]),
                    "crop_id": str(item["id"]),
                    "folio": str(item["folio"]),
                    "word": str(item["word"]),
                    "word_index": int(item["word_index"]),
                    "slot": int(item["slot"]),
                    "n_slots": int(item["n_slots"]),
                    "eva_aligned": str(item["eva_aligned"]),
                    "kind": str(item["kind"]),
                    "hand": str(item["hand"]),
                    "currier": str(item["currier"]),
                    "section": str(item["section"]),
                    "slot_fraction": float(item["slot_fraction"]),
                    "centroid_similarity": float(item["assigned_centroid_similarity"]),
                    "centroid_margin": float(item["centroid_margin"]),
                    "eva_knn_purity10": float(item["eva_knn_purity10"]),
                    "private_crop_path": str(item.get("crop_path_private", "")),
                })

    clusters = pd.DataFrame(cluster_rows).sort_values(["review_class", "review_score"], ascending=[True, False])
    representatives = pd.DataFrame(representative_rows)
    clusters.to_csv(out / "cluster_audit.csv", index=False)
    representatives.to_csv(out / "cluster_review_examples.csv", index=False)

    # Quantify how much of the 768-D geometry the visible 2-D map preserves.
    rng = np.random.default_rng(SEED)
    map_sample = rng.choice(len(df), size=min(5000, len(df)), replace=False)
    tw = float(trustworthiness(X[map_sample], df.loc[map_sample, ["map_x", "map_y"]].to_numpy(), n_neighbors=10, metric="cosine"))
    pair_a = rng.integers(0, len(df), size=100_000)
    pair_b = rng.integers(0, len(df), size=100_000)
    latent_dist = 1.0 - np.sum(X[pair_a] * X[pair_b], axis=1)
    map_xy = df[["map_x", "map_y"]].to_numpy(dtype=np.float32)
    map_dist = np.linalg.norm(map_xy[pair_a] - map_xy[pair_b], axis=1)
    distance_spearman = float(spearmanr(latent_dist, map_dist).statistic)

    # A reduced representation supports stable local-neighbour and cluster checks.
    pca_fit = rng.choice(len(df), size=min(30_000, len(df)), replace=False)
    pca = PCA(n_components=32, svd_solver="randomized", random_state=SEED, iterated_power=3)
    pca.fit(X[pca_fit])
    Z = pca.transform(X).astype(np.float32)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    silhouette_sample = rng.choice(len(df), size=min(12_000, len(df)), replace=False)
    latent_silhouette = float(silhouette_score(Z[silhouette_sample], df.cluster.to_numpy()[silhouette_sample], metric="cosine", sample_size=min(8000, len(silhouette_sample)), random_state=SEED))
    map_silhouette = float(silhouette_score(map_xy[silhouette_sample], df.cluster.to_numpy()[silhouette_sample], metric="euclidean", sample_size=min(8000, len(silhouette_sample)), random_state=SEED))

    # Neighbour overlap on a fixed 10k submap.
    overlap_sample = rng.choice(len(df), size=min(10_000, len(df)), replace=False)
    query_local = rng.choice(len(overlap_sample), size=min(2000, len(overlap_sample)), replace=False)
    latent_nn = NearestNeighbors(n_neighbors=11, metric="cosine", n_jobs=-1).fit(Z[overlap_sample])
    map_nn = NearestNeighbors(n_neighbors=11, metric="euclidean", n_jobs=-1).fit(map_xy[overlap_sample])
    latent_ind = latent_nn.kneighbors(Z[overlap_sample][query_local], return_distance=False)[:, 1:]
    map_ind = map_nn.kneighbors(map_xy[overlap_sample][query_local], return_distance=False)[:, 1:]
    overlap10 = float(np.mean([len(set(a).intersection(set(b))) / 10 for a, b in zip(latent_ind, map_ind, strict=True)]))

    # Recompute cluster stability with three independent MiniBatchKMeans runs.
    labels = df.cluster.to_numpy()
    aris = []
    for offset in (1, 2, 3):
        alt = MiniBatchKMeans(
            n_clusters=len(cluster_ids), random_state=SEED + offset, n_init=3,
            batch_size=4096, max_iter=200,
        ).fit_predict(Z)
        aris.append(float(adjusted_rand_score(labels, alt)))

    map_diagnostics = {
        "dataset": ds.identifier,
        "map_link": MAP_LINK,
        "rows": len(df),
        "latent_dimensions": int(X.shape[1]),
        "pca32_variance": float(pca.explained_variance_ratio_.sum()),
        "trustworthiness_at_10_sample5000": tw,
        "random_pair_distance_spearman": distance_spearman,
        "neighbour_overlap_at_10_submap10000": overlap10,
        "cluster_silhouette_latent_pca32": latent_silhouette,
        "cluster_silhouette_map2d": map_silhouette,
        "cluster_recovery_ari_mean": float(np.mean(aris)),
        "cluster_recovery_ari_runs": aris,
        "centroid_assignment_agreement": float(df.centroid_assignment_agrees.mean()),
    }
    (out / "map_diagnostics.json").write_text(json.dumps(map_diagnostics, indent=2) + "\n")
    checkpoint("map_diagnostics", **map_diagnostics)

    # Query Nomic's original latent index for a bounded, diverse set of review
    # examples. This avoids an O(n^2) all-pairs computation.
    candidate_ids: list[str] = []
    for cluster in cluster_ids:
        g = representatives[representatives.cluster == cluster]
        for role, n in (("representative", 10), ("outlier", 6), ("boundary", 8)):
            candidate_ids.extend(g[g.role == role].sort_values("rank").atlas_id.head(n).tolist())
    # Add globally difficult and high-confidence cases.
    candidate_ids.extend(df.nsmallest(150, "centroid_margin")[id_col].astype(str).tolist())
    candidate_ids.extend(df.nlargest(150, "eva_knn_purity10")[id_col].astype(str).tolist())
    candidate_ids = list(dict.fromkeys(candidate_ids))[:1200]

    row_by_id = {str(row[id_col]): row for _, row in df.iterrows()}
    pair_candidates: list[dict[str, object]] = []
    for start in range(0, len(candidate_ids), 200):
        query_ids = candidate_ids[start : start + 200]
        neighbor_ids, distances = projection.embeddings.vector_search(ids=query_ids, k=30)
        for query_id, neighbors, dists in zip(query_ids, neighbor_ids, distances, strict=True):
            a = row_by_id.get(str(query_id))
            if a is None:
                continue
            for neighbor_id, distance in zip(neighbors, dists, strict=True):
                b = row_by_id.get(str(neighbor_id))
                if b is None or str(neighbor_id) == str(query_id):
                    continue
                same_eva = str(a.eva_aligned) == str(b.eva_aligned) and str(a.eva_aligned) != ""
                different_context = (
                    str(a.folio) != str(b.folio)
                    and (str(a.hand) != str(b.hand) or str(a.currier) != str(b.currier) or int(a.cluster) != int(b.cluster))
                )
                if same_eva and int(a.cluster) == int(b.cluster) and str(a.folio) != str(b.folio):
                    pair_class = "confirmed_positive_candidate"
                elif same_eva and different_context:
                    pair_class = "allograph_candidate"
                elif not same_eva:
                    pair_class = "hard_negative_candidate"
                else:
                    continue
                pair_candidates.append({
                    "pair_id": stable_hash(pair_class, query_id, neighbor_id),
                    "pair_class": pair_class,
                    "atlas_id_a": str(query_id),
                    "atlas_id_b": str(neighbor_id),
                    "crop_id_a": str(a.id),
                    "crop_id_b": str(b.id),
                    "distance": float(distance),
                    "similarity": float(1.0 - distance),
                    "eva_a": str(a.eva_aligned),
                    "eva_b": str(b.eva_aligned),
                    "kind_a": str(a.kind),
                    "kind_b": str(b.kind),
                    "cluster_a": int(a.cluster),
                    "cluster_b": int(b.cluster),
                    "folio_a": str(a.folio),
                    "folio_b": str(b.folio),
                    "hand_a": str(a.hand),
                    "hand_b": str(b.hand),
                    "currier_a": str(a.currier),
                    "currier_b": str(b.currier),
                    "slot_fraction_a": float(a.slot_fraction),
                    "slot_fraction_b": float(b.slot_fraction),
                    "review_label": "",
                    "review_notes": "",
                })

    pairs = pd.DataFrame(dedupe_pairs(pair_candidates))
    selected_pair_frames = []
    targets = {
        "confirmed_positive_candidate": 350,
        "allograph_candidate": 250,
        "hard_negative_candidate": 250,
    }
    for pair_class, target in targets.items():
        subset = pairs[pairs.pair_class == pair_class].sort_values("distance").head(target)
        selected_pair_frames.append(subset)
    benchmark = pd.concat(selected_pair_frames, ignore_index=True) if selected_pair_frames else pd.DataFrame()
    benchmark.to_csv(out / "benchmark_pairs.csv", index=False)

    segmentation = df.sort_values(
        ["proposal_outlier_score", "centroid_margin", "eva_knn_purity10"],
        ascending=[False, True, True],
    ).head(150).copy()
    segmentation["review_label"] = ""
    segmentation["review_notes"] = ""
    segmentation[[
        id_col, "id", "folio", "word", "word_index", "slot", "n_slots", "kind", "eva_aligned",
        "cluster", "cluster_distance", "latent_outlier", "centroid_margin", "eva_knn_purity10",
        "proposal_outlier_score", "slot_fraction", "hand", "currier", "section",
        "crop_path_private", "review_label", "review_notes",
    ]].to_csv(out / "segmentation_review_queue.csv", index=False)

    # Persist all enriched point diagnostics in a compact Parquet file.
    df[[
        id_col, "id", "folio", "word", "word_index", "slot", "n_slots", "word_len", "kind",
        "eva_aligned", "hand", "currier", "section", "cluster", "cluster_distance",
        "cluster_size", "cluster_eva_purity", "cluster_dominant_eva", "cluster_kind_purity",
        "cluster_dominant_kind", "eva_knn_purity10", "proposal_outlier_score", "slot_fraction",
        "position_class", "heldout_eva_prediction", "heldout_eva_correct", "map_x", "map_y",
        "assigned_centroid_similarity", "latent_outlier", "centroid_margin",
        "best_centroid_cluster", "centroid_assignment_agrees", "crop_path_private",
    ]].to_parquet(out / "point_diagnostics.parquet", index=False)

    class_counts = clusters.review_class.value_counts().to_dict()
    top_candidates = clusters.sort_values("review_score", ascending=False).head(12)
    report_lines = [
        "# Existing Nomic 100k Proposal Map Audit",
        "",
        f"Dataset: `{ds.identifier}`  ",
        f"Map: {MAP_LINK}  ",
        f"Rows: {len(df):,}; latent dimensions: {X.shape[1]}; clusters: {len(cluster_ids)}.",
        "",
        "## Purpose",
        "",
        "This audit analyses the existing map in place. It does not create or replace a Nomic dataset. Cluster labels are triage categories, not asserted glyph identities.",
        "",
        "## Map fidelity",
        "",
        f"- Trustworthiness@10 on a fixed 5,000-point sample: **{tw:.4f}**",
        f"- Latent/2-D random-pair distance Spearman correlation: **{distance_spearman:.4f}**",
        f"- Exact neighbour overlap@10 on a fixed 10,000-point submap: **{overlap10:.4f}**",
        f"- Cluster silhouette in PCA-32 latent space: **{latent_silhouette:.4f}**",
        f"- Cluster silhouette in the visible 2-D map: **{map_silhouette:.4f}**",
        f"- Mean ARI against three independent 32-cluster fits: **{np.mean(aris):.4f}**",
        "",
        "The 2-D map is suitable for navigation and review, but conclusions about neighbourhoods and cluster identity should be checked against the latent metrics supplied here.",
        "",
        "## Cluster triage counts",
        "",
    ]
    for name, count in sorted(class_counts.items()):
        report_lines.append(f"- `{name}`: {count}")
    report_lines += [
        "",
        "## Highest-priority clusters",
        "",
        "| cluster | review class | size | dominant EVA | EVA purity | dominant kind | kind purity | initial | final | folios | reason |",
        "|---:|---|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in top_candidates.itertuples():
        report_lines.append(
            f"| {row.cluster} | {row.review_class} | {row.size} | {row.dominant_eva or '—'} | {row.eva_purity:.3f} | "
            f"{row.dominant_kind} | {row.kind_purity:.3f} | {row.initial_fraction:.3f} | {row.final_fraction:.3f} | "
            f"{row.folio_count} | {row.review_reason} |"
        )
    report_lines += [
        "",
        "## Review products",
        "",
        f"- `{len(clusters):,}` cluster summaries in `cluster_audit.csv`.",
        f"- `{len(representatives):,}` medoid, representative, boundary and outlier examples in `cluster_review_examples.csv`.",
        f"- `{len(benchmark):,}` pair candidates in `benchmark_pairs.csv`.",
        f"- `{len(segmentation):,}` highest-priority segmentation cases in `segmentation_review_queue.csv`.",
        "- All 100,000 point-level diagnostics in `point_diagnostics.parquet`.",
        "",
        "## Interpretation rule",
        "",
        "A cluster is only a candidate manuscript unit when its visual coherence is accompanied by cross-folio recurrence, a concentrated but independently reviewed EVA/position profile, and stable representatives. Visible separation in the 2-D map alone is insufficient.",
    ]
    (out / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "schema": "voynich-nomic-100k-audit/1",
        "seed": SEED,
        "dataset": ds.identifier,
        "project_id": ds.id,
        "map_link": MAP_LINK,
        "rows": len(df),
        "dimensions": int(X.shape[1]),
        "clusters": len(cluster_ids),
        "map_diagnostics": map_diagnostics,
        "review_class_counts": class_counts,
        "benchmark_pairs": int(len(benchmark)),
        "segmentation_queue": int(len(segmentation)),
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    manifest = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            manifest.append({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checkpoint(
        "complete",
        cluster_classes=class_counts,
        benchmark_pairs=len(benchmark),
        segmentation_queue=len(segmentation),
        output=str(out),
    )


if __name__ == "__main__":
    main()
