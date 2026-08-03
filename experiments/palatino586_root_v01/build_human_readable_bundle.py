#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
import zipfile
from urllib.parse import quote

import requests

ROOT = pathlib.Path("/tmp/p586_bundle")
BUNDLE_NAME = "Palatino586_Voynich_Root_Morphology_Human_Readable_v01"
BUNDLE = ROOT / BUNDLE_NAME
ZIP_PATH = ROOT / f"{BUNDLE_NAME}.zip"
SUPABASE_PUBLIC = "https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public"
UPLOAD_EDGE = "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/p586-bundle-signed-upload-v01"
UPLOAD_OBJECT = f"p586_root_v01/bundles/{BUNDLE_NAME}.zip"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "P586-human-readable-bundle/0.1"
MANIFEST: list[dict] = []


def fetch(url: str, destination: pathlib.Path, source: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = SESSION.get(url, timeout=180)
            response.raise_for_status()
            destination.write_bytes(response.content)
            MANIFEST.append(
                {
                    "path": str(destination.relative_to(BUNDLE)),
                    "bytes": len(response.content),
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                    "source": source or url,
                }
            )
            print("FETCH", destination.relative_to(BUNDLE), len(response.content), flush=True)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(min(20, 1.8**attempt))
    raise RuntimeError(f"Fetch failed: {url}: {last_error}")


def storage(bucket: str, name: str, destination: str) -> None:
    url = f"{SUPABASE_PUBLIC}/{bucket}/{quote(name, safe='/')}"
    fetch(url, BUNDLE / destination, f"supabase:{bucket}/{name}")


def write_text(relative_path: str, text: str, source: str = "generated") -> None:
    path = BUNDLE / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    MANIFEST.append(
        {
            "path": str(path.relative_to(BUNDLE)),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source": source,
        }
    )


def write_json(relative_path: str, value, source: str = "generated") -> None:
    write_text(relative_path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", source)


def prepare_directories() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for directory in [
        "00_READ_ME",
        "01_FINAL_RESULTS",
        "02_PROTOCOL_AND_CODE",
        "03_PROVENANCE/preliminary",
        "04_FINAL_CORPUS/crops",
        "04_FINAL_CORPUS/context",
        "05_VISUAL_SUMMARIES/all_pages",
        "05_VISUAL_SUMMARIES/retained_roots",
        "05_VISUAL_SUMMARIES/early_pair_sheets",
        "06_SUPERSEDED_DO_NOT_USE",
    ]:
        (BUNDLE / directory).mkdir(parents=True, exist_ok=True)


def download_reports() -> None:
    final_objects = [
        ("bridge", "p586_root_v01/full_run/checkpoint.json", "03_PROVENANCE/original_frozen_checkpoint_through_canvas79.json"),
        ("bridge", "p586_root_v01/resume/late_qa.json", "03_PROVENANCE/terminal_nine_recovery_QA.json"),
        ("bridge", "p586_root_v01/final/frozen_retained_paths.json", "01_FINAL_RESULTS/frozen_retained_paths.json"),
        ("bridge", "p586_root_v01/final/corrected_comparison.json", "01_FINAL_RESULTS/corrected_comparison.json"),
        ("bridge", "p586_root_v01/final/corrected_blind_key.json", "01_FINAL_RESULTS/corrected_blind_key.json"),
        ("bridge", "p586_root_v01/final/blind_visual_adjudication_qwen.json", "01_FINAL_RESULTS/blind_visual_adjudication_qwen.json"),
        ("manuscripts", "p586_root_v01/final/corrected_blind_triptychs.jpg", "05_VISUAL_SUMMARIES/corrected_blind_triptychs.jpg"),
    ]
    for item in final_objects:
        storage(*item)

    preliminary = [
        ("pilots/hier_pilot_result.json", "hierarchical_pilot_result.json"),
        ("screening/contact_sheet_index.json", "contact_sheet_index.json"),
        ("screening/localisation_checkpoint.json", "localisation_checkpoint.json"),
        ("dinov3_scan_77e97abe-931d-41b5-939a-7f3f8b68ca97/protocol.json", "dinov3_scan_protocol.json"),
        ("dinov3_scan_77e97abe-931d-41b5-939a-7f3f8b68ca97/proposals.json", "dinov3_scan_proposals.json"),
        ("dinov3_scan_77e97abe-931d-41b5-939a-7f3f8b68ca97/run_report.json", "dinov3_scan_run_report.json"),
        ("qwen_locator_14b2a740-8392-49bf-887d-9b452c4cce64/protocol.json", "qwen_locator_protocol.json"),
        ("qwen_locator_14b2a740-8392-49bf-887d-9b452c4cce64/accepted_manifest.json", "qwen_locator_accepted_manifest.json"),
        ("qwen_locator_14b2a740-8392-49bf-887d-9b452c4cce64/locator_results.json", "qwen_locator_results.json"),
        ("qwen_locator_14b2a740-8392-49bf-887d-9b452c4cce64/run_report.json", "qwen_locator_run_report.json"),
        ("comparison_de16c95c-f0c6-41c2-aa33-6b15b71039ad/protocol.json", "early_comparison_protocol.json"),
        ("comparison_de16c95c-f0c6-41c2-aa33-6b15b71039ad/comparison_report.json", "early_comparison_report.json"),
        ("comparison_de16c95c-f0c6-41c2-aa33-6b15b71039ad/nearest_pairs.csv", "early_nearest_pairs.csv"),
        ("comparison_de16c95c-f0c6-41c2-aa33-6b15b71039ad/nearest_pairs.json", "early_nearest_pairs.json"),
    ]
    for source, destination in preliminary:
        storage("bridge", f"p586_root_v01/{source}", f"03_PROVENANCE/preliminary/{destination}")

    superseded = [
        ("bridge", "p586_root_v01/full_run/checkpoint_recovered.json", "checkpoint_recovered_181_proposals.json"),
        ("bridge", "p586_root_v01/full_run/checkpoint_repaired.json", "checkpoint_repaired_181_proposals.json"),
        ("bridge", "p586_root_v01/full_run/comparison_result_recovered.json", "comparison_result_139_roots.json"),
        ("bridge", "p586_root_v01/full_run/root_proposals_recovered.csv", "root_proposals_181_rows.csv"),
        ("bridge", "p586_root_v01/full_run/audit/blind_key_recovered.json", "blind_key_recovered.json"),
        ("manuscripts", "p586_root_v01/full_run/audit/blind_triptychs_recovered.jpg", "blind_triptychs_recovered.jpg"),
    ]
    for bucket, source, destination in superseded:
        storage(bucket, source, f"06_SUPERSEDED_DO_NOT_USE/{destination}")


def download_visual_summaries() -> None:
    for index, (start, end) in enumerate([(0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 138)]):
        name = f"p586_root_v01/contact_sheets/all_pages_{index:02d}_{start:03d}-{end:03d}.jpg"
        storage("manuscripts", name, f"05_VISUAL_SUMMARIES/all_pages/{pathlib.Path(name).name}")
    for index, (start, end) in enumerate([(0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 136)]):
        name = f"p586_root_v01/full_run/contact_sheets/retained_{index:02d}_{start:03d}-{end:03d}.jpg"
        storage("manuscripts", name, f"05_VISUAL_SUMMARIES/retained_roots/{pathlib.Path(name).name}")
    for start in list(range(0, 132, 6)) + [132]:
        end = 135 if start == 132 else start + 5
        name = f"p586_root_v01/comparison_de16c95c-f0c6-41c2-aa33-6b15b71039ad/pair_sheets/pairs_{start:03d}-{end:03d}.jpg"
        storage("manuscripts", name, f"05_VISUAL_SUMMARIES/early_pair_sheets/{pathlib.Path(name).name}")


def build_corrected_ledger() -> list[dict]:
    checkpoint = json.loads((BUNDLE / "03_PROVENANCE/original_frozen_checkpoint_through_canvas79.json").read_text())
    late_qa = json.loads((BUNDLE / "03_PROVENANCE/terminal_nine_recovery_QA.json").read_text())
    proposals = list(checkpoint["proposals"])
    late_by_stem = {entry["stem"]: entry for entry in late_qa}
    statuses = {
        "c080_p00_r00": "accept",
        "c080_p02_r01": "partial",
        "c081_p00_r00": "partial",
        "c081_p01_r01": "accept",
        "c081_p02_r02": "partial",
        "c081_p03_r03": "reject",
        "c082_p01_r00": "accept",
        "c084_p00_r00": "accept",
        "c084_p01_r01": "reject",
    }
    for stem, status in statuses.items():
        match = re.fullmatch(r"c(\d+)_p(\d+)_r(\d+)", stem)
        canvas, plant, root = map(int, match.groups())
        qa = late_by_stem.get(stem, {})
        proposals.append(
            {
                "canvas_index": canvas,
                "plant_index": plant,
                "root_index": root,
                "qa_status": status,
                "qa_confidence": qa.get("confidence"),
                "root_fraction": qa.get("root_fraction"),
                "qa_reason": qa.get("reason") or "terminal recovery",
                "crop_path": f"p586_root_v01/full_run/crops/{stem}.png",
                "context_crop_path": f"p586_root_v01/full_run/context/{stem}_plant.jpg",
                "metadata": {"terminal_recovery": True},
            }
        )
    proposals.sort(key=lambda row: (int(row["canvas_index"]), int(row.get("plant_index", 0)), int(row.get("root_index", 0))))
    counts = {status: sum(row.get("qa_status") == status for row in proposals) for status in ["accept", "partial", "reject"]}
    assert len(proposals) == 178
    assert counts == {"accept": 30, "partial": 107, "reject": 41}
    write_json("01_FINAL_RESULTS/corrected_root_proposals_178.json", proposals, "frozen checkpoint plus terminal nine")
    fields = [
        "canvas_index", "plant_index", "root_index", "qa_status", "qa_confidence", "root_fraction",
        "detector_score", "visibility", "crop_path", "context_crop_path", "image_sha256", "qa_reason",
        "bbox", "context_bbox",
    ]
    csv_path = BUNDLE / "01_FINAL_RESULTS/corrected_root_proposals_178.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for proposal in proposals:
            row = {field: proposal.get(field) for field in fields}
            for field in ["bbox", "context_bbox"]:
                if isinstance(row.get(field), (dict, list)):
                    row[field] = json.dumps(row[field], sort_keys=True)
            writer.writerow(row)
    data = csv_path.read_bytes()
    MANIFEST.append({"path": str(csv_path.relative_to(BUNDLE)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "source": "frozen checkpoint plus terminal nine"})
    retained = [row for row in proposals if row.get("qa_status") in {"accept", "partial"}]
    assert len(retained) == 137
    write_json("01_FINAL_RESULTS/corrected_retained_137.json", retained, "derived from corrected ledger")
    return proposals


def download_final_corpus(proposals: list[dict]) -> None:
    for index, proposal in enumerate(proposals, start=1):
        crop_path = proposal["crop_path"]
        context_path = proposal["context_crop_path"]
        storage("manuscripts", crop_path, f"04_FINAL_CORPUS/crops/{pathlib.Path(crop_path).name}")
        storage("manuscripts", context_path, f"04_FINAL_CORPUS/context/{pathlib.Path(context_path).name}")
        if index % 25 == 0:
            print("CORPUS", index, "/", len(proposals), flush=True)


def copy_protocol_and_code() -> None:
    repository = ROOT / "repository"
    subprocess.run(
        [
            "git", "clone", "--quiet", "--depth", "1", "--branch", "gpt56/p586-root-v01-20260803",
            "https://github.com/digitalgoldfisj79/voynich-morphology-compression.git", str(repository),
        ],
        check=True,
    )
    source = repository / "experiments/palatino586_root_v01"
    destination = BUNDLE / "02_PROTOCOL_AND_CODE/github_experiment_directory"
    shutil.copytree(source, destination)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            MANIFEST.append({"path": str(path.relative_to(BUNDLE)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "source": "GitHub branch gpt56/p586-root-v01-20260803"})


def write_guidance() -> None:
    write_text(
        "00_READ_ME/README_FIRST.md",
        """# Palatino 586 ↔ Voynich root morphology — human-readable bundle v0.1

Protocol: `P586-VMS-ROOT-0.1-20260803`  
Run ID: `ef6a0302-8269-4166-8c1c-63b14abb9c47`  
Final result SHA-256: `588e0aa0a759f84fd8468c043f3fb3eb17b173f0c2fc4af3d186a351148eb206`

The final frozen comparison is a strong negative for Palatino 586 as a close Voynich root-morphology comparator. Final counts: 66 pages; 178 proposals; 30 accepts; 107 partials; 41 rejects; 137 retained roots; 128 Voynich roots; 21 BSB control roots.

`01_FINAL_RESULTS` contains corrected statistics and ledgers. `02_PROTOCOL_AND_CODE` contains the immutable experiment directory. `03_PROVENANCE` contains the frozen checkpoint and preliminary reports. `04_FINAL_CORPUS` contains all 178 crops and all 178 context images. `05_VISUAL_SUMMARIES` contains contact and pair sheets. `06_SUPERSEDED_DO_NOT_USE` documents the invalid 139-root replay.

Binary embeddings and model weights are excluded because this is the human-readable bundle. Their identifiers remain in the reports and code.
""",
    )
    write_text(
        "06_SUPERSEDED_DO_NOT_USE/README_DO_NOT_USE.md",
        "# Superseded outputs\n\nThese files contain the nondeterministic canvas-67 replay and report 181 proposals / 139 retained roots. They are included only as an audit trail and must not be used for the final result.\n",
    )


def make_zip() -> None:
    MANIFEST.sort(key=lambda row: row["path"])
    with (BUNDLE / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256", "source"])
        writer.writeheader()
        writer.writerows(MANIFEST)
    (BUNDLE / "SHA256SUMS.txt").write_text(
        "\n".join(f"{row['sha256']}  {row['path']}" for row in MANIFEST) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
    print("ZIP_SIZE", ZIP_PATH.stat().st_size, flush=True)
    print("ZIP_SHA256", hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest(), flush=True)


def upload_zip() -> None:
    token = os.environ["BUNDLE_UPLOAD_TOKEN"]
    response = SESSION.post(
        UPLOAD_EDGE,
        headers={"x-bundle-token": token},
        json={"path": UPLOAD_OBJECT},
        timeout=60,
    )
    response.raise_for_status()
    signed = response.json()["signedUrl"]
    with ZIP_PATH.open("rb") as handle:
        upload = SESSION.put(signed, data=handle, headers={"content-type": "application/zip"}, timeout=1800)
    upload.raise_for_status()
    public_url = f"{SUPABASE_PUBLIC}/bridge/{quote(UPLOAD_OBJECT, safe='/')}"
    print("RESULT_URL=" + public_url, flush=True)


def main() -> None:
    prepare_directories()
    download_reports()
    download_visual_summaries()
    proposals = build_corrected_ledger()
    download_final_corpus(proposals)
    copy_protocol_and_code()
    write_guidance()
    make_zip()
    upload_zip()


if __name__ == "__main__":
    main()
