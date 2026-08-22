#!/usr/bin/env python3
"""V03 boundary-blind r->a runner.

This deliberately derives the executable from the frozen V02 source and applies one
mechanical patch only: the pre-existing DINO connected-component admission threshold
(minimum_area=6) is inserted between adaptive thresholding and line splitting.
The generated effective source is persisted for audit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE = Path("experiments/koen_ra_gap_blind/koen_ra_gap_blind.py")
OUT = Path("results/koen_ra_gap_blind_cc")
EFFECTIVE = OUT / "effective_v03_source.py"

src = BASE.read_text()
base_sha = hashlib.sha256(src.encode()).hexdigest()

src = src.replace(
    "KOEN_RA_GAP_V02_BOUNDARYBLIND_20260822",
    "KOEN_RA_GAP_V03_BOUNDARYBLIND_CC_20260822",
)
src = src.replace(
    'voynich-koen-ra-gap-boundaryblind/0.2',
    'voynich-koen-ra-gap-boundaryblind-cc/0.3',
)
src = src.replace(
    'OUT = Path("results/koen_ra_gap_blind")',
    'OUT = Path("results/koen_ra_gap_blind_cc")',
)
src = src.replace(
    'PROTOCOL = Path("experiments/koen_ra_gap_blind/PROTOCOL.md")',
    'PROTOCOL = Path("experiments/koen_ra_gap_blind_cc/PROTOCOL.md")',
)

needle = '''def ink_mask_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


'''
insert = needle + '''def dino_component_clean_mask(mask: np.ndarray, minimum_area: int = 6) -> np.ndarray:
    """Apply the pre-existing DINO connected_component_proposals admission rule."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for idx in range(1, count):
        x, y, width, height, area = stats[idx]
        if int(area) >= minimum_area and int(width) > 0 and int(height) > 0:
            clean[labels == idx] = 255
    return clean


'''
if src.count(needle) != 1:
    raise SystemExit("V03 patch gate failed: ink-mask source mismatch")
src = src.replace(needle, insert)

old_measure = '''    mask = ink_mask_bgr(crop)
    n = len(zletters)'''
new_measure = '''    raw_mask = ink_mask_bgr(crop)
    mask = dino_component_clean_mask(raw_mask, minimum_area=6)
    n = len(zletters)'''
if src.count(old_measure) != 1:
    raise SystemExit("V03 patch gate failed: measurement source mismatch")
src = src.replace(old_measure, new_measure)

old_instr = '''            "dino_embedding_inference": False, "gpu": False,
'''
new_instr = '''            "dino_embedding_inference": False, "gpu": False,
            "dino_connected_component_minimum_area": 6,
            "component_filter_provenance": "pre-existing DINO connected_component_proposals default; frozen before V03 outcomes",
            "v02_base_source_sha256": "''' + base_sha + '''",
'''
if src.count(old_instr) != 1:
    raise SystemExit("V03 patch gate failed: instrument metadata source mismatch")
src = src.replace(old_instr, new_instr)

old_report = '''        "The image stage saw boundary-stripped glyph indices only. It measured one continuous registered-Yale line raster; internal VT/Takahashi word rectangles were collapsed before character cutting. No DINO embedding inference, Hugging Face Job or GPU was used.",'''
new_report = '''        "The image stage saw boundary-stripped glyph indices only. It measured one continuous registered-Yale line raster; internal VT/Takahashi word rectangles were collapsed before character cutting. The adaptive mask was cleaned only by the pre-existing DINO connected-component admission threshold (area >= 6 px). No DINO embedding inference, Hugging Face Job or GPU was used.",'''
if src.count(old_report) != 1:
    raise SystemExit("V03 patch gate failed: report source mismatch")
src = src.replace(old_report, new_report)

# Add the pre-registered measurement-validity diagnostics and decision gate without
# altering the already-frozen statistical comparisons.
old_decision = '''    result = {
        "protocol_id": "KOEN_RA_GAP_V03_BOUNDARYBLIND_CC_20260822",'''
new_decision = '''    zero_rates = {c: float(np.mean([float(r["gap_px_registered_yale"]) == 0.0 for r in events if r["class"] == c])) for c in ("joined", "uncertain", "certain")}
    pooled_zero_rate = float(np.mean([float(r["gap_px_registered_yale"]) == 0.0 for r in events])) if events else float("nan")
    height_ratios = np.asarray([float(r["line_ink_height"]) / max(1.0, float(r["line_crop_height"])) for r in events], dtype=float)
    median_height_ratio = float(np.median(height_ratios)) if len(height_ratios) else float("nan")
    instrument_valid = bool(pooled_zero_rate < .75 and median_height_ratio < .75)
    if not instrument_valid:
        primary_resolved = False
        continuum_resolved = False

    result = {
        "protocol_id": "KOEN_RA_GAP_V03_BOUNDARYBLIND_CC_20260822",
        "measurement_validity": {
            "zero_gap_rate_by_class": zero_rates,
            "pooled_zero_gap_rate": pooled_zero_rate,
            "median_line_ink_height_over_crop_height": median_height_ratio,
            "gate_zero_gap_lt_075": bool(pooled_zero_rate < .75),
            "gate_height_ratio_lt_075": bool(median_height_ratio < .75),
            "instrument": "VALID" if instrument_valid else "INSTRUMENT_FAILED"
        },'''
if src.count(old_decision) != 1:
    raise SystemExit("V03 patch gate failed: result source mismatch")
src = src.replace(old_decision, new_decision)

old_interp = '''        "",
        "## Decision",
        f"Primary physical distinction: **{result['decision']['primary_physical_distinction']}**.",'''
new_interp = '''        "",
        "## Measurement-validity gate",
        f"Pooled zero-gap rate: **{fmt(result['measurement_validity']['pooled_zero_gap_rate'],4)}** (must be <0.75). Median retained line-ink height / crop height: **{fmt(result['measurement_validity']['median_line_ink_height_over_crop_height'],4)}** (must be <0.75). Instrument: **{result['measurement_validity']['instrument']}**.",
        f"Class zero-gap rates: joined {fmt(result['measurement_validity']['zero_gap_rate_by_class']['joined'],4)}, uncertain {fmt(result['measurement_validity']['zero_gap_rate_by_class']['uncertain'],4)}, certain {fmt(result['measurement_validity']['zero_gap_rate_by_class']['certain'],4)}.",
        "",
        "## Decision",
        f"Primary physical distinction: **{result['decision']['primary_physical_distinction']}**.",'''
if src.count(old_interp) != 1:
    raise SystemExit("V03 patch gate failed: report decision source mismatch")
src = src.replace(old_interp, new_interp)

OUT.mkdir(parents=True, exist_ok=True)
EFFECTIVE.write_text(src)
print(f"V03 base source SHA256={base_sha}", flush=True)
print(f"V03 effective source SHA256={hashlib.sha256(src.encode()).hexdigest()}", flush=True)

code = compile(src, str(EFFECTIVE), "exec")
ns = {"__name__": "__main__", "__file__": str(EFFECTIVE)}
exec(code, ns, ns)
