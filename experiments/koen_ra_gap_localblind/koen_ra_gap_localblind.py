#!/usr/bin/env python3
"""V04 local boundary-blind r->a runner.

Derives the executable from frozen V02 and changes only the physical measurement
geometry. Each target is measured in a small locally translated context positioned
from the complete line envelope and boundary-stripped character ordinal. No VT word
boundary x-coordinate or topology is exposed to the localiser.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

BASE = Path("experiments/koen_ra_gap_blind/koen_ra_gap_blind.py")
OUT = Path("results/koen_ra_gap_localblind")
EFFECTIVE = OUT / "effective_v04_source.py"

src = BASE.read_text()
base_sha = hashlib.sha256(src.encode()).hexdigest()

src = src.replace("KOEN_RA_GAP_V02_BOUNDARYBLIND_20260822", "KOEN_RA_GAP_V04_LOCALBLIND_20260822")
src = src.replace('voynich-koen-ra-gap-boundaryblind/0.2', 'voynich-koen-ra-gap-localblind/0.4')
src = src.replace('OUT = Path("results/koen_ra_gap_blind")', 'OUT = Path("results/koen_ra_gap_localblind")')
src = src.replace('PROTOCOL = Path("experiments/koen_ra_gap_blind/PROTOCOL.md")', 'PROTOCOL = Path("experiments/koen_ra_gap_localblind/PROTOCOL.md")')

needle = '''def ink_mask_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


'''
insert = needle + '''def dino_component_clean_mask(mask: np.ndarray, minimum_area: int = 6) -> np.ndarray:
    """Pre-existing DINO connected-component admission rule."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for idx in range(1, count):
        x, y, width, height, area = stats[idx]
        if int(area) >= minimum_area and int(width) > 0 and int(height) > 0:
            clean[labels == idx] = 255
    return clean


'''
if src.count(needle) != 1:
    raise SystemExit("V04 patch gate failed: ink-mask source mismatch")
src = src.replace(needle, insert)

new_measure = r'''def measure_line(canonical, vline: VLine, zletters: str, targets: list[dict]):
    """Measure targets independently in small boundary-blind local contexts."""
    env = line_envelope(vline, canonical.shape)
    if env is None:
        return [], "no_line_envelope"
    x0, y0, x1, y1, _ = env
    if x1 - x0 < 20 or y1 - y0 < 8:
        return [], "tiny_line_envelope"
    n = len(zletters)
    if n < 2:
        return [], "short_line"
    line_w = float(x1 - x0)
    pitch = line_w / float(n)
    if pitch < 2.0:
        return [], "tiny_pitch"

    context_radius = 3
    shift_grid = (-0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75)
    out = []

    for t in targets:
        ri, ai = int(t["r_index"]), int(t["a_index"])
        if ai != ri + 1 or ri < 0 or ai >= n:
            continue
        lo = max(0, ri - context_radius)
        hi = min(n - 1, ai + context_radius)
        m = hi - lo + 1
        rloc, aloc = ri - lo, ai - lo
        nominal_left = x0 + lo * pitch
        nominal_right = x0 + (hi + 1) * pitch
        nominal_width = nominal_right - nominal_left
        if nominal_width < max(12.0, 2.0 * m):
            continue

        candidates = []
        for shift_u in shift_grid:
            shift_px = shift_u * pitch
            lx0 = int(round(nominal_left + shift_px))
            lx1 = int(round(nominal_right + shift_px))
            if lx0 < 0 or lx1 > canonical.shape[1] or lx1 - lx0 < max(12, 2 * m):
                continue
            crop = canonical[y0:y1, lx0:lx1]
            raw_mask = ink_mask_bgr(crop)
            mask = dino_component_clean_mask(raw_mask, minimum_area=6)
            ys_all, xs_all = np.where(mask > 0)
            if len(xs_all) < 10:
                continue
            cuts = boundary_blind_cuts(mask, m)
            if len(cuts) != m - 1:
                continue
            proj = mask.astype(np.float32).sum(axis=0) / 255.0
            if len(proj) >= 5:
                smoothed = np.convolve(proj, np.array([1, 2, 3, 2, 1], dtype=float) / 9.0, mode="same")
            else:
                smoothed = proj
            positive = smoothed[smoothed > 0]
            denom = float(np.median(positive)) if len(positive) else 1.0
            cut_score = float(np.mean([smoothed[c] for c in cuts]) / max(denom, 1e-9))
            candidates.append((cut_score, abs(float(shift_u)), float(shift_u), lx0, lx1, mask, cuts, proj))

        if not candidates:
            continue
        candidates.sort(key=lambda z: (z[0], z[1], z[2]))
        cut_score, _, shift_u, lx0, lx1, mask, cuts, proj = candidates[0]
        bounds = [0] + cuts + [mask.shape[1]]
        if aloc >= len(bounds) - 1 or rloc < 0:
            continue
        la, lb = bounds[rloc], bounds[rloc + 1]
        aa, ab = bounds[aloc], bounds[aloc + 1]
        if lb <= la or ab <= aa:
            continue
        left = mask[:, la:lb]
        right = mask[:, aa:ab]
        yl, xl = np.where(left > 0)
        yr, xr = np.where(right > 0)
        if len(xl) == 0 or len(xr) == 0:
            continue
        r_right = la + int(xl.max())
        a_left = aa + int(xr.min())
        gap = max(0, int(a_left - r_right - 1))
        ys_all, xs_all = np.where(mask > 0)
        if len(ys_all) < 10:
            continue
        q05, q95 = np.quantile(ys_all, [.05, .95])
        local_ink_height = max(1.0, float(q95 - q05 + 1.0))
        foreground_fraction = float(np.mean(mask > 0))
        positive_proj = proj[proj > 0]
        med_col_ink = float(np.median(positive_proj)) if len(positive_proj) else 1.0
        boundary_x = bounds[aloc]
        boundary_col = float(proj[boundary_x]) if 0 <= boundary_x < len(proj) else float("nan")
        boundary_contrast = float(boundary_col / max(med_col_ink, 1e-9)) if math.isfinite(boundary_col) else float("nan")

        out.append({
            "event_id": t["event_id"], "folio": t["folio"], "locus": t["locus"],
            "hand": t["hand"], "quire": t["quire"], "r_index": ri, "a_index": ai,
            "gap_px_registered_yale": int(gap),
            "gap_px_legacy_equiv": float(gap / CANONICAL_SCALE),
            "gap_norm": float(gap / local_ink_height),
            "line_ink_height": local_ink_height,
            "line_crop_width": int(mask.shape[1]), "line_crop_height": int(mask.shape[0]),
            "local_foreground_fraction": foreground_fraction,
            "translation_pitch_units": float(shift_u),
            "local_cut_score": float(cut_score),
            "boundary_x_in_crop": int(boundary_x),
            "boundary_column_ink": boundary_col,
            "boundary_contrast": boundary_contrast,
            "nominal_pitch_registered_yale": float(pitch),
            "context_lo_index": int(lo), "context_hi_index": int(hi),
            "context_glyph_count": int(m),
        })
    return out, None


'''
pattern = re.compile(r'def measure_line\(canonical, vline: VLine, zletters: str, targets: list\[dict\]\):.*?\n\ndef group_summary\(vals\):', re.S)
if len(pattern.findall(src)) != 1:
    raise SystemExit("V04 patch gate failed: measure_line block mismatch")
src = pattern.sub(new_measure + 'def group_summary(vals):', src)

old_instr = '''            "dino_embedding_inference": False, "gpu": False,
'''
new_instr = '''            "dino_embedding_inference": False, "gpu": False,
            "measurement_geometry": "local boundary-blind ordinal context",
            "context_radius_glyphs": 3,
            "translation_grid_pitch_units": [-0.75,-0.50,-0.25,0.0,0.25,0.50,0.75],
            "dino_connected_component_minimum_area": 6,
            "v02_base_source_sha256": "''' + base_sha + '''",
'''
if src.count(old_instr) != 1:
    raise SystemExit("V04 patch gate failed: instrument metadata mismatch")
src = src.replace(old_instr, new_instr)

old_result = '''    result = {
        "protocol_id": "KOEN_RA_GAP_V04_LOCALBLIND_20260822",'''
new_result = '''    zero_rates = {c: float(np.mean([float(r["gap_px_registered_yale"]) == 0.0 for r in events if r["class"] == c])) for c in ("joined", "uncertain", "certain")}
    pooled_zero_rate = float(np.mean([float(r["gap_px_registered_yale"]) == 0.0 for r in events])) if events else float("nan")
    fg = np.asarray([float(r["local_foreground_fraction"]) for r in events], dtype=float)
    median_foreground = float(np.median(fg)) if len(fg) else float("nan")
    retention_fracs = [float(retention[c]["fraction"]) for c in ("joined", "uncertain", "certain")]
    min_retention = min(retention_fracs) if retention_fracs else float("nan")
    retention_spread = max(retention_fracs) - min(retention_fracs) if retention_fracs else float("nan")
    shifts = np.asarray([abs(float(r["translation_pitch_units"])) for r in events], dtype=float)
    median_abs_shift = float(np.median(shifts)) if len(shifts) else float("nan")
    contrasts = np.asarray([float(r["boundary_contrast"]) for r in events if math.isfinite(float(r["boundary_contrast"]))], dtype=float)
    median_boundary_contrast = float(np.median(contrasts)) if len(contrasts) else float("nan")
    instrument_valid = bool(pooled_zero_rate < .75 and median_foreground < .30 and min_retention >= .40 and retention_spread <= .15)
    if not instrument_valid:
        primary_resolved = False
        continuum_resolved = False

    result = {
        "protocol_id": "KOEN_RA_GAP_V04_LOCALBLIND_20260822",
        "measurement_validity": {
            "zero_gap_rate_by_class": zero_rates,
            "pooled_zero_gap_rate": pooled_zero_rate,
            "median_local_foreground_fraction": median_foreground,
            "minimum_class_retention": min_retention,
            "class_retention_spread": retention_spread,
            "median_abs_translation_pitch_units": median_abs_shift,
            "median_boundary_contrast": median_boundary_contrast,
            "gate_zero_gap_lt_075": bool(pooled_zero_rate < .75),
            "gate_foreground_lt_030": bool(median_foreground < .30),
            "gate_each_class_retention_ge_040": bool(min_retention >= .40),
            "gate_retention_spread_le_015": bool(retention_spread <= .15),
            "instrument": "VALID" if instrument_valid else "INSTRUMENT_FAILED"
        },'''
if src.count(old_result) != 1:
    raise SystemExit("V04 patch gate failed: result block mismatch")
src = src.replace(old_result, new_result)

old_report = '''        "The image stage saw boundary-stripped glyph indices only. It measured one continuous registered-Yale line raster; internal VT/Takahashi word rectangles were collapsed before character cutting. No DINO embedding inference, Hugging Face Job or GPU was used.",'''
new_report = '''        "The image stage saw boundary-stripped glyph indices only. For each target it measured a small registered-Yale context positioned from the complete line envelope and glyph ordinal, searched over the frozen ±0.75-pitch translation grid, and selected the lowest-ink local alignment. No VT/Takahashi internal word edge or topology was available to the localiser. No DINO embedding inference, Hugging Face Job or GPU was used.",'''
if src.count(old_report) != 1:
    raise SystemExit("V04 patch gate failed: report instrument text mismatch")
src = src.replace(old_report, new_report)

old_decision_report = '''        "",
        "## Decision",
        f"Primary physical distinction: **{result['decision']['primary_physical_distinction']}**.",'''
new_decision_report = '''        "",
        "## Measurement-validity gate",
        f"Pooled zero-gap rate: **{fmt(result['measurement_validity']['pooled_zero_gap_rate'],4)}** (must be <0.75). Median local foreground fraction: **{fmt(result['measurement_validity']['median_local_foreground_fraction'],4)}** (must be <0.30).",
        f"Minimum class retention: **{fmt(result['measurement_validity']['minimum_class_retention'],4)}** (must be >=0.40). Retention spread: **{fmt(result['measurement_validity']['class_retention_spread'],4)}** (must be <=0.15).",
        f"Median absolute local translation: {fmt(result['measurement_validity']['median_abs_translation_pitch_units'],3)} glyph pitches. Median boundary-column/positive-column ink ratio: {fmt(result['measurement_validity']['median_boundary_contrast'],3)}.",
        f"Instrument: **{result['measurement_validity']['instrument']}**.",
        "",
        "## Decision",
        f"Primary physical distinction: **{result['decision']['primary_physical_distinction']}**.",'''
if src.count(old_decision_report) != 1:
    raise SystemExit("V04 patch gate failed: report decision block mismatch")
src = src.replace(old_decision_report, new_decision_report)

OUT.mkdir(parents=True, exist_ok=True)
EFFECTIVE.write_text(src)
print(f"V04 base source SHA256={base_sha}", flush=True)
print(f"V04 effective source SHA256={hashlib.sha256(src.encode()).hexdigest()}", flush=True)

code = compile(src, str(EFFECTIVE), "exec")
ns = {"__name__": "__main__", "__file__": str(EFFECTIVE)}
exec(code, ns, ns)
