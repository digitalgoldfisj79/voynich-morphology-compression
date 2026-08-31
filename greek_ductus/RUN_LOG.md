# Greek-ductus experiment run log

Protocol: `2026-08-31.greek-ductus.v1`

This file records failed as well as successful runs. Environment/setup failures are retained explicitly and are not treated as scientific results.

## 2026-08-31

### HF job `6a9600f921c5aa7c8364a480`
- Purpose: CPU-only one-page-per-manuscript crop-extraction QC; no Voynich and no DINO.
- Flavor: `cpu-basic`.
- Result: **environment/setup failure before image analysis**.
- Error: `ModuleNotFoundError: No module named 'cv2'`.
- Scientific status: **no data inspected; no hypothesis result; no protocol change**.
- Action: rerun unchanged algorithm with dependencies installed explicitly.

### HF job `6a96013c0718b0f6d890a706`
- Purpose: exact rerun of CPU-only one-page-per-manuscript crop-extraction QC on the frozen 18-manuscript ordinary-script panel.
- Flavor: `cpu-basic`.
- Dependencies installed explicitly with `uv pip install --system`.
- Status at log creation: running.
- GPU used: **no**.

## Frozen artifacts before scientific results
- `PREREGISTRATION.md` — hypothesis, representations, nulls, decision rule, stop rules.
- `primary_manifest.json` — six Bodleian manuscripts each for Greek, Italian Latin, German controls.
- `vms_manifest.json` — fixed Yale/Voynich page-sampling rule, frozen before target analysis.
- `extract_cpu.py` — identical image-derived crop detector for all families; no transcription/EVA.
- `stroke_features.py` — fixed skeleton/shape descriptor; explicitly does not treat inferred pen order as observed.
