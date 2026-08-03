# Preregistration Amendment 06 — batched reproductive visual QA

Protocol: `P586-VMS-PLANT-0.1-20260803`

This amendment was made before DINOv3 embedding or any target similarity was computed.

After Amendment 05 recovered the detector's frozen proposals, visual QA is executed once per plant rather than once per proposal. Each call receives:

1. the complete masked plant;
2. every recovered proposal crop, in original response order, up to the preregistered maximum of five.

The model remains `Qwen/Qwen2.5-VL-7B-Instruct`. The decision vocabulary and criteria remain those of the original per-proposal QA: `accept|partial|reject|uncertain`, reproductive class, confidence, attachment to the plant, and a brief reason. A proposal is forced to `reject` when the model states it is not attached to the plant. Missing or malformed reviews are `uncertain`, never accepted by default.

Batching changes only execution efficiency and cross-proposal context. It does not change detector proposals, boxes, padding, class eligibility, strict/broad set rules, or any similarity threshold.
