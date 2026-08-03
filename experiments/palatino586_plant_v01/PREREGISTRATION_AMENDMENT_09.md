# Preregistration Amendment 09 — deterministic target repair sharding

Protocol: `P586-VMS-PLANT-0.1-20260803`

This amendment was made before DINOv3 embedding or target similarity analysis.

The Palatino reproductive repair is partitioned by the already frozen whole-plant manifest order:

- shard A: indices 0–99;
- shard B: indices 100–193.

Each shard loads the same pre-repair channel manifest, frozen detector raw responses, parser revision, Qwen model/revision, batch-QA prompt, crop padding, class vocabulary and strict/broad rules. Shards write separate checkpoints and cannot overwrite the public channel manifest.

The merge accepts exactly one completed record for every frozen plant ID, preserving original manifest order. It refuses missing IDs, duplicates, protocol/hash mismatch or overlap outside the fixed partition. The resulting counts and channel SHA are computed exactly as in the unsharded repair.

This is an execution-only change. It does not alter any proposal, box, crop, threshold, label, or comparison set.
