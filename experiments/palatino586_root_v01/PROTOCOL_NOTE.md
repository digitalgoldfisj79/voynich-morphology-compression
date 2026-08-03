# Protocol correction note

The only post-interruption correction is membership restoration, not threshold tuning.

The interrupted job's final log fixed the totals at 178 proposals: 30 accept, 107 partial, 41 reject. A provisional recovery replayed canvas 67 and produced 181 proposals: 30 accept, 109 partial, 42 reject. Since Qwen localisation is nondeterministic, the replayed membership was invalid for a frozen sensitivity set.

The final comparison therefore takes the original persisted checkpoint and appends only the nine terminal decisions recoverable from the interrupted job's per-canvas logs. The common embedding bundle is then filtered to those exact crop paths. No similarity threshold, QA threshold, model, seed, control manuscript or inferential test was changed.
