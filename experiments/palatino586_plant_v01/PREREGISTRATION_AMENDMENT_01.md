# Preregistration Amendment 01 — Control-page screening

Date: 2026-08-03

Protocol: `P586-VMS-PLANT-0.1-20260803`

Status at amendment: Palatino target extraction was in progress. No Palatino–Voynich or control–Voynich DINOv3 embeddings, similarities, statistical tests or blinded comparison results had been computed. No control extraction had begun.

## Reason

An infrastructure audit established that `cat_herbal_folios` is a manuscript-canvas registry, not a registry of botanical-image-positive pages. Therefore selecting only ten fixed quantile pages from an entire manuscript could create structurally underpowered controls by sampling text-only or binding pages. This would violate the preregistered requirement that manuscript controls have sufficient valid plant counts.

## Replacement control-page rule

For each fixed manuscript control sourced from `cat_herbal_folios`:

1. inspect every registered canvas at a fixed 768-pixel maximum side using `Qwen/Qwen2.5-VL-7B-Instruct`, deterministic decoding and a binary page-screen prompt;
2. record `contains_botanical_plant`, candidate count, confidence and reason for every page;
3. admit a page to detailed whole-plant detection only when the screen returns `contains_botanical_plant=true` with confidence at least 0.70;
4. process admitted pages in increasing registered sequence order;
5. stop detailed detection for a manuscript after 20 broad (`accept + partial`) whole plants have been frozen, or after every registered page has been screened;
6. retain the original ten quantile pages as a predeclared sampling audit, not as the main control corpus.

No screen threshold or manuscript cap may be altered after any DINOv3 similarity has been observed. Pages rejected by the screen remain in the page ledger. A manuscript remains eligible for a channel only if it supplies at least eight broad objects in that channel.

Known-answer duplicate manuscripts use the same screened page indices for both records wherever their canvas counts and labels align; this prevents page-selection noise from obscuring the positive control.
