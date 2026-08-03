# Preregistration Amendment 02 — Established checkpoint control

Date: 2026-08-03

Protocol: `P586-VMS-PLANT-0.1-20260803`

Status at amendment: Palatino whole-plant extraction and control extraction were still in progress. No DINOv3 embeddings, Voynich similarities, statistical comparisons or blinded adjudications had been computed.

## Reason

The Supabase corpus audit identified a mature, independently generated whole-plant localisation checkpoint for `bnf_gr_2179` containing 113 registered plant boxes. This is precisely the kind of valid localisation checkpoint that the compute protocol says should be reused rather than regenerated. It also supplies a chronologically distinct painted Dioscoridean control through the same database infrastructure.

Several fixed manifest controls were simultaneously found to contain no screen-positive botanical pages under the frozen page-screen rule. They remain in the ledger and are not reclassified or threshold-tuned, but manuscripts with fewer than eight broad objects remain ineligible under the original rule.

## Added supplementary control

Add `bnf_gr_2179` — BnF Grec 2179, Dioscorides, *De materia medica* — to the manuscript-level control panel.

The 113 existing `herbal_plant_boxes` records are treated only as frozen localisation proposals. Every proposed crop is re-created from its registered source folio, then independently adjudicated under the same experiment QA labels. The first 20 broad plants in deterministic folio/box order are retained; no DINO similarity is available during selection.

The same SAM2 mask, above-ground boundary, reproductive-structure extraction and DINOv3 pipeline is applied. The manuscript is eligible only if it supplies at least eight valid broad objects in a channel.

This amendment adds a control; it does not remove or replace any preregistered control and does not alter any similarity threshold or decision rule.
