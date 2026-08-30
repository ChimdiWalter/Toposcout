# Pre-existing components disclosure

TopoScout's hackathon contribution is the autonomous orchestration layer. The
scientific substance it orchestrates predates the hackathon and was developed
by the author as part of an ongoing research programme. This file separates the
two honestly; nothing pre-existing is claimed as hackathon work.

## PRE-EXISTING SCIENTIFIC WORK (not built during the hackathon)

All under `<private-research-archive>` (the author's research tree), read-only from
TopoScout's perspective. `H` = `code/python/pipeline/segmentors_backbones/hypertopo_maize_gold_parallel`.

| Component | Exact source | Role in TopoScout |
|---|---|---|
| Maize leaf images (normalized leaves) | `maize_lesion_study_v1/derived/normalized_leaves/` | real demo inputs, read in place |
| Gold-corrected expert lesion masks — TRUE ground truth (expert-annotated/corrected) | `gold_corrected_mask` column of `H/manifests/linked_manifest.csv` | the only ground truth TopoScout references (14 held-out demo leaves). Distinct from SAM2 pseudo-labels, which are imperfect, affected only model TRAINING, and are never used as ground truth here |
| Lesion segmentation model (T0 tile baseline): DINOv2 ViT-B/14 backbone with maize continued-pretraining + decoder | `H/src/models/geometry_specialist.py`, `H/v1_src/.../models/foundation_backbones.py`, backbone weights `H/weights/dinov2_maize_cpt.pth` | the real segmentation model |
| Trained checkpoint | `H/experiments/critical_set_ph/results/phase5/p5_t0_tile_base/best_checkpoint.pth` (335 MB, dice_bce, seed 1337) | fixed trusted checkpoint (sha256 reported in every result) |
| Native-resolution tiled inference (512-px tiles, stride 384, 518-px network input, overlap-averaged sigmoid probabilities, 0.5 binarization) | `H/experiments/critical_set_ph/src/p8_significance_eval.py` (`load_model`, `predict_tiled`) | inference engine called by the worker, unmodified |
| Significance filtering of components (min area 500 px) — validated finding S3 of the research project | `p8_significance_eval.filt_count`, MASTER_PLAN settled finding S3 | the scientifically grounded attempt-2 recovery operation |
| Scientific rationale that topological evaluation is required and that tile inference over-fragments (settled findings S2, S4) | `H/experiments/critical_set_ph/MASTER_PLAN.md` | motivates the topology-audited retry design |
| Phenotype extraction utilities (pixel-based; no physical calibration exists, so TopoScout stays in pixels/fractions) | `H/v1_src/.../phenotype/extract_phenotype.py` | reference for future phenotype evidence |
| Python environment for the scientific model | `~/.venvs/lesegenv` (torch, timm, cv2, scipy) | subprocess interpreter for the worker |

Deliberately NOT used (experimental, running, invalidated, or prospective —
firewalled from hackathon claims): Project B geometry instrumentation
(`H/experiments/critical_set_ph/project_b/`, calibration in progress), Project C
prospective topometric-fields architecture, in-progress A3/A28 readout
experiments, retracted claims (e.g. S12 merge-tree hyperbolicity), quarantined
runs and audit archives, and the older standalone
`<private-research-archive>` (superseded SimpleUNet path).

## HACKATHON-BUILT TOPOSCOUT (this repository)

- Gemini 3.5 Flash agent + Google ADK orchestration (`app/`)
- LLM-facing scientific-tool interface layer with schema hardening (no
  filesystem, executable, checkpoint, or threshold control by the model)
- SegmentationAdapter abstraction (`toposcout_core/adapters.py`):
  `demo_dark_structure_v1` (newly written demo fallback) and
  `real_lesion_model_v1` (strict subprocess wrapper, `toposcout_core/real_worker.py`)
- Topology audit (`audit_topology`: beta_0, approximate beta_1, fragmentation)
  and deterministic bounded policy — newly written for the hackathon (simple
  4-connectivity component analysis; the research project's own PH machinery
  is intentionally not imported)
- Bounded autonomous decisions: ACCEPT / RETRY / HUMAN_REVIEW /
  REQUEST_REACQUISITION, max 2 segmentation attempts
- Exact-number evidence reporting: canonical JSON reports, deterministic
  display_summary, in-process evidence cache defeating LLM float truncation
- Deterministic synthetic fixtures + 34-test suite
- Cloud Run deployment, Secret Manager integration (prior milestones)

The demo adapter remains available as a fallback
(`TOPOSCOUT_SEGMENTATION_ADAPTER=demo`, the default).

## Provenance/authorization

The images, annotations, model, and checkpoints are the author's own research
assets (authorized use). Before final submission, add: dates created, license
statement, and whether each component was modified during the hackathon
(current answer: none were modified; all are wrapped read-only).
