# Cloud-safe loader equivalence — real_lesion_model_v1 (Milestone 5, Phase 1)

Date: 2026-08-30. Machine-readable result: `outputs/cloud_loader_equivalence.json`
(written by `scripts/check_cloud_loader_equivalence.py`, run in lesegenv, CPU,
4 torch threads).

## Frozen gate (declared before inspection — docs/MILESTONE5_CLOUD.md)

- probability `max_abs_diff <= 1e-6`
- exact binary-mask equality at threshold 0.5

## Verdict: **PASS** (exact, not merely within tolerance)

| Check | Result |
|---|---|
| Frozen model kwargs vs resolved t0_tile_base.yaml config | match |
| Strict `load_state_dict` missing / unexpected keys | 0 / 0 |
| State-dict bitwise equality (research vs cloud loader) | 200/200 tensors identical |
| Probability map shape (DSC_0059, 612x3900) | equal |
| Probability `max_abs_diff` / `mean_abs_diff` | **0.0 / 0.0** |
| Binary mask equality at 0.5 | exact |
| Foreground fraction | 0.4366750460868108 (both; matches sealed M4 demo attempt-1 value 0.4367) |
| Components min_area=1 / min_area=500 | 65 / 20 (both; matches M4 beta_0=65) |

## Loader identity

- Loader: `scientific_worker/model_loader.py` (`toposcout_cloud_safe_v1`)
- Architecture: `GeometrySpecialist` (dinov2_maize = timm `vit_base_patch14_dinov2`,
  decoder_embed 256, geometry hes, latent {dh:8, de:32, ds:8}, tangent_proxy),
  87,641,215 parameters.
- Built with `pretrained=False`: **no timm weight download, no 1.3 GB Stage-L CPT
  load** (verified in `DINOv2Backbone.__init__` source: both preloads gate on
  `pretrained`).
- Checkpoint (complete model state, proven by strict load + bitwise equality):
  `experiments/critical_set_ph/results/phase5/p5_t0_tile_base/best_checkpoint.pth`
  sha256 `8542675d6ce66de8bb3c0ebe0451b638d2724bb74d5d017d24e4217049ef7cd6`
- Load time: cloud 2.82 s vs research 5.33 s; inference identical (~20 s for
  DSC_0059 at 4 CPU threads). torch 2.11.0+cu128 (lesegenv).

## Why exactness (not just tolerance) was expected

The cloud loader restores the *identical bytes* into the *identical
architecture* and inference runs through the *same* `predict_tiled` code path
on the same CPU — so any nonzero diff would have indicated a real loader bug.
The gate result licenses deploying this loader as the scientific worker's
model source. Any change to loader, checkpoint, or inference code requires
re-running the gate before deployment.
