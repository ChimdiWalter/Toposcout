# TopoScout Cross-Domain Portability Pilots (M6A)

**Validated reference application: maize phenotyping** (the deployed cloud
pipeline). Everything below is a **portability pilot**: an independent public
model wrapped behind the same TopoScout evidence contract and audited by the
same deterministic structural layer, with a domain profile selecting which
measurements matter. Pilots demonstrate that the agent/control architecture is
domain-independent. **No pilot is a validated scientific, clinical, GIS,
safety, or manufacturing application.**

Machine-readable results: `artifacts/pilots/portability_matrix.json`; per-domain
`artifacts/pilots/<domain>/{input.*, mask.png, overlay.jpg, evidence.json, README.md}`.
Every stored audit is exactly recomputable from the frozen mask
(tests/unit/test_cross_domain_pilots.py::test_evidence_metrics_are_recomputable_not_fabricated).

## Portability matrix (2026-08-30)

| Domain | Model | Input | Output | Structural question | Key TopoScout metrics | Result | Status | License | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| Maize (REFERENCE) | validated T0 lesion model (private Cloud Run worker) | field leaf scans | lesion mask | is the lesion prediction over-fragmented; can the validated 500 px filter recover it? | β₀, fragmentation, largest fraction | DSC_0059 RETRY→ACCEPT (β₀ 65→20); DSC_0100 RETRY→HUMAN_REVIEW (707→241) | **VALIDATED** | project-internal | 36 s/attempt (2 CPU) |
| Microscopy | Cellpose 4 (Cellpose-SAM generalist) | cellpose.org sample cell image | instance mask (flattened) | are cells fragmented into debris or implausibly sized? | β₀=40, tiny fraction 0.05, area quartiles [638, 1125, 2033] px, largest 0.126 | structurally plausible (descriptive) | PILOT | code BSD-3; weights trained on CC-BY-NC data (flagged) | 143.2 s |
| Pathology | TIAToolbox HoVer-Net `hovernet_fast-pannuke` | TIAToolbox sample H&E tile (256²) | nucleus instance mask (flattened) | are nucleus instances plausible vs merged/fragmented? | β₀=8, tiny 0, quartiles [730, 983, 1337] px, largest 0.318 | structurally plausible (descriptive); NOT diagnostic | PILOT | code permissive; PanNuke weights CC BY-NC-SA (flagged) | 37.6 s |
| Satellite | Keras U-Net road segmentation (Massachusetts Roads, HF spectrewolf8) | Mass Roads test tile 1500² | road mask | is the road NETWORK connected? | β₀=26 disconnected pieces, fragmentation 1.0, largest network fraction 0.209, skeleton 24 comp / 46 endpoints, β₁=0 | **structurally suspicious → HUMAN_REVIEW** (descriptive) | PILOT | MIT (model) | 24.5 s |
| Materials | CrackenPy model1 (smp FPN resnext50, crack class) | CrackenPy example specimen (5330×1402) | crack mask | is a supposedly continuous crack shattered into fragments? | β₀=335 all tiny, skeleton 334 comp / 573 endpoints, largest 0.041, fg 0.53% | **highly fragmented thin structure → HUMAN_REVIEW** (descriptive) | PILOT | BSD v2 (model + dataset) | 80.4 s |
| Industrial | Anomalib PatchCore (default) on MVTec AD `bottle` | MVTec AD `broken_large/000.png` | anomaly map + thresholded mask (anomalib's computed threshold) | is the anomaly map structurally coherent or speckle? | β₀=1 single coherent defect region, tiny 0, largest fraction 1.0, area 16.9%, fragmentation 0.0 | **structurally coherent defect** (descriptive) | PILOT | anomalib Apache-2.0; MVTec AD CC BY-NC-SA (flagged) | 1648 s (CPU memory-bank fit + predict) |

## Frozen-before-evaluation parameters

- Satellite: tile 256 / stride 256, threshold 0.5, no post-processing, **no recovery rule**.
- Microscopy: Cellpose default weights and diameter estimation, CPU.
- Pathology: `hovernet_fast-pannuke`, tile mode, model's own instance decoding.
- Materials: model1.pt, 416 tiling, argmax classes, crack class only.
- Industrial: anomalib default PatchCore, category `bottle`, anomalib's computed threshold.

No pilot reuses maize policy thresholds (enforced by tests); pilots perform
**descriptive** structural auditing — a validated ACCEPT rule exists only for
maize. Suspicious structure escalates to HUMAN_REVIEW by construction.

## Wording allowed in the submission

- "TopoScout is an autonomous structural-verification and recovery layer for
  imaging models; we validate the complete workflow on maize phenotyping and
  demonstrate portability with five independent public models (microscopy,
  pathology, satellite, materials, industrial)."
- "The satellite pilot shows topology as a control signal: the road model's
  mask splits the network into 26 disconnected pieces — a structural failure
  that overlap metrics do not expose."
- "Pilots are demonstrations of adapter portability, not domain-validated
  applications."

## Claims we MUST NOT make

- That any pilot is scientifically, clinically, or operationally validated.
- Tumor/diagnostic claims (pathology), safety certification (materials),
  manufacturing acceptance (industrial), or operational GIS accuracy (satellite).
- That SAM2 masks are ground truth (they are training-time pseudo-labels;
  gold-corrected masks are the ground truth) — unchanged from M4/M5.
- Commercial use of the NC-licensed assets (Cellpose weights' training data,
  PanNuke weights, MVTec AD) without separate clearance; these are flagged in
  every evidence file and must be reviewed before public redistribution.
