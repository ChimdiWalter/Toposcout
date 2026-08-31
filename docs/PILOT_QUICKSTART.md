# Cross-domain pilot quickstart (judge-runnable)

TopoScout's hosted application demonstrates the validated maize workflow. The
same structural-verification contract can be reproduced locally with the
following portability pilots — real public models, real inference, the same
audit layer. Pilots are descriptive demonstrations, **not** domain-validated
applications.

## 1. Clone + hermetic tests (no models, no cloud, no Gemini)

```bash
git clone https://github.com/ChimdiWalter/Toposcout && cd Toposcout
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m pytest tests/ -q
```

## 2. Turnkey pilot: satellite roads (~1 min setup + ~30 s run)

```bash
bash scripts/setup_pilot.sh satellite
.pilot-envs/satellite/bin/python scripts/pilot_demo.py run satellite
```

The MIT-licensed road U-Net (Hugging Face) runs on the bundled Massachusetts
Roads test tile and the audit reports the frozen reference structure:

```
beta_0                       26
fragmentation_score          1.0
largest_component_fraction   0.20949477351916376
skeleton_components          24
skeleton_endpoints           46
assessment: STRUCTURALLY SUSPICIOUS (descriptive)
```

The road network is predicted in 26 disconnected pieces — the structural
failure overlap metrics don't expose. Artifacts (mask, overlay, evidence.json)
land in `outputs/pilots/satellite-<timestamp>/`; the frozen submission
evidence in `artifacts/pilots/` is never modified.

## 3. Turnkey pilot: materials cracks (BSD model + bundled sample)

```bash
bash scripts/setup_pilot.sh materials
.pilot-envs/materials/bin/python scripts/pilot_demo.py run materials
```

Expected structure: β₀ = 335 tiny crack fragments, fragmentation 1.0.

## 4. Bring-your-own-image domains

Bundled samples are omitted for these domains for licensing reasons
(`docs/THIRD_PARTY_ASSETS.md`); provide an appropriately licensed image:

```bash
bash scripts/setup_pilot.sh microscopy
.pilot-envs/microscopy/bin/python scripts/pilot_demo.py run microscopy --input my_cells.png

bash scripts/setup_pilot.sh pathology
.pilot-envs/pathology/bin/python scripts/pilot_demo.py run pathology --input my_he_tile.png
```

Industrial (PatchCore) additionally needs a normal-image memory bank: obtain
MVTec AD from its owner (CC BY-NC-SA), set `TOPOSCOUT_MVTEC_ROOT`, and expect
~30 minutes on CPU:

```bash
bash scripts/setup_pilot.sh industrial
export TOPOSCOUT_MVTEC_ROOT=/path/to/mvtec_root
.pilot-envs/industrial/bin/python scripts/pilot_demo.py run industrial --input /path/to/test_image.png
```

## 5. Environment status at a glance

```bash
python scripts/pilot_demo.py doctor
```

## Guardrails

The CLI can invoke only registered adapters; there is no flag for models,
checkpoints, thresholds, or executables. Every run prints the domain's
license/provenance notice and its limitations, and writes only under
`outputs/pilots/`.
