# Third-Party Models and Data — Provenance and Redistribution Audit (M7)

Audit date: 2026-08-30. Policy: **numeric evidence produced by TopoScout is
ours and stays public; third-party imagery, weights, or derived imagery are
redistributed only where their license clearly permits it.** Where
redistribution is questionable, the public UI and public repository show a
provenance card ("Dataset imagery omitted from public redistribution") and the
numbers only. Nothing is fabricated; the full frozen artifacts remain in the
private research archive and are regenerable via `scripts/run_cross_domain_pilots.py`.

## Production (validated reference)

| Asset | Source | License / terms | Public exposure |
|---|---|---|---|
| Maize T0 lesion model (checkpoint) | project-internal research | private | NOT distributed; runs behind the private Cloud Run worker |
| Maize demo leaves (DSC_0059/DSC_0100) | project-internal field data | private, project-owned | served through the live app as demo runs |
| DINOv2 backbone architecture (timm) | timm / Meta DINOv2 | Apache-2.0 code | architecture code only; no pretrained weights ship in the image (`pretrained=False`) |

## Pilots

| Domain | Model / weights | Model license | Input imagery | Imagery license | Public imagery? |
|---|---|---|---|---|---|
| Satellite | HF `spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp` | MIT | Massachusetts Roads test tile (Mnih 2013; MassGIS public-record aerial imagery) | research dataset, standard academic redistribution | **YES** |
| Materials | HF `rievil/crackenpy` model1 | BSD v2 (model + dataset, research/education) | CrackenPy GitHub `Examples/Img` | BSD-3 repo | **YES** |
| Microscopy | Cellpose 4 default weights | code BSD-3; **weights trained on CC-BY-NC data** | cellpose.org sample image | no explicit license | **NO — omitted** |
| Pathology | TIAToolbox `hovernet_fast-pannuke` | toolbox permissive; **PanNuke weights CC BY-NC-SA** | TIAToolbox sample H&E tile | sample-data terms not explicit | **NO — omitted** |
| Industrial | Anomalib PatchCore (Apache-2.0) | Apache-2.0 | **MVTec AD** `bottle/test/broken_large/000.png` | **CC BY-NC-SA 4.0 (noncommercial)** | **NO — omitted** |

Enforcement:
- `evidence.json` per domain carries `public_imagery` + `imagery_note`; the UI
  renders a provenance card and the asset route refuses imagery for flagged
  domains (`webui/app.py::_pilot_imagery_public`).
- `.dockerignore` excludes flagged imagery from every container build, so the
  public UI image cannot even contain it.
- The public repository ships imagery only for satellite and materials.
- No model weights of any kind are committed to the repository; adapters
  download from the upstream sources at run time.

## Not claims we make

Pilot results are portability demonstrations, not validated applications.
Noncommercial-licensed assets (Cellpose training data, PanNuke weights,
MVTec AD) are used here for a noncommercial research/hackathon evaluation with
attribution; any commercial use would require separate clearance from their
owners.
