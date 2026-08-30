# INDUSTRIAL PORTABILITY PILOT — NO ACCEPTANCE CLAIMS

- **Model**: Anomalib PatchCore (default config) on MVTec AD 'bottle'
- **Source**: https://github.com/open-edge-platform/anomalib
- **License**: anomalib Apache-2.0; MVTec AD dataset CC BY-NC-SA 4.0 (noncommercial — flagged)
- **Structural question** (sparse_anomaly_regions):
  metrics beta_0, tiny_components, tiny_component_fraction, largest_component_fraction, foreground_fraction, fragmentation_score

Key measurements:
```json
{
  "beta_0": 1,
  "tiny_components": 0,
  "tiny_component_fraction": 0.0,
  "largest_component_fraction": 1.0,
  "foreground_fraction": 0.16911851851851853,
  "fragmentation_score": 0.0
}
```

Limitations: Portability pilot only — no manufacturing acceptance claims; memory bank fitted on the category's normal images at run time; binary mask uses anomalib's computed threshold; no recovery rule.