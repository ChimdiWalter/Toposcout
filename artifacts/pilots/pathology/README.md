# PATHOLOGY PORTABILITY PILOT — NOT FOR CLINICAL USE

- **Model**: TIAToolbox HoVer-Net (hovernet_fast-pannuke)
- **Source**: https://github.com/TissueImageAnalytics/tiatoolbox
- **License**: TIAToolbox code permissive; PanNuke HoVer-Net weights CC BY-NC-SA (noncommercial — flagged); input: TIAToolbox sample tissue tile
- **Structural question** (many_instances):
  metrics beta_0, tiny_components, tiny_component_fraction, component_area_quartiles_px, largest_component_fraction

Key measurements:
```json
{
  "beta_0": 8,
  "tiny_components": 0,
  "tiny_component_fraction": 0.0,
  "component_area_quartiles_px": [
    730,
    983,
    1337
  ],
  "largest_component_fraction": 0.3180031446540881
}
```

Limitations: Portability pilot only — NOT for clinical use, no diagnostic claims; instances flattened to a binary mask; no recovery rule.