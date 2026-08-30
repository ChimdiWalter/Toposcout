# MICROSCOPY PORTABILITY PILOT — NOT BIOLOGICALLY VALIDATED

- **Model**: Cellpose 4 (Cellpose-SAM generalist, default weights)
- **Source**: https://github.com/MouseLand/cellpose
- **License**: code BSD-3; model weights trained on CC-BY-NC data (per Cellpose docs) — noncommercial terms flagged; input: cellpose.org sample image
- **Structural question** (many_instances):
  metrics beta_0, tiny_components, tiny_component_fraction, component_area_quartiles_px, largest_component_fraction

Key measurements:
```json
{
  "beta_0": 40,
  "tiny_components": 2,
  "tiny_component_fraction": 0.05,
  "component_area_quartiles_px": [
    638,
    1125,
    2033
  ],
  "largest_component_fraction": 0.12636173847029797
}
```

Limitations: Portability pilot only — not biologically validated; instance masks flattened to a binary mask for the structural audit; no recovery rule.