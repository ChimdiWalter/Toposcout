# MATERIALS PORTABILITY PILOT — NOT A SAFETY CERTIFICATION

- **Model**: CrackenPy model1 (segmentation-models-pytorch, crack class)
- **Source**: https://huggingface.co/rievil/crackenpy
- **License**: BSD v2 (CrackenPy model + dataset, research/education use)
- **Structural question** (connected_thin_network):
  metrics beta_0, skeleton_components, skeleton_endpoints, tiny_components, largest_component_fraction, fragmentation_score

Key measurements:
```json
{
  "beta_0": 335,
  "skeleton_components": 334,
  "skeleton_endpoints": 573,
  "tiny_components": 335,
  "largest_component_fraction": 0.041271214472914174,
  "fragmentation_score": 1.0
}
```

Limitations: Portability pilot only — structural-integrity visualization, not a safety certification; fixed public checkpoint; crack class only; no recovery rule.