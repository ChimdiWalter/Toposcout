# SATELLITE PORTABILITY PILOT — NOT FOR OPERATIONAL GIS USE

- **Model**: aerial-image-road-segmentation-with-U-NET-xp (Keras U-Net, Massachusetts Roads)
- **Source**: https://huggingface.co/spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp
- **License**: MIT (model); input tile: Massachusetts Roads test set (Mnih 2013)
- **Structural question** (connected_network):
  metrics beta_0, fragmentation_score, largest_component_fraction, skeleton_components, skeleton_endpoints, beta_1

Key measurements:
```json
{
  "beta_0": 26,
  "fragmentation_score": 1.0,
  "largest_component_fraction": 0.20949477351916376,
  "skeleton_components": 24,
  "skeleton_endpoints": 46,
  "beta_1": 0
}
```

Limitations: Portability pilot only — not validated for operational GIS/road mapping. Fixed public checkpoint, frozen 0.5 threshold, no topology-aware training, no recovery rule.