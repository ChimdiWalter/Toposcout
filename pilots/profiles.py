"""Domain profiles (M6A): which structural measurements matter per domain.

Profiles are fixed BEFORE evaluating any pilot output. None of them inherit
the maize policy thresholds — pilots do descriptive structural auditing and
escalate suspicious structure; no pilot claims a validated ACCEPT rule.
"""
from __future__ import annotations

from .base import DomainProfile

MICROSCOPY_CELLS = DomainProfile(
    name="microscopy_cells",
    structure="many_instances",
    primary_metrics=("beta_0", "tiny_components", "tiny_component_fraction",
                     "component_area_quartiles_px", "largest_component_fraction"),
)

PATHOLOGY_NUCLEI = DomainProfile(
    name="pathology_nuclei",
    structure="many_instances",
    primary_metrics=("beta_0", "tiny_components", "tiny_component_fraction",
                     "component_area_quartiles_px", "largest_component_fraction"),
)

SATELLITE_ROADS = DomainProfile(
    name="satellite_roads",
    structure="connected_network",
    primary_metrics=("beta_0", "fragmentation_score", "largest_component_fraction",
                     "skeleton_components", "skeleton_endpoints", "beta_1"),
)

MATERIALS_CRACK = DomainProfile(
    name="materials_crack",
    structure="connected_thin_network",
    primary_metrics=("beta_0", "skeleton_components", "skeleton_endpoints",
                     "tiny_components", "largest_component_fraction",
                     "fragmentation_score"),
)

INDUSTRIAL_DEFECT = DomainProfile(
    name="industrial_defect",
    structure="sparse_anomaly_regions",
    primary_metrics=("beta_0", "tiny_components", "tiny_component_fraction",
                     "largest_component_fraction", "foreground_fraction",
                     "fragmentation_score"),
)

ALL_PROFILES = {p.name: p for p in (
    MICROSCOPY_CELLS, PATHOLOGY_NUCLEI, SATELLITE_ROADS, MATERIALS_CRACK,
    INDUSTRIAL_DEFECT,
)}
