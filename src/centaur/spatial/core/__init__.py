"""
Core spatial utilities: I/O, distance calculations, and CRS handling.
"""

from .io import load_spatial, save_spatial, has_geometry
from .distance import (
    haversine_distance,
    haversine_matrix,
    nearest_neighbor,
    distance_to_nearest,
)
from .crs import ensure_crs, to_projected, estimate_utm_zone

__all__ = [
    "load_spatial",
    "save_spatial",
    "has_geometry",
    "haversine_distance",
    "haversine_matrix",
    "nearest_neighbor",
    "distance_to_nearest",
    "ensure_crs",
    "to_projected",
    "estimate_utm_zone",
]
