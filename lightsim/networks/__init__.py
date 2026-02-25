"""Network generators and importers: grids, arterials, JSON/YAML, and OpenStreetMap."""

from .arterial import create_arterial_network
from .from_dict import from_dict, from_json
from .grid import create_grid_network

__all__ = [
    "create_grid_network",
    "create_arterial_network",
    "from_dict",
    "from_json",
]

# OSM import is optional (requires osmnx)
try:
    from .osm import from_osm_point

    __all__.append("from_osm_point")
except ImportError:
    pass
