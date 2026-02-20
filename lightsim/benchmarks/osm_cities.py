"""Real-world city network scenarios from OpenStreetMap."""

from __future__ import annotations

from ..core.demand import DemandProfile
from ..core.network import Network
from ..networks.osm import from_osm_point, generate_demand

# City definitions: (name, lat, lon, description)
CITIES = {
    # Original 6
    "osm-manhattan-v0": (40.7549, -73.9840, "Midtown Manhattan"),
    "osm-shanghai-v0": (31.2365, 121.5010, "Lujiazui / Pudong"),
    "osm-beijing-v0": (39.9139, 116.4030, "Wangfujing"),
    "osm-shenzhen-v0": (22.5411, 114.0579, "Futian CBD"),
    "osm-losangeles-v0": (34.0522, -118.2437, "Downtown LA"),
    "osm-sanfrancisco-v0": (37.7936, -122.3959, "Financial District"),
    # New 10
    "osm-siouxfalls-v0": (43.5446, -96.7311, "Downtown Sioux Falls"),
    "osm-tokyo-v0": (35.6595, 139.7004, "Shibuya"),
    "osm-chicago-v0": (41.8819, -87.6278, "The Loop"),
    "osm-london-v0": (51.5138, -0.0984, "City of London"),
    "osm-paris-v0": (48.8698, 2.3078, "Champs-Élysées"),
    "osm-singapore-v0": (1.3048, 103.8318, "Orchard Road"),
    "osm-seoul-v0": (37.4979, 127.0276, "Gangnam"),
    "osm-toronto-v0": (43.6510, -79.3810, "Downtown / Bay St"),
    "osm-mumbai-v0": (19.0596, 72.8656, "Bandra-Kurla Complex"),
    "osm-sydney-v0": (-33.8688, 151.2093, "CBD / George St"),
}


def _make_city_factory(lat: float, lon: float):
    """Create a scenario factory for a city at (lat, lon)."""
    def factory(
        dist: float = 500.0,
        rate: float = 0.2,
    ) -> tuple[Network, list[DemandProfile]]:
        net = from_osm_point(lat, lon, dist=dist)
        demand = generate_demand(net, rate=rate)
        return net, demand
    return factory


# Build factories for each city
city_factories: dict[str, callable] = {}
for name, (lat, lon, _desc) in CITIES.items():
    city_factories[name] = _make_city_factory(lat, lon)
