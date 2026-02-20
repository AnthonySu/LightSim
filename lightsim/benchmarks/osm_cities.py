"""Real-world city network scenarios from OpenStreetMap."""

from __future__ import annotations

from ..core.demand import DemandProfile
from ..core.network import Network
from ..networks.osm import from_osm_point, generate_demand

# City definitions: (name, lat, lon, description)
CITIES = {
    "osm-manhattan-v0": (40.7549, -73.9840, "Midtown Manhattan"),
    "osm-shanghai-v0": (31.2365, 121.5010, "Lujiazui / Pudong"),
    "osm-beijing-v0": (39.9139, 116.4030, "Wangfujing"),
    "osm-shenzhen-v0": (22.5411, 114.0579, "Futian CBD"),
    "osm-losangeles-v0": (34.0522, -118.2437, "Downtown LA"),
    "osm-sanfrancisco-v0": (37.7936, -122.3959, "Financial District"),
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
