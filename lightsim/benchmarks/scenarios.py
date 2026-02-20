"""Scenario registry for built-in benchmark scenarios."""

from __future__ import annotations

from typing import Any, Callable

from ..core.demand import DemandProfile
from ..core.network import Network

ScenarioFactory = Callable[..., tuple[Network, list[DemandProfile]]]

_SCENARIO_REGISTRY: dict[str, ScenarioFactory] = {}


def register_scenario(name: str):
    """Decorator to register a scenario factory."""
    def wrapper(fn: ScenarioFactory) -> ScenarioFactory:
        _SCENARIO_REGISTRY[name] = fn
        return fn
    return wrapper


def get_scenario(name: str) -> ScenarioFactory:
    """Retrieve a registered scenario factory by name."""
    if name not in _SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario: {name!r}. "
                       f"Available: {list(_SCENARIO_REGISTRY.keys())}")
    return _SCENARIO_REGISTRY[name]


def list_scenarios() -> list[str]:
    """List all registered scenario names."""
    return list(_SCENARIO_REGISTRY.keys())


# Register built-in scenarios
from .single_intersection import create_single_intersection
from .grid_4x4 import create_grid_4x4
from .arterial_5 import create_arterial_5

register_scenario("single-intersection-v0")(create_single_intersection)
register_scenario("grid-4x4-v0")(create_grid_4x4)
register_scenario("arterial-5-v0")(create_arterial_5)

# OSM city scenarios (requires osmnx)
try:
    from .osm_cities import city_factories as _osm_factories
    for _name, _factory in _osm_factories.items():
        register_scenario(_name)(_factory)
except ImportError:
    pass
