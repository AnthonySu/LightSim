"""Tests for the speed benchmark module."""

import pytest

from lightsim.benchmarks.speed_benchmark import benchmark_scenario, run_all, BenchmarkResult


class TestSpeedBenchmark:
    def test_single_benchmark(self):
        """Benchmark a single scenario."""
        from lightsim.benchmarks.single_intersection import create_single_intersection
        net, demand = create_single_intersection()
        result = benchmark_scenario("test", net, demand, n_steps=500)

        assert isinstance(result, BenchmarkResult)
        assert result.name == "test"
        assert result.n_steps == 500
        assert result.wall_time > 0
        assert result.steps_per_sec > 0
        assert result.speedup_vs_realtime > 0
        assert result.n_cells > 0

    def test_run_all_returns_results(self):
        """run_all should return results for all scenarios."""
        results = run_all(n_steps=200)
        assert len(results) >= 5  # single + grids + arterials
        for r in results:
            assert r.steps_per_sec > 0
            assert r.n_cells > 0

    def test_grid_scales(self):
        """Larger grids should have more cells."""
        from lightsim.benchmarks.speed_benchmark import benchmark_scenario
        from lightsim.networks.grid import create_grid_network
        from lightsim.core.demand import DemandProfile
        from lightsim.core.types import NodeType

        results = []
        for size in [2, 4]:
            net = create_grid_network(rows=size, cols=size, n_cells_per_link=3)
            demand = []
            for link in net.links.values():
                fn = net.nodes.get(link.from_node)
                if fn and fn.node_type == NodeType.ORIGIN:
                    demand.append(DemandProfile(link.link_id, [0.0], [0.1]))
            r = benchmark_scenario(f"grid-{size}", net, demand, n_steps=100)
            results.append(r)

        assert results[1].n_cells > results[0].n_cells
