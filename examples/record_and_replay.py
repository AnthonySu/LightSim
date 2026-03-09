"""Record a simulation and save it for replay in the visualizer."""

try:
    from lightsim.benchmarks.single_intersection import create_single_intersection
    from lightsim.core.engine import SimulationEngine
    from lightsim.core.signal import MaxPressureController
except ImportError:
    print("Please install lightsim: pip install -e '.[all]'")
    raise

try:
    from lightsim.viz.recorder import Recorder
except ImportError:
    print("Visualization support required: pip install -e '.[viz]'")
    raise


def main():
    # Create scenario
    network, demand = create_single_intersection()
    engine = SimulationEngine(
        network=network, dt=1.0,
        controller=MaxPressureController(min_green=5.0),
        demand_profiles=demand,
    )
    engine.reset(seed=42)

    # Record
    recorder = Recorder(engine)
    for step in range(1800):  # 30 minutes
        engine.step()
        recorder.capture()

    # Save
    recorder.save("recording.json")
    print(f"Saved {len(recorder.frames)} frames to recording.json")
    print(f"File contains {engine.net.n_cells} cells across "
          f"{len(network.links)} links")
    print(f"\nReplay with:")
    print(f"  python -m lightsim.viz --replay recording.json")


if __name__ == "__main__":
    main()
