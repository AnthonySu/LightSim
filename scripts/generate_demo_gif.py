"""Generate promotional GIF animations of OSM network simulations.

Usage::

    python scripts/generate_demo_gif.py --city shanghai --frames 600
    python scripts/generate_demo_gif.py --city sf --frames 600 --fps 30
    python scripts/generate_demo_gif.py --all

Outputs GIF files to the project root (or specified --output directory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import MaxPressureController
from lightsim.networks.osm import from_osm_point, generate_demand


# ── City Definitions ──────────────────────────────────────────────

CITIES = {
    "shanghai": {
        "lat": 31.2365,
        "lon": 121.5010,
        "dist": 500,
        "label": "Shanghai, Lujiazui",
        "rate": 0.55,
    },
    "sf": {
        "lat": 37.7936,
        "lon": -122.3959,
        "dist": 500,
        "label": "San Francisco, FiDi",
        "rate": 0.55,
    },
    "siouxfalls": {
        "lat": 43.5446,
        "lon": -96.7311,
        "dist": 500,
        "label": "Sioux Falls, SD",
        "rate": 0.50,
    },
    "tokyo": {
        "lat": 35.6595,
        "lon": 139.7004,
        "dist": 500,
        "label": "Tokyo, Shibuya",
        "rate": 0.45,
    },
    "london": {
        "lat": 51.5138,
        "lon": -0.0984,
        "dist": 500,
        "label": "London, City of London",
        "rate": 0.45,
    },
    "manhattan": {
        "lat": 40.7549,
        "lon": -73.9840,
        "dist": 400,
        "label": "Manhattan, Midtown",
        "rate": 0.50,
    },
}

# ── Color Scale ───────────────────────────────────────────────────

COLOR_STOPS = [
    (0.00, np.array([16, 185, 145]) / 255),
    (0.25, np.array([6, 214, 160]) / 255),
    (0.45, np.array([255, 209, 102]) / 255),
    (0.65, np.array([255, 159, 67]) / 255),
    (0.85, np.array([239, 71, 111]) / 255),
    (1.00, np.array([180, 30, 70]) / 255),
]


def density_color(ratio: float) -> np.ndarray:
    """Map density ratio (0-1) to RGB color."""
    r = max(0.0, min(1.0, ratio))
    lo, hi = 0, len(COLOR_STOPS) - 1
    for i in range(len(COLOR_STOPS) - 1):
        if COLOR_STOPS[i][0] <= r <= COLOR_STOPS[i + 1][0]:
            lo, hi = i, i + 1
            break
    t_range = COLOR_STOPS[hi][0] - COLOR_STOPS[lo][0]
    f = (r - COLOR_STOPS[lo][0]) / t_range if t_range > 0 else 0
    return COLOR_STOPS[lo][1] * (1 - f) + COLOR_STOPS[hi][1] * f


def density_cmap(n: int = 256) -> matplotlib.colors.ListedColormap:
    """Create a matplotlib colormap from our density stops."""
    colors = [density_color(i / (n - 1)) for i in range(n)]
    return matplotlib.colors.ListedColormap(colors)


# ── Network Rendering ─────────────────────────────────────────────

class NetworkRenderer:
    """Renders a LightSim network to matplotlib frames."""

    def __init__(self, engine: SimulationEngine, figsize=(12, 8), dpi=100):
        self.engine = engine
        self.net = engine.net
        self.network = engine.network
        self.figsize = figsize
        self.dpi = dpi

        # Pre-compute link geometry
        self._precompute()

    def _precompute(self):
        """Pre-compute link segments and node positions."""
        net = self.net
        network = self.network

        self.node_xy = {}
        self.node_types = {}
        for node in network.nodes.values():
            self.node_xy[node.node_id] = (node.x, node.y)
            self.node_types[node.node_id] = node.node_type.name

        self.links = []
        for link in network.links.values():
            from_node = network.nodes[link.from_node]
            to_node = network.nodes[link.to_node]
            cell_ids = [c.cell_id for c in link.cells]
            self.links.append({
                "id": link.link_id,
                "x1": from_node.x, "y1": from_node.y,
                "x2": to_node.x, "y2": to_node.y,
                "cells": cell_ids,
                "lanes": link.cells[0].lanes if link.cells else 1,
                "n_cells": len(cell_ids),
            })

        # Bounding box
        xs = [n.x for n in network.nodes.values()]
        ys = [n.y for n in network.nodes.values()]
        self.xmin, self.xmax = min(xs), max(xs)
        self.ymin, self.ymax = min(ys), max(ys)
        pad_x = (self.xmax - self.xmin) * 0.08
        pad_y = (self.ymax - self.ymin) * 0.08
        self.xmin -= pad_x
        self.xmax += pad_x
        self.ymin -= pad_y
        self.ymax += pad_y

    def render_frame(self, title: str = "", step_info: str = "") -> np.ndarray:
        """Render the current engine state to a numpy RGBA array."""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi,
                               facecolor="#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_aspect("equal")
        ax.axis("off")

        density = self.engine.state.density
        net = self.net

        # 1. Draw road backgrounds
        road_segments = []
        for link in self.links:
            road_segments.append([(link["x1"], link["y1"]),
                                  (link["x2"], link["y2"])])
        if road_segments:
            # Dark road background
            lc_bg = LineCollection(road_segments, colors="#0a0a14",
                                  linewidths=5.0, capstyle="round",
                                  zorder=1)
            ax.add_collection(lc_bg)
            lc_road = LineCollection(road_segments, colors="#1e1e30",
                                    linewidths=3.5, capstyle="round",
                                    zorder=2)
            ax.add_collection(lc_road)

        # 2. Draw density-colored cell segments
        cell_segments = []
        cell_colors = []
        for link in self.links:
            x1, y1 = link["x1"], link["y1"]
            x2, y2 = link["x2"], link["y2"]
            dx = x2 - x1
            dy = y2 - y1
            n = link["n_cells"]
            for i, cid in enumerate(link["cells"]):
                d = float(density[cid])
                kj = float(net.kj[cid])
                ratio = d / kj if kj > 0 else 0
                if ratio < 0.005:
                    continue
                t0 = i / n
                t1 = (i + 1) / n
                sx = x1 + dx * t0
                sy = y1 + dy * t0
                ex = x1 + dx * t1
                ey = y1 + dy * t1
                cell_segments.append([(sx, sy), (ex, ey)])
                cell_colors.append(density_color(ratio))

        if cell_segments:
            lc_cells = LineCollection(cell_segments, colors=cell_colors,
                                     linewidths=3.0, capstyle="round",
                                     zorder=3)
            ax.add_collection(lc_cells)

        # 3. Draw signalized nodes
        sig_states = self.engine.signal_manager.states
        for node_id, node in self.network.nodes.items():
            x, y = node.x, node.y
            ntype = node.node_type.name

            if ntype == "SIGNALIZED":
                color = "#555555"
                glow_alpha = 0.3
                if node_id in sig_states:
                    sig = sig_states[node_id]
                    if sig.in_all_red:
                        color = "#ef476f"
                    elif sig.in_yellow:
                        color = "#ffd166"
                    else:
                        color = "#06d6a0"

                # Glow
                glow = plt.Circle((x, y), 12, color=color, alpha=0.25,
                                  zorder=4)
                ax.add_patch(glow)
                # Node
                node_patch = mpatches.FancyBboxPatch(
                    (x - 6, y - 6), 12, 12,
                    boxstyle=mpatches.BoxStyle.Round(pad=2),
                    facecolor=color, edgecolor="white", linewidth=0.5,
                    zorder=5,
                )
                ax.add_patch(node_patch)
            elif ntype == "ORIGIN":
                diamond = mpatches.RegularPolygon(
                    (x, y), 4, radius=5, orientation=np.pi / 4,
                    facecolor="#00b4d8", edgecolor="white",
                    linewidth=0.3, zorder=4,
                )
                ax.add_patch(diamond)
            elif ntype == "DESTINATION":
                diamond = mpatches.RegularPolygon(
                    (x, y), 4, radius=5, orientation=np.pi / 4,
                    facecolor="#5a5a7a", edgecolor="white",
                    linewidth=0.3, zorder=4,
                )
                ax.add_patch(diamond)

        # 4. Title and info overlay
        if title:
            ax.text(0.02, 0.97, title,
                    transform=ax.transAxes, fontsize=14, fontweight="bold",
                    color="#00b4d8", va="top", ha="left",
                    fontfamily="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                              edgecolor="#2a2a4a", alpha=0.9))

        if step_info:
            ax.text(0.02, 0.03, step_info,
                    transform=ax.transAxes, fontsize=10,
                    color="#a0a0c0", va="bottom", ha="left",
                    fontfamily="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f0f1a",
                              edgecolor="#2a2a4a", alpha=0.8))

        # Branding
        ax.text(0.98, 0.03, "LightSim",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                color="#00b4d8", va="bottom", ha="right", alpha=0.6,
                fontfamily="sans-serif")

        # Render to numpy array
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf


# ── GIF Generation ────────────────────────────────────────────────

def generate_gif(
    city_key: str,
    n_frames: int = 600,
    fps: int = 30,
    output_dir: str = ".",
    dt: float = 0.5,
    min_cell_length: float = 8.0,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 100,
    sim_steps_per_frame: int = 3,
) -> Path:
    """Generate a GIF for a city simulation.

    Parameters
    ----------
    city_key : str
        Key from CITIES dict.
    n_frames : int
        Number of GIF frames.
    fps : int
        Frames per second in the output GIF.
    output_dir : str
        Directory for output file.
    dt : float
        Simulation time step.
    min_cell_length : float
        Minimum cell length for OSM import.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output DPI (resolution = figsize * dpi).
    sim_steps_per_frame : int
        Simulation steps between each rendered frame.

    Returns
    -------
    Path
        Path to the generated GIF file.
    """
    import imageio.v3 as iio

    city = CITIES[city_key]
    print(f"Generating GIF for {city['label']}...")

    # 1. Build network
    print(f"  Downloading OSM data ({city['lat']}, {city['lon']}, dist={city['dist']}m)...")
    net = from_osm_point(
        city["lat"], city["lon"],
        dist=city["dist"],
        dt=dt,
        min_cell_length=min_cell_length,
    )
    demand = generate_demand(net, rate=city["rate"])
    compiled = net.compile(dt=dt)
    print(f"  Network: {len(net.nodes)} nodes, {len(net.links)} links, "
          f"{compiled.n_cells} cells")

    # 2. Create engine
    controller = MaxPressureController()
    engine = SimulationEngine(
        network=net, dt=dt,
        controller=controller,
        demand_profiles=demand,
    )
    engine.reset(seed=42)

    # 3. Warm up (let congestion build up)
    warmup_steps = int(300 / dt)  # 300 seconds of warmup
    print(f"  Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        engine.step()

    # 4. Render frames
    renderer = NetworkRenderer(engine, figsize=figsize, dpi=dpi)
    print(f"  Rendering {n_frames} frames...")

    frames = []
    for i in range(n_frames):
        # Advance simulation
        for _ in range(sim_steps_per_frame):
            engine.step()

        # Render
        t = engine.state.time
        veh = engine.get_total_vehicles()
        entered = engine.state.total_entered
        exited = engine.state.total_exited
        step_info = (f"t={t:.0f}s  |  {veh:.0f} vehicles  |  "
                     f"{entered:.0f} entered  |  {exited:.0f} exited")

        frame = renderer.render_frame(
            title=city["label"],
            step_info=step_info,
        )
        frames.append(frame)

        if (i + 1) % 50 == 0:
            print(f"    Frame {i+1}/{n_frames} (t={t:.0f}s, {veh:.0f} veh)")

    # 5. Assemble GIF
    output_path = Path(output_dir) / f"lightsim_{city_key}.gif"
    print(f"  Saving GIF to {output_path}...")

    # Use imageio to write GIF
    duration = 1000 // fps  # ms per frame
    iio.imwrite(
        output_path,
        frames,
        duration=duration,
        loop=0,  # infinite loop
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done! {output_path} ({file_size_mb:.1f} MB)")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate promotional GIF animations of LightSim"
    )
    parser.add_argument("--city", type=str, default="shanghai",
                        choices=list(CITIES.keys()) + ["all"],
                        help="City to render (or 'all')")
    parser.add_argument("--frames", type=int, default=400,
                        help="Number of frames")
    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--dt", type=float, default=0.5,
                        help="Simulation time step")
    parser.add_argument("--min-cell", type=float, default=8.0,
                        help="Minimum cell length (metres)")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Output DPI")
    parser.add_argument("--steps-per-frame", type=int, default=3,
                        help="Sim steps between frames")
    args = parser.parse_args()

    cities = list(CITIES.keys()) if args.city == "all" else [args.city]

    for city in cities:
        generate_gif(
            city_key=city,
            n_frames=args.frames,
            fps=args.fps,
            output_dir=args.output,
            dt=args.dt,
            min_cell_length=args.min_cell,
            dpi=args.dpi,
            sim_steps_per_frame=args.steps_per_frame,
        )


if __name__ == "__main__":
    main()
