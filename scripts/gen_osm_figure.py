"""Generate OSM city network visualization figure for the paper.

Produces a 2x3 grid showing 6 real-world city networks imported from OSM.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "mathtext.fontset": "cm",
})

from lightsim.benchmarks.scenarios import get_scenario
from lightsim.core.types import NodeType

CITIES = [
    ("osm-manhattan-v0", "Manhattan, NYC"),
    ("osm-shanghai-v0", "Shanghai, Pudong"),
    ("osm-beijing-v0", "Beijing, Wangfujing"),
    ("osm-shenzhen-v0", "Shenzhen, Futian"),
    ("osm-losangeles-v0", "Los Angeles, Downtown"),
    ("osm-sanfrancisco-v0", "San Francisco, FiDi"),
]


def plot_network(ax, net, title):
    """Plot a network on an axis."""
    # Collect link segments
    segments = []
    for lid, link in net.links.items():
        fn = net.nodes.get(link.from_node)
        tn = net.nodes.get(link.to_node)
        if fn is None or tn is None:
            continue
        segments.append([(fn.x, fn.y), (tn.x, tn.y)])

    if segments:
        lc = LineCollection(segments, colors="#78909C", linewidths=0.8, alpha=0.6)
        ax.add_collection(lc)

    # Plot nodes by type
    sig_x, sig_y = [], []
    orig_x, orig_y = [], []
    other_x, other_y = [], []

    for nid, node in net.nodes.items():
        if node.node_type == NodeType.SIGNALIZED:
            sig_x.append(node.x)
            sig_y.append(node.y)
        elif node.node_type == NodeType.ORIGIN:
            orig_x.append(node.x)
            orig_y.append(node.y)
        else:
            other_x.append(node.x)
            other_y.append(node.y)

    if other_x:
        ax.scatter(other_x, other_y, s=4, c="#B0BEC5", zorder=2, alpha=0.5)
    if orig_x:
        ax.scatter(orig_x, orig_y, s=15, c="#4CAF50", zorder=3, marker="^",
                   edgecolors="#2E7D32", linewidths=0.3, label="Origin")
    if sig_x:
        ax.scatter(sig_x, sig_y, s=20, c="#FF5722", zorder=4,
                   edgecolors="#BF360C", linewidths=0.3, label="Signalized")

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    # Remove axis ticks (UTM coordinates aren't meaningful to readers)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#CCC")
    # Add 200m scale bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bar_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    bar_y = ylim[0] + (ylim[1] - ylim[0]) * 0.06
    ax.plot([bar_x, bar_x + 200], [bar_y, bar_y], color="#333", linewidth=1.5)
    ax.text(bar_x + 100, bar_y + (ylim[1] - ylim[0]) * 0.03, "200m",
            ha="center", fontsize=6, color="#333")


fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
axes = axes.flatten()

for i, (scenario_name, city_label) in enumerate(CITIES):
    print(f"Loading {city_label}...")
    factory = get_scenario(scenario_name)
    net, demand = factory()
    n_sig = sum(1 for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED)
    n_links = len(net.links)
    label = f"{city_label}\n({n_sig} signals, {n_links} links)"
    plot_network(axes[i], net, label)

# Add legend to last axis
handles = [
    mpatches.Patch(color="#FF5722", label="Signalized intersection"),
    mpatches.Patch(color="#4CAF50", label="Origin / entry point"),
    mpatches.Patch(color="#78909C", label="Road link"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
           frameon=True, fancybox=True, shadow=False,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Real-World City Networks from OpenStreetMap (500m radius)",
             fontsize=11, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save
outpath = r"C:\Users\admin\Projects\69927a89543379cbbfcbc218\figures\osm_cities.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=300, pad_inches=0.1)
print(f"Saved: {outpath}")

png_path = outpath.replace(".pdf", ".png")
fig.savefig(png_path, bbox_inches="tight", dpi=150, pad_inches=0.1)
print(f"Preview: {png_path}")
