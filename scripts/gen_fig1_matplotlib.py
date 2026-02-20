"""Generate Figure 1: CTM Explanation — clean vector PDF with matplotlib."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# Use LaTeX-like rendering — academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman", "serif"],
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.linewidth": 0.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

fig = plt.figure(figsize=(12, 3.2))

# ============================================================
# Panel (a): Cell Discretization
# ============================================================
ax_a = fig.add_axes([0.02, 0.08, 0.30, 0.85])
ax_a.set_xlim(-0.5, 5.5)
ax_a.set_ylim(-1.2, 2.2)
ax_a.set_aspect("equal")
ax_a.axis("off")

# Title
ax_a.text(2.5, 2.1, r"$\mathbf{(a)}$ Cell Discretization", ha="center",
          fontsize=10, fontweight="bold")

# Equation
ax_a.text(2.5, 1.6, r"$\Delta x = v_f \cdot \Delta t$", ha="center",
          fontsize=9, style="italic")

# Cells
densities = [0.02, 0.04, 0.08, 0.12, 0.05]
colors = ["#548235", "#7EA454", "#D4A843", "#C0784D", "#6BA35C"]
cell_labels = ["Cell 1", "Cell 2", "Cell 3", "Cell 4", "Cell 5"]

for i, (k, c, label) in enumerate(zip(densities, colors, cell_labels)):
    rect = FancyBboxPatch((i, 0), 0.9, 0.8, boxstyle="round,pad=0.03",
                           facecolor=c, edgecolor="#333", linewidth=0.8)
    ax_a.add_patch(rect)
    ax_a.text(i + 0.45, 0.4, f"$k$={k}", ha="center", va="center",
              fontsize=7.5, color="white", fontweight="bold")
    ax_a.text(i + 0.45, -0.3, label, ha="center", fontsize=7, color="#555")
    # Flow arrows between cells
    if i < 4:
        ax_a.annotate("", xy=(i + 1.0, 0.4), xytext=(i + 0.9, 0.4),
                       arrowprops=dict(arrowstyle="-|>", color="#1976D2",
                                       lw=1.2, shrinkA=0, shrinkB=0))

# Link brace / label
ax_a.text(2.5, -0.8, "Link (road segment)", ha="center", fontsize=8,
          color="#666", style="italic")

# ============================================================
# Panel (b): Sending & Receiving Flows
# ============================================================
ax_b = fig.add_axes([0.35, 0.08, 0.30, 0.85])
ax_b.set_xlim(-2, 6)
ax_b.set_ylim(-2.0, 3.5)
ax_b.set_aspect("equal")
ax_b.axis("off")

# Title
ax_b.text(2, 3.2, r"$\mathbf{(b)}$ Sending & Receiving Flows", ha="center",
          fontsize=10, fontweight="bold")

# Cell i (sender)
rect_i = FancyBboxPatch((-0.8, -0.2), 2.0, 1.2, boxstyle="round,pad=0.05",
                          facecolor="#FF9800", edgecolor="#333", linewidth=0.8)
ax_b.add_patch(rect_i)
ax_b.text(0.2, 0.4, r"Cell $i$", ha="center", va="center",
          fontsize=10, fontweight="bold")

# Cell i+1 (receiver)
rect_j = FancyBboxPatch((2.8, -0.2), 2.0, 1.2, boxstyle="round,pad=0.05",
                          facecolor="#4CAF50", edgecolor="#333", linewidth=0.8)
ax_b.add_patch(rect_j)
ax_b.text(3.8, 0.4, r"Cell $i\!+\!1$", ha="center", va="center",
          fontsize=10, fontweight="bold")

# Flow arrow
ax_b.annotate("", xy=(2.7, 0.4), xytext=(1.3, 0.4),
               arrowprops=dict(arrowstyle="-|>", color="#D32F2F",
                               lw=2.5, shrinkA=2, shrinkB=2))

# Sending flow formula (above Cell i)
ax_b.text(0.2, 1.7, r"$S_i = \min(v_f k_i,\, Q) \cdot \ell_i$",
          ha="center", fontsize=8.5, color="#8B4513",
          bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF8F0",
                    edgecolor="#C0A080", linewidth=0.4))

# Receiving flow formula (above Cell i+1)
ax_b.text(3.8, 1.7, r"$R_{i+1} = \min(Q,\, w(k_j\!-\!k_{i+1})) \cdot \ell$",
          ha="center", fontsize=8, color="#2E5020",
          bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0F8F0",
                    edgecolor="#80A080", linewidth=0.4))

# Flow equation on arrow
ax_b.text(2.0, -0.7, r"$q = \min(S_i,\, R_{i+1}) \cdot \Delta t$",
          ha="center", fontsize=8.5, color="#B71C1C")

# Bottom label
ax_b.text(2.0, -1.6, "Flow = min(supply, demand)", ha="center",
          fontsize=8, color="#666", style="italic")

# ============================================================
# Panel (c): Signal-Controlled Intersection
# ============================================================
ax_c = fig.add_axes([0.67, 0.08, 0.32, 0.85])
ax_c.set_xlim(-3.5, 3.5)
ax_c.set_ylim(-3.0, 3.5)
ax_c.set_aspect("equal")
ax_c.axis("off")

# Title
ax_c.text(0, 3.2, r"$\mathbf{(c)}$ Signal-Controlled Intersection",
          ha="center", fontsize=10, fontweight="bold")

# Central node
circle = plt.Circle((0, 0), 0.55, facecolor="#4472C4", edgecolor="#333",
                      linewidth=0.8)
ax_c.add_patch(circle)
ax_c.text(0, 0, r"$P_0$", ha="center", va="center", fontsize=11,
          fontweight="bold", color="white")

# Approach cells (small rectangles)
# North-South (GREEN)
for dy in [0.9, 1.35, 1.8]:
    # North inbound
    rect = FancyBboxPatch((-0.22, dy), 0.44, 0.35, boxstyle="round,pad=0.02",
                           facecolor="#4CAF50", edgecolor="#333", linewidth=0.5,
                           alpha=0.8)
    ax_c.add_patch(rect)
    # South inbound
    rect = FancyBboxPatch((-0.22, -dy - 0.35), 0.44, 0.35, boxstyle="round,pad=0.02",
                           facecolor="#4CAF50", edgecolor="#333", linewidth=0.5,
                           alpha=0.8)
    ax_c.add_patch(rect)

# NS flow arrows (solid green)
ax_c.annotate("", xy=(0, 0.6), xytext=(0, 0.85),
               arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2))
ax_c.annotate("", xy=(0, -0.85), xytext=(0, -0.6),
               arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2))

# East-West (RED)
for dx in [0.9, 1.35, 1.8]:
    # East inbound
    rect = FancyBboxPatch((dx, -0.18), 0.35, 0.36, boxstyle="round,pad=0.02",
                           facecolor="#EF9A9A", edgecolor="#333", linewidth=0.5,
                           alpha=0.7)
    ax_c.add_patch(rect)
    # West inbound
    rect = FancyBboxPatch((-dx - 0.35, -0.18), 0.35, 0.36, boxstyle="round,pad=0.02",
                           facecolor="#EF9A9A", edgecolor="#333", linewidth=0.5,
                           alpha=0.7)
    ax_c.add_patch(rect)

# EW flow arrows (dashed red — blocked)
ax_c.annotate("", xy=(0.6, 0), xytext=(0.85, 0),
               arrowprops=dict(arrowstyle="-|>", color="#D32F2F", lw=1.5,
                               linestyle="dashed"))
ax_c.annotate("", xy=(-0.85, 0), xytext=(-0.6, 0),
               arrowprops=dict(arrowstyle="-|>", color="#D32F2F", lw=1.5,
                               linestyle="dashed"))

# GREEN/RED labels outside the cells
ax_c.text(0.55, 2.35, "GREEN", ha="left", fontsize=7.5, color="#2E7D32",
          fontweight="bold")
ax_c.text(2.35, 0.5, "RED", ha="left", fontsize=7.5, color="#D32F2F",
          fontweight="bold")
ax_c.text(-2.3, 0.5, "RED", ha="right", fontsize=7.5, color="#D32F2F",
          fontweight="bold")
ax_c.text(0.55, -2.55, "GREEN", ha="left", fontsize=7.5, color="#2E7D32",
          fontweight="bold")

# Sigma label
ax_c.text(2.2, -2.0, r"$\sigma_m \in \{0, 1\}$", ha="center", fontsize=8.5,
          bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0F4FA",
                    edgecolor="#90A4C4", linewidth=0.4))

# N/S/E/W compass
ax_c.text(0, -2.55, "S", ha="center", fontsize=7, color="#999")
ax_c.text(0, 2.35, "N", ha="center", fontsize=7, color="#999")
ax_c.text(2.35, -0.45, "E", ha="center", fontsize=7, color="#999")
ax_c.text(-2.35, -0.45, "W", ha="center", fontsize=7, color="#999")

# Save
outpath = r"C:\Users\admin\Projects\69927a89543379cbbfcbc218\figures\ctm_explanation.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=300, pad_inches=0.05)
print(f"Saved: {outpath}")

# Also save PNG for preview
png_path = outpath.replace(".pdf", ".png")
fig.savefig(png_path, bbox_inches="tight", dpi=200, pad_inches=0.05)
print(f"Preview: {png_path}")
