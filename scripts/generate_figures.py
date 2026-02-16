"""Generate all figures for the LightSim NeurIPS paper.

Usage::
    python scripts/generate_figures.py

Outputs PDF figures to the Overleaf project figures/ directory.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Output directory
OVERLEAF = Path(r"C:\Users\admin\Projects\69927a89543379cbbfcbc218\figures")
RESULTS = Path(r"C:\Users\admin\Projects\lightsim\results")
OVERLEAF.mkdir(parents=True, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def fig_fundamental_diagram():
    """Figure 2: Fundamental diagram validation."""
    with open(RESULTS / "fundamental_diagram.json") as f:
        data = json.load(f)

    densities = np.array(data['densities'])
    flows = np.array(data['flows'])
    k_theory = np.array(data['k_theory'])
    q_theory = np.array(data['q_theory'])
    r_squared = data['r_squared']

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(k_theory * 1000, q_theory, 'r-', linewidth=2, label='Theoretical', zorder=2)
    ax.scatter(densities * 1000, flows, c='steelblue', s=30, alpha=0.8,
               edgecolors='navy', linewidth=0.5, label=f'LightSim ($R^2={r_squared:.3f}$)', zorder=3)

    ax.set_xlabel('Density (veh/km/lane)')
    ax.set_ylabel('Flow (veh/s/lane)')
    ax.set_title('Fundamental Diagram Validation')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.savefig(OVERLEAF / "fundamental_diagram.pdf")
    plt.close(fig)
    print("  Saved fundamental_diagram.pdf")


def fig_speed_comparison():
    """Figure 3: Speed benchmark bar chart."""
    with open(RESULTS / "speed_benchmark.json") as f:
        data = json.load(f)

    scenarios = data['scenarios']
    names = [s['name'] for s in scenarios]
    steps_per_sec = [s['steps_per_sec'] for s in scenarios]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ['#2196F3' if 'grid' in n else '#4CAF50' if 'arterial' in n else '#FF9800'
              for n in names]
    bars = ax.bar(range(len(names)), steps_per_sec, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, steps_per_sec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('single-intersection', '1-intx') for n in names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Simulation Steps / Second')
    ax.set_title('LightSim Throughput by Network Size')
    ax.set_yscale('log')
    ax.set_ylim(500, 40000)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF9800', label='Single intersection'),
        Patch(facecolor='#2196F3', label='Grid'),
        Patch(facecolor='#4CAF50', label='Arterial'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.savefig(OVERLEAF / "speed_benchmark.pdf")
    plt.close(fig)
    print("  Saved speed_benchmark.pdf")


def fig_sumo_comparison():
    """Figure: LightSim vs SUMO speed comparison."""
    with open(RESULTS / "sumo_comparison.json") as f:
        data = json.load(f)

    scenarios = data['scenarios']
    # Only show subset for clarity
    selected = [s for s in scenarios if s['name'] in
                ['single-intersection', 'grid-2x2', 'grid-4x4',
                 'arterial-3', 'arterial-5', 'arterial-10', 'arterial-20']]

    names = [s['name'] for s in selected]
    ls_sps = [s['ls_sps'] for s in selected]
    sumo_sps = [s['sumo_sps'] for s in selected]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars1 = ax.bar(x - width/2, ls_sps, width, label='LightSim', color='#2196F3',
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, sumo_sps, width, label='SUMO (standalone)', color='#FF5722',
                   edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('single-intersection', '1-intx') for n in names],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Simulation Steps / Second')
    ax.set_title('LightSim vs SUMO Speed Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(OVERLEAF / "sumo_comparison.pdf")
    plt.close(fig)
    print("  Saved sumo_comparison.pdf")


def fig_learning_curves():
    """Figure 4: RL learning curves with error bands."""
    rl_file = RESULTS / "rl_training.json"
    if not rl_file.exists():
        print("  Skipping learning_curves.pdf (no RL training data yet)")
        return

    with open(rl_file) as f:
        data = json.load(f)

    EPISODE_LEN = 720  # RL eval episode length

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    colors = {'DQN': '#2196F3', 'PPO': '#4CAF50'}

    for algo_name in ['DQN', 'PPO']:
        if algo_name not in data['rl']:
            continue

        seeds = data['rl'][algo_name]
        all_timesteps = seeds[0]['timesteps']
        all_rewards = []
        for s in seeds:
            # Convert total episode reward → per-step reward
            all_rewards.append([r / EPISODE_LEN for r in s['mean_rewards']])

        all_rewards = np.array(all_rewards)
        mean = all_rewards.mean(axis=0)
        std = all_rewards.std(axis=0)

        ax.plot(all_timesteps, mean, color=colors[algo_name], linewidth=2, label=algo_name)
        ax.fill_between(all_timesteps, mean - std, mean + std,
                        color=colors[algo_name], alpha=0.2)

    # Add baseline lines (already per-step)
    baselines = data.get('baselines', {})
    if 'FixedTime' in baselines:
        ax.axhline(y=baselines['FixedTime']['avg_reward'], color='#FF9800',
                   linestyle='--', linewidth=1.5, label='FixedTime')
    if 'MaxPressure' in baselines:
        ax.axhline(y=baselines['MaxPressure']['avg_reward'], color='#F44336',
                   linestyle='--', linewidth=1.5, label='MaxPressure')

    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Per-Step Reward (queue)')
    ax.set_title('RL Training Progress on Single Intersection')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.savefig(OVERLEAF / "learning_curves.pdf")
    plt.close(fig)
    print("  Saved learning_curves.pdf")


def fig_cross_validation():
    """Figure 5: Cross-validation metric comparison."""
    cv_file = RESULTS / "cross_validation.json"
    if not cv_file.exists():
        print("  Skipping cross_validation.pdf (no cross-val data)")
        return

    with open(cv_file) as f:
        data = json.load(f)

    # Organize data
    metrics = {}
    for r in data:
        key = (r['simulator'], r['controller'])
        metrics[key] = r

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    simulators = ['LightSim', 'SUMO']
    controllers = ['FixedTimeController', 'MaxPressureController']
    ctrl_labels = ['FixedTime', 'MaxPressure']
    colors = ['#2196F3', '#4CAF50']

    x = np.arange(len(simulators))
    width = 0.3

    for ax_idx, (metric, label) in enumerate([
        ('total_exited', 'Throughput (veh)'),
        ('avg_delay', 'Avg Delay (s)'),
        ('total_queue', 'Queue (veh)'),
    ]):
        ax = axes[ax_idx]
        for i, (ctrl, ctrl_label) in enumerate(zip(controllers, ctrl_labels)):
            vals = [metrics.get((sim, ctrl), {}).get(metric, 0) for sim in simulators]
            ax.bar(x + (i - 0.5) * width, vals, width, label=ctrl_label,
                   color=colors[i], edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(simulators, fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        if ax_idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Cross-Simulator Validation', fontsize=11)
    fig.tight_layout()
    fig.savefig(OVERLEAF / "cross_validation.pdf")
    plt.close(fig)
    print("  Saved cross_validation.pdf")


if __name__ == "__main__":
    print("Generating figures for LightSim paper...\n")

    print("Figure 2: Fundamental diagram")
    fig_fundamental_diagram()

    print("Figure 3: Speed benchmark")
    fig_speed_comparison()

    print("Figure: SUMO comparison")
    fig_sumo_comparison()

    print("Figure 4: Learning curves")
    fig_learning_curves()

    print("Figure 5: Cross-validation")
    fig_cross_validation()

    print("\nDone! Figures saved to:", OVERLEAF)
