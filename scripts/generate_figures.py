"""Generate all figures for the LightSim paper.

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
    """Figure 5: Cross-validation — throughput comparison on both scenarios."""
    cv_file = RESULTS / "cross_validation.json"
    if not cv_file.exists():
        print("  Skipping cross_validation.pdf (no cross-val data)")
        return

    with open(cv_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    scenarios = [
        ("single-intersection-v0", "Single Intersection"),
        ("grid-4x4-v0", "4x4 Grid"),
    ]
    ls_color = '#2196F3'
    sumo_color = '#E91E63'

    for ax_idx, (scenario, title) in enumerate(scenarios):
        ax = axes[ax_idx]
        scen_data = [r for r in data if r['scenario'] == scenario]

        # Collect (controller_name, lightsim_val, sumo_val) for shared controllers
        # Map LightSim names to short labels
        name_map = {
            'FixedTimeController': 'FixedTime',
            'MaxPressureController': 'MaxPressure',
            'SOTLController': 'SOTL',
        }
        shared = ['FixedTimeController', 'MaxPressureController', 'SOTLController']
        ls_only = ['WebsterController', 'LostTimeAwareMaxPressureController']
        sumo_only = ['ActuatedController']
        ls_only_labels = {'WebsterController': 'Webster',
                          'LostTimeAwareMaxPressureController': 'LT-Aware-MP'}

        # Build lists: first shared (paired bars), then LS-only, then SUMO-only
        labels = []
        ls_vals = []
        sumo_vals = []

        for ctrl in shared:
            labels.append(name_map[ctrl])
            ls_v = [r['total_exited'] for r in scen_data
                    if r['simulator'] == 'LightSim' and r['controller'] == ctrl]
            su_v = [r['total_exited'] for r in scen_data
                    if r['simulator'] == 'SUMO' and r['controller'] == ctrl]
            ls_vals.append(ls_v[0] if ls_v else 0)
            sumo_vals.append(su_v[0] if su_v else 0)

        for ctrl in ls_only:
            labels.append(ls_only_labels[ctrl])
            ls_v = [r['total_exited'] for r in scen_data
                    if r['simulator'] == 'LightSim' and r['controller'] == ctrl]
            ls_vals.append(ls_v[0] if ls_v else 0)
            sumo_vals.append(0)

        for ctrl in sumo_only:
            labels.append(ctrl.replace('Controller', ''))
            ls_vals.append(0)
            su_v = [r['total_exited'] for r in scen_data
                    if r['simulator'] == 'SUMO' and r['controller'] == ctrl]
            sumo_vals.append(su_v[0] if su_v else 0)

        x = np.arange(len(labels))
        width = 0.35
        bars_ls = ax.bar(x - width / 2, ls_vals, width, label='LightSim',
                         color=ls_color, edgecolor='white', linewidth=0.5)
        bars_su = ax.bar(x + width / 2, sumo_vals, width, label='SUMO',
                         color=sumo_color, edgecolor='white', linewidth=0.5)

        # Zoom y-axis to data range
        all_vals = [v for v in ls_vals + sumo_vals if v > 0]
        if all_vals:
            lo = min(all_vals) * 0.95
            hi = max(all_vals) * 1.03
            ax.set_ylim(lo, hi)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('Throughput (veh)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='lower left')

    fig.tight_layout()
    fig.savefig(OVERLEAF / "cross_validation.pdf")
    plt.close(fig)
    print("  Saved cross_validation.pdf")


def fig_mesoscopic_crossval():
    """Figure: Mesoscopic cross-validation — controller ranking across modes."""
    cv_file = RESULTS / "cross_validation_mesoscopic.json"
    if not cv_file.exists():
        print("  Skipping (no mesoscopic cross-val data)")
        return

    with open(cv_file) as f:
        data = json.load(f)

    # Focus on key controllers
    key_ctrls = ["FixedTime-30s", "SOTL", "MaxPressure-mg5",
                 "MaxPressure-mg15", "LT-Aware-MP-mg5"]
    short_labels = ["Fixed\nTime", "SOTL", "MP\nmg5", "MP\nmg15", "LT-Aware\nMP"]
    modes = ["default", "mesoscopic"]
    mode_labels = {"default": "Default", "mesoscopic": "Mesoscopic"}
    mode_colors = {"default": "#78909C", "mesoscopic": "#2196F3"}

    def _plot_crossval(scenario_data, title, out_name, show_sumo=True):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

        for ax_i, (metric, ylabel) in enumerate([
            ("total_exited", "Throughput (veh)"),
            ("avg_delay", "Avg Delay (s)"),
        ]):
            ax = axes[ax_i]
            x = np.arange(len(key_ctrls))
            width = 0.35

            all_vals = []
            for m_i, mode in enumerate(modes):
                vals = []
                for ctrl in key_ctrls:
                    row = next((r for r in scenario_data
                                if r["controller"] == ctrl and r["mode"] == mode), None)
                    vals.append(row[metric] if row else 0)
                all_vals.append(vals)
                offset = (m_i - 0.5) * width
                ax.bar(x + offset, vals, width,
                       label=mode_labels[mode], color=mode_colors[mode],
                       edgecolor="white", linewidth=0.5)

            # SUMO reference lines
            if show_sumo:
                for sumo_r in [r for r in scenario_data if r["mode"] == "sumo"]:
                    ctrl_short = "FT" if "Fixed" in sumo_r["controller"] else "MP"
                    val = sumo_r[metric]
                    ax.axhline(y=val, color="#F44336", linestyle="--",
                               linewidth=1, alpha=0.7)
                    ax.text(len(key_ctrls) - 0.5, val, f"SUMO {ctrl_short}",
                            fontsize=6, color="#F44336", va="bottom")

            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, fontsize=7)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis="y")

            # Zoom throughput y-axis to show real differences
            if metric == "total_exited":
                flat = [v for vs in all_vals for v in vs if v > 0]
                if show_sumo:
                    for sr in [r for r in scenario_data if r["mode"] == "sumo"]:
                        flat.append(sr[metric])
                if flat:
                    lo = min(flat) * 0.97
                    hi = max(flat) * 1.02
                    ax.set_ylim(lo, hi)
                    # Broken-axis indicator (two small diagonal marks)
                    d = 0.015
                    kw = dict(transform=ax.transAxes, color="k",
                              clip_on=False, linewidth=0.8)
                    ax.plot((-d, +d), (-d, +d), **kw)
                    ax.plot((1 - d, 1 + d), (-d, +d), **kw)

            if ax_i == 0:
                ax.legend(fontsize=8, loc="lower left")

        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(OVERLEAF / out_name)
        plt.close(fig)
        print(f"  Saved {out_name}")

    # Figure A: Single intersection
    si_data = [r for r in data if r["scenario"] == "single-intersection-v0"]
    _plot_crossval(si_data,
                   "Single Intersection: Default vs Mesoscopic",
                   "meso_crossval_single.pdf", show_sumo=True)

    # Figure B: Grid 4x4
    grid_data = [r for r in data if r["scenario"] == "grid-4x4-v0"]
    if grid_data:
        _plot_crossval(grid_data,
                       "Grid 4\u00d74: Default vs Mesoscopic",
                       "meso_crossval_grid.pdf", show_sumo=True)


def fig_mesoscopic_rl():
    """Figure: RL learning curves — default vs mesoscopic."""
    rl_file = RESULTS / "rl_mesoscopic_experiment.json"
    if not rl_file.exists():
        print("  Skipping (no RL mesoscopic data)")
        return

    with open(rl_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)

    algo_colors = {"DQN": "#2196F3", "PPO": "#4CAF50"}
    baseline_styles = {
        "FixedTime": ("#FF9800", "--"),
        "MaxPressure-mg15": ("#9C27B0", "--"),
        "LT-Aware-MP-mg5": ("#F44336", ":"),
    }

    # Collect final converged values (baselines + last RL eval) for y-range
    ylim_vals = []

    for ax_i, (mode, title) in enumerate([
        ("default", "Default Mode"),
        ("mesoscopic", "Mesoscopic Mode"),
    ]):
        ax = axes[ax_i]
        mode_data = data.get(mode, {})
        rl_data = mode_data.get("rl", {})
        bl_data = mode_data.get("baselines", {})

        for algo in ["DQN", "PPO"]:
            if algo not in rl_data:
                continue
            seeds = rl_data[algo]
            timesteps = seeds[0]["timesteps"]
            all_rewards = np.array([s["mean_rewards"] for s in seeds])
            mean = all_rewards.mean(axis=0)
            std = all_rewards.std(axis=0)

            ax.plot(timesteps, mean, color=algo_colors[algo],
                    linewidth=2, label=algo)
            ax.fill_between(timesteps, mean - std, mean + std,
                            color=algo_colors[algo], alpha=0.15)
            # Only use final converged value for y-range
            ylim_vals.append(mean[-1])
            ylim_vals.append(mean[-1] - std[-1])

        for bl_name, (color, ls) in baseline_styles.items():
            if bl_name in bl_data:
                val = bl_data[bl_name]["avg_reward"]
                short = bl_name.replace("MaxPressure-mg15", "MP-15").replace(
                    "LT-Aware-MP-mg5", "LT-MP")
                ax.axhline(y=val, color=color, linestyle=ls,
                           linewidth=1.2, label=short)
                ylim_vals.append(val)

        ax.set_xlabel("Training Timesteps")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    # Zoom y-axis to converged region — exclude early -170 exploration dip
    if ylim_vals:
        worst = min(ylim_vals)
        best = max(ylim_vals)
        margin = (best - worst) * 0.25
        axes[0].set_ylim(worst - margin, best + margin)

    axes[0].set_ylabel("Per-Step Reward")
    fig.suptitle("RL Training: Default vs Mesoscopic", fontsize=11)
    fig.tight_layout()
    fig.savefig(OVERLEAF / "meso_rl_curves.pdf")
    plt.close(fig)
    print("  Saved meso_rl_curves.pdf")


def fig_mesoscopic_summary():
    """Figure: Combined summary — final reward bar chart across all methods."""
    rl_file = RESULTS / "rl_mesoscopic_experiment.json"
    if not rl_file.exists():
        print("  Skipping (no RL mesoscopic data)")
        return

    with open(rl_file) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Collect all methods and their rewards for both modes
    # Exclude MP-mg5 — its -143 mesoscopic collapse compresses the useful range
    methods = []
    default_vals = []
    default_errs = []
    meso_vals = []
    meso_errs = []

    for algo in ["DQN", "PPO"]:
        methods.append(algo)
        for mode, vals_list, errs_list in [
            ("default", default_vals, default_errs),
            ("mesoscopic", meso_vals, meso_errs),
        ]:
            seeds = data[mode]["rl"][algo]
            rewards = [s["final_eval_reward"] for s in seeds]
            vals_list.append(float(np.mean(rewards)))
            errs_list.append(float(np.std(rewards)))

    for bl_name in ["MaxPressure-mg15", "LT-Aware-MP-mg5", "FixedTime"]:
        methods.append(bl_name.replace("MaxPressure-mg15", "MP-mg15").replace(
            "LT-Aware-MP-mg5", "LT-Aware-MP"))
        for mode, vals_list, errs_list in [
            ("default", default_vals, default_errs),
            ("mesoscopic", meso_vals, meso_errs),
        ]:
            bl = data[mode]["baselines"].get(bl_name, {})
            vals_list.append(bl.get("avg_reward", 0))
            errs_list.append(bl.get("std_reward", 0))

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, default_vals, width, yerr=default_errs,
           label="Default", color="#78909C", edgecolor="white",
           linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})
    ax.bar(x + width / 2, meso_vals, width, yerr=meso_errs,
           label="Mesoscopic", color="#2196F3", edgecolor="white",
           linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})

    # Add value labels on bars
    for i, (dv, mv) in enumerate(zip(default_vals, meso_vals)):
        ax.text(i - width / 2, dv - 0.3, f"{dv:.1f}", ha="center",
                va="top", fontsize=6, color="white", fontweight="bold")
        ax.text(i + width / 2, mv - 0.3, f"{mv:.1f}", ha="center",
                va="top", fontsize=6, color="white", fontweight="bold")

    # Note about excluded controller
    ax.annotate("MP-mg5 excluded\n(meso: \u2212143)",
                xy=(0.98, 0.02), xycoords="axes fraction",
                fontsize=6, color="#999", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Per-Step Reward (higher = better)")
    ax.set_title("Single Intersection: All Controllers (Default vs Mesoscopic)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OVERLEAF / "meso_summary.pdf")
    plt.close(fig)
    print("  Saved meso_summary.pdf")


def fig_rl_crossval():
    """Figure: RL cross-validation — rank agreement between LightSim and SUMO."""
    rl_file = RESULTS / "rl_crossval_results.json"
    if not rl_file.exists():
        print("  Skipping (no RL crossval data)")
        return

    with open(rl_file) as f:
        data = json.load(f)

    # Separate by simulator
    sims = {}
    for r in data:
        if "error" in r:
            continue
        key = (r["simulator"], r["variant"])
        sims.setdefault(key, []).append(r)

    # Get variant lists — split by reward type for ranking
    default_variants = ["DQN", "PPO", "A2C"]
    pressure_variants = ["DQN-pressure", "PPO-pressure"]
    all_variants = default_variants + pressure_variants

    # Check we have both simulators
    has_sumo = any(r["simulator"] == "SUMO" for r in data if "error" not in r)
    if not has_sumo:
        print("  Skipping (no SUMO data yet)")
        return

    # Compute mean reward and median time per (simulator, variant)
    stats = {}
    for (sim, var), runs in sims.items():
        rewards = [r["eval_reward_mean"] for r in runs]
        times = np.array([r["train_time"] for r in runs])
        stats[(sim, var)] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "time_median": np.median(times),
            "time_q25": np.percentile(times, 25),
            "time_q75": np.percentile(times, 75),
        }

    # --- Figure: 2 panels ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Normalized rank comparison
    # Rank within each simulator (lower reward rank = worse)
    for group_name, variants in [("Default reward", default_variants),
                                  ("Pressure reward", pressure_variants)]:
        for sim in ["LightSim", "SUMO"]:
            available = [v for v in variants if (sim, v) in stats]
            if len(available) < 2:
                continue
            # Sort by mean reward (descending = rank 1 is best)
            sorted_v = sorted(available, key=lambda v: -stats[(sim, v)]["mean"])
            for rank, v in enumerate(sorted_v):
                stats[(sim, v)]["rank"] = rank + 1

    # Scatter: LightSim rank vs SUMO rank
    for v in all_variants:
        ls_key = ("LightSim", v)
        su_key = ("SUMO", v)
        if ls_key in stats and su_key in stats and "rank" in stats[ls_key] and "rank" in stats[su_key]:
            color = "#2196F3" if "pressure" not in v else "#FF9800"
            marker = "o" if "pressure" not in v else "s"
            ax1.scatter(stats[ls_key]["rank"], stats[su_key]["rank"],
                       c=color, marker=marker, s=100, zorder=3, edgecolors="white")
            ax1.annotate(v, (stats[ls_key]["rank"], stats[su_key]["rank"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=7)

    max_rank = max(len(default_variants), len(pressure_variants)) + 0.5
    ax1.plot([0.5, max_rank], [0.5, max_rank], "--", color="#ccc", zorder=1)
    ax1.set_xlabel("LightSim Rank")
    ax1.set_ylabel("SUMO Rank")
    ax1.set_title("Variant Ranking Agreement")
    ax1.set_xlim(0.5, max_rank)
    ax1.set_ylim(0.5, max_rank)
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="Default reward"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF9800",
               markersize=8, label="Pressure reward"),
    ]
    ax1.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Training time comparison (speedup) — median + IQR
    available_variants = [v for v in all_variants
                         if ("LightSim", v) in stats and ("SUMO", v) in stats]
    x = np.arange(len(available_variants))
    ls_times = [stats[("LightSim", v)]["time_median"] for v in available_variants]
    su_times = [stats[("SUMO", v)]["time_median"] for v in available_variants]
    # IQR as asymmetric error bars: [median - q25, q75 - median]
    ls_err_lo = [stats[("LightSim", v)]["time_median"] - stats[("LightSim", v)]["time_q25"]
                 for v in available_variants]
    ls_err_hi = [stats[("LightSim", v)]["time_q75"] - stats[("LightSim", v)]["time_median"]
                 for v in available_variants]
    su_err_lo = [stats[("SUMO", v)]["time_median"] - stats[("SUMO", v)]["time_q25"]
                 for v in available_variants]
    su_err_hi = [stats[("SUMO", v)]["time_q75"] - stats[("SUMO", v)]["time_median"]
                 for v in available_variants]

    width = 0.35
    ax2.bar(x - width / 2, ls_times, width, yerr=[ls_err_lo, ls_err_hi],
            label="LightSim", color="#4CAF50", edgecolor="white",
            linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})
    ax2.bar(x + width / 2, su_times, width, yerr=[su_err_lo, su_err_hi],
            label="SUMO", color="#FF5722", edgecolor="white",
            linewidth=0.5, capsize=3, error_kw={"linewidth": 0.8})

    # Speedup annotations
    for i, v in enumerate(available_variants):
        if ls_times[i] > 0:
            speedup = su_times[i] / ls_times[i]
            bar_top = max(ls_times[i] + ls_err_hi[i], su_times[i] + su_err_hi[i])
            ax2.text(i, bar_top + 20,
                    f"{speedup:.0f}×", ha="center", fontsize=7, color="#333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(available_variants, fontsize=8, rotation=25, ha="right")
    ax2.set_ylabel("Training Time (s)")
    ax2.set_title("Training Speed: LightSim vs SUMO")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OVERLEAF / "rl_crossval.pdf")
    plt.close(fig)
    print("  Saved rl_crossval.pdf")


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

    print("\nFigure: Mesoscopic cross-validation (single)")
    fig_mesoscopic_crossval()

    print("Figure: Mesoscopic RL curves")
    fig_mesoscopic_rl()

    print("Figure: Mesoscopic summary bar chart")
    fig_mesoscopic_summary()

    print("\nFigure: RL cross-validation")
    fig_rl_crossval()

    print("\nDone! Figures saved to:", OVERLEAF)
