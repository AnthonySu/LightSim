"""Transfer Experiment: Train RL in LightSim, evaluate learned timing in SUMO.

Demonstrates that policies learned in LightSim produce meaningful signal
timings that transfer to SUMO.

Steps:
  1. Train DQN in LightSim on single-intersection-v0 for 100k timesteps.
  2. Record the learned policy over one evaluation episode (720 RL steps).
  3. Compute learned signal timing statistics.
  4. Replay the exact phase schedule in SUMO.
  5. Compare against default FixedTime (30s/30s cycle) in SUMO.
  6. Save results to results/transfer_experiment.json.

Usage::

    cd lightsim
    python scripts/transfer_experiment.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===========================================================================
# Step 1: Train DQN in LightSim
# ===========================================================================

def train_dqn(total_timesteps=100_000, seed=42):
    """Train a DQN agent on single-intersection-v0 and return the model."""
    from stable_baselines3 import DQN
    from lightsim import make

    print(f"[1/6] Training DQN for {total_timesteps:,} timesteps ...", flush=True)

    env = make(
        "single-intersection-v0",
        sim_steps_per_action=5,
        max_steps=720,          # 720 RL steps x 5 sim steps = 3600 sim seconds
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    env.close()

    print("    Training complete.", flush=True)
    return model


# ===========================================================================
# Step 2: Record the learned policy
# ===========================================================================

def record_policy(model, seed=123):
    """Run one evaluation episode, recording (timestep, phase_action) pairs.

    Returns
    -------
    actions : list[int]
        Phase action at each of the 720 RL decision steps.
    total_reward : float
        Cumulative reward over the episode.
    """
    from lightsim import make

    print("[2/6] Recording learned policy over 1 evaluation episode ...", flush=True)

    env = make(
        "single-intersection-v0",
        sim_steps_per_action=5,
        max_steps=720,
    )
    obs, info = env.reset(seed=seed)

    actions = []
    total_reward = 0.0

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    print(f"    Recorded {len(actions)} decision steps, "
          f"total reward = {total_reward:.2f}", flush=True)
    return actions, total_reward


# ===========================================================================
# Step 3: Compute learned signal timing statistics
# ===========================================================================

def compute_timing_stats(actions, sim_steps_per_action=5, dt=1.0):
    """Analyse the recorded action sequence.

    Returns
    -------
    stats : dict
        Keys: phase_0_avg_green, phase_1_avg_green, num_switches
    schedule : list of (start_time, end_time, phase) tuples in sim seconds.
    """
    print("[3/6] Computing learned signal timing ...", flush=True)

    n_phases = max(actions) + 1
    phase_durations = {p: [] for p in range(n_phases)}
    schedule = []

    current_phase = actions[0]
    run_start = 0
    num_switches = 0

    for i in range(1, len(actions)):
        if actions[i] != current_phase:
            dur = (i - run_start) * sim_steps_per_action * dt
            phase_durations[current_phase].append(dur)
            schedule.append((
                run_start * sim_steps_per_action * dt,
                i * sim_steps_per_action * dt,
                current_phase,
            ))
            num_switches += 1
            current_phase = actions[i]
            run_start = i

    # Close out the last run
    dur = (len(actions) - run_start) * sim_steps_per_action * dt
    phase_durations[current_phase].append(dur)
    schedule.append((
        run_start * sim_steps_per_action * dt,
        len(actions) * sim_steps_per_action * dt,
        current_phase,
    ))

    stats = {"num_switches": num_switches}
    for p in range(n_phases):
        durations = phase_durations[p]
        avg = float(np.mean(durations)) if durations else 0.0
        stats[f"phase_{p}_avg_green"] = round(avg, 2)
        print(f"    Phase {p}: avg green = {avg:.1f}s  "
              f"({len(durations)} intervals)", flush=True)

    print(f"    Total phase switches: {num_switches}", flush=True)
    return stats, schedule



# ===========================================================================
# Step 4 & 5: Run SUMO with FixedTime and RL-learned timing
# ===========================================================================

def _find_sumo_binary():
    """Locate the SUMO executable."""
    from lightsim.benchmarks.sumo_comparison import _find_sumo_binary as _find
    return _find()


def _build_sumo_intersection(tmpdir, sumo_bin, sim_seconds=3600):
    """Build SUMO single intersection, reusing cross_validation logic."""
    from lightsim.benchmarks.cross_validation import (
        _build_sumo_single_intersection,
    )
    return _build_sumo_single_intersection(
        tmpdir, sumo_bin, "FixedTimeController", sim_seconds
    )


def _collect_sumo_metrics(sim_seconds=3600):
    """Collect throughput / delay / queue from a running traci session.

    Must be called after traci.start() and before traci.close().
    """
    import traci

    total_arrived = 0
    cumulative_waiting = 0.0
    step_queues = []

    for step in range(sim_seconds):
        traci.simulationStep()
        total_arrived += traci.simulation.getArrivedNumber()

        queue = 0
        for vid in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(vid) < 0.1:
                queue += 1
                cumulative_waiting += traci.vehicle.getWaitingTime(vid)
        step_queues.append(queue)

    n_remaining = len(traci.vehicle.getIDList())
    throughput = total_arrived
    avg_queue = float(np.mean(step_queues)) if step_queues else 0.0
    n_vehicles = throughput + n_remaining
    avg_delay = cumulative_waiting / max(n_vehicles, 1)

    return {
        "throughput": throughput,
        "delay": round(avg_delay, 2),
        "queue": round(avg_queue, 2),
    }


def run_sumo_fixed_time(sim_seconds=3600):
    """Run SUMO with the default FixedTime controller (30s/30s)."""
    import traci

    print("[4/6] Running SUMO with FixedTime (30s/30s) ...", flush=True)

    sumo_bin = _find_sumo_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _build_sumo_intersection(Path(tmpdir), sumo_bin, sim_seconds)
        traci.start([sumo_bin, "-c", str(cfg), "--no-warnings", "true"])

        # Override TLS with explicit 30s/30s cycle
        tls_ids = traci.trafficlight.getIDList()
        if tls_ids:
            tls_id = tls_ids[0]
            logic = traci.trafficlight.getAllProgramLogics(tls_id)
            if logic:
                existing = logic[0].phases
                greens = [p.state for p in existing
                          if "G" in p.state or "g" in p.state]

                if len(greens) >= 2:
                    nl = len(greens[0])
                    y0 = greens[0].replace("G", "y").replace("g", "y")
                    y1 = greens[1].replace("G", "y").replace("g", "y")
                    ar = "r" * nl

                    phases = [
                        traci.trafficlight.Phase(30, greens[0]),
                        traci.trafficlight.Phase(3, y0),
                        traci.trafficlight.Phase(2, ar),
                        traci.trafficlight.Phase(30, greens[1]),
                        traci.trafficlight.Phase(3, y1),
                        traci.trafficlight.Phase(2, ar),
                    ]
                    prog = traci.trafficlight.Logic(
                        "fixed30", 0, 0, phases
                    )
                    traci.trafficlight.setProgramLogic(tls_id, prog)
                    traci.trafficlight.setProgram(tls_id, "fixed30")

        metrics = _collect_sumo_metrics(sim_seconds)
        traci.close()

    tp = metrics["throughput"]
    dl = metrics["delay"]
    qq = metrics["queue"]
    print(f"    FixedTime: throughput={tp}, delay={dl:.1f}, "
          f"queue={qq:.1f}", flush=True)
    return metrics


def run_sumo_rl_timing(schedule, sim_seconds=3600):
    """Run SUMO replaying the exact RL-learned phase schedule.

    At each simulation second, look up which phase the RL agent selected
    and force that phase via traci.trafficlight.setPhase().
    """
    import traci

    print("[5/6] Running SUMO with RL-learned timing ...", flush=True)

    sumo_bin = _find_sumo_binary()

    # Pre-compute lookup: sim_second -> phase index
    phase_at_second = [0] * sim_seconds
    for start, end, phase in schedule:
        for t in range(int(start), min(int(end), sim_seconds)):
            phase_at_second[t] = phase

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _build_sumo_intersection(Path(tmpdir), sumo_bin, sim_seconds)
        traci.start([sumo_bin, "-c", str(cfg), "--no-warnings", "true"])

        tls_ids = traci.trafficlight.getIDList()
        tls_id = tls_ids[0] if tls_ids else None

        # Discover green phase states from the default program
        greens = []
        if tls_id:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)
            if logic:
                greens = [p.state for p in logic[0].phases
                          if "G" in p.state or "g" in p.state]

        # Build custom TLS program with long green phases for manual control
        if tls_id and len(greens) >= 2:
            nl = len(greens[0])
            y0 = greens[0].replace("G", "y").replace("g", "y")
            y1 = greens[1].replace("G", "y").replace("g", "y")
            ar = "r" * nl

            phases = [
                traci.trafficlight.Phase(9999, greens[0]),   # idx 0: NS green
                traci.trafficlight.Phase(3, y0),             # idx 1: yellow
                traci.trafficlight.Phase(2, ar),             # idx 2: all-red
                traci.trafficlight.Phase(9999, greens[1]),   # idx 3: EW green
                traci.trafficlight.Phase(3, y1),             # idx 4: yellow
                traci.trafficlight.Phase(2, ar),             # idx 5: all-red
            ]
            prog = traci.trafficlight.Logic("rl_replay", 0, 0, phases)
            traci.trafficlight.setProgramLogic(tls_id, prog)
            traci.trafficlight.setProgram(tls_id, "rl_replay")

        # LightSim phase -> SUMO phase index mapping
        # phase 0 -> SUMO idx 0 (NS green)
        # phase 1 -> SUMO idx 3 (EW green)
        ls_to_sumo = {0: 0, 1: 3}

        total_arrived = 0
        cumulative_waiting = 0.0
        step_queues = []
        prev_phase = -1

        for step in range(sim_seconds):
            desired = phase_at_second[step]
            sumo_target = ls_to_sumo.get(desired, 0)

            if tls_id:
                if desired != prev_phase:
                    traci.trafficlight.setPhase(tls_id, sumo_target)
                prev_phase = desired

            traci.simulationStep()
            total_arrived += traci.simulation.getArrivedNumber()

            queue = 0
            for vid in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(vid) < 0.1:
                    queue += 1
                    cumulative_waiting += traci.vehicle.getWaitingTime(vid)
            step_queues.append(queue)

        n_remaining = len(traci.vehicle.getIDList())
        traci.close()

    throughput = total_arrived
    avg_queue = float(np.mean(step_queues)) if step_queues else 0.0
    n_vehicles = throughput + n_remaining
    avg_delay = cumulative_waiting / max(n_vehicles, 1)

    metrics = {
        "throughput": throughput,
        "delay": round(avg_delay, 2),
        "queue": round(avg_queue, 2),
    }

    tp = metrics["throughput"]
    dl = metrics["delay"]
    qq = metrics["queue"]
    print(f"    RL timing: throughput={tp}, delay={dl:.1f}, "
          f"queue={qq:.1f}", flush=True)
    return metrics


# ===========================================================================
# Step 6: Save results
# ===========================================================================

def save_results(lightsim_reward, timing_stats, sumo_fixed, sumo_rl,
                 output_path):
    """Write results to JSON."""
    print(f"[6/6] Saving results to {output_path} ...", flush=True)

    results = {
        "lightsim_rl_reward": round(lightsim_reward, 4),
        "learned_timing": timing_stats,
        "sumo_fixed": sumo_fixed,
        "sumo_rl_timing": sumo_rl,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("    Saved.", flush=True)
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Transfer Experiment: LightSim RL -> SUMO")
    print("=" * 70)
    print()

    # Step 1: Train
    model = train_dqn(total_timesteps=100_000)

    # Step 2: Record
    actions, total_reward = record_policy(model)

    # Step 3: Compute timing
    timing_stats, schedule = compute_timing_stats(actions)

    # Step 4 & 5: Run SUMO
    sumo_fixed = run_sumo_fixed_time(sim_seconds=3600)
    sumo_rl = run_sumo_rl_timing(schedule, sim_seconds=3600)

    # Step 6: Save
    output_path = _PROJECT_ROOT / "results" / "transfer_experiment.json"
    results = save_results(
        total_reward, timing_stats, sumo_fixed, sumo_rl, output_path
    )

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    lr = results["lightsim_rl_reward"]
    print(f"  LightSim RL reward:     {lr}")
    print("  Learned timing:")
    for k, v in results["learned_timing"].items():
        print(f"    {k}: {v}")
    print("  SUMO FixedTime (30/30):")
    for k, v in results["sumo_fixed"].items():
        print(f"    {k}: {v}")
    print("  SUMO RL-learned timing:")
    for k, v in results["sumo_rl_timing"].items():
        print(f"    {k}: {v}")

    # Improvement analysis
    sf_tp = sumo_fixed["throughput"]
    sr_tp = sumo_rl["throughput"]
    sf_dl = sumo_fixed["delay"]
    sr_dl = sumo_rl["delay"]
    sf_qq = sumo_fixed["queue"]
    sr_qq = sumo_rl["queue"]

    if sf_tp > 0:
        tp_diff = sr_tp - sf_tp
        tp_pct = 100 * tp_diff / sf_tp
        print(f"\n  Throughput change: {tp_diff:+.0f} ({tp_pct:+.1f}%)")
    if sf_dl > 0:
        d_diff = sr_dl - sf_dl
        d_pct = 100 * d_diff / sf_dl
        print(f"  Delay change:      {d_diff:+.1f} ({d_pct:+.1f}%)")
    if sf_qq > 0:
        q_diff = sr_qq - sf_qq
        q_pct = 100 * q_diff / sf_qq
        print(f"  Queue change:      {q_diff:+.1f} ({q_pct:+.1f}%)")

    print()


if __name__ == "__main__":
    main()
