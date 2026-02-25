# Experiment Scripts

Scripts for reproducing all paper results. Each script saves results to `results/` as JSON and can be run independently.

## Paper Figures

| Figure | Description | Script | Est. Runtime |
|--------|-------------|--------|-------------|
| Fig. 1 | CTM mechanics diagram | `gen_fig1_matplotlib.py` | 5 min |
| Fig. 2 | Architecture diagram | (TikZ in paper) | — |
| Fig. 3 | OSM city networks | `gen_osm_figure.py` | 2 min |
| Fig. 4 | Grid visualization | `generate_figures.py` → `figure_grid_viz()` | 1 min |
| Fig. 5 | Simulation dynamics | `generate_figures.py` → `figure_visualization()` | 1 min |
| Fig. 6 | Fundamental diagram | `validate_fd_full.py` | 1 min |
| Fig. 7 | Cross-validation bars | `generate_figures.py` → `figure_cross_validation()` | 1 min |
| Fig. 8 | RL learning curves | `generate_figures.py` → `figure_learning_curves()` | 1 min |
| Fig. 9 | RL cross-validation | `rl_cross_validation.py` | 4 hours |
| Fig. 10 | Sample efficiency | `sample_efficiency.py` | 2 hours |
| Fig. 11 | Meso cross-validation | `generate_figures.py` → `figure_meso_crossval()` | 1 min |
| Fig. 12 | Meso RL curves | `generate_figures.py` → `figure_meso_rl()` | 1 min |

## Paper Tables

| Table | Description | Script | Est. Runtime |
|-------|-------------|--------|-------------|
| Tab. 1 | Speed benchmarks | `python -m lightsim.benchmarks.speed_benchmark` | 2 min |
| Tab. 2 | LightSim vs SUMO speed | `python -m lightsim.benchmarks.sumo_comparison` | 10 min |
| Tab. 3 | RL baselines | `python -m lightsim.benchmarks.rl_baselines --train-rl` | 15 min |
| Tab. 4 | Cross-simulator validation | (results in `cross_validation.json`) | — |
| Tab. 5 | Arterial cross-validation | `arterial_cross_validation.py` | 30 min |
| Tab. 6 | Reward ablation | `reward_ablation.py` | 10 min |
| Tab. 7 | Demand sensitivity | `oversaturated_experiment.py` | 10 min |
| Tab. 8 | Multi-agent DQN | `multi_agent_rl.py` | 30 min |
| Tab. 9 | Multi-agent DT | `dt_evaluate_multi.py` | 2 hours |
| Tab. 10 | Mesoscopic validation | `cross_validation_mesoscopic.py` | 20 min |
| Tab. 11 | Mesoscopic RL | `rl_mesoscopic_experiment.py` | 30 min |
| Tab. 12 | OSM city evaluation | `osm_city_evaluation.py` | 2 hours |

## All Figures at Once

```bash
python scripts/generate_figures.py   # generates all figures to Overleaf figures/
```

## Pretrained Weights

```bash
python scripts/train_pretrained.py          # train and save checkpoints
python scripts/evaluate_pretrained.py       # evaluate saved checkpoints
```

## Requirements

- Base: `pip install lightsim`
- Full: `pip install lightsim[all]`
- SUMO scripts: requires SUMO v1.26+ installed separately
- DT scripts: requires `pip install lightsim[dt]` (PyTorch)
- All scripts tested on Python 3.10–3.13, Windows 11 + Ubuntu 22.04
