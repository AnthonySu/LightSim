# Test Coverage Analysis

**Date**: 2026-03-17
**Overall Coverage**: 53% (1,824 statements missed out of 3,884)

## Current Coverage by Module

| Module | Stmts | Miss | Coverage |
|--------|-------|------|----------|
| `core/` | 874 | 69 | **92%** |
| `envs/` | 393 | 21 | **95%** |
| `networks/` | 426 | 176 | **59%** |
| `benchmarks/` | 940 | 691 | **26%** |
| `dt/` | 558 | 526 | **6%** |
| `viz/` | 365 | 268 | **27%** |
| `utils/` | 228 | 55 | **76%** |
| `io/` | 49 | 0 | **100%** |
| `pretrained.py` | 25 | 18 | **28%** |

## Priority Areas for Test Improvement

### 1. Decision Transformer Module — `dt/` (6% coverage, 526 lines uncovered)

This is the **largest coverage gap** in the codebase. Four files have 0% or near-0% coverage:

| File | Coverage | Lines Uncovered |
|------|----------|----------------|
| `dt/model.py` | 0% | All 95 statements — the PyTorch model architecture |
| `dt/train.py` | 0% | All 112 statements — the training loop |
| `dt/controller.py` | 0% | All 141 statements — the DT-based signal controller |
| `dt/dataset.py` | 15% | 174 of 204 statements — expert trajectory dataset |

**Recommended tests:**
- **`dt/model.py`**: Test `DecisionTransformerModel` forward pass with known input shapes, verify output dimensions, test with different sequence lengths, test weight initialization
- **`dt/train.py`**: Test training loop runs for 1 epoch on a tiny synthetic dataset without errors, test checkpoint saving/loading, test learning rate scheduling
- **`dt/controller.py`**: Test `DTController` can select actions given a network state, test trajectory context window management, test integration with the signal manager
- **`dt/dataset.py`**: Test dataset loading from trajectories, test sequence windowing, test normalization/padding logic, test the `collect_expert_trajectories` function

### 2. Visualization Server — `viz/server.py` (10% coverage, 265 lines uncovered)

The FastAPI WebSocket server has almost no test coverage despite being 528 LOC. The existing `test_viz_server.py` tests are all skipped (marked with `anyio` but the dependency isn't configured).

**Recommended tests:**
- Fix the `pytest.mark.anyio` issue (install `anyio`/`pytest-anyio` or switch to `pytest-asyncio`)
- Test WebSocket connection lifecycle (connect, receive state updates, disconnect)
- Test `/api/network` and `/api/state` REST endpoints
- Test simulation control endpoints (start, stop, step, reset)
- Test concurrent client handling
- Test error handling for malformed WebSocket messages

### 3. Benchmarks Module — `benchmarks/` (26% coverage, 691 lines uncovered)

Two large files have 0% coverage:

| File | Coverage | Lines Uncovered |
|------|----------|----------------|
| `benchmarks/cross_validation.py` | 0% | All 343 statements — RL cross-validation |
| `benchmarks/sumo_comparison.py` | 0% | All 246 statements — SUMO comparison logic |
| `benchmarks/rl_baselines.py` | 42% | 83 of 144 statements |

**Recommended tests:**
- **`cross_validation.py`**: Test fold generation logic, test metric aggregation, test that a minimal cross-validation run completes
- **`sumo_comparison.py`**: Test scenario configuration parsing, test metric comparison logic (can mock SUMO calls)
- **`rl_baselines.py`**: Test baseline agent creation, test evaluation loop with a mock/minimal environment

### 4. Utilities — `utils/validation.py` (24% coverage, 42 lines uncovered)

The network validation module is only 24% covered despite being critical for catching invalid network configurations.

**Recommended tests:**
- Test validation of disconnected networks
- Test detection of invalid cell parameters (negative capacity, invalid free-flow speed)
- Test detection of missing or conflicting signal phase definitions
- Test validation of demand profiles against network topology

### 5. Networks — `networks/osm.py` (15% coverage, 161 lines uncovered)

The OpenStreetMap import pipeline (429 LOC) is barely tested. The existing `test_osm.py` (105 LOC) likely only tests the mock/stub path.

**Recommended tests:**
- Test OSM data parsing with a small fixture (saved OSM XML or JSON)
- Test intersection detection and simplification logic
- Test lane count and speed limit extraction
- Test network compilation from parsed OSM data
- Test error handling for malformed or incomplete OSM data

### 6. Pretrained Model Loading — `pretrained.py` (28% coverage, 18 lines uncovered)

**Recommended tests:**
- Test model discovery from the `weights/` directory
- Test loading a checkpoint and verifying the returned controller type
- Test error handling for missing or corrupt weight files

## Lower-Priority Gaps

These modules are well-tested but have a few uncovered branches worth addressing:

| File | Coverage | Uncovered Areas |
|------|----------|----------------|
| `core/engine.py` | 83% | Mesoscopic mode fallback (L172-177), warm-up logic (L213-229) |
| `core/signal.py` | 90% | `GreenWaveController` offset logic, `EfficientMaxPressureController` lost-time paths |
| `core/network.py` | 94% | Network property accessors (L322-362), some compilation edge cases |
| `envs/single_agent.py` | 90% | Render mode handling, `close()` method |
| `envs/multi_agent.py` | 88% | Agent iteration protocol, `state()` method |
| `utils/travel_time.py` | 87% | Summary statistics edge cases, empty tracker paths |

## Recommendations Summary

| Priority | Area | Current | Target | Effort |
|----------|------|---------|--------|--------|
| **P0** | `dt/` module | 6% | 60%+ | High — requires PyTorch fixtures |
| **P0** | `viz/server.py` | 10% | 60%+ | Medium — fix async test setup first |
| **P1** | `utils/validation.py` | 24% | 80%+ | Low — straightforward unit tests |
| **P1** | `benchmarks/` module | 26% | 50%+ | Medium — some functions need mocking |
| **P2** | `networks/osm.py` | 15% | 50%+ | Medium — needs OSM data fixtures |
| **P2** | `pretrained.py` | 28% | 80%+ | Low — small file, simple tests |
| **P3** | Core/envs gaps | 83-95% | 95%+ | Low — fill remaining branches |

Achieving these targets would bring overall coverage from **53% to approximately 75-80%**.
