# Observability Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible observability and sensor-ablation study that ranks sensor suites and maneuvers for vehicle lateral-state estimation.

**Architecture:** Keep the numerical core independent of ROS 2. A metrics module computes finite-difference sensitivities, regularized Gramian/Fisher metrics, effective rank, condition number, information gain, and CRLB. A maneuver module creates deterministic state/measurement trajectories, while an experiment runner evaluates sensor-suite combinations and writes machine-readable results plus plots and a Chinese summary.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Matplotlib, PyYAML (already optional/config-compatible), pytest, Ruff.

---

### Task 1: Observability metrics core

**Files:**
- Create: `src/vehicle_state_estimation/metrics/observability.py`
- Modify: `src/vehicle_state_estimation/metrics/__init__.py`
- Test: `tests/test_observability.py`

- [ ] **Step 1: Write failing tests** for finite-difference Jacobians, Gramian symmetry/PSD, effective rank, regularized condition number, log-determinant information gain, CRLB, invalid shapes, and low-speed ill-conditioning.
- [ ] **Step 2: Run `python -m pytest -q tests/test_observability.py` and confirm RED because the module/API is missing.
- [ ] **Step 3: Implement small pure functions with explicit validation and `np.linalg.pinv`/eigendecomposition; never use an unstable ordinary inverse.
- [ ] **Step 4: Run the focused tests, then Ruff on the new module.
- [ ] **Step 5: Commit `feat: add observability information metrics`.

### Task 2: Maneuver and sensor observation models

**Files:**
- Create: `src/vehicle_state_estimation/simulation/maneuvers.py`
- Test: `tests/test_maneuvers.py`
- Modify: `src/vehicle_state_estimation/simulation/__init__.py`

- [ ] **Step 1: Write failing tests** for deterministic straight, sine-steer, double-lane-change approximation, and low-friction-switch trajectories, including finite states and expected excitation energy.
- [ ] **Step 2: Run the focused tests and confirm RED.
- [ ] **Step 3: Implement `ManeuverConfig`, `ManeuverTrace`, and `generate_maneuver` with fixed seed, bounded speed, steering, friction, and sampling interval.
- [ ] **Step 4: Add sensor observation callbacks for IMU, wheel speed, and GNSS position with documented covariance matrices.
- [ ] **Step 5: Run focused tests and Ruff; commit `feat: add reproducible estimation maneuvers`.

### Task 3: Sensor-ablation experiment runner

**Files:**
- Create: `src/vehicle_state_estimation/experiments/sensor_ablation.py`
- Create: `configs/observability.yaml`
- Create: `tests/test_sensor_ablation.py`
- Modify: `src/vehicle_state_estimation/experiments/__init__.py`

- [ ] **Step 1: Write failing tests** for deterministic sensor-suite ordering, baseline-relative information gain, CRLB shape, CSV/JSON serialization, and invalid suite names.
- [ ] **Step 2: Run the focused tests and confirm RED.
- [ ] **Step 3: Implement `SensorSuite`, `AblationResult`, and `run_sensor_ablation` using the metrics core and maneuver trace; mark ill-conditioned cases explicitly.
- [ ] **Step 4: Add YAML defaults for four maneuvers, four sensor suites, finite-difference step, covariance floor, and regularization.
- [ ] **Step 5: Run focused tests and Ruff; commit `feat: add sensor ablation runner`.

### Task 4: CLI, plots, and Chinese research report

**Files:**
- Create: `experiments/observability_study.py`
- Create: `docs/observability_study.md`
- Create: `tests/test_observability_cli.py`
- Modify: `README.md`, `docs/中文说明.md`

- [ ] **Step 1: Write a failing CLI test** that runs a small study in a temporary directory and asserts `results.json`, `ranking.csv`, `observability.png`, and `summary.md` exist with the expected columns/sections.
- [ ] **Step 2: Run the CLI test and confirm RED.
- [ ] **Step 3: Implement the CLI with `--config`, `--output`, `--seed`, and `--fast` options; use Matplotlib only for plotting and keep headless execution supported.
- [ ] **Step 4: Generate a Chinese summary that states the best suite per maneuver, condition warnings, assumptions, and simulation-only limitations.
- [ ] **Step 5: Run full pytest, Ruff, compileall, and the study CLI; commit `feat: add observability study benchmark` and push `main`.

### Task 5: Final verification

- [ ] Re-run the complete test suite and static checks from a clean checkout.
- [ ] Verify fixed-seed repeatability by comparing two study result JSON files.
- [ ] Confirm GitHub Actions finishes successfully for the final commit.
- [ ] Report the academic contribution and the remaining boundary: local linearization and synthetic data are not real-vehicle validation.
