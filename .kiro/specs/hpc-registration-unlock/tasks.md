# Implementation Plan: hpc-registration-unlock

## Overview

Implementation is in Python 3.12, matching the design and the existing `research/` code. The work builds the `hpc_unlock/` package bottom-up: paths, config and checkpointing first; then the input check, env scripts and the SLURM chain; then each Stage in chain order (prep → gpu_scan → pose_search → joint → pairs → verifier → validate → selftrain_* → assemble/report); then the smoke test, the notebook and the docs. `run_unlock.py` dispatches Stages dynamically from `chain.STAGES` (`importlib.import_module(f"hpc_unlock.{stage}")`), so each new Stage module is wired in as soon as it exists.

Nothing heavy runs locally. The only local runs are the pytest + hypothesis suite in `tests/unlock/` (CPU, small generated inputs, CPU torch for the scan kernel) and the ≤ 300 s smoke test (`python run_unlock.py smoke`). Every full-scale Stage refuses to run outside a SLURM allocation. Each property test lives in its own file `tests/unlock/test_pNN_<name>.py`, tagged `# Feature: hpc-registration-unlock, Property NN (<title>)`, with `@settings(max_examples=100, deadline=None)`.

## Tasks

- [x] 1. Package skeleton, config and checkpointing
  - [x] 1.1 Create the package skeleton and path resolution
    - Create `hpc_unlock/__init__.py` and `hpc_unlock/paths.py`: `ROOT = Path(__file__).resolve().parents[1]`, `DATA`, `RDATA`, `HPC`, `HPC_SMOKE`, `LOGS`, `CONTAINER = ROOT/"hpc/unlock/container"`, `MODELS`; `sys.path` inserts for `ROOT` and `ROOT/"research"`; `work_dir()` (notebook dir if it contains `hpc_unlock/`, else `$CELLMATCH_DIR`, else `/scratch/$USER/cellmatch`); `find_sif()` (name containing `cuda12` first, highest version string, else `cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`, else `None`); `worker_count()` (`SLURM_CPUS_PER_TASK` clamped to `[1, len(os.sched_getaffinity(0))]`)
    - `paths.py` imports only the standard library
    - Create `tests/unlock/conftest.py` (tmp project tree fixture) and `hpc/unlock/requirements-dev.txt` with `pytest` and `hypothesis` pinned to the versions installed in the local `.venv`
    - Add `hpc/unlock/container/`, `research/data/hpc/`, `research/data/hpc_smoke/` and `logs/unlock-*` to `.gitignore`
    - _Requirements: 1.1, 3.2_

  - [ ]* 1.2 Write property test for worker count
    - **Property 5: Worker count bound**
    - **Validates: Requirements 3.2**

  - [x] 1.3 Implement `hpc_unlock/config.py`
    - Frozen `UnlockConfig` dataclass with every field in the design (account `cs_gy_6923-2026fa`, `n2c48m24`, `g2-standard-12`, `env_mode="singularity"`, `sif_name=None`, σ 2.5, K 50, scan steps, joint λ grid, radii 6, self-training params, per-Stage resources)
    - `validate()` returns every error (partition, env mode, σ, K, radii), naming field and value; `fingerprint()` hashes only result-affecting fields (10 hex chars); `save()` / `load()` of `run_config.json`
    - Standard library only, so a plain notebook kernel can import it; no NetID and no other account anywhere
    - _Requirements: 2.2, 2.3, 4.8, 5.11, 12.13_

  - [ ]* 1.4 Write property test for config validation
    - **Property 3: Config validation is exact**
    - **Validates: Requirements 2.3, 4.8, 5.11, 12.13**

  - [x] 1.5 Implement `hpc_unlock/checkpoint.py`
    - `save_atomic` (pickle protocol 5 to `path.tmp-<pid>`, fsync, `os.replace`), `load`, side-file helper for `.npz` with SHA-256, `run_stage(name, compute, root)` implementing REUSE / INCONSISTENT / compute-then-save-then-marker, marker JSON `{"stage", "sha256", "fingerprint", "finished"}`
    - Log lines in fixed `TAG detail` format
    - _Requirements: 3.7, 3.8, 3.9, 3.10, 3.11_

  - [ ]* 1.6 Write property test for checkpoint round trip
    - **Property 1: Checkpoint round trip**
    - **Validates: Requirements 3.11, 3.9**

  - [ ]* 1.7 Write property test for the Stage runner
    - **Property 2: Stage runner state machine**
    - Inject crashes into `compute` and `save_atomic`; truncated and wrong-hash Checkpoints
    - **Validates: Requirements 3.7, 3.8, 3.9, 3.10**

- [x] 2. Input check, pins and the driver
  - [x] 2.1 Implement pins file and `hpc_unlock/inputs.py`
    - Create `hpc/unlock/requirements-unlock.txt` with the design's pins (numpy, scipy, opencv-python-headless, scikit-learn, tifffile, pandas, scikit-image, torch 2.14.0 from default PyPI, cellpose 4.2.1.1) and the `MINICONDA_INSTALLER` line
    - `required_inputs(cfg)`: dataset CSVs, every `invivo.tif` / `exvivo.tif`, the `research/data/` caches, both Baseline CSVs, and in `singularity` mode the overlay file and the `.sif` from `find_sif()` (reported as `hpc/unlock/container/*.sif` when absent)
    - `check_inputs(cfg)` returns every missing or unreadable path (1-byte read; full load for `.pkl` / `.npz`; existence + readability only for overlay and `.sif`)
    - `reuse_env(installed: dict) -> bool` against the six core pins; CLI `python -m hpc_unlock.inputs pins` prints installed versions via `importlib.metadata` as JSON for the shell scripts
    - _Requirements: 1.2, 1.3, 1.6_

  - [ ]* 2.2 Write property test for the missing-input check
    - **Property 6: Missing-input check reports every failure**
    - Cover both env modes (overlay and `.sif` required only in `singularity` mode)
    - **Validates: Requirements 1.6**

  - [ ]* 2.3 Write property test for the environment reuse decision
    - **Property 7: Environment reuse decision**
    - **Validates: Requirements 1.3**

  - [x] 2.4 Implement `run_unlock.py` driver
    - Subcommands: `<stage> [--smoke] --run <fp>`, `check`, `submit`, `monitor`, `smoke`
    - Stage dispatch via `importlib.import_module(f"hpc_unlock.{stage}")` and `checkpoint.run_stage`; per-Stage config re-validation before any region; start/end log of fingerprint, worker count, elapsed time
    - Full-scale Stage without `SLURM_JOB_ID` and without `--smoke` prints "start this Stage through the Job_Chain or the HPC_Notebook" and exits 2 before computing
    - `check` exits non-zero and lists every failing path
    - _Requirements: 1.6, 13.6, 3.2_

  - [ ]* 2.5 Write unit tests for the driver
    - Outside-SLURM guard message and exit code; invalid config exits non-zero before `compute`; `check` output lists every missing path
    - _Requirements: 13.6, 1.6, 4.8_

- [x] 3. SLURM scripts and chain orchestration
  - [x] 3.1 Write the setup and environment scripts
    - `hpc/unlock/run_in_env.sh <mode> <gpu:0|1> -- <cmd>`: `singularity` → `singularity exec [--nv on GPU] --overlay <overlay>:ro <sif> /bin/bash -c "source /ext3/env.sh; cd <WORK>; <cmd>"`; `venv` → `source hpc/env.sh` with `hpc/unlock/.env_path`, then `<cmd>`
    - `hpc/unlock/setup.sbatch <mode> [--no-cellpose]`: `module purge`, `cd "$SLURM_SUBMIT_DIR"`, `run_unlock.py check`; `singularity` mode runs `setup_overlay.sh` inside `singularity exec --overlay <overlay>:rw <sif>`; `venv` mode reuses `env/` or creates `env_unlock/`
    - `hpc/unlock/setup_overlay.sh`: install pinned Miniconda to `/ext3/miniconda3` if missing, write `/ext3/env.sh` (source `conda.sh`, PATH, `conda activate unlock`), `conda install -y pip ipykernel`; check pins with `importlib.metadata` and recreate conda env `unlock` (Python 3.12) when `reuse_env` is false
    - Both modes: one `pip install` per pin under `set -e` with a `PIN_FAILED <package>` trap; `pip install torch==2.14.0` from default PyPI on every run; cellpose + cached `cpsam_v2` weights into `hpc/unlock/models` unless `--no-cellpose`; write `research/data/hpc/setup/setup.json` (env mode, env path, `.sif`, pin versions, `torch.__version__`, `torch.version.cuda`); no CUDA check on the CPU node
    - `hpc/unlock/kernel/kernel.json`: optional OOD Jupyter kernel running `ipykernel_launcher` inside `singularity exec --overlay <overlay>:ro <sif>`
    - _Requirements: 1.3, 1.4, 1.5, 1.7, 1.6_

  - [x] 3.2 Write the Stage sbatch scripts
    - `hpc/unlock/stage.sbatch <mode> <stage> --run <fp>`: `module purge`, `cd "$SLURM_SUBMIT_DIR"`, `run_in_env.sh <mode> 0 -- python run_unlock.py <stage> --run <fp>`
    - `hpc/unlock/gpu_stage.sbatch <mode> <stage> --run <fp>`: same with `--nv`, plus `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 &` into the Stage log, killed by `trap` on exit
    - _Requirements: 3.1, 3.3, 3.4_

  - [x] 3.3 Implement `hpc_unlock/chain.py`
    - `STAGES`, `GPU_STAGES`, `SELFTRAIN`, `plan(cfg, start)` (drops self-training Stages when disabled), `sbatch_argv(cfg, stage, dep)` with `--account`, `--partition`, `--mem`, `--cpus-per-task`, `--time`, `--export=NONE`, `--requeue`, `--gres=gpu:1` on GPU Stages, `--dependency=afterok:<id>`, env mode as first script argument
    - `submit_setup(cfg)` (setup alone, job ID in `research/data/hpc/setup/job.json`); `submit(cfg, start="prep")` with the setup-dependency rule (pending/running → afterok; COMPLETED + `setup.json` → none; otherwise refuse with the setup log path); `SLURM_*` removed from the env; stop and show stderr on the first sbatch error; write `jobs.json`; print stage / partition / job ID
    - `monitor(cfg)`: `squeue` then `sacct` state, last 40 log lines or `log not created yet`, `*.done` list, failure Stage + log path + `resubmit_command`, `scancel` advice for `DependencyNeverSatisfied` dependents; `assemble --report-only` hint after `BASELINE_REPRODUCTION_FAILED`
    - Standard library only (`subprocess`, `json`, `pathlib`)
    - _Requirements: 2.4, 2.5, 2.6, 2.8, 2.9, 3.1, 3.3, 3.6, 12.1, 12.2_

  - [ ]* 3.4 Write property test for chain plan and sbatch commands
    - **Property 4: Chain plan and sbatch commands**
    - Fake `sbatch` runner that fails at a random position; both env modes
    - **Validates: Requirements 2.4, 2.9, 3.1, 3.3, 3.6, 12.1, 12.2**

  - [ ]* 3.5 Write unit tests for monitor, setup gating and scripts
    - Fake `squeue` / `sacct` for each failure state (FAILED, TIMEOUT, CANCELLED, OUT_OF_MEMORY, non-zero exit); log tails shorter and longer than 40 lines; missing logs
    - `submit` refuses when setup failed, uses afterok while setup runs, no dependency when setup is COMPLETED
    - `bash -n` syntax check of every `.sh` / `.sbatch`; each script contains `module purge`; `gpu_stage.sbatch` passes `--nv` and `stage.sbatch` does not; Stage scripts mount the overlay `:ro`, only setup mounts `:rw`; no `sg8304`, `ece_gy_7123-2025sp` or other NetID in any file
    - _Requirements: 2.6, 2.8, 2.9, 3.4, 1.1_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Prep Stage
  - [x] 5.1 Implement `hpc_unlock/prep.py`
    - Held-out `RegionRecord`s from `lab.pkl`; test records via `test_apply.build` on `research/data/submission.csv` with `common.crop_offsets("hidden_test")`
    - `cp_bin` from `cp_pose_lab.cp_map`, `dup_key` = SHA-1 of raw in-vivo + ex-vivo bytes, `group = (subject, ex_shape)`
    - Scan inputs per unique `dup_key` (float32); window/vote candidates (held-out from `cp_pose_train.pkl` / `vote_cands.pkl`; test from `cp_pose_lab.all_window` + `test_apply.cands`; vote modes with duplicates voting once)
    - SHA-256 of `submission_v10_cpgate.csv` and `submission_v7_grow15.csv` stored for the end-of-chain guard
    - Records stored as plain dicts of builtins and arrays (loadable without `hpc_unlock`)
    - _Requirements: 3.6, 4.5, 6.1, 6.4, 10.5, 1.1_

  - [ ]* 5.2 Write unit tests for prep helpers
    - `dup_key` equal only for byte-identical image pairs; `group` keys; record dict round-trips through the Checkpoint
    - _Requirements: 6.1, 6.4_

- [x] 6. GPU dense pose scan
  - [x] 6.1 Implement the scan grid and FFT kernel in `hpc_unlock/gpu_scan.py`
    - Angle grid `linspace(-35, 35, ceil(70/step)+1)`, scale grid `linspace(0.85, 1.13, …)`, 17 stretch hypotheses (`direction=None` for 1.00), linear parts `s·R(θ)·S(k, φ)`
    - Bilinear splats, Gaussian `σ_g√2` in the Fourier domain on `E`, batched `rfft2` / `irfft2` correlation with fast padded sizes (2^a·3^b·5^c), valid-translation mask from the field centre, float32 only
    - Device-agnostic torch code (CUDA on Burst, CPU in tests and smoke); imports none of `registration`, `sklearn`, `validate`, `assemble`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 3.5_

  - [ ]* 6.2 Write property test for the FFT kernel
    - **Property 8: FFT scan equals direct splat correlation**
    - **Validates: Requirements 5.4**

  - [ ]* 6.3 Write unit tests for the scan grids
    - 141 angles with exact ±35, 29 scales with exact 0.85 / 1.13, 17 unique stretch hypotheses, direction `None` for 1.00
    - _Requirements: 5.1, 5.2_

  - [x] 6.4 Implement peak selection and the gpu_scan Stage `compute`
    - Local maxima via `max_pool2d`, top-4 per pose, host-side greedy separation (3°, 20 px landing) up to K, kept count per region, `ScanCandidate` fields incl. translation in px and landing `A(P0 − offset) + t`
    - Batch size from free memory clamped to [16, 1024]; duplicate regions scanned once and copied under both IDs
    - `TORCH_INFO version=… cuda=… available=… device=…` line at start; `NO_GPU_VISIBLE` exit 1 outside smoke; OOM retry once at half batch, then `REGION_FAILED <sid>: <cause>` exit 1 with no marker; smoke grid 5 × 3 × 1 on CPU torch
    - Loads only `prep.pkl`; writes `gpu_scan.pkl`
    - _Requirements: 5.5, 5.6, 5.7, 5.8, 5.9, 5.11, 5.12, 3.5, 3.6, 13.2_

  - [ ]* 6.5 Write property test for peak selection
    - **Property 10: Scan peak selection invariants**
    - **Validates: Requirements 5.3, 5.5, 5.6**

  - [ ]* 6.6 Write property test for planted pose recovery
    - **Property 9: Scan recovers a planted pose**
    - Small canvases and a reduced grid that contains the planted pose
    - **Validates: Requirements 5.1, 5.3, 5.4, 5.7**

  - [ ]* 6.7 Write unit tests for gpu_scan failure paths and imports
    - Mocked `torch.cuda.is_available() → False`; injected `torch.cuda.OutOfMemoryError` (retry then fail, no marker); import graph of `gpu_scan` and `selftrain_gpu` modules excludes `registration`, `sklearn`, `validate`, `assemble`
    - _Requirements: 5.9, 5.12, 3.5_

- [x] 7. Pose search
  - [x] 7.1 Implement `hpc_unlock/soft.py`
    - Soft_Score wrapper over `wide_soft.soft`, Soft_Margin (clearly different: |Δθ| > 3° or ‖Δlanding‖ > 60 px; 0 when none), landing via `window_lab.pose` convention, angle / scale / anisotropy decomposition from the SVD
    - Does not import `objective_probe` or `soft_gate_cv`
    - _Requirements: 4.4_

  - [x] 7.2 Implement `hpc_unlock/pose_search.py`
    - Hough (`wide_soft.wide_candidates`, angles −35..35 step 1°, scales 0.85–1.13 step 0.02, stretch 0.92/1.08 at 0/45/90/135), every GPU candidate refined with `registration.refine`, window and vote candidates; merge in source order; dedup (linear part ≤ 0.01, translation ≤ 4 px, higher Soft_Score wins, ties keep the earlier source)
    - Per-candidate features (refine score, soft, z, refine margin vs vote candidates, soft margin, angle, scale, anisotropy, landing, source); σ validated before any region; identical parameters for all 76 regions; empty list + count for zero-candidate regions; Pool over regions with `worker_count()`
    - Diagnostics for the Run_Report: Correct_Pose present among raw GPU candidates, merged candidates, and ranked first, out of 46, with σ
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 5.7, 5.10_

  - [ ]* 7.3 Write property test for merge and dedup
    - **Property 11: Candidate merge and dedup**
    - **Validates: Requirements 4.3, 4.5, 4.6**

  - [ ]* 7.4 Write unit tests for the Hough grid
    - Angle and scale endpoints and steps, 9 stretch hypotheses (k = 1 plus 2 × 4)
    - _Requirements: 4.1, 4.2_

- [x] 8. Joint registration
  - [x] 8.1 Implement `hpc_unlock/joint.py`
    - Grouping by `(subject, ex_shape)` with `dup_key` collapse; unregistered regions excluded
    - Objective `J` with ψ (τ_l 60 px, τ_A 0.06) and offset weights (ρ 400 px); ICM with descending Soft_Margin sweeps, ≤ 50 sweeps, deterministic restarts, exhaustive enumeration when `Π|L_u| ≤ 10⁵`
    - λ chosen per held-out mouse on the other two mice (ties to smaller λ); test λ on all three; stores `independent` and `joint` selections and the per-mouse / total Correct_Pose counts
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 8.2 Write property test for the λ = 0 reduction
    - **Property 12: Joint selection reduces to independent selection**
    - **Validates: Requirements 6.3**

  - [ ]* 8.3 Write property test for joint validity and optimality
    - **Property 13: Joint selection is valid and optimal where exhaustive**
    - **Validates: Requirements 6.2, 6.6**

  - [ ]* 8.4 Write property test for grouping and duplicates
    - **Property 14: Grouping partition and duplicate agreement**
    - **Validates: Requirements 6.1, 6.4**

- [x] 9. Pair classifier and verifier
  - [x] 9.1 Implement `hpc_unlock/pairs.py`
    - Reuse `pair_clf.candidates`; re-implement dataset and LOO loops over explicit records with the same `HistGradientBoostingClassifier` settings; `loo_probs`, `test_model`, `select` (gated, threshold, greedy one-to-one, stable ties), `choose_threshold` (13 values, lowest on ties)
    - Stage `compute` for `independent` and `joint` selections
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 9.2 Write property test for pair selection
    - **Property 17: Pair selection is one-to-one and gated**
    - **Validates: Requirements 8.3, 8.4, 10.8**

  - [x] 9.3 Implement `hpc_unlock/verifier.py`
    - Features per region (Soft_Score, Soft_Margin, z, refine margin, landing distance to the group median, angle, scale, anisotropy); labels `err < 5` on the 46 GT regions; LOO logistic pipeline, all-mice test model, constant model for single-class folds
    - 21-row gate grid with v10 OR prob ≥ τ, kept / correct / wrong / pair F1; conservative (lowest zero-wrong τ or `unavailable` → v10 only), aggressive (max F1, highest τ on ties); `fold_gate(m)` from the other two mice
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

  - [ ]* 9.4 Write property test for gate and threshold rules
    - **Property 16: Gate and threshold grid selection rules**
    - **Validates: Requirements 7.4, 7.5, 7.6, 7.7, 7.8, 8.2**

- [x] 10. Validation
  - [x] 10.1 Implement the metric functions in `hpc_unlock/validate.py`
    - Per-region PQ via `cellmatch.pq_score` (IoU > 0.75), mean over 47 regions; pooled pair F1 with the TP rule (both masks PQ TPs and a GT pair); `full = 0.25·PQ_iv + 0.25·PQ_ex + 0.5·F1`
    - _Requirements: 9.1, 9.2_

  - [ ]* 10.2 Write property test for the metric
    - **Property 18: Metric matches a reference implementation**
    - **Validates: Requirements 9.1, 9.2**

  - [x] 10.3 Implement Baseline reproduction, configurations and acceptance
    - Exact v10 recipe (`cp_pose_lab.pose_choose`, `margin_lab.margin`, `pair_clf.dataset` / `loo_predict`, threshold 0.025); tolerance check ±0.005 on F1 0.472 and full 0.5186; on failure write `validate_failure.json` with the measured values and exit 1 without a marker
    - Four configurations `indep_cons`, `indep_aggr`, `joint_cons`, `joint_aggr` with their own pair thresholds; metrics to 4 decimals, per-mouse F1, kept-wrong; strict unrounded acceptance `full > baseline_full`; `ConfigResult` with test pairs
    - _Requirements: 9.3, 9.4, 9.5, 9.6, 8.5_

  - [ ]* 10.4 Write property test for acceptance
    - **Property 19: Acceptance is strict and unrounded**
    - **Validates: Requirements 9.5, 12.7, 12.8**

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Self-training Stages
  - [x] 12.1 Implement `hpc_unlock/selftrain_prep.py`
    - Radii validated before any label is built; conservative-gated regions per held-out mouse (fold gate) and for test (full gate; v10 gate if unavailable); poses from `joint` (fallback `indep`); Matched_Invivo_Instance projection; retain / seed / exclude rule with cKDTree; disks clipped to the canvas, painted on background only; consecutive relabelling
    - 96-px tiles (≤ 100 per region, `pipeline.percentile_normalize`), all ex-vivo images as float32 in `selftrain_tiles.npz` side file; `no_confident_regions` flags per mouse and for test
    - _Requirements: 12.3, 12.5, 12.10, 12.11, 12.12, 12.13_

  - [ ]* 12.2 Write property test for pseudo-labels
    - **Property 22: Pseudo-label construction**
    - **Validates: Requirements 12.3**

  - [ ]* 12.3 Write property test for label leakage
    - **Property 15: No label leakage from the evaluated mouse**
    - Synthetic records and fast stand-in estimators through the real fold-splitting code of joint, verifier, pairs and selftrain_prep
    - **Validates: Requirements 6.5, 7.3, 8.1, 12.5**

  - [x] 12.4 Implement `hpc_unlock/selftrain_gpu.py`
    - `TORCH_INFO` line at start; `NO_GPU_VISIBLE` outside smoke; per job [m₁, m₂, m₃, test] skipping no-confident jobs: fresh `pipeline.cellpose_model().net` from `cpsam_v2`, `cellpose.train.train_seg` with the design's settings on ex-vivo tiles only, inference with `pipeline.cellpose_flows` on every ex-vivo image of the job, 6 decode settings
    - Flows (float16) and labels (uint16) saved atomically per job to `selftrain_flows/<job>.npz`; finished jobs skipped on resume; `SELFTRAIN_FAILED <job>: <cause>` exit 1 with no marker
    - Loads only `selftrain_prep.*`; no `registration`, `sklearn`, validation or CSV code
    - _Requirements: 12.4, 12.5, 12.9, 12.14, 3.5, 3.6_

  - [ ]* 12.5 Write unit tests for selftrain_gpu with a mocked `train_seg`
    - Each job starts from pretrained weights; only ex-vivo tiles and pseudo-labels reach training; one test model with inference on all 29 test regions; failure and no-GPU paths leave no marker
    - _Requirements: 12.4, 12.9, 12.14_

  - [x] 12.6 Implement `hpc_unlock/selftrain_pairs.py`
    - New ex features, centroids and links per held-out mouse and decode setting; `pairs.loo_probs` under the same poses and gate; decode setting chosen on the other mice (test: all three); held-out full with Baseline in-vivo PQ and Baseline ex masks for no-confident mice
    - Strict comparison against the best accepted registration-only full, else the Baseline full; if accepted, test pairs with the all-mice model and grow15 on test labels when `st_grow15`
    - _Requirements: 12.5, 12.6, 12.7, 12.8, 12.12_

- [x] 13. Assembly and Run_Report
  - [x] 13.1 Implement `hpc_unlock/report.py`
    - `build(state) -> dict` rendered to `hpc_unlock_report.json` and `hpc_unlock_report.md` from the same dict; ranking key `(-full, -f1, kept_wrong)`; Baseline and per-candidate full, F1, kept, kept-wrong; disclaimer line; recommendations for accepted / `NO_CANDIDATE`; registration diagnostics (4.10, 5.10, 6.7), Baseline reproduction failure, self-training disabled / `NO_CANDIDATE` reasons, no-confident mice
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 4.10, 5.10, 6.7, 9.4, 9.6, 12.2, 12.7, 12.11, 12.12_

  - [ ]* 13.2 Write property test for the report
    - **Property 21: Report ranking and recommendations**
    - **Validates: Requirements 11.1, 11.2, 11.5, 11.6**

  - [x] 13.3 Implement `hpc_unlock/assemble.py`
    - Registration-only writer (Baseline rows copied byte-for-byte, ID assertions, only `match_pairs` set); self-trained writer (Baseline `invivo_instances`, new `exvivo_instances` via `cellmatch.labels_to_rles`, pairs on the new IDs); names `submission_v13_<selection>_<gate>.csv` and `submission_v13_selftrain.csv`
    - Temp-then-rename, then `python validate_submission.py <csv>`; delete and record rejected on failure; Baseline CSV SHA-256 guard against the hashes from `prep`; unlock table per test region; `--report-only` path after a Baseline reproduction failure; self-training-disabled path depends on `validate` only; calls `report.build`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 9.4, 9.5, 12.2, 12.8_

  - [ ]* 13.4 Write property test for candidate CSVs
    - **Property 20: Candidate CSVs preserve Baseline masks**
    - **Validates: Requirements 10.1, 10.2, 10.4, 12.8**

  - [ ]* 13.5 Write unit tests for assemble and report examples
    - Unlock-table example, Format_Checker rejection path, Baseline hash guard, disclaimer text and required report fields, `--report-only` writes no CSV
    - _Requirements: 10.5, 10.6, 10.7, 11.3, 11.4, 9.4, 9.6_

- [x] 14. Local smoke test
  - [x] 14.1 Implement `hpc_unlock/smoke.py` and the `smoke` subcommand
    - 2 held-out regions from 2 different mice, `workers = min(4, cores)`, CPU-torch scan 5 × 3 × 1, outputs only under `research/data/hpc_smoke/`; Baseline tolerances reported `n/a (smoke)`; `selftrain_gpu` reported `SKIPPED (no fine-tune in smoke)`; `assemble` writes only `research/data/hpc_smoke/smoke_candidate.csv` and runs the Format_Checker
    - Before/after hashes of `research/data/hpc/`, root report files and every `submission_*.csv`; 300 s wall-clock budget; `PASS/FAIL <stage>` lines; non-zero exit naming the first failing Stage
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 14.2 Run the local smoke test and fix failures
    - Run `python run_unlock.py smoke` locally (≤ 300 s, ≤ 4 cores) and fix any failing Stage; this is the only local pipeline run
    - _Requirements: 13.1, 13.2, 13.4, 13.5_

- [x] 15. Notebook and docs
  - [x] 15.1 Create `CellMatch_HPC_Unlock.ipynb`
    - Plain Python 3 kernel, standard library plus `hpc_unlock.paths` / `config` / `chain`; a numbered instruction markdown cell before every code cell naming the cell and expected output
    - Step 0 Upload: upload to `/scratch/$USER/cellmatch`, the two `scp` commands from `greene-dtn` (overlay + `gunzip`; CUDA 12 `.sif` if listed, else `cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`), `run_unlock.py check`
    - Step 1 Config: `ENV_MODE`, `GPU_PARTITION`, `DISABLE_SELFTRAIN`, σ, K, radii, resources, `validate()` (errors block every submit)
    - Step 2 Setup submit; Step 3 Chain submit; Step 4 Monitor; Step 5 Report and download command; optional resubmit-from-Stage cell and container kernel registration note
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 12.1, 12.2_

  - [ ]* 15.2 Write unit tests for notebook structure
    - Instruction cell before every code cell; default config values (`g2-standard-12`, `singularity`, self-training enabled, account `cs_gy_6923-2026fa`); no hard-coded user path, NetID or old account; code cells import nothing outside the standard library and `hpc_unlock.{paths,config,chain}`
    - _Requirements: 1.1, 2.1, 2.2_

  - [x] 15.3 Add the pointer cell and README section
    - Insert one markdown cell at the top of `CellMatch_HPC.ipynb` pointing to `CellMatch_HPC_Unlock.ipynb` (leave the existing cells unchanged)
    - Add a README section: upload, the two `scp` commands, env modes, notebook steps 0–5, expected outputs, candidate naming, the 0.65 target-not-guaranteed note, and that v10 stays the safe final
    - _Requirements: 1.1, 2.1, 11.4_

- [x] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. Task 14.2 (local smoke run) is not optional.
- Each task references requirement clauses for traceability; property tests map to the numbered properties in design.md.
- Nothing heavy runs locally: only `pytest tests/unlock` (small generated inputs, CPU torch) and the ≤ 300 s smoke test. Baseline reproduction, the GPU scan, self-training and every candidate CSV are produced only on Burst through the notebook.
- Values that can only be confirmed on Burst (runtimes, `n2c48m24` memory, CUDA 12 `.sif` availability, torch wheel vs host driver, internet for Miniconda / pip) are reported by `setup.json`, `TORCH_INFO`, GPU utilization logs and the `validate` Stage.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.5"] },
    { "id": 2, "tasks": ["1.4", "1.6", "1.7", "2.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4", "3.1", "3.2", "3.3"] },
    { "id": 4, "tasks": ["2.5", "3.4", "3.5", "5.1", "6.1", "7.1"] },
    { "id": 5, "tasks": ["5.2", "6.2", "6.3", "6.4"] },
    { "id": 6, "tasks": ["6.5", "6.6", "6.7", "7.2", "9.1", "10.1"] },
    { "id": 7, "tasks": ["7.3", "7.4", "8.1", "9.2", "10.2"] },
    { "id": 8, "tasks": ["8.2", "8.3", "8.4", "9.3"] },
    { "id": 9, "tasks": ["9.4", "10.3"] },
    { "id": 10, "tasks": ["10.4", "12.1", "13.1"] },
    { "id": 11, "tasks": ["12.2", "12.3", "12.4", "13.2"] },
    { "id": 12, "tasks": ["12.5", "12.6"] },
    { "id": 13, "tasks": ["13.3"] },
    { "id": 14, "tasks": ["13.4", "13.5", "14.1"] },
    { "id": 15, "tasks": ["14.2", "15.1"] },
    { "id": 16, "tasks": ["15.2", "15.3"] }
  ]
}
```
