# Cell Matching x Segmentation (CS-GY-6643, Project 2)

Segment neurons in paired in-vivo / ex-vivo mouse brain images and match the same neuron across
both. Kaggle competition `testingpj-2`.

```
score = 0.5 * (PQ_invivo + PQ_exvivo) / 2 + 0.5 * F1_matching
```

- PQ (panoptic quality) is computed per region and averaged; a mask counts only if IoU > 0.75.
- F1 is pooled over all test regions. A pair counts only if **both** masks are correct and the
  pair is in the hand-verified list.
- Data: 3 training mice (47 regions, 1,139 verified pairs) and 2 hidden test mice (29 regions).

## Score history

| Kaggle score | What changed |
| --- | --- |
| 0.28576 → 0.30514 | Earlier submissions: small U-Net + watershed, affine registration from training priors, greedy nearest-neighbour matching |
| 0.32808 | Colab notebook v1: GPU training, leave-one-mouse-out tuning of decoding thresholds, compactness filters, matching settings |
| 0.37958 | Notebook v2: cell-density cross-correlation registration (`register_ncc`) added as a candidate; checkpoints saved to Drive |
| 0.40202 | Notebook v3: Cellpose-SAM fine-tuned (chosen for in-vivo, held-out PQ 0.727 vs 0.590 U-Net), per-mouse consistency gate (chosen: 0.10). CV estimate 0.450 |
| 0.40870 | Notebook v4: hybrid Cellpose masks filtered by U-Net probability (chosen for ex-vivo, PQ 0.382), ratio test for pairs (2.0), wider grids, PQ error breakdown. CV estimate 0.485 |
| ~0.45074 | Notebook / Cellpose ensemble masks + older matching (baseline masks used by later rematches) |
| 0.44201 | Rematch only: Hough vote + window registration + pair classifier, but low-margin regions included → hurt F1 |
| **0.45520** | Same rematch with **margin ≥ 3** gate (25/25 correct on train). File: `submission_v7.csv` |
| 0.39142 / 0.38233 | Retrained / ensembled ex-vivo masks (`ensb`, `v2b`): smaller ex-vivo cells lost IoU > 0.75 TPs |
| 0.47191 | `submission_v7_grow30.csv`: grow ex-vivo masks into the top 30% of boundary ring by cellprob |
| 0.47488 | `submission_v7_grow15.csv`: same as v7 pairs/IDs, grow ex-vivo by top 15% cellprob ring |
| **0.48893** | **`submission_v10_cpgate.csv` (current best verified):** v7_grow15 masks + cellprob-evidence region gate (unlocks `d7a97c/dd13e1`) |
| 0.44957 | `submission_v11_pairshrink25.csv`: v10 + tighter boundary for the 316 paired ex-vivo cells. Held-out said 0.501 → 0.546; public fell by 0.039. Test mice do **not** share the training mice's tight convention for paired cells |
| not submitted | `submission_v11_pairbase.csv`: paired cells ungrown. Also a shrink relative to v10, so expected to lose too |
| 0.47798 | `submission_v12_pairgrow15.csv`: opposite probe, one extra top-15% cellprob ring on paired ex-vivo cells only. Also below v10 |

Public LB uses ~48% of test; finals use the other 52%. Prefer CV + public when picking two finals.

## v8 experiment (completed; no improvement)

`research/v8_colab.py` fine-tuned CellposeDINO-ViT-B for ex-vivo segmentation on three
leave-one-mouse-out folds. Its selected masks scored 0.287 ex-vivo PQ and 0.509 estimated
combined CV, below the existing masks (~0.397 ex-vivo PQ, ~0.513 estimated combined CV).
The script therefore copied `submission_v7_grow15.csv` to `submission_v8.csv`; **v8 is
byte-identical to v7_grow15, not a new candidate or a new Kaggle result**. Its mask-selection
proxy also overweighted reachable matches and selected a weak threshold. A separate
DINO-plus-existing-mask fusion test reduced ex-vivo PQ, so that route was abandoned.

The follow-up `research/tight_ex_colab.py` and `research/v9_tight_candidate.py` have also
completed. The best 96-px tight-crop Cellpose-SAM fold models reached ex-vivo PQ 0.401;
blending their flows with the previous model reached 0.408 ex-vivo PQ, but the full
held-out score after rematching was only 0.509, below the existing pipeline's roughly
0.513 estimate. The script correctly reported `NO_CANDIDATE` and wrote no new CSV.
`research/adaptive_boundary_cv.py` tested per-cell growth selection on held-out mice;
its best PQ was 0.394, below fixed 10% cellprob-ranked growth (0.398). Neither result
supports replacing `submission_v7_grow15.csv`.

A separate thin-ring pixel classifier (`research/boundary_pixel_cv.py`) was tested
leave-one-mouse-out. It raised ex-vivo PQ on two mice (0.467→0.492 and
0.291→0.305), but the same threshold cut the third mouse from 0.394 to 0.257;
pooled PQ fell to 0.350. This is another example of boundary conventions failing
to transfer, so no test CSV was made from it.

The sparse-label experiment in `research/partial_loss_probe.py`,
`research/partial_loss_cv.py`, and `research/partial_candidate.py` is complete as a
**rejected diagnostic**. Downweighting unlabeled background lifted the individual
held-out ex-vivo PQ on `b2ba5e` from 0.291 to 0.381 and on `db6b8b` from 0.394
to 0.451, but fell from 0.467 to 0.452 on `5d294c`. A single shared decoder
setting reached only **0.367 pooled ex-vivo PQ** versus 0.390 for the existing
masks. Full matching validation reached **0.468 combined CV**, below the
existing ~0.513; no test CSV was generated. Per-region adaptive thresholds
(`research/partial_calibration.py`) also failed out of mouse: 0.340–0.347 PQ
versus 0.367 for the fixed decoder. Colab subsequently recycled the ephemeral
runtime; these results are from the completed logs, not a public Kaggle score.
No 0.65+ Kaggle score has been verified (best verified: 0.48893, v10).

The moderate (0.25) background-weight experiment (`research/mid_loss_cv.py`)
completed with **0.401 pooled ex-vivo PQ**, compared with 0.390 for the older
ex-vivo masks. However, the full held-out score after matching was **0.501**,
below the existing ~0.513 estimate. The conditional follow-up
(`research/mid_candidate.py`) correctly printed `NO_CANDIDATE` and generated
no new CSV. `submission_v7_grow15.csv` remains the best verified submission.
The linked Colab notebook now handles the already-extracted dataset, avoids an
expired temporary ZIP URL, and imports its config dependencies explicitly.
No Drive checkpoint access was granted; Colab checkpoints are ephemeral.

## Later research and candidate (public score unverified)

The supplied `code_submission.zip` is for a **different constellation task**;
its 0.95 score is not transferable. Its useful idea was to score registration
hypotheses with an independent difference-of-Gaussians image signal, then use
the gap to a competing hypothesis as a confidence measure. On held-out mice,
`research/bandpass_rerank_probe.py` improved correct region poses from 31/47
to 33/47 with a conservative gap gate, but the full matching F1 changed only
from 0.457 to 0.461 after blending with existing high-confidence matches.
`research/bandpass_test_candidate.py` produced the valid, **unscored**
`submission_bandpass_candidate.csv`: identical masks to `v7_grow15`, with four
additional pairs in two test regions. Do not treat this as a verified score
gain or as a 0.65 solution.

Two additional segmentation branches were tested before replacing any masks.
Naive fusion of old held-out instance masks barely changed ex-vivo PQ
(0.397 to 0.399 at best). A fluorescence-pretrained StarDist zero-shot pilot
reached at most 0.113 ex-vivo PQ at normal thresholds. A partial-label
StarDist fine-tune (unknown pixels masked from loss) reached just 0.278 PQ on
three probed regions of one held-out mouse at its best tested threshold;
the current masks average ~0.467 over that mouse. Both were rejected. The
remaining pseudo-labelled Cellpose-SAM
experiment is in `research/pseudo_cv_colab.py`; it must pass leave-one-mouse-out
PQ and full-score validation before `research/pseudo_candidate_colab.py`
can create a submission.

## v10 / v11 (October 2026)

**v10 `submission_v10_cpgate.csv` (unscored):** same masks/IDs as `v7_grow15`, plus one
unlocked region. A region's pose now passes if margin ≥ 3 **or** its ex-vivo cellprob
evidence z ≥ 5 (`research/cp_pose_lab.py`, `research/test_cp_apply.py`): the fraction of projected
in-vivo centroids landing on ex-vivo cellprob > 0, versus the same pose shifted 15–45 px.
Held-out: correct-pose z mean 7.1 vs wrong 2.9; pair F1 0.457 → 0.472 (29/31 kept regions
correct vs 25/25), about +0.008 score, slightly optimistic because the gate was picked on the same data. On test it adds
`d7a97c/dd13e1` (10 pairs), whose pairs match 10/10 of the older independent NCC registration.

**v11 `CellMatch_v11_Colab.ipynb` (GPU, not run yet):** runs the pseudo-label Cellpose-SAM
experiment (`pseudo_cv_colab.py` → `pseudo_candidate_colab.py`) from one upload,
`cellmatch_v11_bundle.zip` (build with `make_colab_bundle.py`). It writes a CSV only if the
full held-out score beats 0.5125, and also writes a cellprob-gated variant.

**Near-miss analysis (held-out ex-vivo, IoU 0.5–0.75):** 1,024 near misses vs 2,140 TPs. On
`5d294c` / `b2ba5e` they are too small (median pred/GT area 0.75 / 0.81, ~95% of pred inside GT);
on `db6b8b` too big (1.41, covers the whole GT). Opposite conventions again.
- `research/contour_probe.py`: an oracle-seeded relative-contrast contour (best fixed fraction of
  peak − local background) reaches IoU > 0.75 on only 24% of GT cells, below the model's 58%.
  Annotator boundaries are not a fixed intensity level; rejected.
- `research/multihyp_probe.py`: submitting extra overlapping grown/shrunk copies of each cell adds
  up to +550 ex-vivo TPs but the duplicate FPs drop PQ from 0.390 to 0.27–0.32; rejected.

**Pseudo-label Cellpose-SAM, full run (v11 notebook, completed, rejected):**

| Mouse | Old ex-vivo PQ | Pseudo, shared cellprob −0.5 | Pseudo, best per mouse |
| --- | --- | --- | --- |
| 5d294c | 0.467 | 0.473 | 0.473 (−0.5) |
| b2ba5e | 0.291 | 0.362 | 0.362 (−0.5) |
| db6b8b | 0.394 | 0.352 | 0.448 (+0.5) |

Pooled 0.390 → 0.396 at the shared setting; the mice want opposite thresholds. Full held-out
score after re-registration: 0.478 (best-PQ setting, F1 fell to 0.387) and 0.506 (most reachable
pairs, ex PQ 0.333), both below 0.5125, so `NO_CANDIDATE`. Checkpoints are in
`My Drive/cellmatch_v11/` of the Colab account.

**v11: paired ex-vivo cells are drawn tighter (`research/shrink_probe.py`,
`paired_shrink_cv.py`, `paired_shrink_exact.py`, `make_v11.py`).** Main finding: ex-vivo cells
that annotators put in *verified pairs* are drawn tighter than other ex-vivo cells, in every
mouse. Near-miss masks of verified cells are too big (pred/GT area ~1.35 in all three mice),
while near misses of the whole population go either way. Of the false-positive pairs in the
held-out v10 pipeline, 142 of 389 had the right in-vivo cell but an ex mask that just missed
IoU 0.75. Removing the lowest-cellprob 25% of a cell's inner boundary pixels:

| Verified ex cells IoU > 0.75 | base | ep25 | ep35 | ep50 |
| --- | --- | --- | --- | --- |
| 5d294c | 0.834 | 0.862 | 0.852 | 0.842 |
| b2ba5e | 0.456 | 0.574 | 0.603 | 0.618 |
| db6b8b | 0.750 | 0.847 | 0.851 | 0.874 |

Shrinking *all* ex cells hurts PQ badly (0.390 → 0.351 at ep25), so the rule is applied
only to cells we put in pairs. Exact held-out result (same poses, gate and classifier as v10):

| Ex masks | Ex PQ | Pair F1 | Full held-out score |
| --- | --- | --- | --- |
| grow15 everywhere (v10) | 0.397 | 0.434 | 0.501 |
| grow15, paired cells ungrown (`v11_pairbase`) | 0.405 | 0.484 | 0.528 |
| grow15, paired cells ep25 (`v11_pairshrink25`) | 0.406 | 0.518 | **0.546** |

Per-mouse pair F1 with ep25 rises in all three mice (0.547→0.569, 0.240→0.305, 0.474→0.536).
A learned per-cell variant selector was worse than the fixed rule (0.534). Note that grow15
*lowers* held-out pair F1 (0.472 → 0.434) yet raised the public score, so the test mice may draw
paired cells larger than the training mice; the public score of `v11_pairshrink25` settles this.
Even at the held-out estimate (+0.045), the expected public score is ~0.51–0.53, not 0.65.

**Public result: 0.44957 (−0.039 vs v10).** The boundary convention for paired cells on the test
mice is the opposite of the held-out estimate, consistent with grow15 helping on public while
hurting held-out F1. Held-out CV cannot be trusted for ex-vivo boundary size; only the public
score can, at the risk of overfitting its 48% split. `research/make_v12.py` builds the
opposite probe (`submission_v12_pairgrow15.csv`), which scored **0.47798** (−0.011 vs v10).
Both directions lose, so v10's paired-cell boundaries (grow15) are already at the public optimum.
**Recommended finals: `submission_v10_cpgate.csv` (0.48893) and `submission_v7_grow15.csv` (0.47488).**

## How to run (main path)

1. Upload `CellMatch_Colab.ipynb` to Colab and set `Runtime > Change runtime type > GPU`
   (T4 works; A100/L4 is much faster for Cellpose-SAM).
2. Data: add Colab secrets `KAGGLE_USERNAME` / `KAGGLE_KEY`, or upload `testingpj-2.zip` when asked.
3. `Runtime > Run all`. Checkpoints go to the Colab runtime's `models/` folder;
   reruns in the same runtime skip finished training, but a runtime reset removes them.
4. The last cell validates and downloads `submission.csv`. Stage 2 prints the chosen settings and
   the leave-one-mouse-out CV score (also saved as `best_config.json`).

With Cellpose-SAM, runs are long on a T4: fold fine-tunes plus inference on large, upsampled
ex-vivo images. Checkpoints in the current runtime are reused. The v5 `cellpose_long` variant adds six ~20-minute
fold fine-tunes on the first run. To save time, remove it from `CELLPOSE_VARIANTS` in the config
cell.

## How to run on NYU HPC (Cloud Bursting)

Use **Cloud Bursting**, not the Torch researcher cluster. Portal:
https://ood.burst.hpc.nyu.edu (NYU VPN off-campus).

**SLURM account (CS-GY-6923 DL course):** `cs_gy_6923-2026fa`  
Partitions: `n2c48m24` (CPU), `g2-standard-12` (1× L4), etc. Jobs with low GPU use
are auto-killed after ~20 minutes — our search stage is a separate CPU job for that reason.

Same three-job chain (setup → gpu → cpu). Finished steps are skipped on rerun.

From burst OOD Files / shell (code on burst `/scratch/$USER` — separate from Torch scratch):

```bash
cd /scratch/$USER
# clone or copy the project here if needed
cd cellmatch
bash submit.sh cs_gy_6923-2026fa
squeue -u $USER
tail -f logs/cellmatch-*.out
```

Notes:
- Torch login rejects this course account; submit only on the burst cluster.
- Burst scratch ≠ Torch scratch; copy data if you prepared files on Torch.
- `best_config.json` holds chosen settings. Delete to force a new search.
- If a job fails, cancel dependents (`scancel -u $USER`), fix `logs/`, rerun `submit.sh`.

## HPC registration unlock (Burst)

`CellMatch_HPC_Unlock.ipynb` drives a separate CPU + GPU job chain that attacks registration,
the current bottleneck (only 11/29 test regions get pairs). It adds a GPU dense pose scan with a
soft mutual-nearest score, a wider CPU pose search, joint per-mouse registration, a learned pose
verifier, a retrained pair classifier, and (by default) Cellpose-SAM ex-vivo self-training.
Code: `hpc_unlock/`, `run_unlock.py`, `hpc/unlock/`. Spec: `.kiro/specs/hpc-registration-unlock/`.
The old `CellMatch_HPC.ipynb` is unchanged apart from a pointer cell at the top; do not run both.

**Outlook.** A public score above 0.65 is the target, **not guaranteed**. With v10 masks, better
registration alone is estimated to reach about 0.60–0.63; going beyond needs self-training to
validate out of mouse, which earlier attempts (v8–v11 above) did not. Held-out gains are estimates.
`submission_v10_cpgate.csv` (0.48893) stays the safe final whatever this run produces.

**1. Upload.** Copy the whole project folder to `/scratch/$USER/cellmatch` on Burst (OOD Files
upload or `rsync -av`; `research/data/` is ~800 MB with files over GitHub's 100 MB limit, so not git).
All paths resolve relative to the folder root, so no edits are needed.

**2. Container files** (singularity mode only). Burst cannot see Greene's `/scratch/work/public`,
so copy them once from a Burst login shell (`ssh greene`, then `ssh burst`). Cell 0 prints these
with the real path filled in:

```bash
cd /scratch/$USER/cellmatch
scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz hpc/unlock/container/ \
  && gunzip hpc/unlock/container/overlay-15GB-500K.ext3.gz
ssh greene-dtn ls /scratch/work/public/singularity/ | grep -i cuda   # pick a cuda12 image if listed
scp greene-dtn:/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif hpc/unlock/container/
```

Use a CUDA 12 `.sif` instead of the CUDA 11.8 one if the `ls` lists one; `hpc_unlock/paths.find_sif`
picks it up from `hpc/unlock/container/`.

**Env modes** (`ENV_MODE` in Cell 1):
- `singularity` (default): `.sif` + ext3 overlay. The setup job runs alone and mounts the overlay
  `:rw` to install Miniconda and the pinned packages (numpy, scipy, opencv, scikit-learn, tifffile,
  pandas, torch, plus cellpose unless self-training is disabled) into `/ext3`. Every Stage job
  mounts it `:ro`, so several jobs can share it. Do not resubmit setup while chain jobs (or the
  optional container kernel, `hpc/unlock/kernel/kernel.json`) hold the overlay.
- `venv`: no container files. Setup sources `hpc/env.sh`, reuses `env/` if every core pin
  matches, else creates `env_unlock/`.

**Resources.** Account `cs_gy_6923-2026fa`. CPU Stages run on `n2c48m24`; the two GPU Stages
(`gpu_scan`, `selftrain_gpu`) run alone on `g2-standard-12` (1× L4, default) or `c12m85-a100-1`
(A100), so Burst's low-GPU-utilization auto-kill does not hit CPU work. Every job uses
`--export=NONE` and `--dependency=afterok:<previous job>`.

**3. Notebook steps** (open `CellMatch_HPC_Unlock.ipynb` in Burst OOD Jupyter, plain Python 3
kernel; each code cell has a numbered instruction cell with the expected output):

| Step | Cell | Does | Expected output |
| --- | --- | --- | --- |
| 0 | Upload / check | finds the project, prints the `scp` commands, lists container files, runs `python run_unlock.py check` | `WORK=/scratch/<you>/cellmatch`, `INPUT_CHECK_OK <N> inputs` (or one `INPUT_FAILED` line per missing path) |
| 1 | Config | sets `ENV_MODE`, `GPU_PARTITION`, `DISABLE_SELFTRAIN`, σ, K, radii, `RESOURCES`; runs `validate()` | `config OK, run <fingerprint> ...` and the Stage plan, or `CONFIG ERROR` lines (then nothing can be submitted) |
| 2 | Setup submit | submits setup alone (20–40 min first time) | `setup  n2c48m24  <job id>`, then `setup.json` once done |
| 3 | Chain submit | `prep → gpu_scan → pose_search → joint → pairs → verifier → validate → [selftrain_prep → selftrain_gpu → selftrain_pairs] → assemble` | one `stage  partition  job id` line per Stage (11, or 8 with self-training disabled) |
| 4 | Monitor | `squeue`/`sacct` states, last 40 log lines per Stage (`logs/unlock-<stage>-<job>.out`), done-markers | on failure: `FAILED <stage>`, its log and the resubmit command `python run_unlock.py submit --from <stage> --run <fp>` |
| 5 | Report / download | shows the Run_Report and candidate CSVs, prints `scp` download commands | rendered report and `submission_v13_*.csv` paths, or `NO_CANDIDATE` |

An optional Cell 6 resubmits from a chosen Stage. Finished Stages with valid done-markers are reused.

**Outputs.** Checkpoints and done-markers go to `research/data/hpc/<fingerprint>/` (the fingerprint
covers the settings that change results), setup info to `research/data/hpc/setup/setup.json`, and
the Run_Report to `hpc_unlock_report.json` / `hpc_unlock_report.md` in the project root.

**Candidates.** A CSV is written only if its leave-one-mouse-out full score beats the reproduced
Baseline (v10: held-out full 0.5186, pair F1 0.472; validate stops if the reproduction is off by
more than 0.005). Names: `submission_v13_<selection>_<gate>.csv` (`indep_cons`, `indep_aggr`,
`joint_cons`, `joint_aggr`), which keep v10 masks/IDs and change only `match_pairs`, and
`submission_v13_selftrain.csv`, which keeps v10 in-vivo masks and replaces ex-vivo masks and pairs.
Each CSV passes `validate_submission.py` or is deleted. `submission_v10_cpgate.csv` and
`submission_v7_grow15.csv` are hash-checked and never modified. With no accepted candidate the
report says `NO_CANDIDATE` and the finals stay v10 + v7_grow15.

**Local smoke test** (the only local pipeline run): `python run_unlock.py smoke` runs every Stage
on 2 held-out regions in ≤ 300 s with ≤ 4 cores, writes only to `research/data/hpc_smoke/`, and
prints `PASS/FAIL <stage>` lines. Unit tests: `python -m pytest tests/unlock -q`.

## Files

| File | Role |
| --- | --- |
| `CellMatch_Colab.ipynb` | **Main entry point.** Self-contained: writes the modules below, trains, tunes, writes `submission.csv` |
| `pipeline.py` | Orchestration: fold training, held-out cache, grid search, final ensemble, submission |
| `learned.py` | U-Net, training patches/augmentation, tiled inference (TTA, AMP, input rescaling), instance decoding and shape features |
| `pipeline.py` (Cellpose part) | Cellpose-SAM fine-tuning on cell-centred tiles, flow caching, threshold search on cached flows |
| `registration.py` | Affine priors, prior-based search, cell-density NCC search, refinement, greedy/Hungarian matching |
| `cellmatch.py` | I/O, normalization features, RLE encode/decode, PQ metric, submission writer |
| `validate_submission.py` | Strict format checks (29 rows, RLE bounds, no overlaps, one-to-one pairs) |
| `run_hpc.py`, `submit.sh`, `hpc/` | NYU HPC (Torch) entry point: `gpu` and `cpu` stages, SLURM job scripts, environment setup |
| `solve.py` | Original command-line pipeline (the 0.305 setup), kept for reference; defaults reproduce it |
| `train_model.py` | Original local training CLI |
| `export_features.py`, `tune_*.py` | Earlier analysis scripts; `features_cv.json` / `candidates_cv.json` are their saved held-out outputs |
| `models/` | Local width-16 checkpoints from an interrupted local run (not used by the notebook) |
| `submission*.csv` | Past submissions; **best verified: `submission_v10_cpgate.csv` (0.48893)**; finals: v10 + `submission_v7_grow15.csv` |
| `research/` | Offline labs: Hough/vote/window registration, margin gate, pair classifier, mask grow, Colab helpers |

## Pipeline

### 1. Segmentation
- **Input features:** two channels per image, (a) percentile-normalized intensity and (b) local
  contrast (image minus a Gaussian background), so cell bodies stand out under uneven lighting.
- **Model:** U-Net (default width 32) predicting a cell-body map and an eroded "core" (seed) map.
  Loss is weighted BCE + Dice. Augmentation: rotations, flips, gain/gamma, blur, noise.
- **Scale:** optional 2× upsampling before the network, since cells are only ~10 px wide
  (the Cellpose idea of rescaling cells to a size the network handles well).
  Cross-validation picks the scale per modality.
- **Instances:** marker-controlled watershed on the core map inside the body mask, plus size gates.
- **Compactness filters (ex-vivo):** per-instance core confidence, circularity, eccentricity and
  bounding-box fill remove bright axons and dendrites, which the competition notes look like somata.
- **Inference:** overlapping tiles, flip TTA, mixed precision on GPU, 2-seed ensemble for the final
  models.

### 1b. Cellpose-SAM (pretrained, fine-tuned)
- Model: `cpsam_v2` from Cellpose 4 (a SAM ViT-L encoder with Cellpose flow outputs), fine-tuned per
  modality, leave-one-mouse-out, then on all mice for the final model.
- **Scale:** training uses `rescale=True`, so images are resized so cells are ~30 px, the model's
  native size. Our cells are ~8–10 px, so this is about a 3–4× upsample. At inference the stored
  training diameter (`net.diam_labels`) reproduces the same upsampling.
- **Training tiles:** 128-px crops centred on labelled cells, normalized with whole-image
  percentiles so tiles match full-image inference. Random crops of a full upsampled ex-vivo canvas
  average under one labelled cell each.
- **Thresholds:** flows are cached once per region, then masks are rebuilt for a grid of
  `cellprob_threshold` × `flow_threshold`. Masks get the same size gates as the U-Net.
- **Hybrid source:** Cellpose masks kept or dropped by the U-Net's per-instance core probability and
  the same shape filters, pairing Cellpose boundaries with the U-Net's sense of which blobs
  annotators label.
- The search takes the best setting of each source per modality (U-Net, Cellpose, hybrid), keeps the
  top two per modality by held-out PQ, and picks the pair with the best full score.
- v3 held-out results (leave-one-mouse-out, 47 regions):

  | Source | In-vivo PQ | Ex-vivo PQ |
  | --- | --- | --- |
  | Cellpose-SAM fine-tuned | **0.727** | 0.332 |
  | U-Net ×1 | 0.590 | **0.354** |
  | U-Net ×2 | 0.496 | 0.319 |

  Better in-vivo masks raised eligible verified pairs from 450 to 586 of 1,139, and matching F1 to
  0.36 (precision 0.43, recall 0.31).

### 2. Registration (in-vivo → ex-vivo affine)
- **Priors:** two orientation clusters of the linear part, fitted from verified training pairs. In
  cross-validation they are refit **without** the held-out mouse (no leakage).
- **Candidates**, chosen by CV:
  - `none` / `soma`: translation histogram peaks for each prior ±5° / ±5% scale, scored by pair
    count (`soma` weights ex-vivo cells by brightness × circularity).
  - `ncc`: blur both cell-center sets into density maps and use normalized cross-correlation
    (`cv2.matchTemplate`) over **all** translations for each rotation/scale hypothesis.
    Normalization discounts dense clusters of false detections.
- **Refinement:** alternating nearest-pair search and RANSAC affine fits at shrinking radii.
- **Consistency gate:** within a mouse, the in-vivo field of view lands at nearly the same spot on
  the ex-vivo canvas (spread of 2–6% of the canvas). Regions whose alignment lands far from the
  mouse's consensus spot are almost always misaligned, so their pairs are dropped. The consensus is
  the densest cluster of predicted positions (ties go to the training-mean position). A plain median
  failed on test mouse `78b6a7`: its alignments split into two groups, the median fell between them,
  and all 12 regions (40% of the test set) got zero pairs in the 0.40870 submission. This uses
  predictions only, so it also works on the test mice.

### 3. Matching
- Greedy nearest neighbour or Hungarian one-to-one assignment within a distance threshold, with an
  optional ex-vivo brightness gate. Annotators favoured bright ex-vivo cells.
- Optional ratio test (Lowe-style): keep a pair only if the second-nearest ex-vivo candidate is at
  least 1.5× or 2× farther than the matched one, which drops ambiguous pairs.

### 4. Tuning (leak-free)
- Leave-one-mouse-out: each training region is predicted by a model that never saw its mouse.
- Grid search over scale, thresholds and filters (by PQ), then over registration, gate, matching
  method, distance and brightness (by the full competition score).
- The printed CV score is slightly optimistic because it is the best of many settings.

## Key measurements (held-out training predictions)

| Finding | Number |
| --- | --- |
| Held-out PQ with the original models | in-vivo 0.545, ex-vivo 0.337 |
| Verified pairs whose two masks both pass IoU > 0.75 | 450 / 1,139 → matching F1 can never exceed ~0.57 |
| True alignment is affine | median residual 1.9 px on verified pairs |
| Original registration misaligned (> 50 px) | 29 / 46 regions |
| Matching F1 at 5 px, original registration | 0.174 |
| Matching F1 at 5 px, `ncc` registration | 0.218 |
| Matching F1 with perfect alignment (upper bound, same masks) | 0.320 |
| `ncc` + consistency gate (0.06–0.10) | 0.257–0.260 (a perfect gate would give 0.263) |
| Registration with ground-truth centers (perfect segmentation) | still misaligned in 16 / 46 regions |

## Tried and dropped

| Idea | Result |
| --- | --- |
| Intensity-image cross-correlation for registration | worse: 4–10 / 46 regions aligned vs 20 / 46 for `ncc` |
| Search window around the typical field-of-view position | no gain: 14–18 / 46 aligned |
| Keeping only bright ex-vivo cells, or brightness-weighted density maps, for `ncc` | no gain |
| Gating regions by agreement of two registration methods | no gain |
| Chance-corrected inlier count as the gate | smaller gain than the position gate |
| Hungarian vs greedy matching | identical at 5 px on the saved features (kept as a CV option) |
| Restricting ex-vivo detections to the projected in-vivo field (plus margins up to 600 px) | worse: only 54% of annotated ex-vivo cells lie inside the field, and at wider margins annotations and predictions thin out equally |
| Wider NCC blur (5 px, 8 px) | worse: 22 / 46 and 12 / 46 regions aligned vs 26 / 46 at 2.4 px (ground-truth centers) |
| Coarse-to-fine NCC search (0.5° / 1% steps around the top 5) | no gain (24 / 46 with ground-truth centers). In 11 regions the correlation genuinely prefers a wrong alignment, 6 of them in `b2ba5e` |
| Checking for mirrored sections | none: every true transform has a positive determinant; `b2ba5e` mixes two rotations (about −15° and +9°) |
| Classical ex-vivo soma proposals + leave-one-mouse-out classifier (`research/ex_candidate_boost.py`) | no robust gain: best permissive setting fell from PQ 0.390 to 0.327; strict setting added no cells |
| Cell-level ex-vivo false-positive classifier (`research/ex_precision_filter.py`) | filtering lowered pooled held-out PQ and reachable verified pairs at every nonzero threshold |
| Sparse-label Cellpose loss and adaptive decoder (`research/partial_loss_*`, `research/partial_calibration.py`) | strong on two individual mice but worse with a shared threshold; 0.367 pooled ex-vivo PQ and 0.468 full CV, so no submission |
| Learned per-pixel mask-boundary correction (`research/boundary_pixel_cv.py`) | improves two mice but pooled ex-vivo PQ falls from 0.390 to 0.350; no submission |

## Current matching stack (`research/`)

Used for `submission_v7*.csv` (masks from the Cellpose ensemble; pairs rematched offline):

1. **Per-region Hough** over angle/scale → top refined affine candidates.
2. **Per-(mouse, canvas) vote** for consensus pose modes (angle ~3°, landing ~70 px).
3. **Windowed Hough** (±5°, radius 120) around those modes, then RANSAC refine.
4. **Margin gate:** keep pairs only if chosen score − best alternate pose ≥ 3 (25/25 correct on train).
5. **Pair classifier:** `HistGradientBoosting` on mutual-nearest candidates (threshold ~0.03–0.05).
6. **Ex-vivo grow (LB win):** expand each ex-vivo mask into the highest-cellprob boundary ring
   (15% of ring pixels). IDs and pairs unchanged. Full 1-px grow/shrink still hurts; ranked grow helps.

Held-out with this stack (v7 masks): in-vivo PQ ~0.74, ex-vivo PQ ~0.39, gated pair F1 ~0.46.
Oracle registration with the same masks only reaches pair F1 ~0.52 → score ceiling ~**0.55**.
Only ~10/29 test regions pass the margin gate; unlocking the rest without better masks still
cannot hit 0.65.

## Limits and next steps

- **Binding constraint is ex-vivo segmentation** (held-out PQ ~0.39, recall ~0.39). In-vivo is fine
  (~0.74). To approach **0.65** need roughly ex-vivo PQ ~0.65 and matching F1 ~0.60 together.
- **Retraining / ensemble masks** that shrunk ex-vivo cells lost LB score (0.39) even when CV PQ
  looked similar — IoU > 0.75 is unforgiving on ~10 px cells; test mice behave like the
  “larger GT” training mouse (`5d294c`).
- **Boundary convention differs by mouse:** GT median area is larger than preds on `5d294c`,
  smaller on `db6b8b`. Global shrink/grow fails; cellprob-ranked grow (15%) matched the public test.
- **Registration:** stretch hypothesis `0.92_75` lifts train correct regions 31→35, but on test it
  trades regions and does not unlock the 19 ungated ones. NCC and a learned pose scorer do not
  safely beat the margin ≥ 3 gate.
- **Data quirks:** `b2ba5e` is ~half resolution (cell diam ~8 vs ~11); two orientation modes
  (~−15° / +9°). Test has mixed canvas sizes; regions `7754ed` / `f05266` are duplicates.
- **Most promising next steps:**
  1. Raise ex-vivo recall/PQ (diameter per mouse, longer fine-tunes, threshold/decode tuned for
     reachable verified pairs — not PQ alone).
  2. Then rematch with the vote+window+margin stack and rebuild the pair classifier on new
     held-out masks.
  3. Only after masks improve: try unlocking low-margin test regions (wider search / image cues).
