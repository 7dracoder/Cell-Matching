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
| pending | Notebook v5: consistency check anchored on the densest cluster (fixes a test mouse that got zero pairs), longer Cellpose fine-tune competing with the current one, test-time Cellpose ensemble of final + fold models, grids widened at the edges again |

## How to run (main path)

1. Upload `CellMatch_Colab.ipynb` to Colab and set `Runtime > Change runtime type > GPU`
   (T4 works; A100/L4 is much faster for Cellpose-SAM).
2. Data: add Colab secrets `KAGGLE_USERNAME` / `KAGGLE_KEY`, or upload `testingpj-2.zip` when asked.
3. `Runtime > Run all`. Allow the Google Drive prompt: checkpoints go to `MyDrive/cellmatch_models`,
   so reruns skip finished training.
4. The last cell validates and downloads `submission.csv`. Stage 2 prints the chosen settings and
   the leave-one-mouse-out CV score (also saved as `best_config.json`).

With Cellpose-SAM, runs are long on a T4: fold fine-tunes plus inference on large, upsampled
ex-vivo images. Checkpoints on Drive are reused. The v5 `cellpose_long` variant adds six ~20-minute
fold fine-tunes on the first run. To save time, remove it from `CELLPOSE_VARIANTS` in the config
cell.

## How to run on NYU HPC (Torch)

Same pipeline as the notebook, as three chained SLURM jobs:
1. **Setup (CPU):** installs a self-contained Python environment and pre-downloads the Cellpose
   weights, all inside the project folder on `/scratch`.
2. **`gpu`:** all training plus held-out and test predictions, cached to `cache/`.
3. **`cpu`:** the search plus `submission.csv`.

The search runs as a CPU job because Torch cancels GPU jobs whose GPU utilization stays low, and
the search is CPU-bound. Every step skips work already on disk, so rerunning `submit.sh` after a
cancellation resumes where it stopped.

On your laptop (NYU VPN on):

```bash
scp cellmatch_hpc.tar.gz <NetID>@dtn.torch.hpc.nyu.edu:/scratch/<NetID>/
```

On Torch (`ssh <NetID>@login.torch.hpc.nyu.edu`):

```bash
cd /scratch/$USER && tar xzf cellmatch_hpc.tar.gz && cd cellmatch
my_slurm_accounts                 # pick your account, e.g. torch_pr_XXX_XXXXX
bash submit.sh torch_pr_XXX_XXXXX
squeue -u $USER                   # job states
tail -f logs/cellmatch-*.out      # progress
```

Then copy the result back to your laptop:

```bash
scp <NetID>@dtn.torch.hpc.nyu.edu:/scratch/<NetID>/cellmatch/submission.csv .
```

Notes:
- The chosen settings are saved in `best_config.json`. Delete it to force a new search.
- If a job fails, its dependents stay pending with `DependencyNeverSatisfied`. Cancel them with
  `scancel -u $USER`, fix the problem (see `logs/`), and rerun `submit.sh`.
- `/scratch` is purged after 60 days without access. Copy anything you want to keep.

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
| `submission*.csv` | Past submissions (`submission (2).csv` scored 0.37958) |

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

## Limits and next steps

- **Ceiling:** with the current masks and perfect registration the score would be roughly 0.40. The
  binding constraint is segmentation at IoU > 0.75 on ~10 px cells. It caps matching recall, and
  about half of the regions still misalign even with perfect cell centers, because the two images
  share only ~24 verified cells per region.
- **Ex-vivo segmentation is now the weakest part** (held-out PQ 0.354). The Stage 2 breakdown
  (TP/FP/FN, mean IoU of true positives, near misses at IoU 0.5–0.75) shows whether the errors are
  detections or boundaries.
- **CV vs leaderboard:** v3 CV 0.450 vs leaderboard 0.402; v4 CV 0.485 vs 0.409. Part is selection
  optimism. A larger part was the gate bug above, which dropped every pair for one test mouse.
- **v4 PQ breakdown (held-out):** in-vivo TP 12,125 / FP 1,242 / FN 2,768, mean TP IoU 0.856, 690
  near misses. Ex-vivo TP 2,545 / FP 2,702 / FN 2,971, mean TP IoU 0.825, 1,516 near misses. Each
  near miss counts as both a false positive and a false negative; fixing them all would take
  ex-vivo PQ to roughly 0.60.
- **The near misses are not a correctable bias:** predicted/true area ratio splits ~50/50 (in-vivo)
  and 37/62 (ex-vivo). Growing or shrinking every mask by one pixel is catastrophic (in-vivo PQ 0.51
  → 0.25 on the local U-Net). Centroid offsets are under 0.2 px. Ground-truth masks are not ellipses
  (median IoU 0.88 against a moment-matched ellipse) and never overlap.
- **Mouse `b2ba5e`** (rectangular 737×1085 canvas) is misaligned almost everywhere (F1 ≈ 0.04).
  Mice with rectangular canvases are the weakest case.
- **Most promising next steps:**
  1. If Cellpose-SAM wins, tune its fine-tuning (epochs, tile count, ex-vivo diameter) and try
     combining it with the U-Net (e.g. keep Cellpose masks only where the U-Net core map agrees).
  2. Shape- and appearance-aware partial cell-set matching (Chen et al.-style) instead of
     center-only alignment, followed by non-rigid refinement (e.g. coherent point drift).
  3. A learned pair classifier. For the metric, pairs missing from the verified list count as false
     positives, so learning which cells annotators verify is aligned with the score.
