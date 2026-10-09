# Requirements Document

## Introduction

This feature packages the Cell Matching x Segmentation project (CS-GY-6643, Kaggle `testingpj-2`) so the user can upload the whole project folder to NYU HPC Cloud Bursting and run a stronger registration ("unlock") pipeline from one JupyterLab notebook. Heavy computation runs only on HPC as SLURM jobs; nothing heavy runs on the local machine.

The score gap is registration. Only 11 of 29 test regions currently receive pairs, and held-out training aligns 29–31 of 47 regions. The pipeline adds a wider affine pose search scored by a soft mutual-nearest objective, joint per-mouse registration, a learned pose verifier, and a retrained pair classifier.

The default Job_Chain uses both CPU and GPU Stages. The GPU is used where it helps: (a) a dense GPU pose scan that evaluates the soft objective over the full pose and translation grid, and (b) ex-vivo Cellpose self-training and test inference. CPU-bound work (`registration.refine`, HistGradientBoosting, validation, CSV writing) stays on CPU partition `n2c48m24`, because Burst auto-kills GPU jobs with low GPU utilization after about 20 minutes. Registration-only candidates keep masks and instance IDs identical to `submission_v10_cpgate.csv` and change only `match_pairs`. The self-trained candidate keeps the in-vivo masks and changes only the ex-vivo masks, `exvivo_instances` and `match_pairs`.

The target is a public score above 0.65. That target is not guaranteed. The estimated ceiling with v10 masks and better registration is about 0.60–0.63. Self-training of the ex-vivo segmentation is the route beyond that ceiling, and only if it validates. A candidate CSV is written only when leave-one-mouse-out validation beats the current baseline: full held-out score 0.5186, pair F1 0.472 (v10 gate).

## Glossary

- **Project_Folder**: The project root (`Assign-2/`) that the user uploads to `/scratch/$USER/cellmatch` on Burst. It contains `Project_2_Dataset/`, `research/`, `hpc/` and the notebooks.
- **HPC_Notebook**: The JupyterLab notebook (`CellMatch_HPC.ipynb`, extended) that the user opens through Burst OOD. It submits and monitors the SLURM jobs.
- **Job_Chain**: The SLURM job sequence: environment setup, GPU_Pose_Scan, CPU registration Stages (Pose_Search, Joint_Registrar, Pose_Verifier, Pair_Classifier, Validator), GPU Self_Training_Stage, and final CPU assembly. Each GPU Stage is preceded by a CPU Stage that writes its input Checkpoint.
- **Stage**: One resumable unit of work in the Job_Chain. Each Stage writes its own Checkpoint.
- **Checkpoint**: The output file of a Stage under `research/data/hpc/`. A completed Checkpoint is marked by a done-marker file.
- **Held_Out_Set**: The 47 training regions, evaluated leave-one-mouse-out. 46 regions have a ground-truth affine transform.
- **Test_Set**: The 29 hidden-test regions from 2 test mice.
- **Pose**: A 2×3 affine transform mapping in-vivo centroids onto the ex-vivo canvas.
- **Correct_Pose**: A Pose whose error against the ground-truth transform is below 5 px under `reg_lab.err`.
- **Soft_Score**: The sum over mutual-nearest centroid pairs of exp(−d²/2σ²), with σ configurable in [1.5, 2.5] px.
- **Soft_Margin**: The Soft_Score of the chosen Pose minus the best Soft_Score among clearly different candidates (angle difference > 3° or landing distance > 60 px).
- **GPU_Pose_Scan**: The GPU Stage that evaluates the Soft_Score densely (batched in torch) over angle × scale × stretch × translation for every region, and outputs the top-K candidate Poses per region for CPU refinement.
- **Pose_Search**: The CPU component that generates Pose candidates with a mode-free wide similarity Hough plus anisotropic stretch hypotheses, merges them with the GPU_Pose_Scan candidates, refines them with `registration.refine`, and rescores them with the Soft_Score.
- **Joint_Registrar**: The component that re-selects Poses for all regions of one mouse and canvas together. It uses mosaic crop offsets (`research/common.crop_offsets`), shared-canvas landing consistency, smooth section-to-section variation, and duplicate regions.
- **Pose_Verifier**: A classifier trained leave-one-mouse-out that predicts whether a region's chosen Pose is a Correct_Pose.
- **Pair_Classifier**: The `research/pair_clf` HistGradientBoosting model that keeps or drops mutual-nearest pairs. It is retrained on the new Poses.
- **Validator**: The leave-one-mouse-out evaluator that computes held-out PQ, pair F1 and the full score with the competition formula.
- **Baseline**: `submission_v10_cpgate.csv` (public 0.48893), with held-out full score 0.5186 and pair F1 0.472.
- **Candidate_Writer**: The component that writes candidate submission CSVs from Baseline masks plus new pairs, or from self-trained ex-vivo masks plus new pairs.
- **Self_Training_Stage**: The GPU Stage, run by default, that fine-tunes ex-vivo Cellpose-SAM (`cpsam`) on pseudo-labels from confidently registered regions and runs test inference. The user can disable it with a flag in the HPC_Notebook config cell.
- **Matched_Invivo_Instance**: An in-vivo instance that belongs to a pair output by the Pair_Classifier for its region.
- **Run_Report**: The summary files (`hpc_unlock_report.json` and `hpc_unlock_report.md`) that describe the Stage results and rank the candidates.
- **Format_Checker**: The existing `validate_submission.py`.

## Requirements

### Requirement 1: Upload-and-run project packaging

**User Story:** As the user, I want to upload the whole project folder to HPC and run it there unchanged, so that I do not have to edit paths or copy files by hand.

#### Acceptance Criteria

1. THE Project_Folder SHALL resolve every data, cache and output path relative to the Project_Folder root, so that the Job_Chain runs from `/scratch/$USER/cellmatch` or any other upload location with zero file edits.
2. THE Project_Folder SHALL include every cached input the Stages need (`research/data/` caches, `Project_2_Dataset/`, Baseline CSV, `submission_v7_grow15.csv`, and, in `singularity` environment mode, the ext3 overlay file and the CUDA Singularity image).
3. WHEN the setup Stage runs, THE Job_Chain SHALL reuse an existing Python environment (the conda environment inside the Singularity ext3 overlay in `singularity` mode, the default, or the project venv in `venv` mode) only if it already contains the pinned versions of numpy, scipy, opencv-python-headless, scikit-learn, tifffile and pandas, and SHALL otherwise create a new environment of the same mode with those pinned versions.
4. THE setup Stage SHALL install a pinned torch version into the selected environment in every Job_Chain run.
5. WHILE the self-training disable flag is unset, THE setup Stage SHALL install into the selected environment a pinned cellpose version that provides the Cellpose-SAM (`cpsam`) model.
6. IF one or more required input files are missing or unreadable when the Job_Chain starts, THEN THE Job_Chain SHALL stop before any computation Stage runs, log every missing or unreadable path (not only the first), and exit with a non-zero status.
7. IF installing any pinned package fails during the setup Stage, THEN THE Job_Chain SHALL log the name of the failing package, exit the setup Stage with a non-zero status, and run no dependent Stage.

### Requirement 2: Notebook orchestration on Burst

**User Story:** As the user, I want one notebook that tells me exactly which cells to run and what output to expect, so that I can run the full pipeline from JupyterLab.

#### Acceptance Criteria

1. THE HPC_Notebook SHALL contain a numbered instruction cell before each action cell. Each instruction cell SHALL name the cell to run and the expected output.
2. THE HPC_Notebook SHALL contain a config cell, placed before the submit cell, that sets the GPU partition (`g2-standard-12` by default, `c12m85-a100-1` selectable) and the self-training disable flag (unset by default).
3. IF the configured GPU partition is neither `g2-standard-12` nor `c12m85-a100-1`, THEN THE HPC_Notebook SHALL display an error naming the invalid value and SHALL submit no Stage.
4. WHEN the user runs the submit cell, THE HPC_Notebook SHALL submit each CPU and GPU Stage of the Job_Chain with account `cs_gy_6923-2026fa`, explicit partition and memory flags on the `sbatch` command line, a request for exactly 1 GPU on each GPU Stage, and `--export=NONE`, with each Stage set to start only after its preceding Stage completes successfully (`--dependency=afterok:<job id>`).
5. WHEN the submit cell finishes, THE HPC_Notebook SHALL print the Stage name, partition and SLURM job ID of every submitted Stage.
6. WHEN the user runs the monitor cell, THE HPC_Notebook SHALL show the `squeue` status of each Job_Chain job, the last 40 lines of each Stage log (or a note that the log does not exist yet), and the list of Checkpoints that have a done-marker.
7. WHEN the user runs the report cell after all Stages finish, THE HPC_Notebook SHALL display the Run_Report and the paths of all candidate CSVs, or state that no candidate CSV was written.
8. IF a Stage job ends in a failed state (failed, timed out, cancelled, out of memory, or non-zero exit code), THEN THE HPC_Notebook SHALL show, when the monitor cell runs, the failing Stage name, its log path, and the command that resubmits the Job_Chain from that Stage.
9. IF an `sbatch` submission returns an error, THEN THE HPC_Notebook SHALL display the error output and SHALL submit no later Stage of the Job_Chain.

### Requirement 3: Resource placement and resumability

**User Story:** As the user, I want CPU-heavy work on CPU nodes, GPU-heavy work on GPU nodes, and reruns that skip finished work, so that jobs are not auto-killed and no compute is wasted.

#### Acceptance Criteria

1. THE Job_Chain SHALL run Pose_Search, Joint_Registrar, Pose_Verifier, Pair_Classifier, Validator and Candidate_Writer on partition `n2c48m24`.
2. THE Job_Chain SHALL use a worker count equal to `SLURM_CPUS_PER_TASK` when it is set, and otherwise equal to the number of CPU cores available to the job process, and SHALL never use more workers than the allocated cores.
3. THE Job_Chain SHALL run the GPU_Pose_Scan and the Self_Training_Stage on the GPU partition set in the config cell (`g2-standard-12` by default, `c12m85-a100-1` selectable) as separate jobs that contain only GPU work.
4. WHILE a GPU Stage job runs, THE GPU Stage SHALL log the GPU utilization percentage and GPU memory in use at least once per minute to its Stage log.
5. THE GPU_Pose_Scan and the Self_Training_Stage SHALL process their work in batches on the GPU and SHALL run none of `registration.refine`, HistGradientBoosting training or inference, the Validator, or the Candidate_Writer.
6. THE Job_Chain SHALL run any CPU preprocessing that a GPU Stage needs in a preceding CPU Stage on partition `n2c48m24`. That CPU Stage SHALL write a Checkpoint, and the GPU Stage SHALL load its inputs from that Checkpoint.
7. WHEN a Stage starts and both its done-marker and its Checkpoint exist and the Checkpoint loads without error, THE Stage SHALL skip computation and reuse its Checkpoint.
8. IF a Stage starts and its done-marker exists but its Checkpoint is missing or fails to load, THEN THE Stage SHALL delete the done-marker, log the inconsistency, and recompute the Stage.
9. WHEN a Stage completes, THE Stage SHALL write its Checkpoint atomically (write to a temporary file, then rename) before writing its done-marker.
10. IF a Stage exits before completion, THEN THE Stage SHALL leave no done-marker, so that the next run of that Stage recomputes it.
11. FOR ALL Checkpoints, loading a saved Checkpoint SHALL produce objects equal to those saved, with arrays identical in shape, dtype and every element (round-trip property).

### Requirement 4: Wide affine pose search with soft objective

**User Story:** As the user, I want a denser, mode-free affine pose search scored by the soft objective, so that more regions have the correct Pose available and chosen.

#### Acceptance Criteria

1. THE Pose_Search SHALL generate candidates over angles −35° to 35° and scales 0.85 to 1.13, both ranges inclusive, with step sizes no larger than 1° and 0.02.
2. THE Pose_Search SHALL include anisotropic stretch hypotheses with stretch factors 0.92 and 1.08 at directions 0°, 45°, 90° and 135°.
3. THE Pose_Search SHALL refine each candidate with `registration.refine` and SHALL treat two refined candidates as duplicates when every linear-part entry differs by at most 0.01 and the translations differ by at most 4 px, keeping only the duplicate with the higher Soft_Score.
4. THE Pose_Search SHALL attach to each candidate its refine score, Soft_Score, cellprob z, refine margin, Soft_Margin, angle, scale, anisotropy, landing position and source (Hough, GPU_Pose_Scan, window or vote).
5. THE Pose_Search SHALL include the existing window and vote candidates in each region's candidate list.
6. THE Pose_Search SHALL include every GPU_Pose_Scan candidate of a region in that region's candidate list, refined with `registration.refine` and deduplicated under criterion 3.
7. THE Pose_Search SHALL process all 47 Held_Out_Set regions and all 29 Test_Set regions with identical parameter values, including the same σ, angle and scale ranges, step sizes and stretch hypotheses.
8. IF the configured Soft_Score σ is outside [1.5, 2.5] px, THEN THE Pose_Search SHALL stop before processing any region and log an error indicating the invalid σ value.
9. IF the Pose_Search produces zero candidates for a region, THEN THE Pose_Search SHALL record that region as having zero candidates in its Checkpoint and continue with the remaining regions.
10. WHEN the Pose_Search finishes on the Held_Out_Set, THE Run_Report SHALL record, out of the 46 regions with a ground-truth transform, how many have a Correct_Pose among their candidates and how many have a Correct_Pose ranked first by Soft_Score, together with the σ value used.

### Requirement 5: GPU dense pose scan

**User Story:** As the user, I want the pose search run densely on the GPU, so that the correct Pose is found in more regions than the sparse Hough search finds.

#### Acceptance Criteria

1. THE GPU_Pose_Scan SHALL evaluate a grid of angles from −35° to 35° and scales from 0.85 to 1.13, with both endpoints of each range included in the grid and with step sizes no larger than 0.5° for angle and 0.01 for scale.
2. THE GPU_Pose_Scan SHALL combine every grid angle and scale with each of 17 stretch hypotheses: stretch factors {0.92, 0.96, 1.04, 1.08} at each direction in {0°, 45°, 90°, 135°} (16 hypotheses), plus stretch factor 1.00 evaluated exactly once with no direction, because factor 1.00 gives the same Pose at every direction.
3. THE GPU_Pose_Scan SHALL evaluate, for every grid Pose, every translation on a grid with spacing at most 2 px in both x and y for which the landing position of the transformed in-vivo centroids lies inside the ex-vivo canvas bounds.
4. THE GPU_Pose_Scan SHALL score each grid Pose and translation with either the Soft_Score or the correlation of a Gaussian-splatted ex-vivo centroid map with a Gaussian-splatted transformed in-vivo centroid map, where the Gaussian σ equals the configured Soft_Score σ (in [1.5, 2.5] px) used by the Pose_Search in the same run.
5. THE GPU_Pose_Scan SHALL keep, per region, up to K local maxima of the score over the pose and translation grid, selected in descending score order and skipping any maximum that differs from an already kept candidate by at most 3° in angle and at most 20 px in landing position, where K is a configurable integer in [1, 500] with default 50.
6. IF a region has fewer than K local maxima that satisfy the separation rule in criterion 5 (including zero), THEN THE GPU_Pose_Scan SHALL keep all of them and record the kept count for that region in its Checkpoint.
7. THE GPU_Pose_Scan SHALL write to its Checkpoint, for each kept candidate, the 2×3 Pose, its scan score, angle, scale, stretch factor, stretch direction (recorded as not applicable for stretch factor 1.00) and translation, and THE Pose_Search SHALL load every kept candidate from that Checkpoint for `registration.refine` and merging with the Hough, window and vote candidates.
8. THE GPU_Pose_Scan SHALL process all 47 Held_Out_Set regions and all 29 Test_Set regions with identical angle grid, scale grid, stretch hypotheses, translation spacing, σ and K values.
9. IF no GPU is visible to the GPU_Pose_Scan job outside smoke mode, THEN THE GPU_Pose_Scan SHALL stop before processing any region, log an error stating that no GPU is visible, leave no done-marker, and exit with a non-zero status.
10. WHEN the Pose_Search finishes on the Held_Out_Set, THE Run_Report SHALL record, out of the 46 regions with a ground-truth transform, how many have a Correct_Pose among the raw GPU_Pose_Scan candidates and how many have a Correct_Pose among the merged and refined Pose_Search candidates.
11. IF the configured K is not an integer in [1, 500], THEN THE GPU_Pose_Scan SHALL stop before processing any region, log an error indicating the invalid K value, leave no done-marker, and exit with a non-zero status.
12. IF the GPU_Pose_Scan fails while processing a region (including GPU out-of-memory), THEN THE GPU_Pose_Scan SHALL log the failing region identifier and the failure cause, leave no done-marker, and exit with a non-zero status.

### Requirement 6: Joint per-mouse registration

**User Story:** As the user, I want regions of one mouse registered jointly, so that confident regions constrain uncertain ones.

#### Acceptance Criteria

1. THE Joint_Registrar SHALL group regions by mouse and ex-vivo canvas shape, so that each region belongs to exactly one group.
2. THE Joint_Registrar SHALL pick each region's Pose from that region's Pose_Search candidate list. The pick SHALL maximize a joint objective: summed per-region Soft_Score plus a weighted consistency term over landing positions, mosaic crop offsets (`research/common.crop_offsets`) and Pose linear parts within the group. The consistency weight SHALL be configurable and non-negative.
3. WHEN the consistency weight is 0 or a group contains exactly 1 region, THE Joint_Registrar SHALL pick, for each region, the candidate with the highest Soft_Score (the same result as independent per-region selection).
4. IF two regions have pixel-identical in-vivo images and pixel-identical ex-vivo images (for example `7754ed` and `f05266`), THEN THE Joint_Registrar SHALL give both regions the same Pose.
5. THE Joint_Registrar SHALL use only predicted centroids, image-derived evidence and priors from training mice other than the evaluated mouse. It SHALL use no ground-truth transform or ground-truth pair of the evaluated mouse. On the Test_Set it SHALL use priors from all three training mice.
6. IF a region has zero Pose_Search candidates, THEN THE Joint_Registrar SHALL mark the region as unregistered, leave it out of its group's consistency term, and assign it no Pose.
7. WHEN the Joint_Registrar finishes on the Held_Out_Set, THE Run_Report SHALL record, in total and per mouse, the number of Correct_Pose selections under joint selection and under independent per-region selection.

### Requirement 7: Learned pose verifier and region gate

**User Story:** As the user, I want a learned verifier that decides which regions' Poses are trustworthy, so that more correct regions are unlocked without admitting wrong ones.

#### Acceptance Criteria

1. THE Pose_Verifier SHALL use these features per region: Soft_Score, Soft_Margin, cellprob z, refine margin, landing distance to the group consensus, angle, scale and anisotropy.
2. THE Pose_Verifier SHALL label each Held_Out_Set region as positive if its chosen Pose is a Correct_Pose and negative otherwise. The 1 region without a ground-truth affine transform SHALL be left out of training and out of the correct/wrong counts.
3. THE Pose_Verifier SHALL be trained leave-one-mouse-out on the Held_Out_Set. Each held-out mouse's predictions SHALL come from a model trained without that mouse. Test_Set predictions SHALL come from a model trained on all three training mice.
4. THE Pose_Verifier SHALL output a probability in [0, 1] per region. For each threshold from 0.00 to 1.00 in steps of 0.05 (21 values), it SHALL report the number of kept regions, kept Correct_Poses, kept wrong Poses, and the held-out pair F1.
5. THE Pose_Verifier SHALL set the conservative gate to the lowest grid threshold with zero kept wrong Poses on the Held_Out_Set.
6. THE Pose_Verifier SHALL set the aggressive gate to the grid threshold with the highest held-out pair F1. If several thresholds tie, it SHALL use the highest of them.
7. IF a region passes the v10 gate (margin ≥ 3 or cellprob z ≥ 5) or its verifier probability is at or above the selected gate threshold, THEN THE Pose_Verifier SHALL mark the region as kept. Otherwise it SHALL mark the region as dropped.
8. IF no grid threshold gives zero kept wrong Poses, THEN THE Pose_Verifier SHALL record that the conservative gate is unavailable in the Run_Report and SHALL apply only the v10 gate for the conservative configuration.

### Requirement 8: Retrained pair classifier

**User Story:** As the user, I want the pair classifier retrained on the new Poses, so that pair selection matches the new registration.

#### Acceptance Criteria

1. THE Pair_Classifier SHALL be retrained leave-one-mouse-out on mutual-nearest candidates computed under the newly selected Poses. A candidate SHALL be labeled positive only if it matches a ground-truth pair. Each held-out mouse's predictions SHALL come from a model trained without that mouse.
2. THE Pair_Classifier SHALL pick its probability threshold from 13 values, 0.000 to 0.300 in steps of 0.025, by maximizing held-out pooled pair F1. If several thresholds tie, it SHALL use the lowest of them.
3. THE Pair_Classifier SHALL produce one-to-one pairs within each region: each in-vivo instance and each ex-vivo instance SHALL appear in at most one output pair. Conflicts SHALL be resolved in favor of the candidate with the higher predicted probability.
4. IF the Pose_Verifier marks a region as dropped or the Joint_Registrar marks it as unregistered, THEN THE Pair_Classifier SHALL output zero pairs for that region.
5. WHEN applied to the Test_Set, THE Pair_Classifier SHALL use a model trained on all three training mice, with the same features as the leave-one-mouse-out models and the threshold chosen in criterion 2.

### Requirement 9: Held-out validation and acceptance rule

**User Story:** As the user, I want every candidate validated leave-one-mouse-out against the Baseline, so that only improvements become submissions.

#### Acceptance Criteria

1. THE Validator SHALL compute held-out full score = 0.25·PQ_invivo + 0.25·PQ_exvivo + 0.5·F1_pairs over all 47 Held_Out_Set regions. PQ SHALL be computed per region with IoU > 0.75 as the match rule, then averaged over regions.
2. THE Validator SHALL compute F1_pairs by summing true-positive, false-positive and false-negative pair counts over all held-out regions before computing F1. A predicted pair SHALL count as a true positive only if both masks are PQ true positives and the matched ground-truth instances form a ground-truth pair.
3. WHEN the Validator evaluates the Baseline configuration, THE Validator SHALL reproduce a held-out pair F1 of 0.472 ± 0.005 and a full score of 0.5186 ± 0.005.
4. IF the Baseline reproduction falls outside either tolerance in criterion 3, THEN THE Validator SHALL stop the Validator Stage without writing its done-marker, and SHALL record the measured Baseline values and a reproduction failure in the Run_Report. THE Candidate_Writer SHALL write no candidate CSV in that run.
5. IF a configuration's held-out full score is at most the measured Baseline held-out full score (compared unrounded), THEN THE Candidate_Writer SHALL write no CSV for that configuration and SHALL record `NO_CANDIDATE` with both scores in the Run_Report.
6. THE Validator SHALL report, for every evaluated configuration, the full score, PQ_invivo, PQ_exvivo and pooled pair F1 to 4 decimal places. It SHALL also report the pair F1 of each of the 3 training mice and the count of kept regions whose Pose is not a Correct_Pose.

### Requirement 10: Candidate submission files

**User Story:** As the user, I want ranked candidate CSVs that keep the v10 masks, so that I can submit them safely and keep v10 as a fallback.

#### Acceptance Criteria

1. THE Candidate_Writer SHALL copy the region rows from the Baseline unchanged into each registration-only candidate: the same set of region identifiers, plus each region's `invivo_instances`, `exvivo_instances`, in-vivo masks, ex-vivo masks and instance IDs. It SHALL change only `match_pairs`. The self-trained candidate follows Requirement 12, criterion 8.
2. FOR ALL registration-only candidate CSVs, decoding every RLE mask SHALL produce masks identical to the decoded Baseline masks; FOR the self-trained candidate CSV, decoding every in-vivo RLE mask SHALL produce masks identical to the decoded Baseline in-vivo masks (round-trip property).
3. WHEN the Candidate_Writer writes a CSV, THE Candidate_Writer SHALL run the Format_Checker on that CSV before reporting it as a candidate.
4. THE Candidate_Writer SHALL write at most one conservative and one aggressive candidate per accepted configuration. File names SHALL follow `submission_v13_<config>.csv`, where `<config>` identifies both the configuration and the gate (conservative or aggressive), so that no two candidates share a file name.
5. THE Candidate_Writer SHALL leave `submission_v10_cpgate.csv` and `submission_v7_grow15.csv` byte-identical before and after every Job_Chain run.
6. WHEN a candidate CSV is written, THE Run_Report SHALL list, for each of the 29 Test_Set regions, the candidate's pair count, the Baseline's pair count, and whether the region is newly unlocked. A region is newly unlocked if its Baseline pair count is 0 and its candidate pair count is at least 1.
7. IF the Format_Checker reports a failure for a CSV, THEN THE Candidate_Writer SHALL delete that CSV and record the configuration name and the failure in the Run_Report as a rejected candidate.
8. THE Candidate_Writer SHALL write `match_pairs` so that every pair references an in-vivo instance ID and an ex-vivo instance ID that exist in that region's masks as written in the same CSV (Baseline masks for registration-only candidates; Baseline in-vivo masks and self-trained ex-vivo masks for the self-trained candidate). Each instance ID SHALL appear in at most one pair per region.

### Requirement 11: Run report

**User Story:** As the user, I want a short report that ranks the candidates, so that I can pick my Kaggle submissions and finals.

#### Acceptance Criteria

1. WHEN the final assembly Stage completes, THE Job_Chain SHALL write the Run_Report as both `hpc_unlock_report.json` and `hpc_unlock_report.md` in the Project_Folder root. The two files SHALL contain the same candidate ranking and scores.
2. THE Run_Report SHALL rank accepted candidates by held-out full score, highest first. Ties SHALL be broken by higher held-out pair F1, then by fewer kept wrong-Pose regions.
3. THE Run_Report SHALL state, for each accepted candidate and for the Baseline, these Held_Out_Set values: held-out full score, pair F1, number of kept regions, and number of kept wrong-Pose regions.
4. THE Run_Report SHALL state that held-out gains are estimates and that a public score above 0.65 is the target but not guaranteed.
5. IF at least one configuration is accepted, THEN THE Run_Report SHALL recommend two finals: the highest-ranked accepted candidate and `submission_v10_cpgate.csv`. It SHALL also state that `submission_v10_cpgate.csv` should be replaced by the next-ranked accepted candidate only if the user sees a candidate public Kaggle score above 0.48893.
6. IF no configuration is accepted, THEN THE Run_Report SHALL state `NO_CANDIDATE`, list each rejected configuration with its measured held-out full score, and recommend `submission_v10_cpgate.csv` and `submission_v7_grow15.csv` as the two finals.

### Requirement 12: GPU self-training stage

**User Story:** As the user, I want a default stage that self-trains ex-vivo Cellpose on test images using confident registrations, so that ex-vivo PQ and pair F1 can rise beyond what registration alone reaches.

#### Acceptance Criteria

1. WHILE the self-training disable flag in the HPC_Notebook config cell is unset, THE Job_Chain SHALL run the Self_Training_Stage, including its preceding CPU preprocessing Stage, as part of the Job_Chain.
2. IF the self-training disable flag is set when the user runs the submit cell, THEN THE Job_Chain SHALL submit neither the Self_Training_Stage GPU job nor its CPU preprocessing Stage, SHALL write no self-trained candidate CSV, and SHALL record in the Run_Report that the Self_Training_Stage was disabled.
3. THE Self_Training_Stage SHALL build ex-vivo pseudo-labels only from regions kept under the conservative configuration of Requirement 7 (the v10 gate alone when the conservative gate is unavailable). In each such region it SHALL project the centroid of every Matched_Invivo_Instance through the region's chosen Pose onto the ex-vivo canvas; SHALL retain as a pseudo-label each predicted ex-vivo instance whose centroid lies within the match radius of at least one projected centroid; SHALL seed, at each projected centroid with no predicted ex-vivo instance centroid within the match radius, a new pseudo-label cell as a filled disk of the seed radius clipped to the canvas bounds; and SHALL exclude every other predicted ex-vivo instance from the pseudo-labels. The match radius and the seed radius SHALL each be configurable in [1, 20] px with default 6 px.
4. THE Self_Training_Stage SHALL fine-tune Cellpose-SAM (`cpsam`), starting from the pretrained `cpsam` weights, on ex-vivo images and ex-vivo pseudo-labels only, with no in-vivo images and no ground-truth masks as training targets.
5. THE Self_Training_Stage SHALL be evaluated leave-one-mouse-out on the Held_Out_Set before any Test_Set fine-tuning. For each of the 3 held-out mice, the evaluation SHALL start a separate model from the pretrained `cpsam` weights; SHALL build that mouse's pseudo-labels only from predicted in-vivo and ex-vivo masks, the Poses chosen by the Joint_Registrar, Pair_Classifier and Pose_Verifier predictions from models trained without that mouse, and the conservative gate; SHALL use no ground-truth mask, ground-truth pair or ground-truth transform of that mouse to build pseudo-labels or to fine-tune; SHALL fine-tune on that mouse's ex-vivo images; and SHALL run inference on the ex-vivo images of all of that mouse's regions. The Validator SHALL then compute the held-out full score over all 47 Held_Out_Set regions using the Baseline in-vivo masks and the self-trained ex-vivo masks, using that mouse's ground truth only for scoring.
6. WHEN the Self_Training_Stage produces new ex-vivo masks (held-out or Test_Set), THE Job_Chain SHALL recompute mutual-nearest candidates under the chosen Poses and rerun the Pair_Classifier (same features and threshold-selection rule as Requirement 8) on the new ex-vivo masks in a CPU Stage on partition `n2c48m24` before validation and CSV writing.
7. IF the held-out full score of the self-trained masks is at most the best accepted registration-only held-out full score (or, when no registration-only configuration is accepted, the measured Baseline held-out full score), compared unrounded, THEN THE Self_Training_Stage SHALL write no CSV and SHALL record `NO_CANDIDATE` with the measured score and the score it was compared against in the Run_Report.
8. WHEN the held-out full score of the self-trained masks exceeds the best accepted registration-only held-out full score (or, when no registration-only configuration is accepted, the measured Baseline held-out full score), compared unrounded, THE Candidate_Writer SHALL write exactly one self-trained candidate CSV. That CSV SHALL keep the Baseline in-vivo masks and `invivo_instances` unchanged, SHALL change only the ex-vivo masks, `exvivo_instances` and `match_pairs`, and SHALL pass the Format_Checker.
9. WHEN the Self_Training_Stage fine-tunes on the Test_Set, THE Self_Training_Stage SHALL fine-tune a single model, starting from the pretrained `cpsam` weights, on pseudo-labels built under criterion 3 from the conservatively gated Test_Set regions, and SHALL run inference with that model on the ex-vivo images of all 29 Test_Set regions in the same GPU job.
10. THE Job_Chain SHALL run all CPU-only preprocessing for the Self_Training_Stage (pseudo-label construction and image loading) in a preceding CPU Stage on partition `n2c48m24`. That Stage SHALL write a Checkpoint, and the GPU job SHALL load the Checkpoint and contain only GPU training and inference.
11. IF the conservative gate keeps 0 Test_Set regions, THEN THE Self_Training_Stage SHALL skip Test_Set fine-tuning, write no CSV, and record `NO_CANDIDATE` with the reason "no confident regions" in the Run_Report.
12. IF the conservative gate keeps 0 regions of a held-out mouse, THEN THE Self_Training_Stage SHALL skip fine-tuning for that mouse, SHALL use the Baseline ex-vivo masks for that mouse's regions in the held-out full score, and SHALL record that mouse as having no confident regions in the Run_Report.
13. IF the configured match radius or seed radius is outside [1, 20] px, THEN THE Self_Training_Stage preprocessing Stage SHALL stop before building any pseudo-label, log an error indicating the invalid value, leave no done-marker, and exit with a non-zero status.
14. IF fine-tuning or inference in the Self_Training_Stage GPU job fails (including GPU out-of-memory or no visible GPU outside smoke mode), THEN THE Self_Training_Stage SHALL log the failure cause, leave no done-marker, write no self-trained candidate CSV, and exit with a non-zero status.

### Requirement 13: No heavy local execution

**User Story:** As the user, I want the heavy jobs to run only on HPC, so that my laptop is not tied up.

#### Acceptance Criteria

1. WHEN the user runs the local smoke test, THE Project_Folder SHALL run each CPU Stage on at most 2 Held_Out_Set regions, using at most 4 CPU cores, and SHALL finish within 300 seconds.
2. WHEN the user runs the local smoke test, THE Project_Folder SHALL run the GPU_Pose_Scan in smoke mode with CPU torch on at most 2 Held_Out_Set regions, with a grid of at most 5 angles, 3 scales and 1 stretch hypothesis, within the same 300-second limit.
3. THE local smoke test SHALL run no Cellpose fine-tuning.
4. WHEN the local smoke test finishes, THE Project_Folder SHALL report pass or fail for each Stage. If a Stage fails, it SHALL also report the name of the failing Stage.
5. THE local smoke test SHALL write its outputs separately from full-run Checkpoints. It SHALL not create or modify any full-run Checkpoint, done-marker, Run_Report or candidate CSV.
6. IF a full-scale Stage is started outside a SLURM job allocation, THEN THE Stage SHALL exit before any computation and show a message saying the Stage must be started through the Job_Chain or the HPC_Notebook.
