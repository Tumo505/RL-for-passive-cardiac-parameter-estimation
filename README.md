# Reinforcement learning for passive cardiac parameter estimation

This project trains a reusable Soft Actor-Critic (SAC) policy to estimate two passive material parameters from sparse, noisy displacement observations. It includes a mechanics solver, a Gymnasium environment, reproducible case splits, five comparison methods, evaluation plots, checkpoints, and a four-GPU PBS workflow for a PBS cluster.

The implemented mechanics problem is a **homogeneous myocardial slab under known biaxial traction**. It is a reduced tissue benchmark, not a ventricular finite-element model, patient digital twin, or simulation of biological regeneration. Parameter ranges are illustrative benchmark ranges, not fitted human measurements. No claim of improved calibration performance is built into the implementation.

## Run locally

Use Python 3.10–3.13. After cloning the repository, create an isolated environment:

```bash
git clone https://github.com/Tumo505/RL-for-passive-cardiac-parameter-estimation.git
cd RL-for-passive-cardiac-parameter-estimation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the complete small verification experiment:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m cardiac_rl.cli prepare --config configs/smoke.yaml --data-dir data/smoke
.venv/bin/python -m cardiac_rl.cli train --config configs/smoke.yaml --data-dir data/smoke --output runs/smoke/seed_0 --device cpu --seed 0
.venv/bin/python -m cardiac_rl.cli evaluate --config configs/smoke.yaml --data-dir data/smoke --output runs/smoke/seed_0 --seed 0
```

Smoke training is only 3,000 transitions. It verifies the pipeline; it is not a converged or publication-grade experiment. Output directories are protected against accidental training overwrite. To resume to a **total** of 10,000 transitions:

```bash
.venv/bin/python -m cardiac_rl.cli train --config configs/smoke.yaml --data-dir data/smoke --output runs/smoke/seed_0 --device cpu --seed 0 --resume --steps 10000
```

For the larger experiment use `configs/default.yaml`, `data/default` and `runs/default/seed_0`. The default is 300,000 transitions per independent seed. This is an initial training budget, not a guarantee of convergence. Four simultaneous seeds are configured in [the PBS workflow](#pbs-gpu-workflow).

## Scientific model

The 10 mm slab has a fixed fibre direction along x, incompressibility and a plane-stress thickness face. Unknown parameters are isotropic stiffness `a` and fibre stiffness `af`, both in kPa. Fixed nonlinear shape parameters are `b=4` and `bf=6`. The reduced energy is

```text
W = a/(2b) [exp(b(I1 - 3)) - 1]
  + af/(2bf) [exp(bf max(I4 - 1, 0)^2) - 1]
F = diag(lambda_x, lambda_y, 1/(lambda_x lambda_y))
```

The nonlinear solver balances the two nominal (first Piola–Kirchhoff) tractions at four known load pairs. Through-thickness stress is eliminated using the zero-traction condition. The x and y stretches may be below one under unequal loads; fibre compression switches off the fibre contribution. Volume remains exactly one. Displacements at nine positions in each load case are affine functions of the stretches.

The law is inspired by, but is not the full orthotropic model of, [Holzapfel and Ogden (2009)](https://pubmed.ncbi.nlm.nih.gov/19657007/). It excludes sheet fibres, fibre dispersion, active contraction, fibrosis, viscoelasticity and geometry variation. See [the experiment protocol](#computational-experiment-protocol) before making paper claims.

## SAC environment

- **Action:** two bounded increments in normalised log material coordinates; clipping enforces positive admissible parameters.
- **Observation:** masked measured displacements, mask, predicted displacements, masked residuals, current/best parameter candidates, previous action, remaining budget, and current/best errors. There are 297 features for the default four-load, nine-position model.
- **Reward:** discounted potential shaping based on negative log displacement MSE, a per-evaluation cost, and a terminal objective using the best observed candidate. No ground-truth parameter error enters the reward.
- **Episode:** one fixed material case, noise realisation and observation mask. Training randomises the horizon; the default maximum is 63 updates plus one initial evaluation.
- **Algorithm:** Stable-Baselines3 SAC, twin critics, replay, automatic entropy coefficient, two 128-unit hidden layers, batched action inference across four environments per GPU process.

Training candidates use bilinear interpolation of a mechanics grid. Targets always come from separately solved reference equilibria. `grid.json` records interpolation error on 64 off-grid cases. For a final comparison without interpolation, use `evaluate --backend reference`; all iterative methods then call the same nonlinear solver.

## Baselines and outputs

To calibrate from an observation file, first generate a correctly shaped example:

```bash
.venv/bin/python -m cardiac_rl.cli example --config configs/smoke.yaml --data-dir data/smoke --output runs/smoke/observations.json
.venv/bin/python -m cardiac_rl.cli calibrate --config configs/smoke.yaml --data-dir data/smoke --model-dir runs/smoke/seed_0 --observations runs/smoke/observations.json --output runs/smoke/calibration.json --budget 16
```

The JSON stores displacements in mm, a boolean observation mask, sensor positions and applied nominal tractions. Replace measured values only when geometry, loads, units and the constitutive assumptions match the trained protocol. Missing displacement values may be `null` if their mask is false. Inference reports point estimates and a reference-solver displacement prediction, not uncertainty intervals or clinical recommendations.

Comparators are multistart bounded finite-difference least squares, differential evolution, Gaussian-process Bayesian optimisation, a direct supervised neural estimator, and random search. They receive the same fixed observations and parameter bounds. All online candidate evaluations count, including initialisation, repeated candidates and finite-difference probes. The direct estimator uses one audit prediction per case and no sequential search.

Each run produces:

| File | Purpose |
|---|---|
| `run.json`, `config.json` | Seed, software, device, settings, training time and forward calls |
| `validation.jsonl`, `best.json` | Validation-only model selection history |
| `best.zip` | Selected SAC policy |
| `last.zip`, `last_replay.pkl` | Resume from the last completed/interrupted run |
| `checkpoint.zip`, `checkpoint_replay.pkl` | Periodic recovery checkpoints |
| `direct.pt`, `direct.json` | Supervised comparator and its training cost |
| `evaluation_grid/results.csv` | Per-case, per-condition, per-budget metrics |
| `evaluation_grid/traces.jsonl` | Best objective and parameter trajectory |
| `evaluation_grid/summary.json` | Errors, success rates and case-bootstrap intervals |
| `evaluation_grid/benchmark.png` and `.pdf` | Static comparison plots |

Reference-backend outputs live in `evaluation_reference/`. `aggregate` combines `seed_*/evaluation_*/results.csv` using a bootstrap over both seed and material-case axes. A single-seed confidence interval does not quantify training-seed variability.

## Reproducibility and recovery

Dataset fingerprints reject incompatible mechanics/data configurations. Train, validation and test cases are split by material pair before noise and masking. Validation selects the policy; test data never train the actor or the direct estimator. The same test cases and noise/masks are reused across methods and seeds.

`SIGUSR1`, `SIGTERM` and `SIGINT` request a checkpoint at the next callback boundary. Periodic saves cover abrupt scheduler termination; a hard kill cannot save in-flight updates. Resume restores SAC weights/optimisers and replay when available, then resets the environments. It is **not bitwise continuation** of environment/random-generator state. Checkpoint files are trusted local pickle/zip artifacts: load only your own files.

`requirements.local.lock.txt` records the locally tested package versions. Do not copy the macOS `.venv` to Linux. Cluster setup creates a new environment and records `requirements.cluster.lock.txt`; choose the PyTorch wheel/index compatible with the cluster driver.

## References

- [Stable-Baselines3 SAC documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Passive myocardial constitutive modelling](https://pubmed.ncbi.nlm.nih.gov/19657007/)
- [Physics-informed identification in nonlinear soft-tissue biomechanics](https://link.springer.com/article/10.1007/s00466-024-02516-x)

The conference's manuscript template asks authors to acknowledge AI tools used in manuscript preparation. Actual scientific conclusions must come from the completed experiments, not smoke tests or this scaffold.

## PBS GPU workflow

Connection details, queue names and site-specific resource flags belong in ignored local configuration files. The scripts have not connected to a cluster or submitted a job.

| Setting | Value |
|---|---|
| Gateway | Set `HPC_GATEWAY` in `site.env` |
| Submission target | Set `HPC_TARGET` in `site.env`; reached using ProxyJump |
| Scheduler | PBS: `qsub`, `qstat`, `qdel` |
| Queue | Set `HPC_QUEUE` in `cluster.env` |
| Allocation | One exclusive node, 28 CPU cores, 240 GB RAM, four GPUs |
| Walltime | `168:00:00` |
| Extra resource | Set `HPC_SOFTWARE_RESOURCE` if required |
| Account/project flag | None |

The script lets PBS choose the compute node. Confirm that the local PBS resource spelling is `ngpus` and `mem`; these directives use common PBS select syntax and must be checked against the site configuration.

### Transfer source

On your laptop, from the SACAM project root:

```bash
cp scripts/hpc/site.env.example scripts/hpc/site.env
# Edit HPC_GATEWAY, HPC_TARGET and REMOTE_DIR for your cluster.
bash scripts/hpc/push.sh
```

The transfer uses `ssh -J YOUR_USER@YOUR_GATEWAY YOUR_USER@YOUR_SUBMISSION_HOST` and rsync over that route. It excludes local environments, caches, generated data, runs and local SSH configuration. It does not delete remote files. If the gateway only supports interactive onward SSH and blocks ProxyJump forwarding, use the site-approved SSH route instead; do not assume changing the scheduler scripts will fix gateway policy.

### Create the Linux environment

Log into the submission host through the gateway:

```bash
ssh -J YOUR_USER@YOUR_GATEWAY YOUR_USER@YOUR_SUBMISSION_HOST
cd /your/absolute/writable/path/SACAM
cp scripts/hpc/cluster.env.example scripts/hpc/cluster.env
```

Edit `cluster.env` with the GPU queue in `HPC_QUEUE`, any required resource such as `software=...` in `HPC_SOFTWARE_RESOURCE`, and the site's installed Python/CUDA modules. `HPC_MODULE_INIT` can name an existing shell file containing `module load ...` commands; it is sourced both during setup and in the allocated job. Use Python 3.10–3.13. Choose `TORCH_INDEX_URL` only after checking the node driver and the wheel compatibility. No module names or CUDA version have been guessed.

```bash
bash scripts/hpc/setup.sh
.venv/bin/python -m pytest -q
```

Setup requires package access on the submission host or a site package mirror. It does not install a driver or modify system CUDA. If login-node package installation is disallowed, run setup within an approved allocation or use the site's supported environment. The batch job itself only uses the prepared environment and does not download packages.

### Submit four seeds

From the project root on the submission host:

```bash
mkdir -p logs
bash scripts/hpc/submit-local.sh -v SEED_START=0,RESUME=0,TOTAL_STEPS=300000
qstat -u "$USER"
```

Alternatively, `bash scripts/hpc/submit.sh` performs this submission from the laptop once setup is complete. Record the returned job ID. The small MLP does not need model parallelism: one independent seed runs per allocated GPU, with four vectorised environments per training process. Each process uses six CPU threads, leaving four cores for orchestration. GPU efficiency may be low for this small benchmark; use the pilot to decide whether longer/more diverse training justifies the four-GPU allocation.

`submit-local.sh` supplies the queue and optional software resource to `qsub`; no account flag is added. The batch script honours a comma-separated `CUDA_VISIBLE_DEVICES` allocation (including GPU UUIDs) and verifies that PyTorch sees four devices before starting. It prepares shared mechanics/data once, launches seeds 0–3, checks every exit status, runs the six-method comparison on CPUs, then aggregates results. CPU evaluation is not charged to a GPU speedup claim. PBS stdout is in the submission directory unless an output path is supplied; detailed seed logs are under `logs/`.

For a cluster smoke run before the full allocation, shorten the walltime on submission:

```bash
bash scripts/hpc/submit-local.sh -l walltime=00:30:00 -v CONFIG=configs/smoke.yaml,DATA_DIR=data/hpc_smoke,RUN_ROOT=runs/hpc_smoke,TOTAL_STEPS=3000
```

This still requests four GPUs and runs four short seeds; it is intended to validate device assignment and module setup.

### Resume and evaluate

To continue the same four seeds toward a larger total transition budget:

```bash
bash scripts/hpc/submit-local.sh -v SEED_START=0,RESUME=1,TOTAL_STEPS=600000
```

Configuration and seed fingerprints must match the original run. `--steps` is a total target, not an increment. Replay buffers are saved periodically and on graceful interruption. PBS may kill a process too quickly to save at walltime, so recovery can lose work since the latest completed periodic checkpoint. A partially written checkpoint is not guaranteed recoverable; retain the last completed run and periodic checkpoint files. `qdel JOB_ID` uses site-dependent termination grace; it is not a guaranteed final checkpoint mechanism.

To run the full comparison against exact equilibria after training, submit the same completed training target with `RESUME=1` and `EVAL_BACKEND=reference`:

```bash
bash scripts/hpc/submit-local.sh -v SEED_START=0,RESUME=1,TOTAL_STEPS=300000,EVAL_BACKEND=reference
```

If you trained to 600,000 steps, use `TOTAL_STEPS=600000`. Training is skipped when the saved transition count already meets the target; reference evaluation may be substantially slower. It never retrains against test data.

### Retrieve results

From the laptop:

```bash
bash scripts/hpc/status.sh
bash scripts/hpc/pull.sh
```

Results are copied into `runs/hpc/` and logs into `logs/hpc/`. Nothing is removed remotely. Plot files, CSVs, protocol metadata and policy checkpoints are retained. Save `requirements.cluster.lock.txt` with the experiment archive.

## Computational experiment protocol

The primary question is whether a policy trained across repeated material-calibration tasks improves online accuracy per forward evaluation relative to conventional optimisers and a direct inverse network. The null outcome, that conventional fitting is better, is scientifically valid.

### Fixed assumptions and units

The model is homogeneous, incompressible and passive, with known geometry, fibre direction, constitutive form and four nominal traction pairs. It has two unknown positive material coefficients, not unknown loading or fibre orientation. Coordinates/displacements use mm; coefficients and nominal tractions use kPa. This is a controlled identifiability benchmark, with no clinical validation.

Default parameter bounds are `a=[1,8] kPa` and `af=[2,24] kPa`, sampled uniformly in log coordinates. These are illustrative test bounds. Biological parameter claims require experimental justification and unit/constitutive-convention matching. The two stiffness contributions may have different effective magnitudes because their energy terms differ.

Targets solve nonlinear equilibrium; a separate interpolated parameter-to-displacement grid accelerates training. `grid.json` measures off-grid approximation error. The grid has no finite-element mesh, so mesh-convergence claims are inappropriate. Report interpolation refinement and equilibrium residuals instead. For final quantitative claims, compare every method with the reference backend and assess grid-size sensitivity (e.g. 21, 41, 81 using separate data/run directories).

### Splitting and observation model

The default Latin-hypercube parameter sample is partitioned into 1,024 training, 64 validation and 100 test material cases before augmentation. Identical parameter pairs do not cross splits; this is interpolation to unseen pairs within the same bounds, not out-of-distribution validation. Fixed-test tasks are paired across seeds and methods.

For every condition, Gaussian displacement noise has standard deviation `noise_level * (0.25 * slab_length_mm)`. At length 10 mm, the 1% and 5% settings correspond to 0.025 mm and 0.125 mm standard deviations. These percentages are relative to a fixed scale, not the case's peak displacement. Fractions 1, 0.5 and 0.25 retain a fixed number of randomly chosen displacement components across all positions and loads. Missing values are zero-filled with an explicit binary mask. This is random component loss, not anatomically contiguous dropout or complete missing sensors.

The uniform slab gives affine displacements and redundant spatial measurements. More sensors predominantly improve noise averaging; they do not reveal heterogeneous mechanics. Extend geometry/material heterogeneity only after this baseline is reliable.

### Training and selection

The default four-GPU PBS job trains independent seeds 0–3. Validation on a deterministic subset of observation conditions selects the minimum mean absolute log parameter error. Validation labels may select the model but are never policy inputs or reward targets. Use validation data for hyperparameter selection; do not repeatedly tune against test scores. Rerun the final frozen configuration on all seeds.

Each SAC actor proposes normalised log-parameter increments. The current observed target, mask, prediction/residuals, previous action, current/best candidates and remaining budget are available. Training randomises the number of allowed updates between 8 and 63 (or the smoke configuration's lower cap). Episodes terminate at their intrinsic budget or a fixed observed-MSE tolerance. The known remaining budget makes this a finite-horizon task, so terminal transitions do not bootstrap beyond the budget.

The reward implements `gamma * Phi(next) - Phi(current) - step_cost` with `Phi=-log(MSE+epsilon)`, zero terminal potential, and terminal base reward `-log(best_MSE+epsilon)`. It is not an additional PINN physics penalty: the forward model already enforces mechanics. This does not guarantee noise-robust parameter recovery, which must be evaluated independently.

### Fair comparison and cost accounting

Evaluate each method at budgets 16, 32 and 64 complete four-load forward predictions, counting the initial candidate, finite-difference perturbations, repeated candidates and all restarts. The same mid-bound initial candidate is supplied to iterative methods. SAC can stop before its maximum budget; direct estimation uses one prediction to evaluate its proposed parameters. Gaussian-process acquisition optimisation uses 512 sampled candidates and a fixed kernel; this internal work is included in wall time, not forward-model calls. This is a transparent lightweight BO baseline, not a fully tuned best-possible BO implementation.

Report best-observed candidate error, mean and maximum relative parameter error, log parameter error, observed displacement loss, exact-reference clean displacement error, success rate (both parameters within 10%), and online wall time. Report sensitivity-Jacobian minimum singular value/condition number as a local identifiability diagnostic; these diagnostics use truth only after inference. Local rank is not proof of global uniqueness.

`run.json` records RL training time, including validation and checkpoint overhead; `direct.json` records supervised training time. Shared grid/target-generation costs are in `data/*/*.json`. Count these offline costs explicitly. Reference audits of final estimates and post-hoc identifiability solves are separately declared in `protocol.json`; they do not give optimisation methods extra evaluations. A reference-model candidate evaluation entails one equilibrium solve per load case.

For a proposed amortisation result, at a matched accuracy target compute `N_break_even = ceil((RL_offline - baseline_offline) / (baseline_online - RL_online))` only if RL has a positive online saving. Accuracy must be matched first. If there is no positive saving or RL has lower accuracy, do not claim a break-even advantage. The direct estimator must be part of this comparison.

Noise/mask conditions reuse the same material cases. Bootstrap parameter cases within conditions; across training seeds, use the crossed seed/case bootstrap from `aggregate`. Report paired differences and their uncertainty before asserting superiority. Four seeds provide a limited estimate of training variability, and 100 test cases do not establish clinical generalisation.

### Deliverable boundary

The smoke run establishes software operation and budget accounting, not scientific efficacy. A conference submission should include full multi-seed training curves, reference-solver comparisons, surrogate-refinement checks, identifiability diagnostics, and negative results where relevant. The experiment concerns inverse passive material estimation. A claim about cardiac tissue regeneration requires a separate biological link and supporting experimental evidence.

## Repository contents

Only the root `README.md` is tracked as Markdown. Supplementary local Markdown files are ignored; the HPC instructions and scientific protocol are maintained in this README. Generated data, model checkpoints, logs, local environments and cluster configuration files are also ignored.

## Maximum walltime allocation

`scripts/hpc/max-allocation.pbs` requests the known maximum of 168 hours, one exclusive node, 28 CPUs, 240 GB RAM and four GPUs. Supply the queue and any software resource with `qsub` (or the site's supported submission mechanism). A new allocation is required when an existing job has only a short walltime; a shell variable cannot extend PBS's enforced limit.

```bash
qsub -q YOUR_GPU_QUEUE -l software=YOUR_SITE_RESOURCE scripts/hpc/max-allocation.pbs
```

The launcher reads the **granted** walltime and elapsed time using `qstat -f`, checks CPU affinity and CPU allocation limits, and uses the GPUs visible under the scheduler's device mask. It refuses a granted limit below the requested 168 hours. Four seeds run concurrently: each has its own GPU if four are visible; when only one or two are visible, processes share those GPUs. All allocated CPU threads are divided across the seed processes. If more than four GPUs are visible, increase `SEEDS` to use all GPUs; this creates additional independent seeds.

Training uses `--max-seconds` rather than a fixed transition count. It continues until the training deadline, preserving the best validation checkpoint. Six hours of the allocation are reserved for the full reference-solver comparisons and two minutes for checkpoint grace. `EVALUATION_RESERVE_HOURS` can adjust this estimate. Final evaluation runtime is not guaranteed to fit the reserve; inspect outputs before treating the experiment as complete. Memory remains demand-driven: filling unused RAM does not improve this small model, and algorithm/replay settings remain unchanged for compatibility with existing runs.

Existing material cases and partially trained seeds are reused automatically. Checkpoint loading falls back to the periodic pair if the final pair is unreadable. Complete temporary checkpoint files replace the persistent files, followed by a step-count manifest. A hard kill during replacement can still invalidate a pair; retain both final and periodic pairs. Timed completion is recorded as `time_budget_completed`, which means the assigned training time ended, not that convergence was established. When recovering a run killed before metadata was saved, previous cost totals may be incomplete and are flagged in `run.json`.

For a short **software test** inside an appropriately allocated GPU job, explicitly lower the expected walltime and reserve:

```bash
.venv/bin/python -m cardiac_rl.allocation --expected-walltime-hours 0 --evaluation-reserve-hours 0.01
```

The maximum-allocation workflow does not reinstall dependencies. Keep a previously tested environment, especially on older GPUs/drivers. GPU selection uses the scheduler-visible logical device indices and never expands `CUDA_VISIBLE_DEVICES` to access unallocated devices. Validation, checkpointing and final evaluation consume part of the allocated time; absolute GPU utilisation depends on the small model and CPU-bound mechanics work.

## Finish training early and evaluate

For an already-running maximum-allocation job, editing the launcher source cannot change the code already loaded in that process. The controlled handoff uses a separate evaluation-only job with a PBS `afterany` dependency on the training job. It validates and backs up the currently selected policies, submits evaluation, records the returned job ID, and then sends `USR1` to request a graceful training stop. The old launcher deliberately reports interruption/nonzero exit after saving; `afterany` allows evaluation to start after that exit, when policy files are no longer being written. No `qdel` or forced kill is used. If submission fails, no stop signal is sent. If signalling fails, the evaluation job remains dependent and the training job continues.

Run on the submission host, from the project root, after updating the source:

```bash
bash scripts/hpc/finish-training.sh TRAINING_JOB_ID
```

The queue/resource settings come from the ignored `scripts/hpc/cluster.env` or the environment. This workflow applies to the Python maximum-allocation launcher, which handles `USR1`; it is not a general cancellation script for unrelated jobs. The evaluation-only job requests the same resources and maximum walltime, reuses existing models/data and runs all six methods against reference mechanics. It never starts or resumes SAC training. `evaluation_allocation.json` preserves the new resource record separately from the original training `allocation.json`.

The script records the successor job in `runs/p100_full/evaluation_request_TRAINING_JOB_ID.json`. A repeated request exits without submitting duplicates. Verify successor status with `qstat` and inspect `logs/evaluate_SUCCESSOR_JOB_ID_seed_*.log`. The experiment remains incomplete until all four reference evaluation jobs finish and `aggregate.json` is produced. Successful execution is not evidence of clinical validity or superiority over the baselines; those claims require analysis of the resulting estimates, uncertainty and computational costs.
