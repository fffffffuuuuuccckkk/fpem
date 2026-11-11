# Independent probabilistic baseline adapters (24/48)

These are independent raw-history baselines. They do not use the FPEM encoder,
`Z_inv`, `Z_var`, environment labels, or FPEM heads. All three use the same
Time-Series-Library CSV TRAIN/VAL/TEST split as the existing CSDI comparison:
history length 96, horizons 24/48, all channels, seed 2021, 13 TRAIN epochs,
and 100 full-TEST samples. No TEST-based model or hyperparameter selection is
performed. `ProbTS_CRPS` uses the project's existing ProbTS-compatible scorer;
empirical CRPS is separately named.

The supplied archive called `TimeGrad-main.zip` is TimeGrad, not TimeDiff.
We use the name **TimeGrad** until a different TimeDiff source is supplied.

## What is ported

- `timegrad`: recurrent teacher-forced conditioner; noise-prediction diffusion
  loss for each next-step target; autoregressive reverse-diffusion sampling.
- `tsflow`: conditional flow matching between a GP-regression prior and the
  future trajectory; temporal velocity network; Euler probability-flow
  sampling. The GP prior and flow-matching path mirror TSFlow's defining
  objective. The project's first TSFlow run used an incorrect near-zero
  history scale; it was archived under `tsflow_pre_min_scale_fix_20261013`.
  Active results use the upstream StdScaler-style minimum scale of 1.
- `tsdiff`: unconditional denoising-diffusion training on concatenated
  context/future windows, followed by observation self-guidance at inference.
  The future is never used as inference conditioning.

These are **compact project PyTorch adaptations, not bit-for-bit upstream
executions**. Upstream packages require mutually incompatible versions of
GluonTS, Lightning and Python. The project ports replace upstream S4/other
backbone implementations with a small dilated temporal residual network and
use the project's CSV data pipeline. Their numbers should be labeled
"project adapter" in tables, not advertised as exact paper reproductions.
Source archives: `/data/OuXiaoyu/TimeGrad-main.zip`,
`/data/OuXiaoyu/TSFlow-main.zip`, and
`/data/OuXiaoyu/unconditional-time-series-diffusion-main.zip`.

## Run, resume and outputs

`tools/run_independent_probabilistic_baseline.py` trains exactly one case and
writes `run_config.json`, `history.json`, `final_epoch.pt`,
`test_metrics.json`, and a small `test_trajectory_examples.npz`. If training
finished but evaluation was interrupted, rerun with `--resume_eval` to reuse
the saved final-epoch checkpoint. A complete case has both `complete` and
`test_metrics.json`; the queue script skips it.

`scripts/fpem_probabilistic/launch_independent_probabilistic_24_48.sh` runs
the 30 method×dataset×horizon cases on servers 2 and 3 (seven GPUs). The six
missing CSDI cases run separately on server 1; four existing completed CSDI
24/48 cases are reused. All outputs are under
`results/fpem_probabilistic/independent_baselines_24_48/` on the executing
server. Smoke-test directories are separate and must not be used as full
results.

The scorer's `inv_MSE`/`inv_MAE` fields refer to a last-observation naive
forecast for these independent baselines, **not** FPEM `Z_inv`.
