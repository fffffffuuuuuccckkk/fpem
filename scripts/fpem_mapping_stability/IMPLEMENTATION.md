# SMRVA point-prediction pilot

This is an isolated, frozen-backbone experiment. It does not use the old
reconstruction-error GMM or split the backbone hidden tensor. The A0 forecast
is loaded from the existing checkpoint, and its SHA must match the cached
TRAIN-anchor SHA. No A0 training is performed.

## Inputs and causal policy

- A TRAIN anchor stores raw history, true future, A0 forecast and original
  chronological start. Its residual is normalized using **history-only**
  mean/std and a floor estimated solely from TRAIN histories.
- A query key is each channel's history normalized and interpolated to 64
  points, plus a constant bias. Neither retrieval nor inference uses a query's
  future. During TRAIN, a source is eligible only when its entire future ends
  before the query starts; VAL uses a fixed TRAIN cutoff; TEST reads TRAIN only.
- Four chronological TRAIN blocks are built, with overlapping labels purged
  across boundaries. Shared and centered block-specific linear operators are
  fitted by alternating ridge iterations. Only the shared operator is used at
  inference. A forward-block transfer audit tests its actual portability.
- The shared operator may fall back to zero if it fails to improve the
  designated VAL selection segment over A0. Ordinary global ridge P1 is fitted
  on all TRAIN anchors and selected on that same VAL segment.

## Point modes

P0 A0; P1 ordinary global ridge; P2 shared mapping; P3 P2 plus local
variation ridge with no gate; P4 plus TRAIN-calibrated historical-variation
coverage; P5 plus analytic leave-one-out mapping reliability; P6 plus fixed
history-variation activity. P7 shuffles future labels within each causally
eligible TRAIN pool, P8 uses deterministic random retrieval, P9 removes the
shared mapping, and P10 substitutes a local constant correction. P7-P10 reuse
the P6 VAL-selected K and ridge setting.

The local kernel ridge solve operates on K actual TRAIN history/future pairs.
Its leave-one-out residual is computed analytically from the inverse kernel
diagonal; it is not an in-sample fitting residual. The change descriptor uses
history differences/slopes only. Its nearest-neighbour coverage is calibrated
by a causal TRAIN empirical distribution; the reliability sigmoid scale is
calibrated from TRAIN leave-one-out gains. No gate is trained in this pilot.

P0-P6 configurations are selected by the first half of VAL. The remaining VAL
portion is reserved for any later interval calibration. TEST is reported only
after configuration selection. A complete point result must pass P6 versus
A0, shuffled pairs, random retrieval and local constant correction before the
probability stage is considered. No U0-U7 Flow experiment is started by this
script, because the user explicitly requested a staged decision.

## Run

`bash scripts/fpem_mapping_stability/run_point_pilot.sh CASE GPU`

Supported cases: `patchtst_ETTh1_720`, `itransformer_ETTm1_720`,
`moderntcn_ETTm2_720`. The shell wrapper retries a failed/incomplete case
after 60 seconds without overwriting completed modes.

Outputs reside under `results/fpem_mapping_stability/pilot_20261027/CASE/`:
`run_config.json`, `stable_mapping_audit.json`, `mapping_cache.pt`, per-mode
JSON, `all_point_results.csv`, `P6_sample_diagnostics.csv`,
`point_selection.json`, `summary.txt`, and `run.log`.

The probability phase remains pending. In particular, old Flow residuals and
checkpoints are **not** silently reused with the new prediction anchor; a
future U-phase must cross-fit the full point predictor, rebuild TRAIN-only OOF
residuals, reserve independent calibration data and retrain/adapt the existing
residual Flow under the new anchor.
