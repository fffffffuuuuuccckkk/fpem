# Portable FOIL / ShifTS controls

These are **backbone-adapted** controls, not claims of bit-for-bit upstream
reproduction.  The existing `run_foil_upstream_matrix.sh` remains intact; it
only covers upstream Informer on ExchangeRate-96.

- Backbones: the same project's A0 PatchTST and ModernTCN. Inputs are the
  same 96-step `features=M` windows, split, scaling, seed 2021, 10 final epochs,
  batch sizes, and 1e-4 backbone learning rate used by the FPEM matrix.
- Reference: reuse a compatible TRAIN-only, three-epoch shared checkpoint if
  available. Otherwise create one and share it between the two controls.
- FOIL: freeze the reference while training three environment-specific linear
  heads for two epochs, assign each TRAIN window by lowest normalized horizon
  shape error, then train the common forecast backbone with the upstream
  `log1p(squared error)` / shape-error variance combination (0.8 / 0.2).
  Source Informer decoder embeddings are replaced by channel-wise pooled A0
  tokens. No environment assignment is queried at validation or test time.
- ShifTS: preserve the source's reversible normalization, learnable
  positive-half horizon-window mask, cross-variable translator for the last
  target channel, and future-aligned auxiliary loss. The base forecast loss
  remains multivariate to match FPEM. Future-aligned windows are constructed
  *only* from the TRAIN split; inference receives historical X alone.
- Optimizer: Adam at 1e-4 for every trained parameter. No LR search, no
  scheduler, no TEST-based checkpoint selection. Final epoch is evaluated.
- Reporting: `metrics.json` stores TEST and validation MSE/MAE, while
  `run_config.json` records the reference hash. Smoke tests use a separate
  output directory and never count as final results.

Run a resumable shard with `SERVER_INDEX=0|1|2` and
`scripts/fpem_probabilistic/run_foil_shifts_matrix.sh`. Each server manages
four local GPUs and a disjoint part of the 60 backbone/dataset/horizon cases.
