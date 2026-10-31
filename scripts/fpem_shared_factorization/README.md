# Shared-factorization S1–S8 pilot

This branch is **Zinv-only point prediction**, not the probability Flow branch.
The existing A0 checkpoint, frozen reference head, backbone encoding and TRAIN
predictive-conflict environment inference are reused. No A0 is retrained.

`manifest.csv` contains 36 method/case pairs. Each searches invariant-head
learning rates `2e-5, 5e-5, 1e-4` at 10 epochs, seed 2021, frozen Backbone,
fixed decomposer LR `1e-4`, with the base configuration's environment update
period and K. Four cases are PatchTST/ETTh1/96, PatchTST/ETTh1/720,
ModernTCN/ETTm2/720 and iTransformer/ETTm1/720. Three servers each receive
three methods per case, across four GPUs. Selection uses the **same checkpoint's
VAL MSE**; TEST is reported only after selection and is not a tuning signal.

Launch, on each server: `bash scripts/fpem_shared_factorization/launch_search.sh 0`
(replace 0 with 1 or 2). The launcher is restartable: completed trials with
matching reference hashes are reused, and incomplete attempts get retry paths.

S1/S2/S3/S7/S8 use TRAIN-only statistical buffers, refreshed after each
predictive environment update. S4/S5/S6 use trainable modules; S6's pairwise
loss uses TRAIN q and excludes stride-1 windows whose starting indices differ
by less than `seq_len`. No TEST q or future targets enter factorization.

Source names:

- S1 `common_subspace`
- S2 `shared_spectral`
- S3 `robust_lowrank_sparse`
- S4 `supervised_joint_private`
- S5 `shared_private_lite`
- S6 `pairwise_partial_shared`
- S7 `sample_prevalence`
- S8 `subspace_recurrence`; original polynomial control `subspace_recurrence_poly`

S1 and S3 are not claimed to have `Zinv+Zvar=H`; S3 explicitly emits a rare
residual. S8's global/partial/rare interpretation is descriptive, not causal.
Further environment ablations and multi-seed confirmation should only follow
VAL evidence from this pilot. Existing statistical-pilot baselines must be
compared by the same frozen/joint protocol and A0 hash.
