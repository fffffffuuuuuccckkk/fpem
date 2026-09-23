# Current FPEM research line

The active research line is the alternating predictive-environment and
representation-disentanglement framework:

1. load a pretrained forecasting backbone and obtain `H`;
2. infer/update TRAIN-only soft environments from current predictive conflict;
3. decompose `H` into `Z_inv` and `Z_var`;
4. suppress environment information in `Z_inv` and retain it in `Z_var`;
5. forecast from the two representations;
6. repeat environment discovery and representation learning by stage.

Active implementation:

- `models/PatchTST_PredictiveEnvIV.py`
- `models/predictive_env/`
- `models/domain_iv/{decomposition,feature_variation,losses}.py`
- `tools/run_predictive_env_iv_patchtst.py`
- `scripts/run_predictive_env_*.sh`
- `results/predictive_env_*`
- `results/environment_quality_*`

Historical Pattern Graph, Mapping Graph, EIVPM, raw-pattern mining, and other
superseded experiments are preserved under:

`archive/legacy_non_predictive_fpem_20260915/`

The archive is intentionally ignored by Git and excluded from the sanitized
GitHub publishing script. Nothing in the archive was deleted.
