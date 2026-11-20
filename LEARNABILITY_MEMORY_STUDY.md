# Learnability-aware historical/future memory Pilot

This is an FPEM-specific experimental adaptation. It leaves the three frozen
Backbones, their original heads, and the existing `residual_flow` dynamics code
untouched. New code lives in `models/predictive_env/learned_memory/`,
`consensus_adaptation/`, `residual_difference/` and three standalone tools.

## Exact point path

The source is a real `(TRAIN X, frozen H, TRAIN Y, frozen A0 prediction)`
anchor, not a cluster-averaged future. The detached-H linear history decoder
is trained with TRAIN only. TRAIN-fitted A0 error and decoder error are logged
separately. Two-dimensional robust log-error GMM and reconstruction-only GMM
are compared with one-component alternatives by BIC. The reconstruction-only
component is named by **reconstruction** error; the joint component is named
by **prediction** error. This distinction is essential.

Both GMMs may be used even when BIC favors one component, but that run must
not be interpreted as evidence of genuine two-group structure. P1 is the
explicit no-group fallback. Group labels are not claimed to be environments,
invariant states, or actual future regime labels.

TRAIN retrieval requires each stored source's full future to end before the
query history starts. VAL sources must finish before the first VAL history;
TEST reads only completed TRAIN memory. Retrieval returns whole real entries,
not patch-level synthetic futures. Query target never enters the retriever.

For source hidden features `X` (including a bias column), source normalized
A0 residuals `R`, and diagonal weights `w` derived from Under-Learned
posteriors plus Stable preservation, the unique local operator is calculated
in its five-source dual form:

`delta_y = h_q X_w^T (X_w X_w^T + lambda_r I)^-1 R_w`.

Stable rows have zero target and weight `lambda_s p_stable`; Under-Learned rows
have actual source `(Y-A0)/historical_scale` as targets and weight
`p_underlearned`. All rows therefore train **one** local linear head update.
No source future is directly added to query Y and no source H is concatenated
to query H. Source historical scales receive a TRAIN-only robust floor to
avoid pathological near-constant windows. P3 and P6 in the supplied design
have identical mathematical definitions; their result is intentionally
deduplicated rather than counted as two independent wins.

P7 shuffles future labels only among already-eligible retrieved entries; P8
uses random eligible retrieval. Both controls inherit the best real candidate's
grouping and ridge/Stable weights. They are never allowed to select the main
method. P0–P8 plus the G0 density baseline are saved, including failures.
The first VAL half selects local ridge and Stable weight; TEST is evaluated
only after that choice. The latter VAL half is reserved for interval
calibration.

## Residual dynamics (evidence-gated)

The downstream queue starts only if the VAL-selected real point method beats
A0 and its matched shuffled-pair and random-retrieval controls. It otherwise
writes `probability_skipped.txt` and does not silently run a new uncertainty
model. Existing `ProbabilityHeadBank(heads=("residual_flow",))` and
`HorizonFutureVariant.encode_future` are reused. A `ResidualDifferenceCondition`
adds only a zero-initialized FiLM scale/shift to its condition; U6 is a
separately labeled output-scale calibration ablation. The original Flow and
point model are frozen while fitting each condition adapter.

For OOF residual targets, four chronological TRAIN folds train their **own
fresh A0 reference from the earlier prefix**, fit a prefix-only memory/GMM,
run the complete selected point operator, and save the held-out residual.
Every source's full future ends before the holdout history starts. The new
residual index is independent of A0 training residuals and stores fold/model
provenance. Retrieval statistics use OOF sources only; lacking OOF sources are
masked, not replaced with in-sample errors. U0 is the unchanged Flow condition;
U1 uses source A0 residuals, U2 location, U3 dispersion, U4 location plus
dispersion and magnitude, U5 all seven features, U6 output scale, U7 shuffled
residual pairing and U8 causal random OOF residual retrieval. All train and
TEST results are retained. VAL first half selects; independent VAL latter half
fits a simultaneous-trajectory split-conformal expansion. Marginal and whole
trajectory coverage are reported separately.

**Limitations:** The standard benchmark scaler supplied by the repository is
fitted on the entire TRAIN interval even for prefix fold A0 models. This is a
remaining strict temporal-OOF caveat, recorded in the result protocol. Fold
models are trained from the same initialization but may have different hidden
coordinates from the final full-TRAIN A0. Flow results remain exploratory;
neither exchangeability under overlapping windows nor arbitrary distribution
shift is guaranteed. U6 uses a sample-variance Gaussian-NLL approximation to
train its output scale because Flow Matching has no exact scalar likelihood;
it is not presented as the original Flow loss.

## Entry points and output

```
python tools/run_fpem_learnability_queue.py patchtst_ETTh1_720 --gpu 0 --output results/fpem_learnability/pilot_v4_20261026
python tools/run_fpem_learnability_queue.py itransformer_ETTm1_720 --gpu 1 --output results/fpem_learnability/pilot_v4_20261026
python tools/run_fpem_learnability_queue.py moderntcn_ETTm2_720 --gpu 2 --output results/fpem_learnability/pilot_v4_20261026
python tools/run_fpem_residual_difference_queue.py moderntcn_ETTm2_720 --gpu 2
```

The queues resume completed P/U rows and OOF folds, retry transient failure
up to three times, and pause one minute between attempts. The case folders
contain the A0 SHA, run configuration, GMM/BIC audit, every P mode, OOF fold
provenance, every eligible U mode, calibration metrics, and readable summaries.
The aggregate directory is `results/fpem_learnability/`. Do not compare smoke
results with full results; smoke uses one batch and shorter fold training.
