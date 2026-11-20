# PGIR: Prototype-Guided Information Retention

This is a forecasting adaptation of ORTE's binary information-retention idea,
not a reproduction of ORTE or a modification of the frozen Backbone.
The incomplete local ORTE_public checkout is not imported.

## Point model

1. Fit the existing PTM K=8 historical/future bank on non-overlapping TRAIN
   anchors. Its query uses only the historical part of each sample.
2. Compute segment similarity to the TRAIN historical prototype, weighted by
   TRAIN anchor coverage. The median similarity from TRAIN anchors is stored as
   the invariant-retention threshold. High-score segments retain the actual
   observed history, not prototype copies.
3. A small temporal Conv network uses historical/prototype residual statistics
   to produce a straight-through binary keep mask for other segments.
4. Feed mask * X + (1-mask) * X_proto to frozen, checkpoint-loaded A0.
   The initial hard mask is all ones. P1 forces all ones for the whole run and
   is the exact A0-function control.
5. P2 uses TRAIN-Y forecast MSE, small variation-retention sparsity, and
   temporal continuity. P3 adds a detached counterfactual ranking term:
   removing one selected non-invariant segment should not improve prediction.
   The counterfactual branch is never trained to be worse.

Random and smoothing comparisons are post-hoc diagnostic controls on P2's
selected checkpoint, not separately optimized methods. They preserve the
replacement count or selected mask respectively.

## Probability P4

P4 does not reuse an old CSIF/CFIP checkpoint. A new early-TRAIN-only A0,
PTM bank and P2 mask are fitted on a strict chronological core; later TRAIN
produces target-disjoint OOF residuals. This core-only A0 retrain is required:
normal A0 has already seen all TRAIN windows and is not an OOF predictor.
The core scaler is also fitted on early TRAIN only.

The existing Gaussian/Student-t CFIP class and existing affine Flow head are
reused. Their condition comes from historical X, observed history-minus-
prototype, binary mask and match support; it never consumes future Y at
inference. CFIP only supplies Flow conditioning; the point forecast remains
the selected P2 checkpoint. Flow generates trajectories and intervals, not a
deterministic point residual.

Point and probability checkpoints are selected on VAL MSE and VAL CRPS.
TEST is evaluated after selection.

## Reproduction

From the repository root, with basicts Python:

    python -m pytest -q tests/test_pgir.py
    python tools/run_fpem_pgir_queue.py --worker 0 --workers 3 --gpu 0
    python tools/run_fpem_pgir_queue.py --worker 1 --workers 3 --gpu 1
    python tools/run_fpem_pgir_queue.py --worker 2 --workers 3 --gpu 2
    python tools/run_fpem_pgir_probability_queue.py --gpu 3

Queues are resumable: completed trials are skipped and failed trials receive
one retry after 60 seconds. They cover only the three prespecified pilots and
write under results/fpem_pgir/pilot_20261022.

## Interpretation limits

P1 must equal A0; it is function preservation, not a learned improvement.
Mask selection cannot create forecasting ability frozen A0 lacks. A
high-confidence prototype segment is TRAIN-supported under this score, not
proven causal or invariant. Three pilots and one seed cannot establish broad
superiority. The OOF core model and final full-TRAIN point model have different
training histories; calibration drift remains possible.

## T1-T5 patch-replacement pilot (2026-10-23)

The original P2 is untouched. Its mask already permits a varying number of
replacements; T1 specifically caps that number at two per sample and channel,
replaces only when the learned score predicts benefit, and removes the
old sparsity term that encouraged intervention. Zero replacements exactly
recover A0. T2 changes only the replacement source to a per-segment TRAIN
prototype retrieved from adjacent observed context; it excludes the target
segment's pixels from local matching. T3 adds Huber supervision for the
per-segment replacement score. Labels are obtained from the frozen early-TRAIN
A0 and prototype bank stored by the P4 strict-OOF run, evaluated on later
TRAIN anchors at stride seq_len. They are not computed from VAL or TEST.

T4 applies endpoint alignment to whichever T1/T2/T3 has the best VAL MSE.
T5 adds a stricter replacement threshold for the final two observed segments
to whichever T1/T2/T3/T4 has the best VAL MSE. All methods train 10 epochs
from the same random mask initialization and existing full-TRAIN A0 reference.
Selection is by VAL MSE only; final TEST is read once.

    python -m pytest -q tests/test_pgir.py tests/test_pgir_tricks.py
    python tools/run_fpem_pgir_trick_queue.py --worker 0 --workers 3 --gpu 0
    python tools/run_fpem_pgir_trick_queue.py --worker 1 --workers 3 --gpu 1
    python tools/run_fpem_pgir_trick_queue.py --worker 2 --workers 3 --gpu 2

Outputs go to results/fpem_pgir/tricks_20261023. The OOF supervision uses only
11 independent calibration anchor windows in ETTh1 and 65 in each ETTm case;
its labels may therefore be noisy. Also, core-only A0 and full-TRAIN A0 are
different learned functions, so OOF patch gains are useful TRAIN supervision
but not guaranteed to transfer to the final predictor.
