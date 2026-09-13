# EIVPM PatchTST plugin

This directory is independent from legacy `models/fpem`. Global Pattern and
Mapping statistics are fitted only from the chronological TRAIN loader and its
FOIL-compatible sample IDs. Validation/test inputs may query the frozen banks;
their targets never update prototypes, occurrences, transitions, or mixture
posteriors.

Tensor flow:

```
X [B,L,C]
 -> PatchTST encoder H [B,C,T,D]
 -> multi-scale segments [N,D], scales 1/2/3/4
 -> responsibilities [B,C,T,K]
 -> C_P_inv / C_P_var [B,C,T,D]
 -> H' [B,C,T,D]
 -> flatten [B,C,T*D]
 -> C_M_inv / C_M_var [B,D]
 -> universal head + implicit low-rank update
 -> Y [B,pred_len,C]
```

The ablation switches are E0 baseline; E1 invariant Pattern; E2 forced variant
Pattern; E3 gated variant Pattern; E4 invariant Mapping; E5 forced variant
Mapping; E6 gated variant Mapping; E7 full invariant; E8 full forced; and E9
full selective. Run one experiment with, for example,
`EXPERIMENTS=E3 bash scripts/run_eivpm_patchtst_ablation.sh`, or omit the
variable to run E0--E9 with identical settings.

Pattern environment occurrence is sample-level rather than a segment count.
Existing segment salience is normalized inside each sample and combined with
the soft prototype assignment before evaluating
`1 - product_j(1 - r_ijk)`; environment occurrence is then the mean over
independent TRAIN samples.

Stage A freezes both variant adapters and low-rank head modulation. Stage B
freezes the universal linear head and trains the enabled sample-conditioned
branches. Both Pattern residual outputs and the low-rank left factor are zero
initialized, making the plugin an exact identity at initialization.
