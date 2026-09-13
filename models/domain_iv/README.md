# DomainIV PatchTST

This is an independent experiment path and does not query the Pattern Bank or
Mapping Bank. It reuses only the PatchTST backbone and the repository's
TRAIN-fitted FOIL soft environment assignments.

## Tensor flow

```
X [B,L,C] -> PatchTST H [B,C,T,D]
H -> F_inv -> Z_inv [B,C,T,D]
H -> F_var -> Z_var [B,C,T,D]
Pool(Z_inv), Pool(Z_var) -> [B,D]
Flatten(Z_inv) -> [B,C,T*D]
W_U Z_inv + gate * A diag(code(Z_var)) B Z_inv -> [B,pred_len,C]
```

The low-rank update is evaluated as two sequential einsums and never creates a
batch of full prediction matrices. Its left factor is zero-initialized.

## Ablations

- D0: strict PatchTST baseline
- D1/D2/D3: contrastive constraint with invariant-only/forced/gated Mapping
- D4/D5/D6: mutual-information constraint with invariant-only/forced/gated Mapping

`--domain_constraint none|contrastive|mutual_info` is also available as an
explicit override for isolated diagnostics. The standard D0--D6 run leaves it
unset so each declared experiment selects exactly one constraint.

Environment identities are never used directly. Losses use `q @ q.T`, linear
HSIC with q, or reconstruction of q; therefore renaming all environment columns
does not affect the geometry. FOIL centers and all normalization statistics are
TRAIN-only. Validation/test targets never enter environment discovery.
