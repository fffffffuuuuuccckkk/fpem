# Reuse audit — 2026-11-03

| Existing component | Reuse | Necessary adaptation |
|---|---|---|
| `Model.encode` and frozen A0 head | Yes | Preserve `[B,C,P,D]` flatten order and original history-only mean/std. |
| `DirectInvariantForecaster` | Yes | Shared branch predicts complete future, never an A0 residual by default. |
| `RarePatternLearner(R3)` | Yes | Fit once on TRAIN history keys and complete normalized futures; freeze router while training shared branch. |
| R3 `query` | Yes | `weight` is `[B,C,M]`, with separate NULL mass; no sample-level reinterpretation. |
| `prototype_segment_reliability` / `variation_forecast` | Yes | Recalibrate on new shared anchor using disjoint VAL; never reuse old G1 reliability. |
| `stable_direction_basis` | Conditional | Requires actual forward/held-out evidence; no TEST-derived basis. |
| `ReferenceGuidedDecomposer` | Adapt | Its centers are H-space, whereas R3 centers are 64-D K-space. Compute q-weighted H centers on TRAIN. |
| `EnvironmentMappingEstimator` | Adapt | Existing `add` repeats `[B,M]` across C; use variable-level `[B,C,M]` weighted sufficient statistics instead. |
| `mapping_operator` | Yes | Only after fitting all prototype heads in the *same H feature space*; spectral values are diagnostics, not automatically stable directions. |
| Shared/Private / EIIL | Control | Existing implementations are separate mechanisms, not a default second grouping system. |

The new experiment links the **same frozen R3 router** to shared full-future
training and local full-future inference. Previous G1 fits an A0 residual and
is retained solely as a baseline. Prototype membership is a proxy pattern
assignment, not a demonstrated causal environment.
