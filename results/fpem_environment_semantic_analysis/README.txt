FPEM Predictive Environment Semantic Analysis

Protocol
- Exact saved final-stage q from each current no-future/no-anchor best run.
- TRAIN samples remain chronological and sample_id is asserted to equal 0..N-1.
- Calendar labels use the input-window end timestamp.
- Physical regimes use input-window means and 33/67% quantiles computed on TRAIN samples only.
- Environment inference is untouched; semantic labels are never model inputs.
- Significance uses 1000 fixed-margin random contingency tables, mathematically equivalent to permuting environment labels.

Files per case
- sample_environment_assignments.csv: raw environment IDs and q.
- semantic_labels.csv: aligned post-hoc labels and raw-window summaries.
- environment_composition.csv: both conditional-probability directions, explicitly named.
- environment_enrichment.csv and semantic_soft_assignment.csv: plot source values.
- environment_semantic_metrics.json/txt: statistics and full provenance.

Completed cases: 18
