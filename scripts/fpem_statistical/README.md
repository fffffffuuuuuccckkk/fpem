# H-space statistical factorization pilot

This is a **new FPEM adaptation**, not a line-by-line reproduction of any cited
paper. It runs on Backbone `H=[B,C,P,D]` only; raw-input trend/seasonality,
EIIL, GRL, environment labels, A0 forecast bypass and probability heads are
not part of this stage.

| Mode | H-space operation | Statistical factor | Inspiration |
| --- | --- | --- | --- |
| `t1_group` | Groupwise `(H-mean)/std` | Group mean and log of the *used* std | Basic latent statistics |
| `t2_orthogonal` | Paired learnable Householder rotation, then group norm, inverse rotation | Rotated group mean and log-std | Orthogonal coordinate learning; new FPEM design |
| `t0_stnorm` | Group, token and variable normalized views, then Linear fusion | Statistics from each nondegenerate view | [ST-Norm](https://doi.org/10.1145/3447548.3467330), H-space adaptation |
| `t4_learned` | Predict coefficients from group statistics, then normalize | Actually used learned coefficients | [Dish-TS](https://ojs.aaai.org/index.php/AAAI/article/view/25914), H-space adaptation |
| `c1_covariance` | Group covariance eigendecomposition and whitening | Group mean and matrix log-covariance | [GDWCT](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cho_Image-To-Image_Translation_via_Group-Wise_Deep_Whitening-And-Coloring_Transformation_CVPR_2019_paper.pdf), H-space adaptation |
| `c3_dual` | Group norm plus group whitening followed by Linear fusion | Both sets of statistics | Dual normalization/whitening ideas; new FPEM design |
| `c8_mixed` | Instance group norm on half D, batch feature norm on half D | Per-instance stats; batch branch recorded separately | [IBN-Net](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Xingang_Pan_Two_at_Once_ECCV_2018_paper.php), H-space adaptation |

Controls: unchanged A0, A0 continued (`a1`), Complementary Gate (`gate`),
LayerNorm, `groupnorm` with G=1, identity-initialized raw Linear and raw
residual MLP. All methods use the same A0 reference, TRAIN labels for MSE,
VAL to select the epoch and TEST once for the selected checkpoint. The joint
fine-tuning phase picks two candidates per case by VAL, never TEST. Every
independent experiment recreates the same seeded shuffled TRAIN loader.

First-round scope: four predefined cases, one seed, Zinv-only. The statistic
descriptor is **not** converted to an H-shaped legacy Zvar. Only T1/T2/T4/C1
are asserted to reconstruct H; C3 and C8 are not claimed reversible. The
second-stage statistic-conditioned prediction/probability and three-seed
confirmation require screening evidence and are not silently mixed into this
first-round loss.

Run: `bash scripts/fpem_statistical/run_pilot.sh` from the project root.
