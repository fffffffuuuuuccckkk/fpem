# FPem-PMG：Pattern-Mapping Graph FPem

## 1. Motivation

FPem-PMG 用训练集中跨窗口反复出现、且对未来具有一致预测意义的 temporal pattern 与 temporal mapping 定义 invariant。它不是固定 K 的 prototype bank，也不需要环境、域、季节或负载标签。

核心语义是：

```text
Invariant = recurrent predictive patterns + recurrent predictive mappings
Variant   = stable graph 无法解释的 pattern/mapping information
```

特别地，系统能区分：seen pattern + seen mapping、unseen pattern、seen pattern + unseen mapping、seen mapping pair + abnormal delta dynamics。

## 2. Architecture

```text
X [B,L,C]
  │
  ├─ per-window normalization
  ▼
PatchTST backbone
  ▼
H [B,C,P,D] ── PatternProjector ── q [B,C,P,d_p]
  │                                  │
  │                    ┌─────────────┴─────────────┐
  │                    ▼                           ▼
  │          Stable Pattern Graph        Stable Mapping Graph
  │                    │                           │
  │                    └──── invariant evidence ──┘
  │                                      │
  ├──────── graph contexts ──────────────┤
  ▼                                      ▼
InvariantPatternEncoder                a=1-c_inv
  │                                      │
  ▼                                      ▼
Z_inv                         VariantPatternEncoder(H,sg(Z_inv))
  │                                      │
  │                                      ▼
  │                                    Z_var
  │                                      │
  │                                      ▼
  │                         one latent environment e
  │                                      │
  ├──────── chronological cross-fit r ───┤
  │                                      ▼
  └──────────────────────────── representation modulation
                                         │
                                         ▼
                                       Z_f
                                         │
                                  shared forecast head
                                         ▼
                                       Y_hat
```

## 3. Formulas

Pattern query：

```math
q_{bcp}=\operatorname{normalize}(P(H_{bcp})).
```

Pattern responsibility：

```math
r_{tk}=\operatorname{softmax}_k[-\tfrac12 d_M(q_t,P_k)+\log(S_k+\epsilon)].
```

Invariant and variant：

```math
c_{inv}=c_{pattern}c_{mapping},\quad a=1-c_{inv},
```

```math
H_{current}=E_{stable}(H),\quad C_{graph}=E_c(C_{pattern},C_{mapping}),
```

```math
Z_{inv}=c_{inv}H_{current}+(1-c_{inv})C_{graph},
```

```math
Z_{var}=a\,E_{var}(H,\operatorname{sg}(Z_{inv})).
```

Environment and fusion：

```math
e=E_{env}(Z_{var},a),
```

```math
Z_f=Z_{inv}+r\,m(e,Z_{inv})\odot[\gamma(e)\odot LN(Z_{inv})+\beta(e)-Z_{inv}].
```

`Y_hat` 与 `Y_inv` 使用同一个 forecasting head。

## 4. Stable Pattern Graph

每个 active node 保存：`mean`、diagonal `variance`、token sufficient-statistics count、distinct anchor support、predictive support 和连续 stability。节点通过 candidate → stable 两阶段产生：构图器从按时间排序的训练窗口中抽取候选；覆盖半径由训练 embedding 最近邻距离的 `median + 1.4826 MAD` 自动估计。`mean/variance/delta` 使用全部 stride=1 窗口，但 recurrence support 以 `anchor_id=start_index//seq_len` 去重，同一 non-overlapping 时间块最多贡献一次。节点统计由 count/sum/sum-square 得到，没有 prototype momentum。

预测支持使用同一节点所关联的完整 prediction horizon，经固定的归一化 target representation 后估计一致性，不再使用 future mean。高频但未来行为不一致的节点不会获得高 stability：

```math
S_k=\operatorname{norm}(window\_support_k)\operatorname{norm}(predictive\_consistency_k).
```

图大小完全由数据覆盖与跨窗口 recurrence 决定；没有 `num_patterns` 参数。

## 5. Stable Mapping Graph

mapping 只沿同一变量的连续 patch：`q[b,c,p] -> q[b,c,p+1]`。构图时使用完整 soft responsibilities 的外积，并仅按累计 posterior mass 0.99 做稀疏化；没有固定 Top-2。边 recurrence 同样按 anchor 去重。最终边是稀疏结构：

```text
src_index, dst_index, count, window_support,
delta_mean, delta_variance, stability
```

其中 `delta=q_{t+1}-q_t`。forward 通过矩阵乘法展开 Mahalanobis 二次式，不构造 `[B,C,P,K,K,D]`。因此既可发现不存在的 `A -> B`，也可发现见过 `A -> B` 但本次 delta 异常。

## 6. Z_inv

pattern context 是 responsibility 对节点均值的期望；mapping context 是稀疏边权重对 destination pattern 均值的期望。`InvariantPatternEncoder` 分别得到 `H_current` 和 `C_graph`，再严格按 `c_inv` 插值。低 evidence 时退回 stable graph context，绝不通过 `H + ...` 无条件复制原始表示。

## 7. Z_var

`VariantPatternEncoder` 独立读取 `H` 与 `stopgrad(Z_inv)`。它的输出再乘 `a=1-c_inv`。代码从未使用 `H-Z_inv`，也没有 prediction residual 环境分支。

## 8. Latent environment

系统仅产生一个 latent `e`。`LatentEnvironmentEncoder` 以 variation activation 加权所有 channel/patch token 后投影到默认 16 维。没有 environment ID、固定环境数、环境 prototype、环境重建或多专家 router。

## 9. Novelty

pattern novelty 来自训练 best-match Mahalanobis distance 的经验 CDF；mapping novelty同时包含未被 active sparse edge 覆盖的 responsibility mass 与 edge delta empirical-CDF compatibility。所有分数方向统一为 `0=familiar, 1=novel`，NULL probability 等于 pattern unexplained probability，并不是可学习 prototype。

## 10. Retrospective validity

旧 retrospective 接口仍被保留，但其 forecast origin 与窗口内部 observed tail 错位，因此默认关闭，也不再作为 reliability 输入。后续只有在实现真正 historical cut-point forecasting head 后才可重新启用。

该函数仍不接收预测任务的 future Y，仅作为关闭状态下的兼容接口保留；文档不再把它描述为有效的预测校验。

## 11. Reliability

唯一 reliability head 是 `Linear(3,1)+sigmoid`，输入 `[mean(a), mean(u_pat), mean(u_map)]`。训练完成后进行 TRAIN-only chronological cross-fit：前 40% 构图预测 40%-60%，前 60% 构图预测 60%-80%，前 80% 构图预测 80%-100%。监督量为 `Loss(Y_inv,Y)-Loss(Y_env,Y)`，样本保存到 `reliability_crossfit.pt`；随后恢复完整 TRAIN graph 并拟合线性 calibrator。`r=0` 时融合严格退回 `Z_inv`。

## 12. Training stages

1. Stage 0：若目标 graph 尚不存在，先对 backbone + PatternProjector + shared head 做一次固定的 TRAIN-only forecasting warmup，使 pattern space 具有预测语义。
2. Stage 1：`Exp_Long_Term_Forecast` 从 `train_data` 新建 `shuffle=False` loader；backbone/projector 在 eval/no-grad 下提取 canonical token；只用 TRAIN 构图并保存 `pattern_mapping_graph.pt`。
3. Stage 2：冻结 backbone、PatternProjector、pattern graph 和 mapping graph，并将前两者保持 eval；训练 invariant/variant encoder、environment encoder/fusion 与共享 head。reliability head 只在随后的 chronological cross-fit stage 中训练。
4. `fpem_pmg_refresh_every=0` 默认永不刷新；正整数才进行 epoch-level TRAIN-only refresh。

每个 epoch 另存 `training_state.pth`（model、optimizer、epoch、graph metadata）。中断后传入 `--fpem_pmg_resume_checkpoint <path>/training_state.pth` 可继续训练；best `checkpoint.pth` 仍保持现有工程的纯 model state_dict 格式。

图 artifact 同时保存构图时的 backbone/projector state；加载图时先恢复这两个坐标定义再冻结。旧 artifact 若缺少该状态会自动重建。

Stage 2 总损失：

```math
L=L_{full}+L_{inv}+\lambda_{cons}L_{cons}+\lambda_{sep}L_{sep}.
```

## 13. Inference

普通 `forward` 仍返回 prediction tensor，兼容现有训练/测试代码。需要诊断时调用 `forward_with_diagnostics`，返回 prediction、prediction_inv、pattern_novelty、mapping_novelty、variation_activation、environment 和 environment_reliability。

## 14. Leakage safety

- Pattern graph、mapping graph、novelty CDF：只由按时间排序的 TRAIN dataset 构建。
- graph API 不接收 val/test loader；artifact metadata 固定记录 `source_split=train`。
- reliability 样本只来自三个 chronological TRAIN prefix/heldout 切分；最终恢复完整 TRAIN graph，val/test 不参与。
- inference forward 不接收 Y；retrospective function 的签名只有 `x_norm`。
- 代码在推理 reliability 边界前有明确的 `NO TEST-FUTURE INFORMATION BEYOND THIS POINT` 注释。

## 15. Configuration

真实运行配置由 `run.py` CLI 提供；`configs/fpem_pmg.yaml` 是同名键的集中参考。默认总开关为关闭，所以旧 FPem 不受影响。主要尺寸只有 `pattern_dim=32`、`env_dim=16`；Stage 2 使用 cons/sep auxiliary 权重。

## 16. Ablations

`scripts/run_fpem_pmg.sh` 接受环境变量 `ABLATION=A0..A7`：

| ID | 执行路径 |
|---|---|
| A0 | 原始 PatchTST backbone |
| A1 | pattern-only decomposition |
| A2 | pattern + transition-count mapping，不查 delta |
| A3 | pattern + full mapping distribution，Z_inv only |
| A4 | 禁用 variant/environment，invariant fallback |
| A5 | Z_inv + Z_var + environment，但 reliability 固定为 1 |
| A6 | 完整 FPem-PMG |
| A7 | 与当前默认 A6 相同，显式确认 retrospective validity 关闭 |

这些开关分别跳过 mapping、delta、variant、environment、reliability 或 retrospective 的真实执行路径。

## 17. Implementation files

- `models/fpem/pattern_mapping_graph.py`：projector、pattern/mapping graph query、retriever、Z_inv/Z_var 与整体 PMG。
- `models/fpem/pattern_graph_builder.py`：TRAIN-only candidate/stable graph builder。
- `models/fpem/pattern_mapping_reliability.py`：single e、modulation、reliability。
- `models/fpem/pattern_mapping_losses.py`：consensus 与 separation。
- `models/PatchTST_FPEM.py`：最小集成、共享 head、retrospective、diagnostics。
- `exp/exp_long_term_forecasting.py`：Stage 1、loss/logging、可选 refresh。
- `tools/inspect_pattern_mapping_graph.py`：文本/JSON/PNG 诊断。
- `tests/test_pattern_mapping_graph.py`：关键研究现象、checkpoint、leakage 测试。

## 18. How to run

完整 A6：

```bash
ABLATION=A6 bash scripts/run_fpem_pmg.sh
```

ETTh1 一轮 smoke：

```bash
SMOKE=1 ABLATION=A6 bash scripts/run_fpem_pmg.sh
```

图诊断：

```bash
python tools/inspect_pattern_mapping_graph.py \
  checkpoints/<setting>/pattern_mapping_graph.pt \
  --output_dir graph_diagnostics/<setting>
```

旧 FPem 保持原命令；只要不传 `--fpem_pmg_enabled 1`，仍执行原双编码器/预测 delta 路径。
