# 当前 FPEM：模块与实现说明

> 依据主服务器 `/data/OuXiaoyu/Time-Series-Library-FPEM` 在 2026-10-14 的代码整理（HEAD `f4774a0`，包含工作区中尚未提交的实验代码）。本文描述当前 **A2 + complementary gate + predictive environment + 可选概率修正头** 路径，而不是早期 Pattern/Mapping Graph 路径。符号中 `B` 为 batch、`C` 为变量数、`L` 为历史长度、`T` 为预测长度、`P` 为骨干 token 数、`D` 为表示维度、`M=ceil(T/16)` 为未来 patch 数。

## 1. 一句话概括

FPEM 先从预训练骨干得到预测表示 `H`，依据当前模型的 **TRAIN 预测误差** 交替发现软环境 `q`；再将 `H` 信息守恒地拆成稳定的 `Zinv` 与环境相关的 `Zvar`。`Zinv` 独立给出确定性预测 `Yinv`；`Zvar` 预测未来 patch 的变化条件，概率头在 **不改变 `Yinv` 锚点** 的前提下生成多条可能未来。

```text
历史 X ──归一化──> 预训练 Backbone ──> H [B,C,P,D]
                                         │
                              complementary gate
                                ┌────────┴────────┐
                             Zinv              Zvar
                               │                  │
                       shared forecast head   Future-Zvar [B,C,M,D]
                               │                  │
                         Yinv [B,T,C]      概率修正头 / 多轨迹采样
                               └────────┬─────────┘
                                      Y_samples

TRAIN 的当前预测 Ŷ 与真实 Y ──> EIIL 式预测冲突环境 q
q ──> Zinv 环境对抗约束 + Zvar 环境分类约束 ──> 下一阶段重新发现 q
```

## 2. 各主要模块

| 模块 | 当前具体实现 | 主要代码 |
| --- | --- | --- |
| A0 预测骨干 | 历史窗口逐变量归一化；PatchTST、iTransformer、ModernTCN 均转成 `[B,C,P,D]` 接口。`head_linear` 将 token 展平后预测未来，再按历史均值/尺度反归一化。当前实验先选已有候选中 TEST MSE 最低的 A0 checkpoint，再作为 FPEM 初始化。 | `models/PatchTST_PredictiveEnvIV.py` 的 `encode()`、`forecast_head()`；`scripts/fpem_probabilistic/run_test_best_patchtst_a0.sh` |
| Predictive Environment | 只对完整、按时间顺序排列的 TRAIN 样本，用当前预测 `Ŷ` 与标签 `Y` 算每个样本的二维 scale/bias probe gradient。优化样本×环境 logits，让环境平均梯度的方差增大，同时约束环境质量平衡、分配熵；softmax 得到软归属 `q∈[0,1]^{N×K}`。每次更新后用 Hungarian 匹配对齐环境列。 | `models/predictive_env/eiil_environment.py` 的 `PredictiveConflictEnvironment.update()` |
| 动态交替训练 | 每个 stage 先对 TRAIN 全部样本重新推理 `q`，再固定、停梯度使用该 `q` 训练若干 epoch；随后依据新预测再更新。当前实验 `K=3`、`stage_epochs=2`、`eiil_steps=50`。环境编号本身没有语义。 | `tools/run_predictive_env_iv_patchtst.py` 的 stage loop |
| 信息守恒分解 | 元素级 `g=tanh(gate_net(H))`，`Zinv=H⊙(1+g)/2`、`Zvar=H⊙(1-g)/2`，故逐元素 `Zinv+Zvar=H`。正 gate 偏向不变，负 gate 偏向变化；不是两个独立 projection。 | `models/predictive_env/signed_gate_decomposition.py` 的 `ComplementaryGateDecomposer` |
| 环境监督 | 对池化后的 `Zvar` 做软环境交叉熵；对 `Zinv` 经 Gradient Reversal Layer 做同样的分类，使表示难以泄露环境。标签 `q` 停梯度，训练损失仍保留 `Yinv` 独立预测 MSE，避免不变表示塌缩。 | `models/predictive_env/classification_constraint.py` |
| 不变预测锚点 | `Yinv=ForecastHead(Zinv)`，沿用预训练骨干的预测头结构；`h_forecast_head` 是预训练头的冻结副本，主要用于 H 预测及诊断，不等于最终不变头。概率模式的普通 `forward()` 默认输出 `Yinv`，不会默认采样 100 条。 | `models/PatchTST_PredictiveEnvIV.py` 的 `fuse_predictions()`、`forward_components()`、`forward()` |
| Future-Zvar | 对历史 `Zvar` 沿 token 维池化；按预测范围每 16 步划一个 future patch，用 patch 中心的归一化 horizon embedding 和轻量 MLP 预测 `[B,C,M,D]` 未来变化条件。推理只使用历史输入。 | `models/predictive_env/horizon_future_variant.py` 的 `encode_future()` |
| 概率仿射动力学（IAPAD） | `AffineCenterHead` 给出未来各 patch 的典型 `(γ,β)`；联合 Temporal Conv + 条件 FiLM 的速度场通过 conditional flow matching 生成整段 `(γ,β)` 轨迹；可选 AR(1) innovation 补充仿射无法解释的残差。patch 参数按中心位置连续插值到各预测点。 | `models/predictive_env/probabilistic_affine_dynamics.py`、`affine_target_projection.py`、`stochastic_innovation.py` |
| 多概率头对比 | 除 IAPAD `legacy_full` 外，同一条件上比较 deterministic affine、Gaussian affine、3-component affine MDN、affine flow、独立 Gaussian residual、低秩 Gaussian residual、residual flow。它们是**对比头**，不是同时串联进最终预测的八层模块。 | `models/predictive_env/probability_heads.py`、`tools/run_fpem_head_comparison.py` |

环境发现的核心量为：

\[
g_i^{scale}=2\,\operatorname{mean}\big[(\hat Y_i-Y_i)\hat Y_i\big],\qquad
g_i^{bias}=2\,\operatorname{mean}(\hat Y_i-Y_i),
\]

\[
q_{ie}=\operatorname{softmax}(\ell_i)_e,\qquad
\bar g_e=\frac{\sum_iq_{ie}g_i}{\sum_iq_{ie}}.
\]

优化器最小化“负的跨环境梯度方差 + 环境 mass 平衡惩罚 + `sum(q log q)` 熵项”。这不是按 `Zvar` 直接聚类；不过当前 `environment mode=current` 会使用当前模型预测，所以表示更新能够**通过预测行为**影响下一轮环境。验证集/测试集不进入环境发现。

## 3. 概率修正头如何工作

### 3.1 训练目标与条件

`Yinv` 与历史输入尺度 `s_x` 是锚点。每个 TRAIN 样本、变量、未来 patch，先用真实未来离线投影出正则化仿射目标 `θ*=(γ*,β*)`：

\[
u_t=(Y_{inv,t}-\overline Y_{inv,k})/s_x,\quad
r_t=(Y_t-Y_{inv,t})/s_x,
\]
\[
\beta_k^*=\overline r_k,\qquad
\gamma_k^*=\frac{\sum_{t\in k}u_t(r_t-\overline r_k)}
{\sum_{t\in k}u_t^2+\lambda_{ridge}}.
\]

`Yinv`、`s_x` 和 `θ*` 在概率目标中停梯度；最后一个不足 16 步的 patch 单独按实际长度处理。`γ/β` 用 TRAIN 样本的稳健尺度（绝对值的 75% 分位）标准化，训练记录截断比例。该投影只是 TRAIN 监督和 oracle 诊断，**测试生成绝不使用真实未来**。

IAPAD 的损失由 center Huber、joint affine flow matching、可选 innovation Gaussian NLL 与跨 patch smoothness 构成。Flow 初始状态为 `stopgrad(μ)+σ₀ε`；速度网络通过卷积耦合相邻 patch，不是每个 patch 独立的 Gaussian 输出。当前推理以 Euler 积分，实验脚本采用 12 步、100 条轨迹，并按 chunk 生成以控制显存。

### 3.2 从变化参数还原未来

对每条采样轨迹，将 patch `(γ,β)` 连续插值到时间点，再计算：

\[
Y_t^{(s)}=Y_{inv,t}+\gamma_t^{(s)}
(Y_{inv,t}-\overline Y_{inv,k})+s_x\bigl(\beta_t^{(s)}+e_t^{(s)}\bigr).
\]

`γ=β=e=0` 时精确回退 `Yinv`。返回 `Y_samples [B,N,T,C]`、均值、中位数、95% 区间和 patch 的 `γ/β` 样本。无 innovation 的 affine flow 是独立消融；`shuffled_zvar`、`unconditional_flow`、Gaussian baseline 和 invariant-only 也可单独评价。

## 4. 当前实际训练协议：MSE 搜索与概率头分开辨认

1. **A0：**选取当前可用、已完成 TEST 评价的骨干候选中 MSE 最低者，校验/转换 checkpoint，作为共享 reference。这里的“最优”是**现有候选中的最优**，不是穷尽所有骨干超参数。
2. **A2 锚点：**从该 A0 初始化，使用 `complementary_gate + classification + K=3` 训练 10 epoch；每 2 epoch 根据 TRAIN 当前预测重新推断环境。该步采用 `prob_ablation=invariant_only`，目标是形成 `Yinv`，不训练概率修正损失。骨干与不变头分别使用小学习率；decomposer、环境头、Future-Zvar 的 LR 尽量复用此前坐标搜索结果，缺少历史结果时使用脚本默认值。
3. **现在优先执行的 MSE 搜索：**暂停新的概率头训练，固定同一 A0、骨干 LR 及其他 FPEM LR，只搜索 A2 的不变预测头 LR `2e-5 / 5e-5 / 1e-4`。已完成的 `5e-5` A2 checkpoint/TEST 指标直接复用；其他两档新训 A2（仍动态更新 TRAIN 环境），按完整 TEST `inv_MSE` 选择。结果写入独立的 `results/fpem_probabilistic/inv_head_lr_search_20261014/`，不覆盖旧概率结果。
4. **已暂停的多头 LR 比较：**此前实验加载并冻结整个 A2 锚点（包括 `Yinv`、骨干和分解器），只训练 Future-Zvar 的 horizon/future 网络与概率头 bank。`Zvar` 输入在概率路径显式 `detach`。八个头共享同一个 Future-Zvar/条件编码器，概率损失取算术平均；不是八次独立重训编码器。头 LR 原网格为 `5e-5 / 1e-4 / 5e-4`，每个头按 TEST `ProbTS_CRPS` 选择。现保留已产出的结果，不继续此搜索。

完整的单阶段 IAPAD 训练接口仍保留：若不传冻结锚点，则训练 `L_inv + λ_env L_env + w(L_center+L_FM+L_innovation+L_smooth)`，概率损失可传至 Future-Zvar；这是**另一种协议**，不能与当前“固定 A2、联合比较多头”结果混称。

## 5. 输出、评价与边界

- 确定性比较：A0 与 `Yinv` 的 TEST MSE/MAE；概率样本均值的 MSE/MAE另列，不能把其中较好的那个自动称为主预测结果。
- 概率比较：经验 ensemble CRPS、与 ProbTS `quantiles_num=20` 对齐的 `ProbTS_CRPS`、95% 覆盖率 `PICP_95`、区间宽度 `MPIW_95`、WIS、EnergyScore、样本多样性和 patch 边界跳变。两种 CRPS 归一化不同，不能混报。
- 当前 `best_heads_test_selected.csv` **按 TEST 的 ProbTS_CRPS 选每个头的 LR**；A0 也按已有 TEST 候选选取。因此这些结果是探索性、带测试集选参偏差的比较，不应直接当作严格未见测试集上的论文结论。若用于正式结论，应改为 VAL 选参、TEST 只评价一次。
- TRAIN 标签只用于环境更新、仿射投影监督及训练损失；验证/测试真实未来只用于评价（或明确标识的离线 oracle 诊断），不能进入真实生成路径。
- 当前路径不使用早期 Pattern Graph、Mapping Graph、Variant Pattern Bank，也不需要手工环境标签。

当前 MSE 搜索输出位于各服务器的 `results/fpem_probabilistic/inv_head_lr_search_20261014/`，每个配置保存 `candidate_metrics.csv`、`best_inv_head_lr.json` 和新候选训练日志。此前多头结果仍分别位于 `results/fpem_probabilistic/test_best_patchtst_a0_20261013/`、`results/fpem_probabilistic/test_best_backbones_a0_20261013/`，保留 `best_heads_test_selected.csv`。

## 6. 主要入口索引

| 目的 | 文件 |
| --- | --- |
| 骨干、分解、`Yinv` 与概率推理接口 | `models/PatchTST_PredictiveEnvIV.py` |
| 环境推断及 Hungarian 对齐 | `models/predictive_env/eiil_environment.py` |
| 软环境分类与 GRL | `models/predictive_env/classification_constraint.py` |
| Future-Zvar | `models/predictive_env/horizon_future_variant.py` |
| IAPAD center/flow/innovation | `models/predictive_env/probabilistic_affine_dynamics.py`、`affine_target_projection.py`、`stochastic_innovation.py` |
| 八个即插即用对比头 | `models/predictive_env/probability_heads.py` |
| 动态环境 + A2 训练 | `tools/run_predictive_env_iv_patchtst.py` |
| 多头共享条件训练 | `tools/run_fpem_head_comparison.py` |
| 概率损失权重与评价 | `tools/fpem_probabilistic_training.py`、`tools/evaluate_fpem_probabilistic.py` |
| 当前三骨干不变头 LR 搜索 | `scripts/fpem_probabilistic/run_inv_head_lr_search.sh` |
| 已暂停的多概率头搜索 | `scripts/fpem_probabilistic/run_test_best_patchtst_a0.sh` |
