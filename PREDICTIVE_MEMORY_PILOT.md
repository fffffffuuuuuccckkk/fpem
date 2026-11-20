# FPEM 环境感知预测记忆库 Pilot（2026-10-24）

此实验位于独立文件，不改现有 Backbone、PTM、PGIR、CSIF、Feature Fusion 或概率头。A0 使用同一已保存 checkpoint，输出头沿用 `head_linear`，所有新模块默认不参与旧命令。

## 当前代码审计

- A0：`Model.encode(X)` 得到 `[B,C,P,D]` 和只由历史 X 计算的均值/尺度；`frozen_h_head(H)` 是冻结参考预测，`forecast_head(H)` 使用现有共享 `head_linear`。两者最后按历史均值/尺度反标准化。
- 现有 Complementary Gate：`Zinv=H(1+tanh(g(H)))/2`，`Zvar=H(1-tanh(g(H)))/2`，故 `Zinv+Zvar=H`。原框架的独立 `Yinv` 是辅助预测；当前新 Pilot 的主预测改为 H 空间融合后同一个 Head 输出。
- EIIL `PredictiveConflictEnvironment.update` 只接受按 ID 排列的完整 TRAIN 预测及标签，输出 soft q，并已实现 Hungarian 编号对齐。环境的原始尺度/偏移梯度定义没有改动。
- PTM 的 `PrototypeBank` 可复用历史归一化与 KMeans 原型思路，但它的全 TRAIN 未来均值 V 不适合直接给 TRAIN 自身查询；PGIR 的原型输入替换在 PatchTST/iTransformer 上无稳定收益，简单 Feature Fusion 也未证明跨 Backbone 改善。因此本 Pilot 不替换输入，改为 H 空间小幅、可回退的检索修正。

## 新模块与防泄漏边界

- `predictive_memory.py`：从 TRAIN 历史构建固定 KMeans Key；用冻结 A0 在 TRAIN 产生历史尺度标准化响应 `R=(Y-Y_A0)/sigma_X`。每个环境 Stage 仅重估隐藏原型、共享响应、软环境条件响应、支持量和组内/组间离散度。
- TRAIN 查询使用 3 个 chronological fold；每折 Value 从完整历史—未来区间与该折查询区间**不相交**的训练锚点重算。不是仅从 Top-K 检索结果中删掉自己。每折的环境软标签也只使用该折允许锚点的预测与标签重新推断，按预测梯度中心的尺度方向作确定性编号排序，不沿用看过查询标签的全 TRAIN q。相邻滑窗按长度 `seq_len+pred_len` 的时间块/变量计独立支持。VAL/TEST 只查完整 TRAIN Bank，推理检索不接触自己的未来标签。
- `retrieval_guided_adapter.py`：复用 Complementary Gate。检索可靠时少量软对齐 Zinv；Zvar 门控初始接近全保留，环境响应只提供 TRAIN 聚合统计的摘要，轻量变化适配器最后一层为零初始化；最终 `H_fused=Zinv'+Zvar'`，同一 Head 输出点预测。另保留 `Yinv` 辅助监督及环境分类/GRL。
- 初始无检索旁路严格返回 H 和 A0；新检索的对齐强度、变化适配强度初始都很小。若环境从历史不可辨识，条件映射可信度会下降，且正式队列不训练 P3/P4。

## 运行及解释

队列：`tools/run_fpem_predictive_memory_queue.py <case> --gpu <id>`；三组为 `patchtst_ETTh1_720`、`itransformer_ETTm1_720`、`moderntcn_ETTm2_720`。每组 Phase 0 后运行 B0/B1/P1/P2；仅 Phase 0 同时显示历史可辨识的环境、历史/未来响应相关性、跨环境支持和 permutation 显著性时才运行 P3/P4。每次训练以 VAL MSE 选一个 checkpoint，TEST 仅作为探索性结果。队列跳过已有 `run_complete`，失败等 60 秒重试一次并保留日志。

Phase 0 的环境可预测性是按时间顺序切分 TRAIN 的诊断，不代表因果不变性；permutation test 只衡量环境响应差异相对于打乱 q 的强度。未来残差可能仍含确定性模型遗漏，不能统称“物理创新”。概率头此轮没有默认启动：必须先看到点预测及映射证据，再以新的点预测训练 OOF 残差与概率模型。

预计成本：每组 A0/环境 Phase 0 需遍历全 TRAIN，最多 3 GPU 并行；每个 Pilot 为 10 Epoch、冻结 A0 Backbone、只训练共享 Head 和新增适配/分类器。原型数 8，响应 Value 大小主要为 `K×(1+env_num)×pred_len`，无需逐样本保留全量未来序列到推理 Bank。
