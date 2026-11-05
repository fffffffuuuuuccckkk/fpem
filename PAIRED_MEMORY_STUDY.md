# FPEM 实配对记忆与直接拼接 Pilot

此实验位于独立模块，不改动已有 A0、PTM、CSIF、GCID、CFIP 或上一轮原型融合。借鉴历史检索与分层组织的研究动机；以下实现为 FPEM 自己的适配设计，不是 PFRP/ProtoTS/TFPS 的完整复现。

## 实际前向

1. 现成 A0 checkpoint 冻结，`encode(X)` 给出 `H:[B,C,P,D]`。真实 TRAIN anchor 记忆保存 `X,H,Y,A0,R,start,end`，而不是聚类中心的平均未来。
2. TRAIN 查询仅检索 `source_end <= query_start` 的已完成历史记录；VAL 查询仅用在第一个 VAL 历史窗口之前完整可见的 TRAIN 记录；TEST 使用 TRAIN 记忆。查询只用 X。Kc/Kr 默认 3/2，Common/Rare 由 TRAIN 历史 Key 的局部密度中位数划分；不等同于真实不变/变化标签。
3. `cat(H_q,H_1,...,H_K)` 沿 D 拼接，再通过同一个扩展线性 Head；原 A0 列与 bias 复制且固定，新列零初始化。缺失候选填零并保留 valid mask。无 Attention、H Adapter、Hypernetwork 或强制 Zinv/Zvar 分解。
4. 将检索的真实 TRAIN 未来使用其对应历史均值与受保护标准差归一化，与当前样本的归一化点预测直接拼接，简单 Linear ResidualHead 输出修正；残差头零初始化。原始真实未来仍另存于记忆，数值极端的归一化输入仅在进入 Head 时裁剪到 ±20；记录该裁剪比例。最后按当前查询历史均值/标准差恢复预测尺度。

## 阶段、选择与证据门

- Phase A：A0、随机 H、全部最近 H、Common H、Rare H、混合 H。无未来修正。
- Phase B：固定 Phase A 的 VAL 最佳 H 视图，比较无未来、真实未来、历史残差、打乱真实配对、仅 Common、仅 Rare、Common+Rare、随机未来。B0 可复用数学上相同的 A 阶段结果；混合视图的 B6 可复用 B1。
- 每个训练 run 的 checkpoint 只按 VAL 前半段 MSE 选择。后半段 VAL 专供区间校准；TEST 只做探索性报告。所有 run 独立重建同 seed 的训练 DataLoader。
- 风险阶段只有在 VAL 上最佳 B 胜过 A0，且真实未来 B1 胜过错配未来 B3 时才自动启动；否则明确写出 `risk_evidence_gate_skipped.txt`，不宣称机制成立。
- 风险训练重新训练三折时间前向点模型：TRAIN 前 40%→40–60%，前 60%→60–80%，前 80%→80–100%。每折的点模型与记忆均只看该折查询之前完整观测的数据。用其 OOF 误差训练历史可观测风险尺度，后半段 VAL 独立做同时轨迹覆盖的 split-conformal 校准。固定尺度、检索残差尺度、新颖性、分歧、完整风险并列比较。
- 风险值仅指历史关系迁移/误差尺度风险，不是有标签校准的“真实未见环境概率”。普通 conformal 对任意分布漂移不保证 95% 覆盖。

## 结果与执行

结果根目录：`results/fpem_paired_memory/pilot_20261026/`，含每个运行的 `run_config.json`、`memory_audit.json`、`best.pt`、`result.json`、日志；完成后生成分案例和全局 CSV/Markdown。每个队列失败会休息一分钟重试一次；独立的风险队列等待对应点预测队列结束，未通过证据门则安静跳过。

```bash
python -m pytest -q tests/test_fpem_paired_memory.py
python tools/run_fpem_paired_memory_queue.py patchtst_ETTh1_720 --gpu 0
python tools/run_fpem_paired_memory_queue.py itransformer_ETTm1_720 --gpu 1
python tools/run_fpem_paired_memory_queue.py moderntcn_ETTm2_720 --gpu 2
python tools/run_fpem_paired_risk_queue.py patchtst_ETTh1_720 --gpu 0
python tools/run_fpem_paired_risk_queue.py itransformer_ETTm1_720 --gpu 1
python tools/run_fpem_paired_risk_queue.py moderntcn_ETTm2_720 --gpu 2
python tools/summarize_paired_memory.py --wait
```

## 尚未声称完成的扩展

Phase C 的 P1 KMeans 辅助、P2 medoids、P3 预测感知、P4 层次与 P5 时频真实样本检索，以及 Phase D 的软频率划分/多 seed，未在本批实现或运行。先用 Raw KNN 确定直接 H/未来拼接是否有价值；不能因为引入记忆或风险分数就宣称已发现不变模式、未知环境或取得增益。后续是否扩展以当前 Pilot 的 VAL、真实未来对错配未来的差距及 OOF 转移证据为前提。
