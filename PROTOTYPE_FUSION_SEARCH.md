# FPEM 历史预测原型 × 多空间融合 Pilot

此入口是独立实验，不修改原 PTM、PredictiveMemory、CSIF 或 A0。第一批仅实现并调度 P0/P2/P3 与 F1/F2/F3；M0/M1/M2 是开关，但主队列只跑 M1。以前的 TRAIN held-out 诊断表明历史表示不足以可靠预测环境，因此 M2 不作为默认主实验。F3 为检索条件低秩动态 Head；这是针对 FPEM 的设计，不是 HN-MVTS 的原样复现。

## 数据边界

- A0 Backbone 和预测头从既有 checkpoint 加载并冻结，不重新训练。
- P0：历史窗口归一化后 KMeans。P2：仅源 TRAIN anchor 的 A0 残差监督历史 Key 度量，再 KMeans。P3：仅历史 Key 的 2×4 粗细聚类。
- TRAIN 查询使用 chronological purged folds。每折的 Key 度量、原型、响应值、环境标签均只从与完整 query 历史及未来区间不重叠的源 anchor 拟合。
- VAL/TEST 查询完整 TRAIN 记忆；原型选择和融合推理只读取历史 X 及训练所得参数。未来真实值只在 TRAIN 建库、训练损失和最终离线评价使用。
- 每个新融合模块输出从零修正初始化，初始预测等于 A0。F1 在 H 空间残差修正，F2 用历史 Key 对训练原型 Value 注意力聚合后修正 H，F3 对共享 A0 线性预测 Head 施加低秩动态增量。F1/F2/F3 不复制 A0 预测头。
- M0 不使用响应轨迹，M1 使用训练集共享未来响应，M2 使用从历史 H 预测的 qhat 加环境响应；不在测试时读取真实未来推断环境。

## 实验与选择

每个案例按 Phase A（P0/P2/P3 TRAIN-only 质量诊断）→ Phase B（三种 P 均固定 F1/M1）→ Phase C（P0 与 VAL 上最佳的 P，各测 F2/F3）顺序运行。Phase C 的选择只看 VAL，不看 TEST。每个 run 以 VAL MSE 选择 checkpoint，TEST 仅做探索性报告。不做 27 格笛卡尔积；现阶段也未开始概率头或全量预测长度矩阵。

三个先行设置：PatchTST/ETTh1/720、iTransformer/ETTm1/720、ModernTCN/ETTm2/720。每个有独立的现成 A0 checkpoint，使用相同数据切分。结果写入 `results/fpem_prototype_fusion_search/pilot_20261025/`，各 run 保存 `run_config.json`、`prototype_diagnostics.json`、`memory_report.json`、`train_history.json`、`best.pt` 和 `result.json`；案例汇总为 `all_results.csv` 与 `summary.md`。

```bash
python -m pytest -q tests/test_prototype_fusion.py
python tools/run_fpem_prototype_fusion_queue.py patchtst_ETTh1_720 --gpu 0
python tools/run_fpem_prototype_fusion_queue.py itransformer_ETTm1_720 --gpu 1
python tools/run_fpem_prototype_fusion_queue.py moderntcn_ETTm2_720 --gpu 2
```

## 尚未实现/运行

P1 medoid、P4 时频、P5 regime-residual、P6 refinement 及 F4/F5/F6 尚未实现。未进行 M2 的正式比较、随机环境/错配 Mapping 消融、概率区间评估、跨 seed 验证或全数据集扩展。它们应在本 Pilot 的 VAL 和检索诊断支持下再实现，不应把本阶段结果描述成完整组合搜索结论。
