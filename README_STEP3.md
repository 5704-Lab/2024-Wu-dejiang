# Step 3.2-B：Homography 合法性与质量验收

Step 3 读取 Step 1 的只读样本索引和无泄漏 split，使用 SIFT、Lowe ratio 与 RANSAC 估计 Query→Template Homography。Step 3.2-B 在 Step 3.2-A masked 指标基础上增加几何合法性验收、near-identity 跳过和 original/warped 质量选择。当前范围仍是宏观配准验证，不包含 Siamese、DCN、注意力候选区域或 RT-DETR；Step 3.3 尚未开始。

## 环境与运行

```bash
python -m pip install -r requirements-step2.txt
python scripts/run_step3.py align --config configs/step3_alignment.yaml
```

`validation.sample_count` 是 train/val/test 合计验证样本数，`validation.seed` 固定确定性抽样。程序只向 `artifacts/step3/` 写入报告和可视化。

## 三阶段语义

schema v4 明确区分：

- `homography_estimated`：OpenCV 返回了 Homography 和 RANSAC mask；不代表矩阵合法。
- `geometry_accepted`：RANSAC 证据、矩阵数值、投影四边形、变换强度和重叠比例全部通过。
- `quality_improved`：仅对实际生成的非 near-identity Warp 候选评估；nullable，未评估时为 `null`。

兼容字段 `success` 等同于 `geometry_accepted`。终态包括：

- `homography_not_estimated`
- `geometry_rejected`
- `already_aligned`
- `warped_selected`
- `original_retained`

返回图像始终是 `selected_variant`。几何拒绝和 near-identity 都保留 original，不执行 Warp。

## Homography 验收

配置位于 `configs/step3_alignment.yaml`：

1. 最小 RANSAC 内点数和内点比例；
2. 最大内点重投影误差，同时保留平均误差用于诊断；
3. H 必须是有限 3×3 矩阵，并可通过 `H[2,2]` 归一化；
4. 在中心化、尺度归一坐标中计算条件数；
5. Query 四角的齐次分母和投影坐标必须有限；
6. 投影四边形面积比例处于范围内、保持方向且不自交；
7. 中心 Jacobian 的尺度、各向异性、旋转，以及归一化 H 的透视强度必须在阈值内；
8. 投影四边形与 Template 矩形的连续面积重叠比例达到下限。

危险或被拒绝的 H 不会传给 `cv2.warpPerspective`。逐样本 `criteria` 记录每项判据的适用性、值、运算符、阈值和通过状态，`reasons` 记录全部拒绝或选择原因。

## near-identity 与质量选择

当四角最大位移、尺度偏差、绝对旋转和透视强度同时低于 near-identity 阈值时，样本标记为 `already_aligned`，直接使用原 Query，避免插值造成微小退化。

对非 near-identity 且几何通过的 H：

1. 生成 warped candidate 和有效 footprint；
2. original 与 candidate 在完全相同的 footprint 上计算 SSIM/互信息，SSIM mask 可腐蚀边界；
3. 至少一个指标达到最小改善，并且两个指标都没有超过允许回退时，才选择 warped；否则状态为 `original_retained`。

报告分别记录 original、warped candidate 和 selected 指标。`quality_improved_rate` 以实际完成质量评估的候选数为分母；没有候选时为 `null`。

## 输出

- `artifacts/step3/alignment_report.json`：`step3-alignment-v4`，包含阈值快照、阶段/决策统计、拒绝原因、判据失败计数、逐 split 和逐样本诊断，纳入 Git。
- `artifacts/step3/alignment/<split>/*.png`：选择结果对比图、差异热图和 RANSAC 内点图，可重复生成，不纳入 Git。

报告不记录机器相关的绝对数据集路径。不可用数值写为 `null`，并通过严格 JSON 序列化，禁止 NaN/Infinity。

## 只读保证

运行前验证输出目录不等于、也不位于原始数据集目录内；每个输入路径必须解析在数据集根目录内。真实运行前后可执行 Step 1 verify，对比已记录的 source digest：

```bash
python scripts/run_step1.py verify --config configs/step1_dataset.yaml
```

## 测试

```bash
python -m pytest tests/step3 -q
python -m pytest tests/step1 tests/step2 tests/step3 -q
```

合成测试覆盖有限性/归一化、条件与投影几何、面积/方向/自交、尺度/旋转/透视、重叠、near-identity、质量选择、已知平移和相同图像跳过 Warp；报告测试覆盖 schema v4、严格 JSON、阶段分母、拒绝原因和 already-aligned 非失败语义。
