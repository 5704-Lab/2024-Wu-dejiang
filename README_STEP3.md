# Step 3.2-C：完整 Validation 与配准参数冻结

Step 3.2-C 使用 Step 1 生成的外部分组 split，只运行完整 Validation 集，对 Step 3.2-B 的 SIFT/RANSAC、Homography 几何验收、near-identity 和质量选择参数进行最终审查并冻结。Step 3.2-D 才会使用冻结参数运行 Test；本步骤不读取、不评估 Test，也不能根据 Test 调参。

## 正式运行范围

- split：仅 `val`
- 样本数：400
- source groups：40
- 选择方式：`val.txt` 固定顺序的完整 split，不抽样
- Test 访问：禁止
- 原始数据集：严格只读

历史 Step 3.2-B 的 10 个跨 split 样本仅为功能 smoke probe，不是参数冻结依据。

## 运行

```bash
python -m pip install -r requirements-step2.txt
python scripts/run_step1.py verify --config configs/step1_dataset.yaml
python scripts/run_step3.py validate-freeze --config configs/step3_alignment.yaml
python scripts/run_step1.py verify --config configs/step1_dataset.yaml
```

`validate-freeze` 没有 split、limit、include-test 或阈值覆盖参数。配置必须声明：

```yaml
roadmap_step: 3.2-C
run:
  purpose: validation_freeze
  split: val
  selection: full_split
  expected_sample_count: 400
  expected_group_count: 40
```

Runner 只打开 `samples.jsonl` 和 `splits/val.txt`，不会构造或读取 `test.txt`。它会交叉验证 Step 1 split manifest、split summary 和 Step 2 index manifest 中的 Validation hashes。

## 参数审查与冻结原则

完整 Validation 报告用于检查：

- Homography 估计、几何接受/拒绝和 near-identity 比例；
- 内点数、内点比例、平均/最大内点重投影误差；
- 条件数、投影面积与重叠比例；
- 尺度、各向异性、旋转、透视强度和角点位移；
- original/warped/selected SSIM、互信息和 candidate delta；
- 所有 criterion failure 和拒绝原因。

只有 Validation 数值分布和诊断图能够支持参数调整。不得为了提高接受率自动放宽安全阈值。若调整任一参数，必须重新运行全部 400 个 Validation 样本，再生成冻结产物。Test 阶段不得重新冻结参数。

## 产物

### `artifacts/step3/alignment_report.json`

schema：`step3-alignment-validation-v1`。

包含：

- Validation-only scope；
- processed/unique sample count、group count、complete 状态；
- `test_accessed: false`；
- 输入 provenance 和 effective parameters；
- stages、decisions、status/reason/criterion failure counts；
- 每个样本的完整判据和 Homography 诊断；
- 数值分布的 count、missing count、min/max/mean/std；
- p01/p05/p25/p50/p75/p95/p99，使用 NumPy `quantile(method="linear")`；
- 按终态分组的分布。

不可用或非有限数值写为 `null`，严格禁止 NaN/Infinity。

### `artifacts/step3/alignment_parameters_frozen.json`

schema：`step3-alignment-freeze-v1`。

包含所有会影响配准结果的：

- SIFT 参数；
- matcher 与 Lowe ratio；
- RANSAC 参数；
- Homography geometry 阈值；
- near-identity 阈值；
- masked SSIM/MI 和质量选择阈值。

这些参数以排序、紧凑 JSON 规范化后生成 `parameters_sha256`。诊断图片限制和输出路径不进入 digest。Step 3.2-D 必须读取该冻结产物或校验相同 digest。

本次完整 Validation 将 `ransac.max_inlier_reprojection_error` 从 Step 3.2-B 候选值 5.5 px 调整并冻结为 9.0 px。Validation 中 3 个超过 5.5 px 的样本仅存在孤立最大误差（6.66–8.40 px），其平均内点误差仅 0.049–0.087 px、内点比例为 0.953–0.977、条件数为 1.0001–1.0014、投影重叠至少 0.99927，因此整体 Homography 稳定且接近单位阵。9.0 px 仍保留高于 Validation 最大值的有限保护。最终 400 个 Validation Homography 全部通过几何验收；398 个判定为 already-aligned，另外 2 个候选 Warp 均因 SSIM 和互信息显著下降而保留 original。

### `artifacts/step3/run_manifest.json`

记录软件版本、Validation-only scope、source snapshot digest、Step 1/Step 2 Validation hashes、ordered Validation IDs digest、配置 hash，以及 report/freeze artifact hashes。追踪产物中不记录绝对数据集路径，也不记录 Test hash。

## 诊断图片

完整 Validation 默认不会为 400 个普通样本生成全部图片。程序按确定性优先级选择有限样本：

1. Homography 未估计；
2. 几何拒绝；
3. Warp 后保留 original；
4. Warp 被选择；
5. 最大内点误差较高的 near-identity 样本。

默认最多 20 个样本，只生成 comparison 和 RANSAC matches。图片位于 `artifacts/step3/alignment/val/`，不纳入 Git；JSON 证据纳入 Git。

## 只读与完整性保证

运行前后对原始数据集执行完整 tree snapshot；任何文件增删、大小或 hash 变化都会阻止冻结。以下情况也会 fail closed：

- Validation 样本不是恰好 400 个唯一 ID；
- Validation group 不是恰好 40 个；
- ID 不存在或属于非 Validation group；
- Step 1/Step 2 Validation hash 不一致；
- 输入路径逃逸数据集根目录；
- 输出目录位于数据集内；
- runner 异常或处理不完整。

单个样本的 Homography 估计失败属于有效评估结果，不会使完整 Validation 失效。

## 测试

```bash
python -m pytest tests/step3 -q
python -m pytest tests/step1 tests/step2 tests/step3 -q
```

测试覆盖 Validation-only 契约、禁止 Test 访问、固定顺序、重复/未知 ID、group/count 完整性、上游 hashes、参数 digest、分布统计、严格 JSON 和路径安全。

本步骤完成后停止，不运行 Test，不自动开始 Step 3.2-D。
