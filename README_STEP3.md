# Step 3.2-C / 3.2-D / 3.3-A / 3.3-B / 3.4-A / 3.4-B / 3.4-C / 3.5-A：配准冻结、Test 评估、共享多尺度 Backbone、Offset Predictor、DCN Feature Warp、微对齐训练与特征差异热力图

Step 3.5-A 已实现多尺度特征差异热力图：逐尺度计算 warped template 与 query 特征的相对 L2 差异，双线性上采样到 800×800，渲染为 5 列面板（query / template / diff_C3 / diff_C4 / diff_C5），使用 INFERNO colormap 着色。使用 zero-init predictor 建立预训练基线。

## Step 3.3-B 多尺度 Backbone

正式命令：

```bash
python scripts/run_step3.py backbone-smoke --config configs/step3_backbone.yaml
```

模型使用 torchvision `ResNet50_Weights.IMAGENET1K_V2`，只保留 stem 和 `layer1`–`layer4`，移除分类 avgpool/fc。Query 和 Template 调用同一个 encoder，单次前向中按以下固定顺序返回多尺度特征：

| key | 来源 stage | channels | stride | 800×800 空间尺寸 |
|-----|-----------|----------|--------|------------------|
| C3 | layer2 | 512 | 8 | 100×100 |
| C4 | layer3 | 1024 | 16 | 50×50 |
| C5 | layer4 | 2048 | 32 | 25×25 |

- 输入：RGB float32 `[B, 3, 800, 800]`，值域 `[0,1]`
- 模型内 normalization：mean `[0.485,0.456,0.406]`，std `[0.229,0.224,0.225]`
- 输出：有序 `dict[str, Tensor]`，key 顺序固定 `["C3", "C4", "C5"]`
- 参数：23,508,032，全部 trainable，无新增层；多尺度为纯 tap，不含 FPN/neck/通道投影
- `feature_stages` 配置必须与代码常量 `FEATURE_STAGES` 严格一致，否则 fail closed

真实 smoke 使用 ordered Validation prefix 的 2 对图像、batch size 1，并在 CUDA 可用时选择 CUDA。共享证明包括：只注册一个 encoder、state dict 无 branch-specific keys、159 个参数对象全部唯一、同一 encoder 被双分支实际调用、同输入逐尺度输出逐 bit 完全一致（每个尺度 max-abs-diff 均为 0）。

权重为官方 `resnet50-11ad3fa6.pth`（已在 PyTorch Hub cache），checkpoint SHA-256 为 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`。权重保存在 cache，不复制到仓库，也不生成模型 checkpoint。

产物（schema v2）：

- `artifacts/step3/backbone/backbone_contract.json`
- `artifacts/step3/backbone/smoke_report.json`
- `artifacts/step3/backbone/run_manifest.json`

Step 3.3-B 不实现 FPN/neck、channel projection、feature fusion、attention、DCN、loss、optimizer、训练或 RT-DETR。Offset Predictor 属于 Step 3.4-A，尚未开始。

## Step 3.4-A：Offset Predictor

Step 3.4-A 在 Step 3.3-B 的多尺度共享 Backbone 之上实现 Offset Predictor，为后续 Step 3.4-B 的 DCN/Feature Warp 预测逐尺度形变采样偏移。本步骤只建立预测器模块、契约、测试与真实 smoke，不做 DCN warp（3.4-B）、不做训练（3.4-C）。

正式命令：

```bash
python scripts/run_step3.py offset-smoke --config configs/step3_offset.yaml
```

### 设计

- **逐尺度独立 head**：`3×3 conv(2C→C/4) → GELU → 3×3 conv(C/4→18)`，C3/C4/C5 各一个 head，参数独立。
- **输入 pairing**：每尺度拼接 `[query_feature, template_feature]`（2×C 通道）。
- **零初始化**：最终 conv 权重和偏置为 0，初始偏移严格为 0，使后续 DCN 初始行为等价于普通卷积（安全 identity）。
- **偏移限幅**：`tanh` × 对应 stride（C3≤8、C4≤16、C5≤32 像素），防止初始随机权重产生灾难性采样位置。
- 输出形状：每尺度 `[B, 18, H, W]`，对应 `deform_conv2d` 的 offset 布局 `[B, 2×kH×kW, Hout, Wout]`（k=3）。

### 参数

- 总参数：24,918,710（backbone 23,508,032 + predictor 1,410,678）
- 所有 Backbone 参数在 smoke 中冻结；predictor 保持 trainable（为 3.4-C 训练准备）

### 真实 Smoke 结果

- 2 对 ordered Validation pairs，CUDA 设备
- 三尺度 offset shape 完全符合预期：C3 `[1,18,100,100]`、C4 `[1,18,50,50]`、C5 `[1,18,25,25]`
- 所有 offset 严格为 0（zero-init 验证通过）
- 数据集只读、Step 3.2 证据 hashes 不变

### 产物

- `artifacts/step3/offset_predictor/offset_contract.json`
- `artifacts/step3/offset_predictor/smoke_report.json`
- `artifacts/step3/offset_predictor/run_manifest.json`

Step 3.4-A 不实现 DCN warp（属于 3.4-B）、不训练（属于 3.4-C）、不修改 Step 3.2/3.3 证据。

## Step 3.4-B：DCN / Feature Warp

Step 3.4-B 在 Step 3.4-A 的 Offset Predictor 之上实现 DCN Feature Warp：使用逐尺度 identity kernel + deformable convolution 对模板多尺度特征进行形变采样，使其在特征空间向 Query 对齐。本步骤只建立 warp 模块、契约、测试与真实 smoke，不做偏移训练（3.4-C）。

正式命令：

```bash
python scripts/run_step3.py feature-warp-smoke --config configs/step3_feature_warp.yaml
```

### 设计

- **identity kernel**：逐尺度注册 `[C, C, 3, 3]` 恒等卷积核为 buffer（`weight[c, c, 1, 1] = 1.0`），stride=1, padding=1。零偏移时 `deform_conv2d` 等价于 identity：`warped[c, h, w] = template[c, h, w]`。
- **逐尺度独立 DCN**：对 template feature 的每个尺度调用 `torchvision.ops.deform_conv2d`，使用对应尺度的 identity kernel 和 offsets。
- **输入契约**：template_features 与 offsets 两个 dict，key 顺序固定 `["C3","C4","C5"]`，逐尺度验证 shape、finite、dtype/device 一致。
- **zero-offset identity 保证**：OffsetPredictor 零初始化，offsets 全零 → warped features 逐 bit 等于 template features（经真实 smoke 验证）。

### 参数

- FeatureWarp 无 trainable 参数：identity kernel 注册为 buffer，不含 `nn.Parameter`。
- 端到端 pipeline（Backbone + OffsetPredictor + FeatureWarp）总参数不变：24,918,710（backbone 23,508,032 + predictor 1,410,678）。

### 真实 Smoke 结果

- 2 对 ordered Validation pairs，CUDA 设备
- 三尺度 warped 输出 shape 与 template feature 完全一致：C3 `[1,512,100,100]`、C4 `[1,1024,50,50]`、C5 `[1,2048,25,25]`
- zero-offset identity 逐 bit 验证通过（所有尺度 max-abs-diff 均为 0）
- 非零合成偏移产生不同于 template 的有限输出（确认 DCN 运行为非退化）
- 数据集只读、Step 3.2 证据 hashes 不变

### 产物

- `artifacts/step3/feature_warp/feature_warp_contract.json`
- `artifacts/step3/feature_warp/smoke_report.json`
- `artifacts/step3/feature_warp/run_manifest.json`

Step 3.4-B 不实现偏移预测训练（属于 3.4-C）、不修改 loss/optimizer、不保存 checkpoint、不修改 Step 3.2/3.3/3.4-A 证据。

## Step 3.4-C：微对齐训练

Step 3.4-C 在 Step 3.4-A/B 的 offset predictor + feature warp 之上实现完整的训练循环：冻结 Backbone、冻结 Warp（无参数）、仅训练 OffsetPredictor 的 1,410,678 参数，使 warped template features 向 query features 对齐。

正式命令：

```bash
python scripts/run_step3.py train-alignment --config configs/step3_train_alignment.yaml
```

### 设计

- **损失函数**：`MicroAlignmentLoss` —— 逐尺度加权 MSE：`loss = Σ α_s * MSE(warped[s], query_features[s])`。每个尺度的 MSE 内部已按 C×H×W 归一化，`α_C3=α_C4=α_C5=1.0` 保证三尺度等权贡献。
- **训练 pipeline**：Backbone(冻结) → Predictor(训练) → Warp(冻结) → loss(warped, qf) → backward → optimizer.step
- **优化器**：Adam(lr=0.001, weight_decay=0)
- **调度器**：CosineAnnealingLR(T_max=epochs)
- **数据增强**：训练集用 `PairTransform(training=True)` 进行几何翻转/旋转 + 独立两路光度增强（亮度/对比度/饱和度/色相/模糊/噪声），验证集无增强。
- **确定性训练**：`derive_seed` 保证给定 (global_seed, split, sample_id, epoch, view) 下每个 epoch 的增强结果可复现，epoch 参数推进随机种子。

### 参数

- Trainable：仅 OffsetPredictor 的 1,410,678 参数
- Frozen：Backbone 23,508,032 + Warp 0 = 23,508,032
- 总参数不变：24,918,710

### 产物

- `artifacts/step3/training/training_report.json`
- `artifacts/step3/training/run_manifest.json`
- `artifacts/step3/training/checkpoint_best.pt`
- `artifacts/step3/training/checkpoint_final.pt`

Step 3.4-C 不进行超参搜索、不实现 early stopping、不进行 mixed-precision training、不实现多 GPU 训练、不修改 Step 3.2/3.3/3.4-A/3.4-B 证据。

## Step 3.5-A：特征差异热力图

Step 3.5-A 在 DCN 对齐 pipeline 之上输出多尺度特征差异的可视化热力图，为人类理解模型行为提供第一扇窗口。

正式命令：

```bash
python scripts/run_step3.py heatmap-smoke --config configs/step3_heatmap.yaml
```

### 设计

- **差异度量**：`diff[s] = ||warped[s] - query[s]||₂ / (||query[s]||₂ + ε)`，逐像素跨通道 L2 范数，除以 query 特征范数做相对归一化。
- **上采样**：各尺度 diff map 通过 cv2.INTER_LINEAR 上采样到 800×800（C3: 8×, C4: 16×, C5: 32×）。
- **可视化面板**：5 列水平拼接 —— query | template | diff_C3 | diff_C4 | diff_C5。Query/template 用原图 RGB，diff 列用 `cv2.COLORMAP_INFERNO` 着色（感知均匀，适合出版物）。
- **预训练基线**：使用 zero-init OffsetPredictor，热力图反映训练前的特征差异分布。后续训练后重新运行可对比变化。

### 产物

- `artifacts/step3/heatmap/heatmap_contract.json`
- `artifacts/step3/heatmap/smoke_report.json`
- `artifacts/step3/heatmap/run_manifest.json`
- `artifacts/step3/heatmap/panels/*.png`（不被 Git 追踪）

Step 3.5-A 不实现注意力机制（属于 3.5-B）、不进行候选框生成（属于 3.5-C）、不修改 Step 3.2/3.3/3.4 证据。

## Step 3.3-A 历史记录

Step 3.3-A 建立了共享 ResNet-50 Backbone 基础与单尺度 layer4 契约（真实 smoke 已验证共享权重、CUDA 前向与源数据只读）；3.3-B 在其上公开了有序多尺度输出，A 的单尺度 API 已由 B 的多尺度契约替代，A 阶段的 artifacts 由 B 的 v2 artifacts 继承覆盖。


Step 3.2-C 使用完整 Validation split 冻结配准参数；Step 3.2-D 使用这份不可变参数运行完整外部 Test split。Test 结果只作描述性泛化评估，不用于调参、阈值选择或重新冻结。Step 3.3-A 尚未开始。

## Step 3.2-D 正式范围

- split：仅 `test`
- 样本数：400
- source groups：40
- 参数来源：`artifacts/step3/alignment_parameters_frozen.json`
- 冻结参数 digest：`2262bcc10235465a2138e602de584eff5ea976289b67ea920b96e694051efe8c`
- tuning/refreeze：禁止
- Validation 证据：保持字节内容不变，不重新计算
- 原始数据集：严格只读

运行命令：

```bash
python scripts/run_step1.py verify --config configs/step1_dataset.yaml
python scripts/run_step3.py test-evaluate --config configs/step3_alignment_test.yaml
python scripts/run_step1.py verify --config configs/step1_dataset.yaml
```

Test 配置不能包含任何 SIFT、matcher、RANSAC、geometry、near-identity 或 quality 参数。Runner 会先校验 Step 3.2-C Validation report、frozen parameters 和 manifest 的固定 hashes，再仅从 freeze artifact 构造 `SIFTAligner`。

在 Windows 的 CRLF checkout 中，文本证据 hash 会先明确规范化为 LF；无效 UTF-8 或裸 CR 会被拒绝。该规则只用于验证已提交的 JSON 文本证据，不通过 JSON 重序列化改变其语义或覆盖 C 产物。

## Step 3.2-D 产物

- `artifacts/step3/test/alignment_report.json`：完整 Test 报告，400 条逐样本记录、判据、状态、分布和有限诊断。
- `artifacts/step3/test/val_test_comparison.json`：Validation 与 Test 的描述性 count/rate/分布差异，不含综合分数、新阈值或自动通过结论。
- `artifacts/step3/test/run_manifest.json`：Test provenance、冻结证据 hashes、输出 hashes、软件版本和 no-tuning/no-refreeze 控制状态。
- `artifacts/step3/alignment/test/*.png`：最多 20 个确定性诊断样本的 comparison/matches，不纳入 Git。

Test report 的 scope 必须声明完整 Test、参数从 freeze 加载、未调参、未重新冻结。Val-vs-Test comparison 的 policy 必须声明：未执行参数/阈值选择、Test 未用于 tuning、未应用自动 acceptance gate、未开始下一步。

本次外部 Test 结果为：400/400 Homography 均成功估计并通过几何验收；399 个样本为 already-aligned，1 个样本 `0153_b1_test` 超过 near-identity 角点位移阈值并生成候选 Warp，但 SSIM 下降 0.00277、互信息下降 0.1100，因此保留 original；没有任何 warped candidate 被选择。相对 Validation，already-aligned 比例由 99.5% 变为 99.75%，original-retained 由 0.5% 变为 0.25%，差异仅作描述，不用于修改冻结参数。

## 完整性与泄漏保护

Step 3.2-D 只打开 `samples.jsonl`、`splits/test.txt`、split manifest/summary 和 Step 2 index manifest。它不会打开 `train.txt` 或 `val.txt`，也不会重新加载 Validation 图像。运行会验证：

- 400 个唯一 Test IDs 和 40 个 Test groups；
- train/val/test group sets 两两不重叠；
- 每个 Test ID 属于 Test group，且所有 Test groups 均被覆盖；
- Step 1/Step 2 samples 和 Test split hashes 一致；
- 所有源路径位于 dataset root；
- Test 输出根不会覆盖 Step 3.2-C artifacts；
- 运行前后 dataset snapshot 完全相同；
- 诊断读取后再执行最终 snapshot；
- C 三份冻结证据在 Test 运行前后 hashes 不变。

任何结构、来源、冻结证据、只读或完整性错误都会 fail closed；单个样本的 Homography 拒绝属于合法评估结果。

## Step 3.2-C 冻结摘要

Validation 为 400 samples / 40 groups。最大内点重投影误差阈值基于 Validation 从 5.5 px 调整并冻结为 9.0 px；其余参数保持候选值。最终 Validation：400 个 Homography 全部几何通过，398 个 already-aligned，2 个候选 Warp 因 SSIM/MI 下降保留 original。

冻结产物与 Validation 报告位于：

- `artifacts/step3/alignment_report.json`
- `artifacts/step3/alignment_parameters_frozen.json`
- `artifacts/step3/run_manifest.json`

Step 3.2-D 绝不写入这些路径。

## 测试

```bash
python -m pytest tests/step3 -q
python -m pytest tests/step1 tests/step2 tests/step3 -q
```

测试覆盖 Validation freeze 回归、Test-only contract、算法参数覆盖拒绝、冻结证据完整性、CRLF/LF digest、Test 固定顺序、无 train/val split 访问、group/hash/path 安全和严格 JSON。

Step 3.2-D 完成后停止，不修改冻结参数，不开始 Step 3.3-A。

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
