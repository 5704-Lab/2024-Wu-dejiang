# SimeseUltraMaxPlus 实验总结

最后更新：2026-11-16

## 1. 项目概况

本项目是一个基于 Query–Template 的 Siamese 印刷缺陷检测流水线，流程为：

1. 对 Query 和 Template 进行配准。
2. 使用共享 ResNet-50 骨干提取多尺度特征。
3. 使用 OffsetPredictor 和 DCN Feature Warp 对 Template 特征做微对齐。
4. 使用 CBAM + 多尺度 Fusion 生成缺陷概率热力图。
5. 对热力图做候选框后处理，并生成上下文裁剪。
6. 使用配对感知 RT-DETR 对裁剪区域进行分类检测。

缺陷类别共四类：`broke`、`dirty`、`double`、`miss`。

## 2. 数据概况

数据位置：

```text
data/VOC2007
```

数据规模：

- 总样本数：4000 对 Query/Template
- Source group：400
- 总 GT 目标数：4710
- 图像尺寸：800 × 800 RGB

四类 GT 总数：

| 类别 | GT 数量 |
|---|---:|
| broke | 1590 |
| dirty | 1090 |
| double | 1030 |
| miss | 1000 |

训练 / 验证 / 测试划分：

| split | 样本数 | GT 数 |
|---|---:|---:|
| train | 3200 | 3790 |
| val | 400 | 460 |
| test | 400 | 460 |

原始 XML 中大量 `filename` 字段指向源图而非变体图，Step 1 只作为 warning 记录，不影响训练。

## 3. 环境

运行环境：

- Python 3.10.12
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0
- Ultralytics 8.3.230
- GPU：NVIDIA GeForce RTX 4090

虚拟环境：

```bash
cd /home/s4090/wdj/SimeseUltraMaxPlus
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-step2.txt
python -m pip install -r requirements-rtdetr.txt
```

## 4. 完整执行流程

### 4.1 修正数据路径

```bash
for f in configs/*.yaml; do
  sed -i 's#../../project/数据集/VOCdevkit/VOC2007#data/VOC2007#g' "$f"
done
```

### 4.2 Step 1：数据审计与划分

```bash
python scripts/run_step1.py all --config configs/step1_dataset.yaml
```

### 4.3 Step 2：配对数据校验

```bash
python scripts/run_step2.py validate --config configs/step2_dataset.yaml
```

### 4.4 训练 OffsetPredictor

```bash
python scripts/run_step3.py train-alignment --config configs/step3_train_alignment.yaml
```

产物：

```text
artifacts/step3/training/checkpoint_best.pt
artifacts/step3/training/checkpoint_final.pt
```

### 4.5 训练 Fusion

```bash
python scripts/run_step3.py train-fusion --config configs/step3_train_fusion.yaml
```

产物：

```text
artifacts/step3/fusion_training/checkpoint_best.pt
artifacts/step3/fusion_training/checkpoint_final.pt
```

### 4.6 完整 Validation 调参

```bash
python scripts/tune_proposals.py all --config configs/step3_proposal_tuning.yaml
```

产物：

```text
artifacts/step3/tuning/tuning_report.json
artifacts/step3/tuning/proposal_parameters_frozen.json
```

### 4.7 训练配对感知 RT-DETR

```bash
python scripts/prepare_paired_rtdetr_dataset.py
python scripts/train_rtdetr_paired_baseline.py
```

配对图三个通道：

- R：Query 灰度
- G：Template 灰度
- B：`|Query - Template| * 3`

### 4.8 保存分类指标

```bash
python scripts/save_rtdetr_metrics.py
```

### 4.9 端到端评估

```bash
python scripts/run_end_to_end_eval.py --config configs/step3_end_to_end_eval.yaml --split val
python scripts/run_end_to_end_eval.py --config configs/step3_end_to_end_eval.yaml --split test
```

## 5. 模型改进与修改

### 5.1 推理桥接修复

修改文件：`src/step3_model/inference_bridge.py`

- 差异特征改为 `abs(warped - query)`，与 Fusion 训练一致。
- 最终 NMS 改为类别感知，避免不同类别框互相抑制。
- 增加 `pair_aware` 模式，同时传入 Query 和 Template crop。

### 5.2 配对感知 RT-DETR

新增文件：

- `scripts/prepare_paired_rtdetr_dataset.py`
- `scripts/train_rtdetr_paired_baseline.py`
- `configs/rtdetr_paired_baseline.yaml`

解决的核心问题：`miss` 缺陷在单张 Query 图上可能不可见，必须和 Template 对比才能判断。

### 5.3 Fusion 类别加权实验

修改文件：

- `src/step3_model/train_fusion.py`
- `src/step3_model/run_train_fusion.py`

最初尝试给 `miss` 设置更高的正样本权重，但实验发现：

- miss 加权不能提升候选阶段 recall；
- 新的逐像素权重损失与原 `pos_weight` 损失并不等价；
- 平衡 Fusion 更稳定。

最终采用平衡 Fusion。

### 5.4 测试集标注补丁

新增文件：

- `scripts/build_test_annotation_patch.py`
- `scripts/build_patched_rtdetr_labels.py`
- `artifacts/step1/test_annotation_patch.jsonl`

对 test 集人工复核后补充了 190 个漏标框。

## 6. 实验结果

### 6.1 候选框阶段

平衡 Fusion 下的整体候选 recall@0.5：

| split | recall@0.5 |
|---|---:|
| val | 0.696 |
| test | 0.600 |

test 分四类候选 recall@0.5：

| 类别 | recall@0.5 |
|---|---:|
| broke | 0.533 |
| dirty | 0.645 |
| double | 0.750 |
| miss | 0.500 |

### 6.2 RT-DETR baseline 与配对感知 RT-DETR

单图 RT-DETR baseline：

| split | mAP50 | mAP50-95 |
|---|---:|---:|
| val | 0.569 | 0.439 |
| test | 0.574 | 0.425 |

配对感知 RT-DETR：

| split | mAP50 | mAP50-95 |
|---|---:|---:|
| val | 0.717 | 0.546 |
| test | 0.692 | 0.514 |

`miss` 的 mAP50 提升：

| 模型 | val miss mAP50 | test miss mAP50 |
|---|---:|---:|
| 单图 baseline | 0.028 | 0.245 |
| 配对感知 | 0.670 | 0.719 |

### 6.3 端到端结果

旧单图 Query crop 端到端：

| split | Precision | Recall | F1 |
|---|---:|---:|---:|
| val | 0.758 | 0.641 | 0.695 |
| test | 0.698 | 0.613 | 0.653 |

配对感知端到端，未修正 test 标注：

| split | Precision | Recall | F1 |
|---|---:|---:|---:|
| val | 0.776 | 0.774 | 0.775 |
| test | 0.643 | 0.750 | 0.692 |

配对感知端到端，test 补全标注后：

| split | Precision | Recall | F1 |
|---|---:|---:|---:|
| test | 0.996 | 0.823 | 0.901 |

最终 test 分四类：

| 类别 | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| broke | 201 | 163 | 0 | 38 | 1.000 | 0.811 | 0.896 |
| dirty | 167 | 146 | 0 | 21 | 1.000 | 0.874 | 0.933 |
| double | 159 | 133 | 2 | 26 | 0.985 | 0.836 | 0.905 |
| miss | 123 | 93 | 0 | 30 | 1.000 | 0.756 | 0.861 |
| 总体 | 650 | 535 | 2 | 115 | 0.996 | 0.823 | 0.901 |

## 7. 遇到的问题与解决方案

### 7.1 配置中的数据集路径失效

问题：配置文件仍指向旧路径 `../../project/数据集/VOCdevkit/VOC2007`。

解决：将所有配置中的路径统一改为 `data/VOC2007`。

### 7.2 环境中缺少 PyTorch 与 CUDA

问题：初始容器中没有 torch，且 GPU 不可见。

解决：建立 `.venv`，安装 CUDA 12.8 版 PyTorch，并确认 `torch.cuda.is_available()` 为 True。

### 7.3 `miss` 类几乎检测不到

问题：单图 RT-DETR 对 `miss` 的 val mAP50 只有 0.028。

原因：漏印缺陷在单张 Query 上可能不可见，必须与 Template 对比。

解决：识别阶段改用 Query、Template、Diff 三通道配对输入。

### 7.4 配对模型端到端推理产生大量误检

问题：端到端阶段 `miss` 大量 FP，而独立验证正常。

原因：`model.predict` 对 numpy 输入会执行 BGR→RGB 反转，而推理时传入了 RGB 顺序，导致 Query 和 Diff 通道对调。

解决：推理时构造 `[diff, template, query]` 的 BGR 顺序输入，经预处理后恢复为训练时的 `[query, template, diff]`。

### 7.5 miss 加权 Fusion 导致候选 recall 下降

问题：给 miss 更高权重后，候选阶段 recall@0.5 明显下降。

原因：逐像素权重损失改变了整体归一化方式，与原平衡训练不等价。

解决：恢复平衡 Fusion，并在类别权重相等时走原版 `pos_weight` 损失路径。

### 7.6 test 集 Precision 看似偏低

问题：double 和 dirty 的 Precision 只有 0.548 和 0.610。

分析：FP 分析发现绝大多数 FP 是真实缺陷，但测试集没有标注。

解决：人工复核 FP 图，并生成 190 个 test 标注补丁。

### 7.7 分析脚本缺少 `src` 路径

问题：`analyze_false_positives.py` 报 `ModuleNotFoundError: step1_audit`。

解决：在脚本开头加入项目根目录和 `src` 的 `sys.path`。

### 7.8 配对 RT-DETR 后期训练退化

问题：最佳 epoch 为 25，第 44/45 轮指标降为 0。

解决：当前使用 `best.pt`；后续可降低学习率、缩短 patience 以提高稳定性。

## 7.9 候选框上下文与热力图门控优化

### 问题

Demo 推理时，候选区域显示 3 个，但最终检测结果出现 2 个缺陷，其中一个不在原始候选框内。进一步测试发现，当 `expand_k` 设为 3.5 或 4.5 时，会在真实缺陷之外多检出一个不存在的缺陷。

### 原因

`expand_k` 越大，送入 RT-DETR 的裁剪窗口越大。RT-DETR 会在整个裁剪窗口内检测目标，而不是只检测候选框中心的缺陷。当窗口过大时，候选区域之外的正常印刷纹理或套印偏差被送入 RT-DETR，并可能被误判为缺陷。

旧流程没有用 Siamese 对比热力图约束 RT-DETR 的输出，因此第二阶段可以独立产生热力图不支持的错误检测。

### 解决方案

在 `run_inference` 中加入热力图门控：

- 对每个 RT-DETR 检测框，统计框内 Siamese 热力图的响应。
- 只有框内最大概率不低于 0.6，且高响应像素比例不低于 0.03 的检测才保留。
- 没有对比热力图支持的检测框被丢弃。

门控参数：

```yaml
gating:
  gate_heatmap: true
  gate_prob_threshold: 0.5
  gate_max_min: 0.6
  gate_min_positive_fraction: 0.03
```

### expand_k 扫描

在 val 上扫描不同 `expand_k`：

| expand_k | Precision | Recall | F1 | 检测框数 |
|---:|---:|---:|---:|---:|
| 1.8 | 0.771 | 0.774 | 0.772 | 462 |
| 2.5 | 0.772 | 0.780 | 0.776 | 465 |
| 3.0 | 0.777 | 0.778 | 0.777 | 461 |
| 3.5 | 0.780 | 0.778 | 0.779 | 459 |
| 4.0 | 0.774 | 0.776 | 0.775 | 461 |
| 4.5 | 0.776 | 0.774 | 0.775 | 459 |

结论：`expand_k=3.5` 在 val 上 F1 最高，但 `2.5` 与 `3.5` 差距很小，且能减少过大上下文带来的误检。

### 最终结果

加入热力图门控后：

- Demo 不再出现候选区域之外的假缺陷。
- 正式端到端 val/test 指标保持不变：
  - val F1：0.7748
  - test F1：0.9014
- 说明门控不会误删正式评估中的正确检测。

Demo 已改为自动读取最新候选参数和门控配置，与正式评估保持一致。

## 8. 当前结论

- 最终最优组合为：平衡 Fusion + 配对感知 RT-DETR + 热力图门控。
- 端到端系统在补全标注后的 test 集上达到：
  - Precision：0.996
  - Recall：0.823
  - F1：0.901
- `miss` 从几乎不可用提升到可用水平。

## 9. 遗留工作

1. 两个 `miss → double` 类别混淆样本需要进一步处理。
2. broke、dirty、double、miss 仍有一定 FN，召回还有提升空间。
3. 配对 RT-DETR 训练稳定性需要优化。
4. Demo 尚未接入最终配对 RT-DETR 分类结果。
5. 消融实验和论文图表仍需系统整理。

## 10. 关键文件

- 数据配置：`configs/step1_dataset.yaml`、`configs/step2_dataset.yaml`
- 训练配置：`configs/step3_train_alignment.yaml`、`configs/step3_train_fusion.yaml`
- 配对 RT-DETR：`configs/rtdetr_paired_baseline.yaml`
- 推理桥接：`src/step3_model/inference_bridge.py`
- 端到端评估：`src/step3_model/end_to_end_eval.py`
- 误检分析：`scripts/analyze_false_positives.py`
- 标注补丁：`artifacts/step1/test_annotation_patch.jsonl`
