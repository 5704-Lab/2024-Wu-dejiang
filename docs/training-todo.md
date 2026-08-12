# 剩余工作待办文档

最后更新：2026-08-12

---

## 一、Step 3.4-C：训练 OffsetPredictor（DCN 微对齐）

### 目标

训练 OffsetPredictor（1,410,678 参数），使 DCN 学会将 template 特征"掰"到与 query 对齐。正常区域对齐后 diff 消失 → 候选框只响应真正的缺陷区域。

### 当前状态

- ✅ 损失函数（`MicroAlignmentLoss`）：逐尺度加权 MSE，C3/C4/C5 等权
- ✅ 训练循环（`train_one_epoch` / `validate_one_epoch`）
- ✅ 完整 runner（`run_train_alignment.py`）：Adam + CosineAnnealingLR
- ✅ 配置（`configs/step3_train_alignment.yaml`）
- ✅ 测试（17 个 unit/config 测试）
- ❌ **实际训练未执行**

### 训练命令

```bash
python scripts/run_step3.py train-alignment --config configs/step3_train_alignment.yaml
```

### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | train split（3200 对，320 组） | 数据增强：几何翻转/旋转 + 独立双路光度增强 |
| Batch size | 8 | |
| Epochs | 20 | |
| 优化器 | Adam（lr=0.001, weight_decay=0） | |
| 调度器 | CosineAnnealingLR（T_max=20） | |
| 损失 | MSE(warped, query) per scale | α_C3=α_C4=α_C5=1.0，各尺度等权 |
| Trainable | OffsetPredictor（1,410,678） | |
| Frozen | Backbone（23,508,032）+ Warp（0） | |
| 验证 | val split（400 对），每 epoch 验证 | |
| 设备 | auto（优先 CUDA） | |

### 产物

```
artifacts/step3/training/
  training_report.json    # 每 epoch 的 train/val loss
  run_manifest.json       # 训练运行记录
  checkpoint_best.pt      # 最佳 val loss 的 predictor 权重
  checkpoint_final.pt     # 最终 epoch 的 predictor 权重
```

### 预估时间

- 每 epoch：~400 batches × ~3-5s = 约 3-5 分钟
- 总计 20 epochs：约 60-100 分钟（CUDA）

### 已知问题

1. **训练脚本无实时日志**：当前只在最后输出 JSON 摘要——长时间看不到进度。建议添加 `tqdm` 进度条或每 N batch 打印 loss。
2. **无 early stopping**：固定 20 epochs，不会自动停止。
3. **checkpoint 体积**：state_dict + optimizer state 约 17 MB（git-ignored）。

### 训练后验证

训练完成后，重新运行 Step 3.5-C（候选框 recall）和 Step 4-A（上下文裁剪），对比 zero-init vs trained 的 recall@0.5 和候选框数量：

```bash
python scripts/run_step3.py proposals-smoke --config configs/step3_proposals.yaml
python scripts/run_step3.py crop-smoke --config configs/step3_crop.yaml
```

期待变化：
- recall@0.5 从 15% → 60%+
- 候选框数量从 ~10K → ~50-200（大幅下降）
- 裁剪覆盖率从 32.5% → 80%+

---

## 二、Step 5-B：训练 RT-DETR（缺陷分类）

### 目标

训练 RT-DETR 模型在原 800×800 数据集上进行缺陷分类（4 类：broke/dirty/double/miss）。

### 当前状态

- ✅ COCO 格式数据集已导出（Step 5-A）：train 3200 / val 400 / test 400
- ✅ 推理桥接代码已就绪（`inference_bridge.py`）
- ❌ **RT-DETR 模型尚未集成**
- ❌ **训练脚本未写**

### 训练命令（待实现）

```bash
# 预期命令格式
python scripts/run_detr.py train --coco artifacts/step3/detection/coco_train.json \
                                  --val artifacts/step3/detection/coco_val.json
```

### 配置参数（建议）

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | COCO train（3200 张 800×800） | |
| 类别数 | 4（broke/dirty/double/miss） | |
| Backbone | ResNet-50（ImageNet 预训练） | 与 Siamese pipeline 一致 |
| 输入尺寸 | 640×640 或 800×800 | RT-DETR 标准尺寸 |
| Epochs | 100-300 | 取决于收敛速度 |
| Batch size | 4-8（取决于 GPU 显存） | RT-DETR 显存消耗较大 |
| 优化器 | AdamW（lr=1e-4, weight_decay=1e-4） | RT-DETR 标准配置 |
| 数据增强 | 多尺度 resize + flip + 随机裁剪 | 已在 COCO 数据上标准 |

### 产物（预期）

```
artifacts/step5/
  rtdetr_checkpoint.pth    # 训练好的 RT-DETR 模型
  training_log.json        # 训练指标（mAP, loss per epoch）
  eval_results.json        # val/test split 评估
```

### 注意事项

1. **RT-DETR 官方实现**：建议使用 `lyuwenyu/RT-DETR` GitHub 仓库或 `huggingface/transformers` 集成版本
2. **显存需求**：RT-DETR 在 800×800 训练时显存消耗约 8-12 GB（batch=4），需要确认 GPU 显存是否充足
3. **训练增强**：需要加入多尺度 resize（480-800）+ 随机裁剪 + padding，让模型适应推理时的 pad 场景
4. **mAP 基准**：训练完成后在 test split 上评估 COCO mAP@0.5:0.95

---

## 三、Step 6：端到端推理

### 目标

串联整个 pipeline：原始图像对 → Siamese 筛选 → 候选框 → 上下文裁剪 → padding → RT-DETR → inverse mapping → 最终检测结果。

### 当前状态

- ✅ Siamese 筛选管线完整（Backbone → OffsetPredictor → Warp → Fusion → 候选框 → 裁剪 → inverse map）
- ✅ 推理桥接代码已就绪（`inference_bridge.py`）
- ✅ Padding 工具已就绪
- ❌ **缺少训练好的 OffsetPredictor 权重**（依赖 3.4-C）
- ❌ **缺少训练好的 RT-DETR 模型**（依赖 5-B）

### 推理流程

```
输入: query.jpg, template.jpg
  ↓
Step 3.5-B: Backbone → OffsetPredictor(best.pt) → Warp → MultiScaleFusion
  ↓
Step 3.5-C: enhanced heatmap → postprocess → 候选框 [K, 4]
  ↓
Step 4-A: 候选框 → merge → expand(k=1.8) → 上下文裁剪 → 裁剪小图 [K, 3, h, w]
  ↓
Step 5-A: 裁剪小图 → pad(800×800) → RT-DETR → 检测 [K_i, 6]
  ↓
Step 4-B: inverse_map_boxes → 全图坐标
  ↓
Step 6: NMS 合并 → 最终检测结果
输出: [boxes, scores, classes] 全图坐标
```

### 期望输出

```json
[
  {
    "box": [120, 200, 180, 280],
    "score": 0.92,
    "class": 1,
    "class_name": "dirty"
  },
  ...
]
```

### 验证指标

- 端到端 recall：与 GT 的 IoU@0.5 匹配率
- 端到端 precision：检测结果中真正缺陷的比例
- 推理速度：每对图像的处理时间（前向 + warp + 后处理 + RT-DETR）

---

## 四、Step 7：对比与消融实验

### 目标

量化每个模块的贡献，写进论文。

### 实验设计

#### 实验 1：DCN 微对齐消融

| 配置 | OffsetPredictor | 候选框 recall@0.5 |
|------|----------------|-------------------|
| A（baseline） | zero-init（未训练） | ~15% |
| B（proposed） | 训练后（step3.4-C） | 预期 60%+ |

- 对比 A vs B 的热力图差异：正常区域的假阳性应该大幅减少
- 对比候选框数量的变化：从 10K → ~100

#### 实验 2：CBAM 注意力消融

| 配置 | 融合方式 | 候选框 recall@0.5 |
|------|---------|-------------------|
| A | 无注意力（直接 L2 diff，Step 3.5-A） | 预期低 |
| B | 简单平均融合（去掉 CBAM，纯加权） | 预期中 |
| C | CBAM + 学习融合（Step 3.5-B） | ✅ 当前方案 |

#### 实验 3：上下文裁剪消融

| 配置 | 裁剪方式 | 候选框 recall |
|------|---------|-------------|
| A | tight crop（无上下文，expand_k=1.0） | — |
| B | 上下文裁剪（expand_k=1.8） | ✅ 当前方案 |

#### 实验 4：RT-DETR baseline

| 配置 | 方法 | mAP |
|------|------|-----|
| A | 直接 RT-DETR 在原图（无 Siamese 筛选） | 预期 baseline |
| B | Siamese 筛选 + RT-DETR | ✅ proposed |

#### 实验 5：SIFT 配准 vs DCN 对齐

| 配置 | 对齐方式 | 候选框 recall |
|------|---------|-------------|
| A | SIFT+RANSAC 单应性 warp（Step 3.2） | — |
| B | DCN 微对齐（Step 3.4） | ✅ 当前方案 |

### 每个实验需要的产物

- 定量指标表（recall/mAP/precision）
- 消融对比热力图/检测结果可视化

---

## 五、Step 8：开发演示软件

### 目标

构建一个可交互的 GUI 应用，让用户上传印刷图像对，看到完整的缺陷检测流程和结果。

### 当前状态

- ⏳ 尚未开始

### 功能需求

1. **图像上传**：选择 query 和 template 两张图像
2. **一键推理**：点击后自动运行完整 pipeline
3. **结果可视化**：
   - 原始图像（query / template）
   - 多尺度差异热力图（Step 3.5-A 的 5 列面板）
   - CBAM 增强热力图（Step 3.5-B）
   - 候选框叠加（在 query 图上画候选框）
   - 最终检测结果（RT-DETR 分类框 + 类别标签 + 置信度）
4. **参数调整面板**：可选调整 proposal 参数（min_area, expand_k, threshold 等）和查看效果变化
5. **导出结果**：保存带标注的检测图、导出 JSON 结果

### 技术选型建议

| 选项 | 优点 | 缺点 |
|------|------|------|
| **Gradio** ✅ | 最简、Hugging Face 生态、3 行代码启动 | 布局灵活性有限 |
| Streamlit | 灵活、社区大 | 需要写 state management |
| PyQt/PySide | 完全自定义 | 开发量大 |
| Flask + HTML/JS | Web 部署 | 需要写前后端 |

**推荐 Gradio**：与 PyTorch 生态无缝集成、`gr.Interface` 一行启动、支持 Image/JSON 输入输出。

### 预期界面

```
┌──────────────────────────────────────────────┐
│  基于深度学习的印刷缺陷检测系统                   │
│──────────────────────────────────────────────│
│                                              │
│  ┌──────────┐ ┌──────────┐                   │
│  │ Query 图  │ │Template图│ ← 拖拽上传         │
│  └──────────┘ └──────────┘                   │
│            [ 开始检测 ]                        │
│                                              │
│  ┌──────────┬──────────┬──────────┐           │
│  │ 热力图C3  │ 热力图C4  │ 热力图C5  │           │
│  ├──────────┼──────────┼──────────┤           │
│  │ 候选框图  │ 最终检测  │          │           │
│  └──────────┴──────────┴──────────┘           │
│                                              │
│  检测结果:                                     │
│  [broke x=120 y=200 w=60 h=80 conf=0.92]     │
│  [dirty x=350 y=150 w=40 h=50 conf=0.87]     │
└──────────────────────────────────────────────┘
```

### 注意事项

1. **模型加载**：需要提前加载 Backbone + OffsetPredictor(best.pt) + Warp + Fusion + RT-DETR 权重到 GPU，避免每次推理重新加载
2. **文件组织**：演示代码和训练代码分离，放到 `src/demo/` 目录
3. **无数据集依赖**：演示模式不访问训练/测试集——纯推理
4. **依赖**：`gradio>=4.0`（需加入 requirements）

---

## 六、Step 9：论文图表、表格与结论

### 目标

从已有实验结果中提取图表数据，撰写论文章节。

### 需要生成的图表

1. **pipeline 架构图**：使用 draw.io / matplotlib / TikZ 绘制完整方法流程图
2. **热力图对比**：正常区域 vs 缺陷区域的差异热力图可视化
3. **候选框效果**：候选框叠加在 query 图上的实例（含 TP/FP 标注）
4. **消融实验柱状图**：recall/mAP 在各消融条件下的对比
5. **检测结果样例**：TP 检测（成功）和 FP 检测（误检）的案例对比

### 需要生成的表格

1. 数据集统计表（图像数、GT boxes 数、类别分布）
2. 候选框 recall@IoU 表（各阈值 × 各方法）
3. 消融实验定量表（各模块的量化贡献）
4. 与 baseline 方法的对比表（RT-DETR only vs proposed）
5. 推理速度/效率对比表

### 论文章节对应

| 论文章节 | 对应 Step | 关键数据来源 |
|---------|----------|------------|
| 3.2 宏观配准 | Step 3.2 | alignment_report.json |
| 3.3 Siamese 多尺度特征 | Step 3.3 | backbone smoke report |
| 3.4 DCN 微对齐 | Step 3.4 | training report + offset statistics |
| 3.5 注意力差异筛选 | Step 3.5 | proposals smoke + heatmap PNGs |
| 4 上下文裁剪 | Step 4 | crop statistics |
| 5 精细分类 | Step 5 | RT-DETR mAP + per-class results |
| 6 实验与分析 | Step 7 | 消融实验数据 |

---

## 执行顺序建议

```
3.4-C 训练 (OffsetPredictor)
  → 验证候选框 recall 改善
    → 可选：重新运行 3.5-C/4-A smoke 收集新数据
      → 5-B RT-DETR 训练
        → 6 端到端推理 + 指标收集
          → 7 消融实验 (跑完前两部即可撰写大部分)
            → 8 演示软件 (可并行)
              → 9 论文撰写 (数据有了即可开始)
```

## 关键依赖链

```
3.4-C 训练 → 候选框质量 → 裁剪质量 → 端到端 recall
                                  ↓
5-B 训练 (独立，不依赖 3.4-C) → RT-DETR mAP → 端到端检测
```

**3.4-C 和 5-B 可以部分并行**（RT-DETR 训练不依赖 OffsetPredictor 权重）。
