# Step 3.4-C 训练待办（Training TODO）

## 状态

代码已就绪（loss、training loop、CLI、配置、测试），但**实际训练尚未执行**。

## 训练命令

```bash
python scripts/run_step3.py train-alignment --config configs/step3_train_alignment.yaml
```

## 配置摘要

| 参数 | 值 |
|------|-----|
| 数据集 | train split（3200 对，320 组） |
| Batch size | 8 |
| Epochs | 20 |
| 优化器 | Adam（lr=0.001, weight_decay=0） |
| 调度器 | CosineAnnealingLR |
| 损失 | 逐尺度加权 MSE（C3/C4/C5 等权 α=1.0） |
| Trainable | 仅 OffsetPredictor（1,410,678 参数） |
| Frozen | Backbone（23,508,032）+ FeatureWarp（0） |
| 增强 | 几何翻转/旋转 + 独立双路光度增强 |
| 输出目录 | artifacts/step3/training/ |

## 产物（训练完成后生成）

- `artifacts/step3/training/training_report.json`
- `artifacts/step3/training/run_manifest.json`
- `artifacts/step3/training/checkpoint_best.pt`
- `artifacts/step3/training/checkpoint_final.pt`

## 已知问题

1. **训练脚本无实时日志**：当前只在最后输出 JSON 摘要，长时间运行时看不到进度。建议后续为 `train_one_epoch` 和 `validate_one_epoch` 添加 tqdm 进度条或每 N batch 打印 loss。
2. **预估时间**：3200 对 × 20 epochs，batch_size=8，CUDA 设备，每 epoch 约需 3–5 分钟（取决于 backbone 推理速度），总计约 60–100 分钟。
3. **checkpoint 体积**：OffsetPredictor 只有 1,410,678 参数，state_dict 约 5.6 MB + optimizer state ~11.2 MB，总计约 17 MB（在 gitignore 中已排除 `.pt` 文件，不会被提交）。

## 后续步骤

完成 Step 3.4-C 的代码实现（已完成），训练应在最后统一进行或按需触发。

---

最后更新：2026-08-12
