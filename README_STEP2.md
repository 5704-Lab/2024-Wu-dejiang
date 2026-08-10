# Step 2：Query–Template配对加载、增强与可视化

Step 2只读取Step 1生成的外部索引和无泄漏split。原始VOC数据集继续保持严格只读。

## 环境

当前实现面向Python 3.12和现代PyTorch/torchvision配对。根据CUDA环境从官方PyTorch索引安装，例如：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-step2.txt
```

## 运行

```bash
python scripts/run_step2.py validate --config configs/step2_dataset.yaml
python scripts/run_step2.py probe --config configs/step2_dataset.yaml
python scripts/run_step2.py visualize --config configs/step2_dataset.yaml
python scripts/run_step2.py all --config configs/step2_dataset.yaml
```

默认验收配置只抽样验证40条记录，并对40条训练样本执行确定性与增强探针；完整的3200/400/400 split成员关系仍会全部检查，但不会解码全部4000对图片。

仅生成验证集联系表：

```bash
python scripts/run_step2.py visualize --config configs/step2_dataset.yaml --split val --limit 6
```

## 数据约定

每条数据返回Query Tensor、Template Tensor、两份独立检测target和metadata。两个分支共享完全相同的几何增强，分别使用独立光度增强。target采用XYXY、int64 labels和可变长度list collate，便于后续接入RT-DETR。

所有报告和PNG都写入 `artifacts/step2/`，不会复制或改写原始图片和XML。
