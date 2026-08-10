# Step 3.1：SIFT + RANSAC 宏观几何配准

Step 3.1 读取 Step 1 生成的样本索引和无泄漏 split，对 Query–Template 图像执行 SIFT 特征匹配、Lowe ratio 筛选和 RANSAC Homography 估计，并把 Query 粗配准到 Template 坐标系。原始 VOC 数据集严格只读。

当前阶段只实现宏观配准验证，不包含后续计划中的 Siamese、DCN、注意力候选区域或 RT-DETR。

## 环境

Step 3.1 复用 Step 2 的 Python 依赖：

```bash
python -m pip install -r requirements-step2.txt
```

其中包含 NumPy、OpenCV、Pillow、PyYAML 和 pytest。

## 运行

先完成 Step 1 并确保 `artifacts/step1/samples.jsonl` 和 `artifacts/step1/splits/` 存在，然后运行：

```bash
python scripts/run_step3.py align --config configs/step3_alignment.yaml
```

配置中的 `validation.sample_count` 表示本次验证的总样本数。程序按 train/val/test 规模确定性分配样本，并使用 `validation.seed` 固定抽样。

## 输出

- `artifacts/step3/alignment_report.json`：汇总指标和逐样本结果，纳入 Git。
- `artifacts/step3/alignment/<split>/*.png`：对比图、差异热图和 RANSAC 内点匹配图，可重复生成，不纳入 Git。

报告中的 `success` 表示 Homography 成功求得并满足最小内点数量，不代表 SSIM 或互信息一定改善。应分别检查：

- 关键点、ratio matches 和 RANSAC 内点数量；
- RANSAC 内点重投影误差；
- 配准前后的 SSIM 与互信息变化；
- 逐样本 Homography 和失败原因。

当前 10 样本探针能够完成几何估计，但既有结果的平均 SSIM/MI 略有下降。因此 Step 3.1 当前结论是“宏观配准链路可运行”，不是“配准质量已达到最终验收标准”。下一步需要使用有效重叠区域 mask 和更严格的 Homography 合法性/质量阈值继续评估。

## 只读保证

运行前会验证输出目录不等于、也不位于原始数据集目录内。输入路径还必须解析在数据集根目录下；程序只向 `artifacts/step3/` 写入报告和可视化文件。

## 测试

```bash
python -m pytest tests/step3 -q
python -m pytest tests/step1 tests/step2 tests/step3 -q
```
