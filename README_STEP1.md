# Step 1：原始数据只读审计与外部索引

本步骤把规范数据集作为严格只读数据源，所有索引、报告和新划分只写入本工程的 `artifacts/step1/`。

## 数据源

`../../project/数据集/VOCdevkit/VOC2007`（相对仓库根目录）

程序不会修改、移动、重命名、删除或复制其中的图片、模板、XML和原始划分。

## 命令

```bash
python scripts/run_step1.py all --config configs/step1_dataset.yaml --dry-run
python scripts/run_step1.py audit --config configs/step1_dataset.yaml
python scripts/run_step1.py split --config configs/step1_dataset.yaml
python scripts/run_step1.py verify --config configs/step1_dataset.yaml
```

也可一次执行：

```bash
python scripts/run_step1.py all --config configs/step1_dataset.yaml
```

## 主要产物

- `samples.jsonl`：样本级统一索引，以XML文件stem作为sample_id。
- `objects.csv`：目标级扁平索引。
- `groups.jsonl`：400个source group及其10个变体。
- `issues.csv`：警告和致命问题；XML filename错误仅记录，不修改。
- `audit_report.json`：数据统计和原始划分泄漏报告。
- `duplicate_report.json`：Query和Template内容重复统计。
- `splits/*.txt`：工程外部的80/10/10无泄漏划分。
- `source_snapshot_before/after.json`：验证规范数据集前后完全一致。

后续模型必须通过 `samples.jsonl` 和工程外部split清单读取数据，不依赖XML内部filename字段。
