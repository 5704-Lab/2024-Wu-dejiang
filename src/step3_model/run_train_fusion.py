"""Train MultiScaleFusion from annotated defect heatmaps."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from step1_audit.hashing import sha256_file
from step1_audit.paths import resolved, write_json
from step2_dataset.collate import build_dataloader
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config

from .backbone import SharedSiameseBackbone
from .feature_warp import FeatureWarp
from .fusion import MultiScaleFusion
from .model_io import load_model_checkpoint
from .offset_predictor import OFFSET_SPEC, OffsetPredictor
from .train_fusion import FusionHeatmapLoss, build_defect_heatmaps, build_defect_heatmaps_with_weights


def load_fusion_train_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workspace = path.resolve().parent.parent
    config["_config_path"] = str(path.resolve())
    config["_step2_config"] = str(resolved(config["inputs"]["step2_config"], workspace))
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_predictor_checkpoint"] = str(resolved(config["inputs"]["predictor_checkpoint"], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "step3-fusion-training-config-v1":
        raise ValueError("Invalid fusion training config schema")
    training = config.get("training", {})
    if int(training.get("epochs", 0)) <= 0 or int(training.get("batch_size", 0)) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if float(training.get("lr", 0)) <= 0:
        raise ValueError("lr must be positive")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def run_train_fusion(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)
    train_data = PairedQueryTemplateDataset(index, "train", step2, training=True)
    val_data = PairedQueryTemplateDataset(index, "val", step2, training=False)
    training = config["training"]
    train_loader = build_dataloader(train_data, batch_size=int(training["batch_size"]),
                                    num_workers=int(training["num_workers"]), shuffle=True)
    val_loader = build_dataloader(val_data, batch_size=int(training["batch_size"]),
                                  num_workers=int(training["num_workers"]), shuffle=False)
    device = _device(training["device"])
    backbone = SharedSiameseBackbone("IMAGENET1K_V2").to(device).eval()
    predictor = OffsetPredictor(max_offset=float(config["model"].get("max_offset", 3.0))).to(device).eval()
    load_model_checkpoint(predictor, Path(config["_predictor_checkpoint"]), device, "OffsetPredictor",
                          "defect-masked-alignment-v1")
    warp = FeatureWarp().to(device).eval()
    fusion = MultiScaleFusion(hidden=int(config["fusion"]["hidden"]), reduction=int(config["fusion"]["reduction"])).to(device)
    for module in (backbone, predictor, warp):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    loss_cfg = config["loss"]
    base_positive_weight = float(loss_cfg.get("positive_weight", 10.0))
    miss_positive_weight = float(loss_cfg.get("miss_positive_weight", base_positive_weight))
    class_positive_weights = {
        0: base_positive_weight,
        1: base_positive_weight,
        2: base_positive_weight,
        3: miss_positive_weight,
    }
    balanced_class_weights = all(
        weight == base_positive_weight for weight in class_positive_weights.values()
    )
    criterion = FusionHeatmapLoss(base_positive_weight, float(loss_cfg["dice_weight"]))
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=float(training["lr"]),
                                  weight_decay=float(training.get("weight_decay", 0.0)))
    output = Path(config["_output_root"]); output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf"); best_epoch = -1; log = []
    epochs = int(training["epochs"])
    trainable_params = sum(parameter.numel() for parameter in fusion.parameters() if parameter.requires_grad)
    # 在耗时较长的双分支特征提取开始前输出完整运行上下文，便于确认设备、
    # 数据规模、权重来源和产物目录均符合预期，避免训练数小时后才发现配置错误。
    print(f"\n{'=' * 60}")
    print("  Step 3.5-B: MultiScaleFusion 监督训练开始")
    print(f"  Train: {len(train_data)} pairs | Val: {len(val_data)} pairs")
    print(f"  Batch: {training['batch_size']} | Epochs: {epochs} | LR: {training['lr']} | Device: {device}")
    print(f"  Trainable: MultiScaleFusion ({trainable_params:,} params)")
    print(f"  Predictor: {config['_predictor_checkpoint']}")
    print(f"  Output: {output}")
    print(f"{'=' * 60}\n", flush=True)

    training_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_data.set_epoch(epoch - 1)
        epoch_stats = {}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            fusion.train(phase == "train")
            totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
            count = 0
            phase_start = time.time()
            total_batches = len(loader)
            # 每个阶段约输出十次进度；小数据集至少每个 batch 输出一次。
            log_every = max(1, total_batches // 10)
            for batch_index, batch in enumerate(loader, 1):
                query, template = batch["query"].to(device), batch["template"].to(device)
                with torch.no_grad():
                    qf, tf = backbone(query, template)
                    warped = warp(tf, predictor(qf, tf))
                    differences = {name: (warped[name] - qf[name]).abs() for name, _, _ in OFFSET_SPEC}
                if balanced_class_weights:
                    targets = build_defect_heatmaps(batch["query_targets"], device=device)
                    pixel_weights = None
                else:
                    targets, pixel_weights = build_defect_heatmaps_with_weights(
                        batch["query_targets"],
                        class_positive_weights=class_positive_weights,
                        device=device,
                    )
                with torch.set_grad_enabled(phase == "train"):
                    logits = fusion(differences)
                    loss, loss_stats = criterion(logits, targets, pixel_weights=pixel_weights)
                    if phase == "train":
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                totals["loss"] += loss_stats["loss_total"]
                totals["bce"] += loss_stats["loss_bce"]
                totals["dice"] += loss_stats["loss_dice"]
                count += 1
                if batch_index % log_every == 0 or batch_index == total_batches:
                    # ETA 使用本阶段已经完成 batch 的平均耗时估算，仅用于观察训练状态，
                    # 不参与训练调度或 checkpoint 选择。
                    elapsed = time.time() - phase_start
                    eta = elapsed / batch_index * (total_batches - batch_index)
                    label = "Train" if phase == "train" else "Val  "
                    print(
                        f"  {label} [{epoch:3d}/{epochs:3d}] batch {batch_index:4d}/{total_batches:4d} | "
                        f"loss={totals['loss'] / count:.6f} | "
                        f"bce={totals['bce'] / count:.6f} | dice={totals['dice'] / count:.6f} | "
                        f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s",
                        flush=True,
                    )
            denominator = max(count, 1)
            epoch_stats[f"{phase}_loss"] = totals["loss"] / denominator
            epoch_stats[f"{phase}_loss_bce"] = totals["bce"] / denominator
            epoch_stats[f"{phase}_loss_dice"] = totals["dice"] / denominator
        log.append({"epoch": epoch, **epoch_stats})
        # 先记录是否刷新最佳结果，再更新 best_loss，确保汇总行中的星号语义准确。
        improved = epoch_stats["val_loss"] < best_loss
        if epoch_stats["val_loss"] < best_loss:
            best_loss, best_epoch = epoch_stats["val_loss"], epoch
            torch.save({"epoch": epoch, "model_state_dict": fusion.state_dict(), "val_loss": best_loss,
                        "training_contract": "supervised-fusion-heatmap-v1"}, output / "checkpoint_best.pt")
        print(
            f"  ── Epoch {epoch:3d}/{epochs} | train={epoch_stats['train_loss']:.6f} | "
            f"val={epoch_stats['val_loss']:.6f}{' *' if improved else ''} | "
            f"time={time.time() - epoch_start:.0f}s | best_epoch={best_epoch}\n",
            flush=True,
        )
    torch.save({"epoch": epochs, "model_state_dict": fusion.state_dict(),
                "training_contract": "supervised-fusion-heatmap-v1"}, output / "checkpoint_final.pt")
    report = {"schema_version": "step3-fusion-training-report-v1", "processed": True,
              "best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_log": log,
              "epochs_completed": epochs, "selected_device": str(device),
              "trainable_parameter_count": trainable_params,
              "total_training_seconds": time.time() - training_start,
              "predictor_checkpoint_sha256": sha256_file(Path(config["_predictor_checkpoint"]))}
    write_json(output / "training_report.json", report)
    write_json(output / "run_manifest.json", {"config_sha256": sha256_file(Path(config["_config_path"])),
              "predictor_checkpoint_sha256": report["predictor_checkpoint_sha256"],
              "fusion_checkpoint_sha256": sha256_file(output / "checkpoint_best.pt")})
    json.dumps(report, allow_nan=False)
    print(
        f"\nFusion 训练完成：best_epoch={best_epoch}, best_val_loss={best_loss:.6f}, "
        f"total_time={report['total_training_seconds'] / 60:.1f} min",
        flush=True,
    )
    return report
