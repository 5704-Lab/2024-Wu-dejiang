"""Training runner for Step 3.4-C micro-alignment (OffsetPredictor training)."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from step1_audit.hashing import sha256_file, snapshot_digest, snapshot_tree
from step1_audit.paths import resolved, validate_roots, write_json
from step2_dataset.collate import build_dataloader
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config

from .backbone import SharedSiameseBackbone
from .feature_warp import FeatureWarp
from .offset_predictor import OffsetPredictor
from .train_alignment import (
    MicroAlignmentLoss,
    _build_pipeline,
    train_one_epoch,
    validate_one_epoch,
)

_EVIDENCE_KEYS = (
    "alignment_validation_report", "alignment_frozen_parameters",
    "alignment_validation_manifest", "alignment_test_report", "alignment_test_manifest",
)


def load_train_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workspace = path.resolve().parent.parent
    config["_config_path"] = str(path.resolve())
    config["_workspace"] = str(workspace)
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    config["_dataset_root"] = str(resolved(config["inputs"]["dataset_root"]))
    config["_step2_config"] = str(resolved(config["inputs"]["step2_config"], workspace))
    for key in _EVIDENCE_KEYS:
        config[f"_{key}"] = str(resolved(config["inputs"][key], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("roadmap_step") != "3.4-C" or config.get("schema_version") != "step3-training-config-v1":
        raise ValueError("Invalid Step 3.4-C config schema")
    training = config.get("training", {})
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive")
    if float(training.get("lr", 0)) <= 0:
        raise ValueError("lr must be positive")
    if training.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported device")
    loss_cfg = config.get("loss", {}).get("per_scale_weights", {})
    for key in ("C3", "C4", "C5"):
        if not isinstance(loss_cfg.get(key), (int, float)):
            raise ValueError(f"Missing per_scale_weight for {key}")
    model = config.get("model", {})
    if int(model.get("kernel_size", 0)) != 3:
        raise ValueError("kernel_size must be 3")
    if model.get("zero_init") is not True:
        raise ValueError("zero_init must be true for 3.4-C")
    if training.get("scheduler") not in {"cosine", "none"}:
        raise ValueError("Unsupported scheduler")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _evidence_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(Path(config[f"_{key}"])) for key in _EVIDENCE_KEYS}


def run_train_alignment(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    dataset_root, output_root = Path(config["_dataset_root"]), Path(config["_output_root"])
    validate_roots(dataset_root, output_root)
    sample_dir = Path(config["_alignment_validation_report"]).parent.resolve()
    if output_root.resolve() == sample_dir:
        raise ValueError("Training output must not overwrite Step 3.2 evidence")
    output_root.mkdir(parents=True, exist_ok=True)
    before, evidence_before = snapshot_tree(dataset_root), _evidence_hashes(config)
    source_digest = snapshot_digest(before)

    step2 = load_step2_config(Path(config["_step2_config"]))
    index = build_dataset_index(step2)

    training = config["training"]
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    lr = float(training["lr"])
    weight_decay = float(training.get("weight_decay", 0.0))
    val_interval = int(training.get("validation_interval", 1))

    train_dataset = PairedQueryTemplateDataset(index, "train", step2, training=True, epoch=0)
    val_dataset = PairedQueryTemplateDataset(index, "val", step2, training=False)
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = _resolve_device(training["device"])
    backbone, predictor, warp = _build_pipeline(
        device,
        kernel_size=int(config["model"]["kernel_size"]),
        hidden_ratio=int(config["model"]["hidden_ratio"]),
        zero_init=bool(config["model"]["zero_init"]),
        max_offset=float(config["model"].get("max_offset", 3.0)),
    )

    loss_weights = config["loss"]["per_scale_weights"]
    criterion = MicroAlignmentLoss(
        alpha_c3=float(loss_weights["C3"]),
        alpha_c4=float(loss_weights["C4"]),
        alpha_c5=float(loss_weights["C5"]),
    )
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if training.get("scheduler") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    epochs_log = []
    best_val_loss = float("inf")
    best_epoch = -1
    checkpoint_best_path = output_root / "checkpoint_best.pt"
    checkpoint_final_path = output_root / "checkpoint_final.pt"

    print(f"\n{'='*60}")
    print(f"  Step 3.4-C: OffsetPredictor 训练开始")
    print(f"  Train: {len(train_dataset)} pairs | Val: {len(val_dataset)} pairs")
    print(f"  Batch: {batch_size} | Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"  Trainable: OffsetPredictor ({sum(p.numel() for p in predictor.parameters()):,} params)")
    print(f"  Output: {output_root}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch - 1)
        train_stats = train_one_epoch(backbone, predictor, warp, train_loader, optimizer, criterion,
                                      device, epoch=epoch, total_epochs=epochs)
        val_stats: dict[str, float] = {}
        if epoch % val_interval == 0 or epoch == epochs:
            val_stats = validate_one_epoch(backbone, predictor, warp, val_loader, criterion,
                                           device, epoch=epoch, total_epochs=epochs)
        entry = {"epoch": epoch, **train_stats}
        if val_stats:
            entry.update(val_stats)
            val_loss = val_stats.get("val_loss", float("inf"))
            improved = " *" if val_loss < best_val_loss else ""
            print(f"  ── Epoch {epoch:3d}/{epochs}  train_loss={train_stats['train_loss']:.6f}  "
                  f"val_loss={val_loss:.6f}{improved}  time={time.time() - epoch_start:.0f}s\n", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state_dict": predictor.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss}, checkpoint_best_path)
        else:
            print(f"  ── Epoch {epoch:3d}/{epochs}  train_loss={train_stats['train_loss']:.6f}  "
                  f"time={time.time() - epoch_start:.0f}s\n", flush=True)
        if scheduler is not None:
            entry["lr"] = float(scheduler.get_last_lr()[0])
            scheduler.step()
        epochs_log.append(entry)

    torch.save({"epoch": epochs, "model_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, checkpoint_final_path)

    # Check dataset integrity
    if before != snapshot_tree(dataset_root):
        raise RuntimeError("Read-only dataset changed during training")
    if evidence_before != _evidence_hashes(config):
        raise RuntimeError("Step 3.2 evidence changed during training")

    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    report = {
        "schema_version": "step3-training-report-v1", "roadmap_step": "3.4-C",
        "processed": True, "epochs_completed": epochs, "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss) if best_epoch > 0 else None,
        "selected_device": str(device),
        "trainable_parameter_count": trainable_params,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "scheduler": training.get("scheduler"),
        },
        "epochs_log": epochs_log,
    }
    json.dumps(report, allow_nan=False)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "training_report.json"
    write_json(report_path, report)

    # Verify checkpoint hashes
    artifacts_hashes = {"training_report.json": sha256_file(report_path)}
    if checkpoint_best_path.is_file():
        artifacts_hashes["checkpoint_best.pt"] = sha256_file(checkpoint_best_path)
    if checkpoint_final_path.is_file():
        artifacts_hashes["checkpoint_final.pt"] = sha256_file(checkpoint_final_path)

    manifest = {
        "schema_version": "step3-training-run-manifest-v1", "roadmap_step": "3.4-C",
        "command": "train-alignment", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
                     "selected_device": str(device),
                     "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "inputs": {"config_sha256": sha256_file(Path(config["_config_path"])),
                   "step2_config_sha256": sha256_file(Path(config["_step2_config"])),
                   "source_snapshot_digest": source_digest, "step3_2_evidence_sha256": evidence_before},
        "artifacts": artifacts_hashes,
        "controls": {"source_read_only_verified": True, "alignment_evidence_integrity_verified": True,
                     "train_started": True, "training_completed": True,
                     "offset_predictor_trained": True, "step_3_4_c_started": True},
    }
    write_json(output_root / "run_manifest.json", manifest)
    return report
