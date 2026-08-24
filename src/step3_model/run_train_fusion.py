"""Train MultiScaleFusion from annotated defect heatmaps."""

from __future__ import annotations

import json
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
from .train_fusion import FusionHeatmapLoss, build_defect_heatmaps


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
    criterion = FusionHeatmapLoss(float(config["loss"]["positive_weight"]), float(config["loss"]["dice_weight"]))
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=float(training["lr"]),
                                  weight_decay=float(training.get("weight_decay", 0.0)))
    output = Path(config["_output_root"]); output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf"); best_epoch = -1; log = []
    for epoch in range(1, int(training["epochs"]) + 1):
        train_data.set_epoch(epoch - 1)
        epoch_stats = {}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            fusion.train(phase == "train"); total = 0.0; count = 0
            for batch in loader:
                query, template = batch["query"].to(device), batch["template"].to(device)
                with torch.no_grad():
                    qf, tf = backbone(query, template)
                    warped = warp(tf, predictor(qf, tf))
                    differences = {name: (warped[name] - qf[name]).abs() for name, _, _ in OFFSET_SPEC}
                targets = build_defect_heatmaps(batch["query_targets"], device=device)
                with torch.set_grad_enabled(phase == "train"):
                    logits = fusion(differences)
                    loss, _ = criterion(logits, targets)
                    if phase == "train":
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.item()); count += 1
            epoch_stats[f"{phase}_loss"] = total / max(count, 1)
        log.append({"epoch": epoch, **epoch_stats})
        if epoch_stats["val_loss"] < best_loss:
            best_loss, best_epoch = epoch_stats["val_loss"], epoch
            torch.save({"epoch": epoch, "model_state_dict": fusion.state_dict(), "val_loss": best_loss,
                        "training_contract": "supervised-fusion-heatmap-v1"}, output / "checkpoint_best.pt")
    torch.save({"epoch": int(training["epochs"]), "model_state_dict": fusion.state_dict(),
                "training_contract": "supervised-fusion-heatmap-v1"}, output / "checkpoint_final.pt")
    report = {"schema_version": "step3-fusion-training-report-v1", "processed": True,
              "best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_log": log,
              "predictor_checkpoint_sha256": sha256_file(Path(config["_predictor_checkpoint"]))}
    write_json(output / "training_report.json", report)
    write_json(output / "run_manifest.json", {"config_sha256": sha256_file(Path(config["_config_path"])),
              "predictor_checkpoint_sha256": report["predictor_checkpoint_sha256"],
              "fusion_checkpoint_sha256": sha256_file(output / "checkpoint_best.pt")})
    json.dumps(report, allow_nan=False)
    return report
