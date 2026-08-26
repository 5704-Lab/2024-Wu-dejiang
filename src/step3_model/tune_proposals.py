"""在完整 Validation 热力图缓存上搜索 Proposal 与 Crop 后处理参数。

本模块严格只读取 Validation split。神经网络前向与 CPU 参数搜索分离，避免每组
后处理参数都重复运行 Backbone、OffsetPredictor 和 MultiScaleFusion。
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from step1_audit.hashing import sha256_file
from step1_audit.paths import resolved, write_json
from step2_dataset.collate import pair_collate_fn
from step2_dataset.dataset import PairedQueryTemplateDataset
from step2_dataset.index import build_dataset_index, load_config as load_step2_config

from .backbone import SharedSiameseBackbone
from .crop import expand_box, merge_overlapping_boxes
from .feature_warp import FeatureWarp
from .fusion import MultiScaleFusion
from .model_io import load_model_checkpoint
from .offset_predictor import OFFSET_SPEC, OffsetPredictor
from .proposals import postprocess_probability_heatmap


def load_tuning_config(path: Path) -> dict[str, Any]:
    """读取调参配置并解析所有相对工作区路径。"""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workspace = path.resolve().parent.parent
    config["_config_path"] = str(path.resolve())
    config["_step2_config"] = str(resolved(config["inputs"]["step2_config"], workspace))
    config["_predictor_checkpoint"] = str(resolved(config["inputs"]["predictor_checkpoint"], workspace))
    config["_fusion_checkpoint"] = str(resolved(config["inputs"]["fusion_checkpoint"], workspace))
    config["_output_root"] = str(resolved(config["output"]["root"], workspace))
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """拒绝任何非完整 Validation 的调参配置，防止 Test 泄漏。"""
    if config.get("schema_version") != "step3-proposal-tuning-config-v1":
        raise ValueError("无效的 Proposal 调参配置版本")
    run = config.get("run", {})
    if run.get("split") != "val" or run.get("selection") != "full_split":
        raise ValueError("Proposal 调参只允许使用完整 Validation split")
    if int(run.get("expected_sample_count", 0)) != 400:
        raise ValueError("完整 Validation 必须包含 400 个样本")


def _device(name: str) -> torch.device:
    """解析运行设备；显式请求 CUDA 但不可用时立即失败。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 被显式请求但当前不可用")
    return torch.device(name)


def cache_validation_heatmaps(config: dict[str, Any]) -> dict[str, Any]:
    """对完整 Validation 前向一次，并以 uint8 缓存缺陷概率热力图。"""
    _validate_config(config)
    output_root = Path(config["_output_root"])
    cache_root = output_root / "val_heatmaps"
    cache_root.mkdir(parents=True, exist_ok=True)
    step2 = load_step2_config(Path(config["_step2_config"]))
    dataset = PairedQueryTemplateDataset(build_dataset_index(step2), "val", step2, training=False)
    expected = int(config["run"]["expected_sample_count"])
    if len(dataset) != expected:
        raise ValueError(f"Validation 样本数不符合预期：{len(dataset)} != {expected}")
    loader = DataLoader(
        dataset,
        batch_size=int(config["run"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config["run"].get("num_workers", 0)),
        collate_fn=pair_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    device = _device(config["run"].get("device", "auto"))
    model_cfg, fusion_cfg = config["model"], config["fusion"]
    backbone = SharedSiameseBackbone("IMAGENET1K_V2").to(device).eval()
    predictor = OffsetPredictor(
        kernel_size=int(model_cfg["kernel_size"]), hidden_ratio=int(model_cfg["hidden_ratio"]),
        zero_init=True, max_offset=float(model_cfg.get("max_offset", 3.0)),
    ).to(device).eval()
    warp = FeatureWarp(kernel_size=int(model_cfg["kernel_size"])).to(device).eval()
    fusion = MultiScaleFusion(
        hidden=int(fusion_cfg["hidden"]), reduction=int(fusion_cfg["reduction"]),
    ).to(device).eval()
    load_model_checkpoint(predictor, Path(config["_predictor_checkpoint"]), device, "OffsetPredictor",
                          "defect-masked-alignment-v1")
    load_model_checkpoint(fusion, Path(config["_fusion_checkpoint"]), device, "MultiScaleFusion",
                          "supervised-fusion-heatmap-v1")
    for module in (backbone, predictor, warp, fusion):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    metadata: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            query, template = batch["query"].to(device), batch["template"].to(device)
            qf, tf = backbone(query, template)
            warped = warp(tf, predictor(qf, tf))
            differences = {name: (warped[name] - qf[name]).abs() for name, _, _ in OFFSET_SPEC}
            probabilities = torch.sigmoid(fusion(differences)).cpu().numpy()
            for item_index, meta in enumerate(batch["meta"]):
                sample_id = meta["sample_id"]
                # uint8 将缓存缩小到 float32 的四分之一；搜索时除以 255 恢复概率。
                heatmap = np.clip(np.rint(probabilities[item_index, 0] * 255.0), 0, 255).astype(np.uint8)
                np.save(cache_root / f"{sample_id}.npy", heatmap, allow_pickle=False)
                target = batch["query_targets"][item_index]
                metadata.append({
                    "sample_id": sample_id,
                    "source_id": meta["source_id"],
                    "heatmap_file": f"val_heatmaps/{sample_id}.npy",
                    "boxes": target["boxes"].tolist(),
                    "labels": target["labels"].tolist(),
                })
            if batch_index % max(1, len(loader) // 10) == 0 or batch_index == len(loader):
                elapsed = time.time() - started
                eta = elapsed / batch_index * (len(loader) - batch_index)
                print(f"缓存 Validation 热力图 {batch_index}/{len(loader)} | elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)

    metadata_path = output_root / "val_cache_manifest.json"
    manifest = {
        "schema_version": "step3-val-heatmap-cache-v1",
        "split": "val", "sample_count": len(metadata), "samples": metadata,
        "predictor_checkpoint_sha256": sha256_file(Path(config["_predictor_checkpoint"])),
        "fusion_checkpoint_sha256": sha256_file(Path(config["_fusion_checkpoint"])),
        "config_sha256": sha256_file(Path(config["_config_path"])),
    }
    write_json(metadata_path, manifest)
    return {"sample_count": len(metadata), "cache_manifest": str(metadata_path), "device": str(device)}


def _coverage_statistics(predictions: list[torch.Tensor], targets: list[torch.Tensor],
                         expand_k: float, merge_iou: float) -> dict[str, float | int]:
    """仅用候选框和 GT 坐标计算裁剪覆盖率，无需重复读取原图。"""
    total_crops = covered = crops_with_gt = total_gt = 0
    areas: list[int] = []
    for candidates, gt_boxes in zip(predictions, targets):
        merged = merge_overlapping_boxes(candidates, merge_iou)
        total_crops += int(merged.shape[0]); total_gt += int(gt_boxes.shape[0])
        matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        for box in merged:
            x, y, width, height = expand_box(box, expand_k, (800, 800))
            areas.append(width * height)
            inside = ((gt_boxes[:, 0] >= x) & (gt_boxes[:, 1] >= y) &
                      (gt_boxes[:, 2] <= x + width) & (gt_boxes[:, 3] <= y + height))
            if inside.any():
                crops_with_gt += 1
            matched |= inside
        covered += int(matched.sum().item())
    return {
        "gt_coverage": covered / max(total_gt, 1), "covered_gt_boxes": covered,
        "total_gt_boxes": total_gt, "total_crops": total_crops,
        "crops_per_image": total_crops / max(len(predictions), 1),
        "crops_with_gt": crops_with_gt, "empty_crops": total_crops - crops_with_gt,
        "crop_area_mean": sum(areas) / max(len(areas), 1),
    }


def search_cached_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """在缓存概率图上完成 Proposal 粗搜索、候选筛选和 Crop 参数搜索。"""
    output_root = Path(config["_output_root"])
    cache_manifest = json.loads((output_root / "val_cache_manifest.json").read_text(encoding="utf-8"))
    if cache_manifest["sample_count"] != int(config["run"]["expected_sample_count"]):
        raise ValueError("热力图缓存不完整")
    heatmaps = [
        np.load(output_root / sample["heatmap_file"], allow_pickle=False).astype(np.float32) / 255.0
        for sample in cache_manifest["samples"]
    ]
    targets = [torch.tensor(sample["boxes"], dtype=torch.float32).reshape(-1, 4) for sample in cache_manifest["samples"]]
    search = config["search"]
    proposal_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[float, int, float, float], list[torch.Tensor]] = {}
    combinations = list(itertools.product(
        search["probability_thresholds"], search["morph_kernels"],
        search["min_areas"], search["max_aspect_ratios"],
    ))
    print(f"开始搜索 {len(combinations)} 组 Proposal 参数", flush=True)
    for index, (threshold, kernel, min_area, aspect) in enumerate(combinations, 1):
        key = (float(threshold), int(kernel), float(min_area), float(aspect))
        predictions = [
            torch.from_numpy(postprocess_probability_heatmap(
                heatmap, probability_threshold=key[0], morph_kernel=key[1], min_area=key[2],
                max_aspect_ratio=key[3], max_area_ratio=float(search.get("max_area_ratio", 0.8)),
            ))
            for heatmap in heatmaps
        ]
        prediction_cache[key] = predictions
        total_gt = sum(target.shape[0] for target in targets)
        total_candidates = sum(prediction.shape[0] for prediction in predictions)
        matched = {0.3: 0, 0.5: 0, 0.7: 0}
        for prediction, target in zip(predictions, targets):
            if prediction.numel() == 0 or target.numel() == 0:
                continue
            best = box_iou(prediction, target).max(dim=0).values
            for threshold_iou in matched:
                matched[threshold_iou] += int((best >= threshold_iou).sum().item())
        row = {
            "probability_threshold": key[0], "morph_kernel": key[1],
            "min_area": key[2], "max_aspect_ratio": key[3],
            "recall@0.3": matched[0.3] / max(total_gt, 1),
            "recall@0.5": matched[0.5] / max(total_gt, 1),
            "recall@0.7": matched[0.7] / max(total_gt, 1),
            "total_candidate_boxes": total_candidates,
            "candidates_per_image": total_candidates / len(targets),
        }
        # 粗搜索优先保留高召回，同时对极端候选数量施加轻度惩罚。
        row["proposal_score"] = row["recall@0.3"] + row["recall@0.5"] - 0.001 * row["candidates_per_image"]
        proposal_rows.append(row)
        if index % max(1, len(combinations) // 10) == 0 or index == len(combinations):
            print(f"Proposal 搜索 {index}/{len(combinations)}", flush=True)

    proposal_rows.sort(key=lambda row: row["proposal_score"], reverse=True)
    top_count = min(int(search.get("top_proposal_configs", 10)), len(proposal_rows))
    crop_rows: list[dict[str, Any]] = []
    for proposal in proposal_rows[:top_count]:
        key = (proposal["probability_threshold"], proposal["morph_kernel"],
               proposal["min_area"], proposal["max_aspect_ratio"])
        for expand_k, merge_iou in itertools.product(search["expand_factors"], search["merge_iou_thresholds"]):
            coverage = _coverage_statistics(prediction_cache[key], targets, float(expand_k), float(merge_iou))
            row = {**proposal, **coverage, "expand_k": float(expand_k), "merge_iou_threshold": float(merge_iou)}
            # Coverage 是进入检测器前的硬上限，因此权重最高；同时兼顾 IoU 召回和计算负担。
            row["score"] = (2.0 * row["gt_coverage"] + row["recall@0.5"] +
                            0.5 * row["recall@0.3"] - 0.002 * row["crops_per_image"])
            crop_rows.append(row)
    crop_rows.sort(key=lambda row: row["score"], reverse=True)
    best = crop_rows[0]
    report = {
        "schema_version": "step3-proposal-tuning-report-v1",
        "scope": {"split": "val", "selection": "full_split", "sample_count": len(targets)},
        "best": best, "proposal_ranking": proposal_rows[:20], "combined_ranking": crop_rows[:50],
        "checkpoint_sha256": {
            "predictor": cache_manifest["predictor_checkpoint_sha256"],
            "fusion": cache_manifest["fusion_checkpoint_sha256"],
        },
    }
    write_json(output_root / "tuning_report.json", report)
    frozen = {
        "schema_version": "step3-proposal-parameters-frozen-v1", "split_used": "val",
        "test_used_for_tuning": False,
        "heatmap_method": "probability_threshold",
        "parameters": {key: best[key] for key in (
            "probability_threshold", "morph_kernel", "min_area", "max_aspect_ratio",
            "expand_k", "merge_iou_threshold",
        )},
        "validation_metrics": {key: best[key] for key in (
            "recall@0.3", "recall@0.5", "recall@0.7", "gt_coverage",
            "candidates_per_image", "crops_per_image", "empty_crops", "crop_area_mean",
        )},
        "checkpoint_sha256": report["checkpoint_sha256"],
    }
    write_json(output_root / "proposal_parameters_frozen.json", frozen)
    return report
