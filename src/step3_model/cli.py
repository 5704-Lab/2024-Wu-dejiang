"""Step 3 CLI for alignment, backbone, and offset predictor evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_alignment import load_config, run_validation_freeze
from .run_backbone import load_backbone_config, run_backbone_smoke
from .run_crop import load_crop_config, run_crop_smoke
from .run_detection import load_detection_config, run_coco_export, run_detection_smoke
from .run_feature_warp import load_feature_warp_config, run_feature_warp_smoke
from .run_fusion import load_fusion_config, run_fusion_smoke
from .run_heatmap import load_heatmap_config, run_heatmap_smoke
from .run_offset import load_offset_config, run_offset_smoke
from .run_proposals import load_proposals_config, run_proposals_smoke
from .run_train_alignment import load_train_config, run_train_alignment
from .test_evaluation import load_test_config, run_test_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 alignment and model evaluation")
    parser.add_argument("command", choices=("validate-freeze", "test-evaluate", "backbone-smoke", "offset-smoke", "feature-warp-smoke", "train-alignment", "heatmap-smoke", "fusion-smoke", "proposals-smoke", "crop-smoke", "export-coco", "detection-smoke"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "validate-freeze":
        result = run_validation_freeze(load_config(args.config or Path("configs/step3_alignment.yaml")))
        summary = {key: result[key] for key in ("schema_version", "scope", "stages", "decisions", "parameters_sha256")}
    elif args.command == "test-evaluate":
        result = run_test_evaluation(load_test_config(args.config or Path("configs/step3_alignment_test.yaml")))
        summary = {key: result[key] for key in ("schema_version", "scope", "stages", "decisions", "parameters_sha256")}
    elif args.command == "backbone-smoke":
        result = run_backbone_smoke(load_backbone_config(args.config or Path("configs/step3_backbone.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "ordered_validation_sample_ids": result["ordered_validation_sample_ids"],
            "sharing": result["sharing"], "weights": result["weights"],
        }
    elif args.command == "offset-smoke":
        result = run_offset_smoke(load_offset_config(args.config or Path("configs/step3_offset.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "ordered_validation_sample_ids": result["ordered_validation_sample_ids"],
            "offset_max_abs_per_scale": result["offset_max_abs_per_scale"],
            "parameter_count": result["parameter_count"],
        }
    elif args.command == "feature-warp-smoke":
        result = run_feature_warp_smoke(load_feature_warp_config(args.config or Path("configs/step3_feature_warp.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "ordered_validation_sample_ids": result["ordered_validation_sample_ids"],
            "dcn_warp_started": result["dcn_warp_started"],
            "train_started": result.get("train_started", False),
        }
    elif args.command == "train-alignment":
        result = run_train_alignment(load_train_config(args.config or Path("configs/step3_train_alignment.yaml")))
        summary = {
            "schema_version": result["schema_version"], "processed": result["processed"],
            "selected_device": result["selected_device"],
            "epochs_completed": result["epochs_completed"],
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "trainable_parameter_count": result["trainable_parameter_count"],
        }
    elif args.command == "heatmap-smoke":
        result = run_heatmap_smoke(load_heatmap_config(args.config or Path("configs/step3_heatmap.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "ordered_validation_sample_ids": result["ordered_validation_sample_ids"],
            "png_count": result["png_count"],
        }
    elif args.command == "fusion-smoke":
        result = run_fusion_smoke(load_fusion_config(args.config or Path("configs/step3_fusion.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "ordered_validation_sample_ids": result["ordered_validation_sample_ids"],
            "trainable_parameter_count": result["trainable_parameter_count"],
            "png_count": result["png_count"],
        }
    elif args.command == "proposals-smoke":
        result = run_proposals_smoke(load_proposals_config(args.config or Path("configs/step3_proposals.yaml")))
        recall = result["recall_statistics"]
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "recall@0.3": recall.get("recall@0.3"),
            "recall@0.5": recall.get("recall@0.5"),
            "recall@0.7": recall.get("recall@0.7"),
            "total_gt_boxes": recall.get("total_gt_boxes"),
            "total_candidate_boxes": recall.get("total_candidate_boxes"),
        }
    elif args.command == "crop-smoke":
        result = run_crop_smoke(load_crop_config(args.config or Path("configs/step3_crop.yaml")))
        stats = result["crop_statistics"]
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "processed_sample_count": result["processed_sample_count"],
            "total_crops": stats.get("total_crops"),
            "crops_with_gt": stats.get("crops_with_gt_boxes"),
            "gt_coverage": stats.get("gt_coverage"),
            "crop_area_mean": stats.get("crop_area_mean"),
        }
    elif args.command == "export-coco":
        result = run_coco_export(load_detection_config(args.config or Path("configs/step3_detection.yaml")))
        summaries = result.get("summaries", {})
        summary = {
            "schema_version": result["schema_version"],
            "train_images": summaries.get("train", {}).get("image_count", 0),
            "val_images": summaries.get("val", {}).get("image_count", 0),
            "test_images": summaries.get("test", {}).get("image_count", 0),
            "total_annotations": sum(s.get("annotation_count", 0) for s in summaries.values()),
        }
    else:
        result = run_detection_smoke(load_detection_config(args.config or Path("configs/step3_detection.yaml")))
        summary = {
            "schema_version": result["schema_version"], "verified": result["verified"],
            "selected_device": result["selected_device"],
            "sample_count": result["sample_count"],
            "results": result["results"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())