"""完整 Validation 热力图缓存与 Proposal/Crop 参数搜索入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step3_model.tune_proposals import (  # noqa: E402
    cache_validation_heatmaps,
    load_tuning_config,
    search_cached_parameters,
)


def main() -> int:
    """执行缓存、搜索或两阶段完整调参。"""
    parser = argparse.ArgumentParser(description="完整 Validation Proposal/Crop 参数调优")
    parser.add_argument("command", choices=("cache", "search", "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/step3_proposal_tuning.yaml"))
    args = parser.parse_args()
    config = load_tuning_config(args.config)
    result: dict
    if args.command in {"cache", "all"}:
        result = cache_validation_heatmaps(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"search", "all"}:
        result = search_cached_parameters(config)
        print(json.dumps({"best": result["best"], "scope": result["scope"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
