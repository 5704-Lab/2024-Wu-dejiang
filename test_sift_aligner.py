"""直接测试 SIFTAligner 的几何变换能力。

思路: 取一张已有图片作为 template → 人工旋转+平移得到 query
→ 用 SIFTAligner.align() 尝试配准 → 检查配准质量、可视化对比。

用法:
    cd <仓库根目录>
    python -X utf8 test_sift_aligner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2
import numpy as np

from step3_model.alignment import SIFTAligner, load_image_rgb, _imwrite


def apply_transform(img: np.ndarray, rotation_deg: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    """绕图像中心旋转 + 平移, 返回 (变换后图像, 变换矩阵)。

    矩阵 T2W 将 template 坐标映射到 warped 坐标:  p_warped = T2W @ p_template
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), rotation_deg, 1.0)
    T2W = np.eye(3)
    T2W[:2, :] = M_rot
    T2W[0, 2] += tx
    T2W[1, 2] += ty
    transformed = cv2.warpPerspective(img, T2W, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(128, 128, 128))
    return transformed, T2W


def main():
    print("=" * 60)
    print("  SIFTAligner 几何变换能力测试")
    print("=" * 60)

    # ---------- 1. 准备模板图 ----------
    candidates = sorted(Path("artifacts/step2/visualizations/train").glob("*.png"))
    if not candidates:
        print("ERROR: 找不到测试图片")
        return
    img_path = candidates[0]
    template = load_image_rgb(img_path)
    h, w = template.shape[:2]
    print(f"\n[1] Template: {img_path.name}  size={w}x{h}")

    # ---------- 2. 施加已知变换 ----------
    ROT, TX, TY = 5.0, 15.0, 10.0
    query, T2W = apply_transform(template, ROT, TX, TY)
    # T2W: template → warped_query   (p_warped = T2W @ p_template)
    # 逆矩阵 W2T: warped_query → template  (p_template = inv(T2W) @ p_warped)
    W2T_true = np.linalg.inv(T2W)  # SIFT 估计的 H 应该接近这个

    print(f"[2] 施加变换: 旋转{ROT}deg + 平移({TX},{TY})px")

    # ---------- 3. SIFTAligner 配准 ----------
    # 使用冻结参数 (max_inlier_reprojection_error=9.0 而非默认 4.0)
    config = {
        "sift": {"nfeatures": 2000},
        "matcher": {"lowe_ratio": 0.7},
        "ransac": {
            "reprojection_threshold": 5.0,
            "min_inliers": 10,
            "min_inlier_ratio": 0.3,
            "max_inlier_reprojection_error": 9.0,  # 关键: 与 frozen config 一致
        },
        "geometry": {
            "min_projected_area_ratio": 0.3,
            "max_projected_area_ratio": 3.0,
            "min_overlap_ratio": 0.5,
        },
        "near_identity": {"enabled": False},
    }
    aligner = SIFTAligner(config)
    aligned, report = aligner.align(query, template, sample_id="transform_test")

    # ---------- 4. 打印报告 ----------
    print(f"\n[3] 配准报告:")
    print(f"    status          = {report.status}")
    print(f"    success         = {report.success}")
    print(f"    warp_applied    = {report.warp_applied}")
    print(f"    内点 / 匹配     = {report.num_inlier_matches} / {report.num_raw_matches}")
    print(f"    内点率          = {report.inlier_ratio:.3f}" if report.inlier_ratio else "")
    print(f"    重投影误差均值  = {report.ransac_reprojection_error:.2f} px" if report.ransac_reprojection_error else "")
    if report.reasons:
        print(f"    reasons         = {report.reasons}")

    # 打印每项验收
    print(f"\n    逐项验收结果:")
    for name, c in report.criteria.items():
        p = c.get("passed")
        icon = "PASS" if p else ("FAIL" if p is False else "N/A")
        print(f"      [{icon}] {name}  value={c.get('value')}")

    # ---------- 5. 评估配准精度 ----------
    if report.warp_applied and report.homography_matrix:
        H_est = np.array(report.homography_matrix)  # query → template (即 W2T)

        # 检查 H_est 是否接近 W2T_true
        # 注意 H 是齐次坐标, 允许全局缩放因子
        scale = W2T_true[2, 2] / H_est[2, 2] if abs(H_est[2, 2]) > 1e-8 else 1.0
        error = np.linalg.norm(W2T_true - scale * H_est)
        print(f"\n[4] 配准精度评估:")
        print(f"    H_est =\n{H_est}")
        print(f"    W2T_true =\n{W2T_true}")
        print(f"    ||W2T_true - H_est||_F = {error:.4f}")

        # 像素级误差: 用估计的 H 将 query 的四角投影到 template 坐标,
        # 再用真实的 W2T 投影, 比较差异
        corners_q = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        proj_est = cv2.perspectiveTransform(corners_q, H_est).reshape(-1, 2)
        proj_true = cv2.perspectiveTransform(corners_q, W2T_true).reshape(-1, 2)
        corner_error = np.linalg.norm(proj_est - proj_true, axis=1)
        print(f"    四角重投影误差 (px): {corner_error.round(2)}")
        print(f"    最大角点误差: {corner_error.max():.2f} px")

        # 差异热力图
        diff_before = np.abs(query.astype(np.float32) - template.astype(np.float32)).mean(axis=2)
        diff_after = np.abs(aligned.astype(np.float32) - template.astype(np.float32)).mean(axis=2)
        print(f"    平均像素差异: {diff_before.mean():.1f} -> {diff_after.mean():.1f}")

        if diff_after.mean() < diff_before.mean() * 0.3:
            print("\n    VERDICT: PASS - SIFTAligner 几何变换能力正常")
        else:
            print("\n    VERDICT: WEAK - 差异降低不够显著")

    else:
        print("\n[4] 配准未执行 warp, 无法评估精度")

    # ---------- 6. 保存可视化 ----------
    out = Path("artifacts/step3/demo")
    out.mkdir(parents=True, exist_ok=True)
    gap = 4
    canvas = np.full((h, w * 3 + gap * 2, 3), 248, dtype=np.uint8)
    canvas[:, :w] = template
    canvas[:, w + gap:2 * w + gap] = query
    canvas[:, 2 * (w + gap):] = aligned
    for i, txt in enumerate(["Template", "Query (旋转+平移后)", "SIFT 配准后"]):
        cv2.putText(canvas, txt, (i * (w + gap) + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    _imwrite(out / "sift_test_comparison.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # 差异图
    vmax = max(float(np.abs(query.astype(np.float32) - template.astype(np.float32)).mean(axis=2).max()),
               float(np.abs(aligned.astype(np.float32) - template.astype(np.float32)).mean(axis=2).max()), 1.0)
    for tag, diff in [("before", query), ("after", aligned)]:
        d = np.abs(diff.astype(np.float32) - template.astype(np.float32)).mean(axis=2)
        colored = cv2.applyColorMap((d / vmax * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        _imwrite(out / f"sift_test_diff_{tag}.png", colored)

    print(f"\n[5] 可视化保存到: {out / 'sift_test_*.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
