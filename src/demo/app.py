"""SimeseUltraMaxPlus+ 印刷缺陷检测流水线 Gradio 演示应用（Step 8）。

完整推理流程:
    SIFT+RANSAC 宏观配准 → Backbone 特征提取 → OffsetPredictor DCN偏移
    → FeatureWarp 特征形变 → MultiScaleFusion + CBAM → 候选框后处理

用法:
    python src/demo/app.py
    # 浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 将 src/ 加入 sys.path，以便通过绝对路径导入 step3_model 包
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from step3_model.alignment import SIFTAligner
from step3_model.backbone import SharedSiameseBackbone
from step3_model.feature_warp import FeatureWarp
from step3_model.fusion import MultiScaleFusion
from step3_model.heatmap import compute_diff_maps, render_heatmap_panel, upsample_diff_maps
from step3_model.offset_predictor import OffsetPredictor
from step3_model.proposals import postprocess_heatmap
from step2_dataset.transforms import pil_to_tensor

# ============================================================================
# 常量定义
# ============================================================================

CLASS_NAMES = ("破损", "脏污", "重影", "漏印")
"""缺陷类别名称"""

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (42, 120, 214),   # 破损:  蓝色
    1: (27, 175, 122),   # 脏污:  绿色
    2: (237, 161, 0),    # 重影:  橙色
    3: (0, 131, 0),      # 漏印:  深绿色
}
"""缺陷类别对应颜色（BGR 格式）"""

# 运行时产物输出目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "step3" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SIFT+RANSAC 宏观配准器（使用项目冻结参数，禁用 near_identity 以适配 demo 场景）
# ============================================================================

_aligner = SIFTAligner({
    "sift": {"nfeatures": 2000},
    "matcher": {"lowe_ratio": 0.7},
    "ransac": {
        "reprojection_threshold": 5.0,
        "min_inliers": 10,
        "min_inlier_ratio": 0.3,
        "max_inlier_reprojection_error": 9.0,   # 与 frozen config 一致
    },
    "geometry": {
        "min_projected_area_ratio": 0.3,
        "max_projected_area_ratio": 3.0,
        "min_overlap_ratio": 0.5,
        "max_abs_rotation_degrees": 180.0,  # demo 场景不限制旋转（两张不同印刷品可能方向不同）
    },
    "near_identity": {"enabled": False},  # 禁用 near-identity，确保始终尝试 warp
})
print("[demo] SIFTAligner 配准器已初始化")

# ============================================================================
# 全局模型组件（应用启动时加载一次，避免每次推理重复初始化）
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 骨干网络 —— 始终使用 ImageNet 预训练权重
backbone = SharedSiameseBackbone("IMAGENET1K_V2").eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad_(False)

# 偏移量预测器 —— 优先加载训练权重，回退至零初始化
predictor = OffsetPredictor(kernel_size=3, hidden_ratio=4, zero_init=True).eval().to(DEVICE)
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts" / "step3" / "training" / "checkpoint_best.pt"
)
if CHECKPOINT_PATH.exists():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    predictor.load_state_dict(ckpt["model_state_dict"])
    print(f"[demo] 已加载训练后的 OffsetPredictor: {CHECKPOINT_PATH}")
else:
    print("[demo] 使用零初始化 OffsetPredictor（未找到训练权重）")

for p in predictor.parameters():
    p.requires_grad_(False)

# 特征形变模块
warp = FeatureWarp(kernel_size=3).eval().to(DEVICE)
for p in warp.parameters():
    p.requires_grad_(False)

# 多尺度融合模块
fusion = MultiScaleFusion(hidden=128, reduction=16).eval().to(DEVICE)
for p in fusion.parameters():
    p.requires_grad_(False)

print(f"[demo] 所有模型已加载完成，运行设备: {DEVICE}")

# ============================================================================
# 可视化辅助函数
# ============================================================================


def render_single_heatmap(heatmap_2d: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """将单通道热力图 [H, W] 渲染为 RGB 彩色图像 [H, W, 3]。"""
    vmax = max(float(heatmap_2d.max()), 1.0)
    norm = np.clip(heatmap_2d / vmax, 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def draw_boxes(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在 RGB 图像上绘制检测框。"""
    canvas = image_rgb.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return canvas


def draw_boxes_with_labels(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    colors: dict[int, tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制带类别标签的彩色检测框。"""
    canvas = image_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        class_idx = idx % len(colors)
        color = colors[class_idx]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_alignment_comparison(
    original: np.ndarray,
    aligned: np.ndarray,
    template: np.ndarray,
    info: dict,
) -> np.ndarray:
    """渲染 SIFT 配准对比面板：原始 Query | 配准后 Query | Template | 配准指标。"""
    h, w = template.shape[:2]
    gap = 4
    info_h = 80
    title_h = 28
    total_h = h + title_h + info_h
    total_w = w * 3 + gap * 2
    canvas = np.full((total_h, total_w, 3), 248, dtype=np.uint8)

    canvas[title_h:title_h + h, :w] = original
    canvas[title_h:title_h + h, w + gap:2 * w + gap] = aligned
    canvas[title_h:title_h + h, 2 * (w + gap):2 * (w + gap) + w] = template

    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, label in enumerate(["原始 Query（待检测）", "SIFT 配准后 Query", "Template（模板）"]):
        cv2.putText(canvas, label, (idx * (w + gap) + 4, title_h - 6), font, 0.45, (44, 44, 44), 1, cv2.LINE_AA)

    y0 = title_h + h + 6
    for i, line in enumerate([
        f"配准状态: {info.get('status', 'N/A')}",
        f"特征点: Q={info.get('kp_q', 0)}  T={info.get('kp_t', 0)}  高质量匹配: {info.get('good_matches', 0)}",
        f"RANSAC内点: {info.get('inliers', 0)}  内点率: {info.get('inlier_ratio', 'N/A')}",
        f"重投影误差: {info.get('reproj_error', 'N/A')}",
    ]):
        cv2.putText(canvas, line, (6, y0 + i * 16), font, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

    return canvas


def transform_boxes_to_original(
    boxes: np.ndarray,
    H_inv: np.ndarray | None,
    src_h: int,
    src_w: int,
) -> np.ndarray:
    """将检测框从 aligned 坐标系映射回原始 query 坐标系。

    参数:
        boxes: [K, 4] xyxy 框，aligned 坐标系。
        H_inv:  3×3 逆单应矩阵 (aligned → original)，为 None 时直通。
        src_h, src_w: 原始 query 图像的高和宽。

    返回:
        [K, 4] xyxy 框，原始 query 坐标系。
    """
    if boxes.size == 0 or H_inv is None:
        return boxes

    # 矩形四角 → 透视变换 → 重新取包围盒
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y1]),
        np.column_stack([x2, y2]),
        np.column_stack([x1, y2]),
    ], axis=1).reshape(-1, 1, 2).astype(np.float32)

    projected = cv2.perspectiveTransform(corners, H_inv)
    projected = projected.reshape(-1, 4, 2)

    projected[..., 0] = np.clip(projected[..., 0], 0, src_w - 1)
    projected[..., 1] = np.clip(projected[..., 1], 0, src_h - 1)

    x_min = projected[..., 0].min(axis=1)
    y_min = projected[..., 1].min(axis=1)
    x_max = projected[..., 0].max(axis=1)
    y_max = projected[..., 1].max(axis=1)

    return np.column_stack([x_min, y_min, x_max, y_max]).astype(np.float32)


# ============================================================================
# 核心推理流水线
# ============================================================================


def run_full_pipeline(
    query_pil: Image.Image | None,
    template_pil: Image.Image | None,
    min_area: float = 64.0,
    expand_k: float = 1.8,
    adaptive_block: int = 51,
    adaptive_c: int = -5,
) -> dict:
    """执行完整的 Siamese 缺陷筛选流水线。

    流程:
        SIFT+RANSAC 宏观配准 → Backbone → OffsetPredictor → FeatureWarp
        → MultiScaleFusion → 候选框后处理 → 可视化

    参数:
        query_pil:     待检测（缺陷）图像。
        template_pil:  参考（标准模板）图像。
        min_area:      候选框最小面积阈值（像素）。
        expand_k:      候选框扩展系数（预留后续裁剪阶段）。
        adaptive_block: 自适应二值化块大小。
        adaptive_c:     自适应二值化偏移常数。

    返回:
        包含所有可视化图像和检测元数据的字典。
    """
    if query_pil is None or template_pil is None:
        raise gr.Error("请同时上传待检测图像和模板图像后再运行检测。")

    # ================ 第 0 步：图像预处理（PIL → numpy） ====================
    query_pil = query_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    template_pil = template_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    query_rgb = np.array(query_pil)        # [800, 800, 3] uint8 RGB
    template_rgb = np.array(template_pil)

    # ================ 第 1 步：SIFT+RANSAC 宏观几何配准 =====================
    # 将 query 透视变换对齐到 template，消除拍照位置偏差、旋转、缩放。
    # 同时保存 H_inv 用于后续将检测框映射回原始 query 坐标系。
    align_info: dict = {"status": "未执行"}
    aligned_rgb = query_rgb  # 默认回退：不配准
    H_inv: np.ndarray | None = None  # 逆单应矩阵: aligned → original query

    try:
        aligned_result, align_report = _aligner.align(query_rgb, template_rgb, sample_id="demo")
        align_info["kp_q"] = align_report.num_keypoints_query
        align_info["kp_t"] = align_report.num_keypoints_template
        align_info["good_matches"] = align_report.num_raw_matches
        align_info["inliers"] = align_report.num_inlier_matches
        align_info["inlier_ratio"] = (
            f"{align_report.inlier_ratio:.3f}" if align_report.inlier_ratio is not None else "N/A"
        )
        align_info["reproj_error"] = (
            f"{align_report.ransac_reprojection_error:.2f} px"
            if align_report.ransac_reprojection_error is not None
            else "N/A"
        )
        align_info["success"] = align_report.success

        # 打印所有验收项到控制台，方便排查
        print(f"\n[demo] SIFTAligner 验收详情:")
        for name, crit in align_report.criteria.items():
            p = crit.get("passed")
            icon = "PASS" if p else ("FAIL" if p is False else "N/A")
            print(f"  [{icon}] {name}: value={crit.get('value')}  threshold={crit.get('threshold')}")
        if align_report.reasons:
            print(f"  reasons={align_report.reasons}")

        # 获取 H 矩阵（aligned = warpPerspective(query, H)，即 H 将 query 映射到 aligned）
        H = (
            np.array(align_report.homography_matrix, dtype=np.float32)
            if align_report.homography_matrix is not None
            else None
        )

        # 优先使用 SIFTAligner 的标准结果
        if align_report.success and not align_report.already_aligned:
            aligned_rgb = aligned_result
            align_info["status"] = "配准成功 ✓"
        # 回退方案：几何验收未通过但 RANSAC 有足够内点，直接用 H 矩阵 warp
        elif (H is not None
              and align_report.num_inlier_matches >= 10
              and align_report.inlier_ratio is not None
              and align_report.inlier_ratio >= 0.3):
            h, w = template_rgb.shape[:2]
            aligned_rgb = cv2.warpPerspective(
                query_rgb, H, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            align_info["status"] = f"配准成功（绕过几何验收: {', '.join(align_report.reasons)}）"
            align_info["success"] = True
        else:
            align_info["status"] = f"配准失败: {align_report.error or '几何验收未通过'}"

        # 保存逆矩阵: aligned → original query（后续将检测框映射回用户原始图像）
        if H is not None and align_info["success"]:
            H_inv = np.linalg.inv(H)
            align_info["has_inverse"] = True
        else:
            align_info["has_inverse"] = False
    except Exception as exc:
        align_info["status"] = f"配准异常: {str(exc)}"
        align_info["success"] = False

    # ================ 第 1.5 步：配准对比可视化 =============================
    alignment_viz = render_alignment_comparison(query_rgb, aligned_rgb, template_rgb, align_info)

    # ================ 第 2 步：配准后图像 → Tensor ==========================
    q_tensor = pil_to_tensor(Image.fromarray(aligned_rgb)).unsqueeze(0).to(DEVICE)  # [1, 3, 800, 800]
    t_tensor = pil_to_tensor(template_pil).unsqueeze(0).to(DEVICE)

    # ================ 第 3-7 步：神经网络推理 ===============================
    with torch.inference_mode():
        # 第 3 步：共享骨干网络提取多尺度特征
        qf, tf = backbone(q_tensor, t_tensor)

        # 第 4 步：偏移量预测器 → DCN 采样偏移
        offsets = predictor(qf, tf)

        # 第 5 步：特征形变模块 → 对齐后的模板特征（微观形变补偿）
        warped = warp(tf, offsets)

        # 第 6 步：差异特征 → 多尺度融合 → CBAM 增强热力图
        diff_features = {k: warped[k] - qf[k] for k in ["C3", "C4", "C5"]}
        enhanced = fusion(diff_features)  # [1, 1, 800, 800]

        # 第 7 步：各尺度标量差异图（用于面板可视化）
        diff_maps = compute_diff_maps(warped, qf)
        diff_upsampled = upsample_diff_maps(diff_maps)

    # ================ 第 8 步：候选框后处理 =================================
    enhanced_np: np.ndarray = enhanced[0, 0].cpu().numpy()  # [800, 800] float32
    candidate_boxes = postprocess_heatmap(
        enhanced_np,
        min_area=min_area,
        max_aspect_ratio=5.0,
        adaptive_block=adaptive_block,
        adaptive_c=adaptive_c,
        morph_kernel=5,
    )  # → [K, 4] xyxy（aligned 坐标系）

    candidate_count = int(candidate_boxes.shape[0])

    # ================ 第 8.5 步：SIFT 逆变换 → 原始 query 坐标系 ============
    src_h, src_w = query_rgb.shape[:2]
    original_boxes = transform_boxes_to_original(candidate_boxes, H_inv, src_h, src_w)

    # ================ 第 9 步：渲染多尺度热力图面板 ==========================
    panel_path = OUTPUT_DIR / "heatmap_panel.png"
    render_heatmap_panel(aligned_rgb, template_rgb, diff_upsampled, panel_path)
    panel_data = np.fromfile(str(panel_path), dtype=np.uint8)
    heatmap_panel_img = cv2.cvtColor(cv2.imdecode(panel_data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # ================ 第 10 步：渲染 CBAM 增强热力图 =========================
    enhanced_heatmap_rgb = render_single_heatmap(enhanced_np)

    # ================ 第 11 步：候选框叠加可视化（画在原始 query 上）==========
    candidate_overlay = draw_boxes(query_rgb, original_boxes, color=(255, 60, 60), thickness=2)

    # ================ 第 12 步：最终检测结果叠加图（画在原始 query 上）========
    # 注意：RT-DETR 分类权重尚未集成，当前统一标注为"候选"。
    # 后续接入 RT-DETR 后，用实际类别预测结果替换此处标签。
    labels = ["候选"] * max(candidate_count, 0)
    final_overlay = draw_boxes_with_labels(
        query_rgb, original_boxes, labels, CLASS_COLORS, thickness=2,
    )

    # ================ 生成结果 JSON（原始坐标系）=============================
    results_json = [
        {
            "框坐标": [float(x1), float(y1), float(x2), float(y2)],
            "面积": float((x2 - x1) * (y2 - y1)),
            "类别": "候选",
        }
        for x1, y1, x2, y2 in original_boxes
    ]

    alignment_json = {
        "SIFT配准状态": align_info["status"],
        "坐标映射": "已从 aligned 逆变换回原始 query 坐标系" if H_inv is not None else "无逆变换（SIFT 配准未执行）",
        "Query特征点数": align_info.get("kp_q", 0),
        "Template特征点数": align_info.get("kp_t", 0),
        "匹配点对数": align_info.get("matches", 0),
        "高质量匹配数": align_info.get("good_matches", 0),
        "RANSAC内点数": align_info.get("inliers", 0),
        "内点率": align_info.get("inlier_ratio", "N/A"),
        "重投影误差均值": align_info.get("reproj_error", "N/A"),
    }

    return {
        "heatmap_panel": heatmap_panel_img,
        "enhanced_heatmap": enhanced_heatmap_rgb,
        "alignment_viz": alignment_viz,
        "candidate_overlay": candidate_overlay,
        "final_overlay": final_overlay,
        "candidate_count": candidate_count,
        "alignment_info": alignment_json,
        "results_json": results_json,
    }


def run_pipeline_wrapper(query_img, template_img, min_area, expand_k, adaptive_block, adaptive_c):
    """流水线包装函数：统一异常捕获并转换为 Gradio 所需的结构化输出。"""
    try:
        result = run_full_pipeline(
            query_img, template_img,
            min_area=float(min_area),
            expand_k=float(expand_k),
            adaptive_block=int(adaptive_block),
            adaptive_c=int(adaptive_c),
        )

        align_status = result["alignment_info"]["SIFT配准状态"]
        detect_msg = f"✅ 检测完成 —— 共检出 {result['candidate_count']} 个候选区域。"
        align_msg = f"SIFT配准: {align_status}"
        status_text = f"{detect_msg}\n{align_msg}"

        return (
            result["heatmap_panel"],
            result["enhanced_heatmap"],
            result["alignment_viz"],
            json.dumps(result["alignment_info"], indent=2, ensure_ascii=False),
            result["candidate_overlay"],
            result["final_overlay"],
            json.dumps(result["results_json"], indent=2, ensure_ascii=False),
            status_text,
        )
    except Exception as exc:
        raise gr.Error(str(exc))


# ============================================================================
# Gradio 用户界面
# ============================================================================

CSS = """
footer {visibility: hidden}
.status-box textarea {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
    font-size: 14px;
}
.gr-markdown h1 { text-align: center; }
.gr-markdown h5 { font-weight: 600; }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_text_color="*neutral_900",
    body_text_size="*text_sm",
    block_label_text_color="*neutral_700",
    block_label_text_size="*text_sm",
    block_title_text_color="*neutral_800",
    block_title_text_size="*text_md",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
    button_large_text_size="*text_md",
)


def build_ui() -> gr.Blocks:
    """构建完整的中文 Gradio 交互界面。"""

    with gr.Blocks(
        title="SimeseUltraMaxPlus+ · 印刷缺陷智能检测系统",
    ) as demo:

        gr.Markdown(
            """
            <div style="text-align: center; padding: 12px 0 4px 0;">
                <h1 style="margin-bottom: 4px;">🔬 印刷缺陷智能检测系统</h1>
                <p style="color: #666; font-size: 0.95em;">
                    基于并行孪生网络的端到端缺陷检测演示平台
                </p>
            </div>
            """
        )

        gr.Markdown(
            """
            ---
            ### 📋 操作说明

            1. 在下方左侧上传 **待检测图像**（Query），右侧上传 **标准模板图像**（Template）
            2. 点击 **开始检测** 按钮，系统将自动执行 SIFT+RANSAC 宏观配准 → 孪生网络筛选流水线
            3. 在结果区域切换标签页，查看各阶段的可视化输出
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                query_input = gr.Image(label="📷 待检测图像（Query）", type="pil", image_mode="RGB", height=320)
            with gr.Column(scale=1):
                template_input = gr.Image(label="📐 标准模板图像（Template）", type="pil", image_mode="RGB", height=320)

        with gr.Accordion("⚙️ 高级检测参数", open=False):
            gr.Markdown("以下参数用于控制候选框后处理阶段的行为，通常情况下保持默认值即可。")
            with gr.Row():
                min_area = gr.Slider(16, 256, value=64, step=8, label="最小面积阈值 (px²)", info="面积低于该值的连通域将被过滤")
                expand_k = gr.Slider(1.0, 3.0, value=1.8, step=0.1, label="框扩展系数 (k)", info="候选框向外扩展的比例（预留，后续裁剪阶段启用）")
                adaptive_block = gr.Slider(11, 101, value=51, step=2, label="自适应阈值块大小", info="高斯自适应二值化的邻域尺寸（取奇数）")
                adaptive_c = gr.Slider(-20, 10, value=-5, step=1, label="自适应阈值常数 (C)", info="阈值偏移量，负值使阈值更宽松")

        run_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

        status = gr.Textbox(
            label="运行状态",
            value="就绪 —— 请上传待检测图像和模板图像，然后点击「开始检测」。",
            interactive=False,
            elem_classes=["status-box"],
            lines=2,
        )

        gr.Markdown("---")
        gr.Markdown("### 📊 检测结果")

        with gr.Tabs():
            with gr.TabItem("🔥 热力图分析"):
                gr.Markdown("**热力图面板** 展示五列对比，**增强热力图** 为 CBAM 注意力增强后的多尺度融合结果。")
                with gr.Row():
                    with gr.Column(scale=2):
                        panel_output = gr.Image(label="多尺度差异热力图面板", type="numpy", format="png")
                    with gr.Column(scale=1):
                        enhanced_output = gr.Image(label="CBAM 增强融合热力图", type="numpy", format="png")

                gr.Markdown("---")
                gr.Markdown("##### 📐 SIFT 宏观配准结果")
                gr.Markdown("展示原始 Query → SIFT+RANSAC 透视变换 → 配准后 Query 的对比以及配准指标。")
                with gr.Row():
                    with gr.Column(scale=3):
                        alignment_viz_output = gr.Image(label="SIFT 配准对比面板", type="numpy", format="png")
                    with gr.Column(scale=2):
                        alignment_json_output = gr.Code(label="配准指标", language="json", lines=10)

            with gr.TabItem("📦 候选区域"):
                gr.Markdown("自适应阈值二值化 + 形态学闭运算 + 连通域分析得到的候选框，叠加在配准后的待检测图像上。")
                candidate_output = gr.Image(label="候选框可视化", type="numpy", format="png")

            with gr.TabItem("🏷️ 检测结果"):
                gr.Markdown("左侧为最终检测结果叠加图，右侧为结构化的 JSON 检测数据。")
                with gr.Row():
                    with gr.Column(scale=3):
                        final_output = gr.Image(label="最终检测结果叠加图", type="numpy", format="png")
                    with gr.Column(scale=2):
                        json_output = gr.Code(label="检测结果 (JSON)", language="json", lines=18)

        run_btn.click(
            fn=run_pipeline_wrapper,
            inputs=[query_input, template_input, min_area, expand_k, adaptive_block, adaptive_c],
            outputs=[panel_output, enhanced_output, alignment_viz_output, alignment_json_output,
                     candidate_output, final_output, json_output, status],
        )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #999; font-size: 0.82em; padding: 8px 0;">
                SimeseUltraMaxPlus+ · 基于并行孪生网络的印刷缺陷检测 · 演示系统 v1.1
            </div>
            """
        )

    return demo


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1)  # 串行处理请求，避免 GPU 显存溢出
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=THEME,
        css=CSS,
    )
