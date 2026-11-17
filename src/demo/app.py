"""SimeseUltraMaxPlus+ 印刷缺陷检测流水线 Gradio 演示应用（Step 8）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import RTDETR

from step3_model.alignment import SIFTAligner
from step3_model.backbone import SharedSiameseBackbone
from step3_model.crop import expand_box, merge_overlapping_boxes
from step3_model.feature_warp import FeatureWarp
from step3_model.fusion import MultiScaleFusion
from step3_model.heatmap import compute_diff_maps, render_heatmap_panel, upsample_diff_maps
from step3_model.inference_bridge import run_inference
from step3_model.offset_predictor import OffsetPredictor
from step3_model.proposals import postprocess_heatmap
from step3_model.model_io import load_model_checkpoint
from step2_dataset.transforms import pil_to_tensor

# ============================================================================
# 常量定义
# ============================================================================

CLASS_NAMES = ("破损", "脏污", "重影", "漏印")

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (42, 120, 214),
    1: (27, 175, 122),
    2: (237, 161, 0),
    3: (0, 131, 0),
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "step3" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SIFT+RANSAC 宏观配准器
# ============================================================================

_aligner = SIFTAligner({
    "sift": {"nfeatures": 2000},
    "matcher": {"lowe_ratio": 0.7},
    "ransac": {
        "reprojection_threshold": 5.0,
        "min_inliers": 10,
        "min_inlier_ratio": 0.3,
        "max_inlier_reprojection_error": 9.0,
    },
    "geometry": {
        "min_projected_area_ratio": 0.3,
        "max_projected_area_ratio": 3.0,
        "min_overlap_ratio": 0.5,
        "max_abs_rotation_degrees": 180.0,
    },
    "near_identity": {"enabled": False},
})
print("[demo] SIFTAligner 配准器已初始化")

# ============================================================================
# 全局模型组件
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = SharedSiameseBackbone("IMAGENET1K_V2").eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad_(False)

predictor = OffsetPredictor(kernel_size=3, hidden_ratio=4, zero_init=True).eval().to(DEVICE)
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts" / "step3" / "training" / "checkpoint_best.pt"
)
load_model_checkpoint(predictor, CHECKPOINT_PATH, DEVICE, "OffsetPredictor", "defect-masked-alignment-v1")
print(f"[demo] 已加载训练后的 OffsetPredictor: {CHECKPOINT_PATH}")
for p in predictor.parameters():
    p.requires_grad_(False)

warp = FeatureWarp(kernel_size=3).eval().to(DEVICE)
for p in warp.parameters():
    p.requires_grad_(False)

fusion = MultiScaleFusion(hidden=128, reduction=16).eval().to(DEVICE)
FUSION_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts" / "step3" / "fusion_training" / "checkpoint_best.pt"
)
load_model_checkpoint(fusion, FUSION_CHECKPOINT_PATH, DEVICE, "MultiScaleFusion", "supervised-fusion-heatmap-v1")
print(f"[demo] 已加载训练后的 MultiScaleFusion: {FUSION_CHECKPOINT_PATH}")
for p in fusion.parameters():
    p.requires_grad_(False)

RTDETR_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts" / "step5" / "rtdetr_paired_baseline" / "train" / "weights" / "best.pt"
)
rtdetr_model = RTDETR(str(RTDETR_CHECKPOINT_PATH))
print(f"[demo] 已加载配对感知 RT-DETR: {RTDETR_CHECKPOINT_PATH}")


def paired_rtdetr_callable(query_tensor: torch.Tensor, template_tensor: torch.Tensor) -> torch.Tensor:
    def to_gray(tensor: torch.Tensor) -> np.ndarray:
        gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        return np.clip(np.rint(gray.detach().cpu().numpy() * 255.0), 0, 255).astype(np.float32)

    query_gray = to_gray(query_tensor[0])
    template_gray = to_gray(template_tensor[0])
    diff = np.clip(np.abs(query_gray - template_gray) * 3.0, 0, 255)
    image = np.stack([diff, template_gray, query_gray], axis=2).astype(np.uint8)
    results = rtdetr_model.predict(
        source=image, imgsz=800, conf=0.25, iou=0.7, device=str(DEVICE), verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return torch.empty((0, 6), dtype=torch.float32)
    boxes = results[0].boxes.xyxy.cpu()
    scores = results[0].boxes.conf.cpu().unsqueeze(1)
    classes = results[0].boxes.cls.cpu().unsqueeze(1)
    return torch.cat([boxes, scores, classes], dim=1)


print(f"[demo] 所有模型已加载完成，运行设备: {DEVICE}")

PROPOSAL_PARAMETERS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts" / "step3" / "tuning" / "proposal_parameters_frozen.json"
)
E2E_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs" / "step3_end_to_end_eval.yaml"
)
PROPOSAL_PARAMETERS = json.loads(PROPOSAL_PARAMETERS_PATH.read_text(encoding="utf-8"))["parameters"]
GATING_PARAMETERS = yaml.safe_load(E2E_CONFIG_PATH.read_text(encoding="utf-8")).get("gating", {})

# ============================================================================
# 可视化辅助函数
# ============================================================================


def render_single_heatmap(heatmap_2d: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    vmax = max(float(heatmap_2d.max()), 1.0)
    norm = np.clip(heatmap_2d / vmax, 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def draw_boxes(image_rgb: np.ndarray, boxes: np.ndarray, color=(255, 0, 0), thickness=2) -> np.ndarray:
    canvas = image_rgb.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return canvas


def draw_boxes_with_labels(image_rgb, boxes, labels, colors, thickness=2) -> np.ndarray:
    canvas = image_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = colors[idx % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_alignment_comparison(original, aligned, template, info) -> np.ndarray:
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


def transform_boxes_to_original(boxes, H_inv, src_h, src_w) -> np.ndarray:
    if boxes.size == 0 or H_inv is None:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([
        np.column_stack([x1, y1]), np.column_stack([x2, y1]),
        np.column_stack([x2, y2]), np.column_stack([x1, y2]),
    ], axis=1).reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 4, 2)
    projected[..., 0] = np.clip(projected[..., 0], 0, src_w - 1)
    projected[..., 1] = np.clip(projected[..., 1], 0, src_h - 1)
    return np.column_stack([
        projected[..., 0].min(axis=1), projected[..., 1].min(axis=1),
        projected[..., 0].max(axis=1), projected[..., 1].max(axis=1),
    ]).astype(np.float32)


# ============================================================================
# 核心推理流水线
# ============================================================================


def run_full_pipeline(query_pil, template_pil) -> dict:
    if query_pil is None or template_pil is None:
        raise gr.Error("请同时上传待检测图像和模板图像后再运行检测。")

    query_pil = query_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    template_pil = template_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    query_rgb = np.array(query_pil)
    template_rgb = np.array(template_pil)

    align_info = {"status": "未执行"}
    aligned_rgb = query_rgb
    H_inv = None

    try:
        aligned_result, align_report = _aligner.align(query_rgb, template_rgb, sample_id="demo")
        align_info["kp_q"] = align_report.num_keypoints_query
        align_info["kp_t"] = align_report.num_keypoints_template
        align_info["good_matches"] = align_report.num_raw_matches
        align_info["inliers"] = align_report.num_inlier_matches
        align_info["inlier_ratio"] = f"{align_report.inlier_ratio:.3f}" if align_report.inlier_ratio is not None else "N/A"
        align_info["reproj_error"] = f"{align_report.ransac_reprojection_error:.2f} px" if align_report.ransac_reprojection_error is not None else "N/A"
        align_info["success"] = align_report.success

        H = np.array(align_report.homography_matrix, dtype=np.float32) if align_report.homography_matrix is not None else None

        if align_report.success and not align_report.already_aligned:
            aligned_rgb = aligned_result
            align_info["status"] = "配准成功 ✓"
        elif H is not None and align_report.num_inlier_matches >= 10 and align_report.inlier_ratio is not None and align_report.inlier_ratio >= 0.3:
            h, w = template_rgb.shape[:2]
            aligned_rgb = cv2.warpPerspective(query_rgb, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            align_info["status"] = f"配准成功（绕过几何验收: {', '.join(align_report.reasons)}）"
            align_info["success"] = True
        else:
            align_info["status"] = f"配准失败: {align_report.error or '几何验收未通过'}"

        if H is not None and align_info["success"]:
            H_inv = np.linalg.inv(H)
            align_info["has_inverse"] = True
        else:
            align_info["has_inverse"] = False
    except Exception as exc:
        align_info["status"] = f"配准异常: {str(exc)}"
        align_info["success"] = False

    alignment_viz = render_alignment_comparison(query_rgb, aligned_rgb, template_rgb, align_info)

    q_tensor = pil_to_tensor(Image.fromarray(aligned_rgb)).unsqueeze(0).to(DEVICE)
    t_tensor = pil_to_tensor(template_pil).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        qf, tf = backbone(q_tensor, t_tensor)
        offsets = predictor(qf, tf)
        warped = warp(tf, offsets)
        diff_features = {k: (warped[k] - qf[k]).abs() for k in ["C3", "C4", "C5"]}
        enhanced = torch.sigmoid(fusion(diff_features))
        diff_maps = compute_diff_maps(warped, qf)
        diff_upsampled = upsample_diff_maps(diff_maps)

    enhanced_np = enhanced[0, 0].cpu().numpy()
    proposals_cfg = {
        "heatmap_method": "probability_threshold",
        "probability_threshold": float(PROPOSAL_PARAMETERS["probability_threshold"]),
        "morph_kernel": int(PROPOSAL_PARAMETERS["morph_kernel"]),
        "min_area": float(PROPOSAL_PARAMETERS["min_area"]),
        "max_aspect_ratio": float(PROPOSAL_PARAMETERS["max_aspect_ratio"]),
        "max_area_ratio": float(PROPOSAL_PARAMETERS.get("max_area_ratio", 0.8)),
        "expand_k": float(PROPOSAL_PARAMETERS.get("expand_k", 3.5)),
        "merge_iou": float(PROPOSAL_PARAMETERS.get("merge_iou_threshold", 0.1)),
    }
    proposals_cfg.update(GATING_PARAMETERS)

    with torch.inference_mode():
        detections, candidate_boxes = run_inference(
            q_tensor[0], t_tensor[0], backbone, predictor, warp, fusion,
            rtdetr_model=paired_rtdetr_callable, proposals_cfg=proposals_cfg,
            image_size=(800, 800), device=DEVICE, pair_aware=True, return_proposals=True,
        )

    merged_candidates = merge_overlapping_boxes(candidate_boxes, iou_threshold=float(proposals_cfg["merge_iou"]))
    crop_windows = torch.stack([
        torch.tensor(expand_box(box, float(proposals_cfg["expand_k"]), (800, 800)), dtype=torch.float32)
        for box in merged_candidates
    ]) if merged_candidates.numel() > 0 else torch.empty((0, 4), dtype=torch.float32)
    crop_windows_xyxy = torch.stack([
        torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        for x, y, w, h in [tuple(v.tolist()) for v in crop_windows]
    ]) if crop_windows.numel() > 0 else torch.empty((0, 4), dtype=torch.float32)

    src_h, src_w = query_rgb.shape[:2]
    original_boxes = transform_boxes_to_original(crop_windows_xyxy.cpu().numpy(), H_inv, src_h, src_w)

    panel_path = OUTPUT_DIR / "heatmap_panel.png"
    render_heatmap_panel(aligned_rgb, template_rgb, diff_upsampled, panel_path)
    panel_data = np.fromfile(str(panel_path), dtype=np.uint8)
    heatmap_panel_img = cv2.cvtColor(cv2.imdecode(panel_data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    enhanced_heatmap_rgb = render_single_heatmap(enhanced_np)
    candidate_overlay = draw_boxes(query_rgb, original_boxes, color=(255, 60, 60), thickness=2)

    final_boxes, final_labels, final_results = [], [], []
    for detection in detections:
        box = np.array([detection["box"]], dtype=np.float32)
        if H_inv is not None:
            box = transform_boxes_to_original(box, H_inv, src_h, src_w)
        box = box[0]
        class_id = int(detection["class"])
        class_name = CLASS_NAMES[class_id]
        score = float(detection["score"])
        final_boxes.append(box)
        final_labels.append(f"{class_name} {score:.2f}")
        final_results.append({"框坐标": [float(v) for v in box], "类别": class_name, "置信度": score})

    final_boxes_np = np.array(final_boxes, dtype=np.float32) if final_boxes else np.empty((0, 4), dtype=np.float32)
    final_overlay = draw_boxes_with_labels(query_rgb, final_boxes_np, final_labels, CLASS_COLORS, thickness=2)

    alignment_json = {
        "SIFT配准状态": align_info["status"],
        "坐标映射": "已从 aligned 逆变换回原始 query 坐标系" if H_inv is not None else "无逆变换（SIFT 配准未执行）",
        "Query特征点数": align_info.get("kp_q", 0),
        "Template特征点数": align_info.get("kp_t", 0),
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
        "candidate_count": int(crop_windows_xyxy.shape[0]),
        "detection_count": len(detections),
        "alignment_info": alignment_json,
        "results_json": final_results,
    }


def run_pipeline_wrapper(query_img, template_img):
    try:
        result = run_full_pipeline(query_img, template_img)
        align_status = result["alignment_info"]["SIFT配准状态"]
        detect_msg = f"✅ 检测完成 —— 共检出 {result['detection_count']} 个缺陷，候选区域 {result['candidate_count']} 个。"
        status_text = f"{detect_msg}\nSIFT配准: {align_status}"
        return (
            result["heatmap_panel"], result["enhanced_heatmap"], result["alignment_viz"],
            json.dumps(result["alignment_info"], indent=2, ensure_ascii=False),
            result["candidate_overlay"], result["final_overlay"],
            json.dumps(result["results_json"], indent=2, ensure_ascii=False), status_text,
        )
    except Exception as exc:
        raise gr.Error(str(exc))


# ============================================================================
# Gradio 用户界面
# ============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

:root {
    --app-primary: #1E3A5F;
    --app-secondary: #2563EB;
    --app-background: #F8FAFC;
    --app-foreground: #0F172A;
    --app-card: #FFFFFF;
    --app-border: #E4E7EB;
}

html, body {
    font-family: 'Noto Sans SC', Inter, system-ui, -apple-system, sans-serif;
    background: var(--app-background);
    color: var(--app-foreground);
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100vw !important;
    width: 100vw !important;
}

/* ====== 暴力撑开所有外层容器，不管 Gradio 5.x 改了什么 class ====== */
.gradio-container,
.gradio-container > div,
.gradio-container > div > div,
main,
main > div,
#root > div,
body > div {
    max-width: 100% !important;
    width: 100% !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    padding-left: 40px !important;
    padding-right: 40px !important;
    padding-top: 24px !important;
    padding-bottom: 56px !important;
    box-sizing: border-box !important;
}

/* Row 强制水平 */
.gr-row, .row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 20px !important;
    width: 100% !important;
}
.gr-row > .gr-column, .row > .column {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    max-width: 100% !important;
}

/* Image 内部允许收缩 */
.gr-image, .image, .gr-image > div, .image > div {
    min-width: 0 !important;
    max-width: 100% !important;
}

/* 头部横幅 */
.app-header {
    background: linear-gradient(135deg, var(--app-primary) 0%, #16324F 100%);
    border-radius: 20px;
    padding: 36px 40px;
    color: #FFFFFF;
    box-shadow: 0 12px 30px rgba(30, 58, 95, 0.16);
    margin-bottom: 24px;
}
.app-header h1 {
    margin: 0 0 8px;
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
}
.app-header p {
    margin: 0;
    color: rgba(255, 255, 255, 0.82);
    font-size: 15px;
}

.gr-box, .gr-panel {
    background: var(--app-card);
    border: 1px solid var(--app-border);
    border-radius: 14px;
}

.gr-button-primary {
    background: var(--app-primary);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    cursor: pointer;
}
.gr-button-primary:hover {
    background: var(--app-secondary);
}

footer { visibility: hidden; }
.status-box textarea {
    font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}
"""


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#F8FAFC",
    body_text_color="#0F172A",
    block_background_fill="#FFFFFF",
    block_border_color="#E4E7EB",
    block_border_width="1px",
    block_radius="*radius_lg",
    button_primary_background_fill="#1E3A5F",
    button_primary_background_fill_hover="#2563EB",
    button_primary_text_color="#FFFFFF",
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="SimeseUltraMaxPlus+ · 印刷缺陷智能检测系统",
        css=CSS,
        theme=THEME,
    ) as demo:
        gr.Markdown("""
        <div class="app-header">
            <h1>印刷缺陷智能检测系统</h1>
            <p>基于并行孪生网络的端到端缺陷检测演示平台</p>
        </div>
        """)

        gr.Markdown("""
        ---
        ### 操作说明
        1. 在下方左侧上传 **待检测图像**（Query），右侧上传 **标准模板图像**（Template）
        2. 点击 **开始检测** 按钮，系统将自动执行 SIFT+RANSAC 宏观配准 → 孪生网络筛选流水线
        3. 在结果区域切换标签页，查看各阶段的可视化输出
        """)

        # ====== 上传区：高度从 320 提到 400，两个框等宽 ======
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=0):
                query_input = gr.Image(label="待检测图像（Query）", type="pil", image_mode="RGB", height=400)
            with gr.Column(scale=1, min_width=0):
                template_input = gr.Image(label="标准模板图像（Template）", type="pil", image_mode="RGB", height=400)
        run_btn = gr.Button("开始检测", variant="primary", size="lg")

        status = gr.Textbox(
            label="运行状态",
            value="就绪 —— 请上传待检测图像和模板图像，然后点击「开始检测」。",
            interactive=False, elem_classes=["status-box"], lines=2,
        )

        gr.Markdown("---")
        gr.Markdown("### 检测结果")

        with gr.Tabs():
            with gr.TabItem("热力图分析"):
                gr.Markdown("**热力图面板** 展示五列对比，**增强热力图** 为 CBAM 注意力增强后的多尺度融合结果。")
                with gr.Row():
                    with gr.Column(scale=2, min_width=0):
                        panel_output = gr.Image(label="多尺度差异热力图面板", type="numpy", format="png")
                    with gr.Column(scale=1, min_width=0):
                        enhanced_output = gr.Image(label="CBAM 增强融合热力图", type="numpy", format="png")

                gr.Markdown("---")
                gr.Markdown("##### SIFT 宏观配准结果")
                with gr.Row():
                    with gr.Column(scale=3, min_width=0):
                        alignment_viz_output = gr.Image(label="SIFT 配准对比面板", type="numpy", format="png")
                    with gr.Column(scale=2, min_width=0):
                        alignment_json_output = gr.Code(label="配准指标", language="json", lines=10)

            with gr.TabItem("候选区域"):
                with gr.Row():
                    with gr.Column(scale=3, min_width=0):
                        candidate_output = gr.Image(label="检测裁剪窗口可视化", type="numpy", format="png")
                    with gr.Column(scale=1, min_width=0):
                        gr.Markdown(
                            """
                            Siamese 筛选得到的候选区域，经过合并与上下文扩展后送入 RT-DETR。

                            最终检测框应位于这些裁剪窗口内部。
                            """
                        )

            with gr.TabItem("检测结果"):
                gr.Markdown("左侧为最终检测结果叠加图，右侧为结构化的 JSON 检测数据。")
                with gr.Row():
                    with gr.Column(scale=3, min_width=0):
                        final_output = gr.Image(label="最终检测结果叠加图", type="numpy", format="png")
                    with gr.Column(scale=2, min_width=0):
                        json_output = gr.Code(label="检测结果 (JSON)", language="json", lines=18)

        run_btn.click(
            fn=run_pipeline_wrapper,
            inputs=[query_input, template_input],
            outputs=[panel_output, enhanced_output, alignment_viz_output, alignment_json_output,
                     candidate_output, final_output, json_output, status],
        )

        gr.Markdown("""
        ---
        <div style="text-align: center; color: #999; font-size: 0.82em; padding: 8px 0;">
            SimeseUltraMaxPlus+ · 基于并行孪生网络的印刷缺陷检测 · 演示系统 v1.1
        </div>
        """)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=THEME, css=CSS)
