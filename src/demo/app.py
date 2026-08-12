"""SimeseUltraMaxPlus+ 印刷缺陷检测流水线 Gradio 演示应用（Step 8）。

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

from step3_model.backbone import SharedSiameseBackbone
from step3_model.feature_warp import FeatureWarp
from step3_model.fusion import MultiScaleFusion
from step3_model.heatmap import compute_diff_maps, render_heatmap_panel, upsample_diff_maps
from step3_model.offset_predictor import OffsetPredictor
from step3_model.proposals import postprocess_heatmap
from step2_dataset.transforms import pil_to_tensor

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CLASS_NAMES = ("broke", "dirty", "double", "miss")
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (42, 120, 214),   # broke:  蓝色
    1: (27, 175, 122),   # dirty:  绿色
    2: (237, 161, 0),    # double: 橙色
    3: (0, 131, 0),      # miss:   深绿色
}

# 运行时产物输出目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "step3" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 全局模型组件（应用启动时加载一次，避免每次推理重新加载）
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Backbone（始终使用 ImageNet 预训练权重）
backbone = SharedSiameseBackbone("IMAGENET1K_V2").eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad_(False)

# OffsetPredictor —— 尝试加载训练后的 checkpoint；若不存在则回退到 zero-init
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
    print("[demo] 使用 zero-init OffsetPredictor（未找到 checkpoint）")

for p in predictor.parameters():
    p.requires_grad_(False)

warp = FeatureWarp(kernel_size=3).eval().to(DEVICE)
for p in warp.parameters():
    p.requires_grad_(False)

fusion = MultiScaleFusion(hidden=128, reduction=16).eval().to(DEVICE)
for p in fusion.parameters():
    p.requires_grad_(False)

print(f"[demo] 所有模型已加载到 {DEVICE}")

# ---------------------------------------------------------------------------
# 可视化辅助函数
# ---------------------------------------------------------------------------


def render_single_heatmap(heatmap_2d: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """将 [H,W] 单通道热力图渲染为 [H,W,3] RGB colormap 图像。"""
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
    """在 RGB 图像上绘制 xyxy 候选框。boxes 形状: [K, 4]。"""
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
    """绘制带类别标签和彩色边框的 xyxy 候选框。

    参数:
        image_rgb: [H, W, 3] uint8 RGB 背景图。
        boxes:     [K, 4] xyxy float32 坐标。
        labels:    长度 K 的标签字符串列表。
        colors:    类别索引到 BGR 颜色元组的映射。
        thickness: 边框粗细。

    返回:
        [H, W, 3] uint8 RGB 标注图像。
    """
    canvas = image_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        class_idx = idx % len(colors)
        color = colors[class_idx]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        # 绘制标签背景
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 核心推理流水线
# ---------------------------------------------------------------------------


def run_full_pipeline(
    query_pil: Image.Image | None,
    template_pil: Image.Image | None,
    min_area: float = 64.0,
    expand_k: float = 1.8,
    adaptive_block: int = 51,
    adaptive_c: int = -5,
) -> dict:
    """运行完整的 Siamese 筛选流水线。

    参数:
        query_pil:     query（缺陷）图像，PIL Image。
        template_pil:  template（参考）模板图像，PIL Image。
        min_area:      候选框最小轮廓面积。
        expand_k:      扩展系数（本 demo 尚未使用；预留后续裁剪阶段使用）。
        adaptive_block: 自适应阈值块大小。
        adaptive_c:     自适应阈值常数。

    返回:
        包含可视化图像和检测元数据的字典。
    """
    if query_pil is None or template_pil is None:
        raise gr.Error("请同时上传 Query 图像和 Template 图像。")

    # --- 1. 预处理: PIL → Tensor，缩放到 800×800 ----------------------------
    query_pil = query_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    template_pil = template_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    query_rgb = np.array(query_pil)        # [800, 800, 3] uint8 RGB
    template_rgb = np.array(template_pil)

    q_tensor = pil_to_tensor(query_pil).unsqueeze(0).to(DEVICE)       # [1, 3, 800, 800]
    t_tensor = pil_to_tensor(template_pil).unsqueeze(0).to(DEVICE)

    # --- 2. Backbone 提取多尺度特征 -----------------------------------------
    with torch.inference_mode():
        qf, tf = backbone(q_tensor, t_tensor)

        # --- 3. OffsetPredictor 预测 DCN 偏移 --------------------------------
        offsets = predictor(qf, tf)

        # --- 4. FeatureWarp 对模板特征做形变采样 ----------------------------
        warped = warp(tf, offsets)

        # --- 5. 差异特征 → MultiScaleFusion → CBAM 增强热力图 ----------------
        diff_features = {k: warped[k] - qf[k] for k in ["C3", "C4", "C5"]}
        enhanced = fusion(diff_features)  # [1, 1, 800, 800]

        # --- 6. 标量差异图（用于面板可视化） ---------------------------------
        diff_maps = compute_diff_maps(warped, qf)
        diff_upsampled = upsample_diff_maps(diff_maps)

    # --- 7. 候选框后处理 ----------------------------------------------------
    enhanced_np: np.ndarray = enhanced[0, 0].cpu().numpy()  # [800, 800] float32
    candidate_boxes = postprocess_heatmap(
        enhanced_np,
        min_area=min_area,
        max_aspect_ratio=5.0,
        adaptive_block=adaptive_block,
        adaptive_c=adaptive_c,
        morph_kernel=5,
    )  # → [K, 4] xyxy

    candidate_count = int(candidate_boxes.shape[0])

    # --- 8. 渲染热力图面板（写入磁盘再读回） --------------------------------
    panel_path = OUTPUT_DIR / "heatmap_panel.png"
    render_heatmap_panel(query_rgb, template_rgb, diff_upsampled, panel_path)
    heatmap_panel_img = cv2.cvtColor(cv2.imread(str(panel_path)), cv2.COLOR_BGR2RGB)

    # --- 9. 渲染 CBAM 增强热力图 --------------------------------------------
    enhanced_heatmap_rgb = render_single_heatmap(enhanced_np)

    # --- 10. 候选框叠加图 ---------------------------------------------------
    candidate_overlay = draw_boxes(query_rgb, candidate_boxes, color=(255, 60, 60), thickness=2)

    # --- 11. 最终检测结果叠加图 ---------------------------------------------
    # 注意: RT-DETR 分类权重尚未集成。
    # 后续可用实际分类结果替换以下标签。
    labels = ["candidate"] * max(candidate_count, 0)
    final_overlay = draw_boxes_with_labels(
        query_rgb, candidate_boxes, labels, CLASS_COLORS, thickness=2,
    )

    # --- 12. 结果 JSON ------------------------------------------------------
    results_json = [
        {
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "area": float((x2 - x1) * (y2 - y1)),
            "label": "candidate",
        }
        for x1, y1, x2, y2 in candidate_boxes
    ]

    return {
        "heatmap_panel": heatmap_panel_img,
        "enhanced_heatmap": enhanced_heatmap_rgb,
        "candidate_overlay": candidate_overlay,
        "final_overlay": final_overlay,
        "candidate_count": candidate_count,
        "results_json": results_json,
    }


def run_pipeline_wrapper(query_img, template_img, min_area, expand_k, adaptive_block, adaptive_c):
    """包装函数：捕获异常并返回 Gradio 所需的结构化输出。"""
    try:
        result = run_full_pipeline(
            query_img, template_img,
            min_area=float(min_area),
            expand_k=float(expand_k),
            adaptive_block=int(adaptive_block),
            adaptive_c=int(adaptive_c),
        )
        return (
            result["heatmap_panel"],           # Tab 1a: 热力图面板
            result["enhanced_heatmap"],        # Tab 1b: CBAM 增强热力图
            result["candidate_overlay"],       # Tab 2:  候选框叠加图
            result["final_overlay"],           # Tab 3a: 最终检测叠加图
            json.dumps(result["results_json"], indent=2, ensure_ascii=False),  # Tab 3b: JSON
            f"✅ 检测完成 —— 共发现 {result['candidate_count']} 个候选框。",
        )
    except Exception as exc:
        raise gr.Error(str(exc))


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------

CSS = """
footer {visibility: hidden}
.status-box textarea {font-family: 'Consolas', 'JetBrains Mono', monospace; font-size: 13px;}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_text_color="*neutral_900",
    block_label_text_color="*neutral_700",
    block_title_text_color="*neutral_800",
    button_primary_background_fill="*primary_500",
)


def build_ui() -> gr.Blocks:
    """构建 Gradio Blocks 界面。"""
    with gr.Blocks(
        title="SimeseUltraMaxPlus+ — 缺陷检测演示",
    ) as demo:
        gr.Markdown(
            """
            # 🔬 SimeseUltraMaxPlus+ — 印刷缺陷检测演示

            上传一张 **Query**（缺陷）图像和一张 **Template**（参考模板）图像，
            然后点击 **运行检测**，即可可视化 Siamese 流水线每个阶段的结果。
            """
        )

        # ---- 输入区域 --------------------------------------------------------
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                query_input = gr.Image(
                    label="📷 Query 图像",
                    type="pil",
                    image_mode="RGB",
                    height=300,
                )
            with gr.Column(scale=1):
                template_input = gr.Image(
                    label="📐 Template 图像",
                    type="pil",
                    image_mode="RGB",
                    height=300,
                )

        # ---- 参数区（可折叠） ------------------------------------------------
        with gr.Accordion("⚙️ 高级参数", open=False):
            with gr.Row():
                min_area = gr.Slider(
                    16, 256, value=64, step=8,
                    label="最小面积 (px²)",
                    info="候选框最小面积阈值",
                )
                expand_k = gr.Slider(
                    1.0, 3.0, value=1.8, step=0.1,
                    label="扩展系数 (k)",
                    info="候选框扩展倍数（预留后续裁剪阶段使用）",
                )
                adaptive_block = gr.Slider(
                    11, 101, value=51, step=2,
                    label="自适应块大小",
                    info="自适应阈值块大小（奇数）",
                )
                adaptive_c = gr.Slider(
                    -20, 10, value=-5, step=1,
                    label="自适应常数 C",
                    info="自适应阈值常数",
                )

        run_btn = gr.Button("🚀 运行检测", variant="primary", size="lg")

        # ---- 状态栏 ----------------------------------------------------------
        status = gr.Textbox(
            label="状态",
            value="就绪 —— 请上传图像并点击运行。",
            interactive=False,
            elem_classes=["status-box"],
        )

        # ---- 输出 Tab 页 ----------------------------------------------------
        with gr.Tabs():
            with gr.TabItem("🔥 热力图分析"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("##### 五列差异面板（Query | Template | C3 | C4 | C5）")
                        panel_output = gr.Image(
                            label="热力图面板",
                            type="numpy",
                            format="png",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("##### CBAM 增强融合热力图")
                        enhanced_output = gr.Image(
                            label="增强热力图",
                            type="numpy",
                            format="png",
                        )

            with gr.TabItem("📦 候选框"):
                gr.Markdown("候选框叠加在 Query 图像上的可视化（分类前）。")
                candidate_output = gr.Image(
                    label="候选框叠加图",
                    type="numpy",
                    format="png",
                )

            with gr.TabItem("🏷️ 检测结果"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("##### 最终检测叠加图")
                        final_output = gr.Image(
                            label="检测叠加图",
                            type="numpy",
                            format="png",
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("##### 结果 JSON")
                        json_output = gr.Code(
                            label="检测 JSON",
                            language="json",
                            lines=18,
                        )

        # ---- 事件绑定 --------------------------------------------------------
        run_btn.click(
            fn=run_pipeline_wrapper,
            inputs=[query_input, template_input, min_area, expand_k, adaptive_block, adaptive_c],
            outputs=[panel_output, enhanced_output, candidate_output, final_output, json_output, status],
        )

        # ---- 页脚 -----------------------------------------------------------
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #888; font-size: 0.85em;">
            SimeseUltraMaxPlus+ · 并行 Siamese 网络印刷缺陷检测 · Step 8 演示
            </div>
            """
        )

    return demo


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1)  # 串行推理，避免 GPU 显存溢出
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=THEME,
        css=CSS,
    )
