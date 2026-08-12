"""Gradio demo application for SimeseUltraMaxPlus+ defect detection pipeline (Step 8).

Usage:
    python src/demo/app.py
    # Opens browser at http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path for absolute imports of step3_model packages
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
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ("broke", "dirty", "double", "miss")
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (42, 120, 214),   # broke:  blue
    1: (27, 175, 122),   # dirty:  green
    2: (237, 161, 0),    # double: orange
    3: (0, 131, 0),      # miss:   dark-green
}

# Artifacts output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "step3" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Globally loaded model components (loaded once at startup)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Backbone (always ImageNet pre-trained)
backbone = SharedSiameseBackbone("IMAGENET1K_V2").eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad_(False)

# OffsetPredictor – attempt to load trained checkpoint; fall back to zero-init
predictor = OffsetPredictor(kernel_size=3, hidden_ratio=4, zero_init=True).eval().to(DEVICE)
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts" / "step3" / "training" / "checkpoint_best.pt"
)
if CHECKPOINT_PATH.exists():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    predictor.load_state_dict(ckpt["model_state_dict"])
    print(f"[demo] Loaded trained OffsetPredictor from {CHECKPOINT_PATH}")
else:
    print("[demo] Using zero-init OffsetPredictor (no checkpoint found)")

for p in predictor.parameters():
    p.requires_grad_(False)

warp = FeatureWarp(kernel_size=3).eval().to(DEVICE)
for p in warp.parameters():
    p.requires_grad_(False)

fusion = MultiScaleFusion(hidden=128, reduction=16).eval().to(DEVICE)
for p in fusion.parameters():
    p.requires_grad_(False)

print(f"[demo] All models loaded on {DEVICE}")

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def render_single_heatmap(heatmap_2d: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Render a [H,W] single-channel heatmap as [H,W,3] RGB colormap image."""
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
    """Draw xyxy boxes onto an RGB image.  boxes: [K,4]."""
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
    """Draw xyxy boxes with class labels and colored borders.

    Args:
        image_rgb: [H, W, 3] uint8 RGB background.
        boxes: [K, 4] xyxy float32.
        labels: length-K list of label strings.
        colors: mapping from class index to BGR color tuple.
        thickness: box border thickness.

    Returns:
        [H, W, 3] uint8 RGB annotated image.
    """
    canvas = image_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(v) for v in box]
        class_idx = idx % len(colors)
        color = colors[class_idx]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(
    query_pil: Image.Image | None,
    template_pil: Image.Image | None,
    min_area: float = 64.0,
    expand_k: float = 1.8,
    adaptive_block: int = 51,
    adaptive_c: int = -5,
) -> dict:
    """Run the full Siamese screening pipeline on a query/template pair.

    Args:
        query_pil: query (defective) image as PIL Image.
        template_pil: reference (golden) template image as PIL Image.
        min_area: minimum contour area for candidate boxes.
        expand_k: expansion factor (unused in this demo; kept for future use).
        adaptive_block: adaptive threshold block size.
        adaptive_c: adaptive threshold constant.

    Returns:
        dict with visualization images and detection metadata.
    """
    if query_pil is None or template_pil is None:
        raise gr.Error("Please upload both Query and Template images.")

    # --- 1. Preprocess: PIL → Tensor, resize to 800×800 --------------------
    query_pil = query_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    template_pil = template_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    query_rgb = np.array(query_pil)        # [800, 800, 3] uint8 RGB
    template_rgb = np.array(template_pil)

    q_tensor = pil_to_tensor(query_pil).unsqueeze(0).to(DEVICE)       # [1,3,800,800]
    t_tensor = pil_to_tensor(template_pil).unsqueeze(0).to(DEVICE)

    # --- 2. Backbone -------------------------------------------------------
    with torch.inference_mode():
        qf, tf = backbone(q_tensor, t_tensor)

        # --- 3. OffsetPredictor → offsets ------------------------------------
        offsets = predictor(qf, tf)

        # --- 4. FeatureWarp → warped template features -----------------------
        warped = warp(tf, offsets)

        # --- 5. Diff features → MultiScaleFusion → enhanced heatmap ----------
        diff_features = {k: warped[k] - qf[k] for k in ["C3", "C4", "C5"]}
        enhanced = fusion(diff_features)  # [1, 1, 800, 800]

        # --- 6. Scalar diff maps (for panel visualization) -------------------
        diff_maps = compute_diff_maps(warped, qf)
        diff_upsampled = upsample_diff_maps(diff_maps)

    # --- 7. Candidate box post-processing ----------------------------------
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

    # --- 8. Render heatmap panel (write to disk, read back) ----------------
    panel_path = OUTPUT_DIR / "heatmap_panel.png"
    render_heatmap_panel(query_rgb, template_rgb, diff_upsampled, panel_path)
    heatmap_panel_img = cv2.cvtColor(cv2.imread(str(panel_path)), cv2.COLOR_BGR2RGB)

    # --- 9. Render CBAM enhanced heatmap -----------------------------------
    enhanced_heatmap_rgb = render_single_heatmap(enhanced_np)

    # --- 10. Candidate overlay ---------------------------------------------
    candidate_overlay = draw_boxes(query_rgb, candidate_boxes, color=(255, 60, 60), thickness=2)

    # --- 11. Final detection overlay ---------------------------------------
    # NOTE: RT-DETR classification weights are not yet integrated.
    # When available, replace the labels below with actual class predictions.
    labels = ["candidate"] * max(candidate_count, 0)
    final_overlay = draw_boxes_with_labels(
        query_rgb, candidate_boxes, labels, CLASS_COLORS, thickness=2,
    )

    # --- 12. Results JSON --------------------------------------------------
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
    """Wrapper that catches errors and returns structured outputs for Gradio."""
    try:
        result = run_full_pipeline(
            query_img, template_img,
            min_area=float(min_area),
            expand_k=float(expand_k),
            adaptive_block=int(adaptive_block),
            adaptive_c=int(adaptive_c),
        )
        return (
            result["heatmap_panel"],           # Tab 1a
            result["enhanced_heatmap"],        # Tab 1b
            result["candidate_overlay"],       # Tab 2
            result["final_overlay"],           # Tab 3a
            json.dumps(result["results_json"], indent=2, ensure_ascii=False),  # Tab 3b
            f"✅ Detection complete — {result['candidate_count']} candidate box(es) found.",
        )
    except Exception as exc:
        raise gr.Error(str(exc))


# ---------------------------------------------------------------------------
# Gradio UI
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
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(
        title="SimeseUltraMaxPlus+ — Defect Detection Demo",
    ) as demo:
        gr.Markdown(
            """
            # 🔬 SimeseUltraMaxPlus+ — 印刷缺陷检测演示

            Upload a **Query** (defective) image and a **Template** (reference) image,
            then click **Run Detection** to visualise every stage of the Siamese pipeline.
            """
        )

        # ---- Input row --------------------------------------------------------
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                query_input = gr.Image(
                    label="📷 Query Image",
                    type="pil",
                    image_mode="RGB",
                    height=300,
                )
            with gr.Column(scale=1):
                template_input = gr.Image(
                    label="📐 Template Image",
                    type="pil",
                    image_mode="RGB",
                    height=300,
                )

        # ---- Parameters (collapsible) -----------------------------------------
        with gr.Accordion("⚙️ Advanced Parameters", open=False):
            with gr.Row():
                min_area = gr.Slider(
                    16, 256, value=64, step=8,
                    label="Min Area (px²)",
                    info="Minimum candidate box area",
                )
                expand_k = gr.Slider(
                    1.0, 3.0, value=1.8, step=0.1,
                    label="Expand Factor (k)",
                    info="Box expansion multiplier (reserved for future crop stage)",
                )
                adaptive_block = gr.Slider(
                    11, 101, value=51, step=2,
                    label="Adaptive Block",
                    info="Adaptive threshold block size (odd)",
                )
                adaptive_c = gr.Slider(
                    -20, 10, value=-5, step=1,
                    label="Adaptive C",
                    info="Adaptive threshold constant",
                )

        run_btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")

        # ---- Status bar -------------------------------------------------------
        status = gr.Textbox(
            label="Status",
            value="Ready — upload images and click Run.",
            interactive=False,
            elem_classes=["status-box"],
        )

        # ---- Output tabs ------------------------------------------------------
        with gr.Tabs():
            with gr.TabItem("🔥 Heatmap Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("##### Five-column diff panel (Query | Template | C3 | C4 | C5)")
                        panel_output = gr.Image(
                            label="Heatmap Panel",
                            type="numpy",
                            format="png",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("##### CBAM-enhanced fusion heatmap")
                        enhanced_output = gr.Image(
                            label="Enhanced Heatmap",
                            type="numpy",
                            format="png",
                        )

            with gr.TabItem("📦 Candidate Boxes"):
                gr.Markdown("Candidate boxes overlaid on the query image (pre-classification).")
                candidate_output = gr.Image(
                    label="Candidate Box Overlay",
                    type="numpy",
                    format="png",
                )

            with gr.TabItem("🏷️ Detection Results"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("##### Final detection overlay")
                        final_output = gr.Image(
                            label="Detection Overlay",
                            type="numpy",
                            format="png",
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("##### Results JSON")
                        json_output = gr.Code(
                            label="Detection JSON",
                            language="json",
                            lines=18,
                        )

        # ---- Wire up events ---------------------------------------------------
        run_btn.click(
            fn=run_pipeline_wrapper,
            inputs=[query_input, template_input, min_area, expand_k, adaptive_block, adaptive_c],
            outputs=[panel_output, enhanced_output, candidate_output, final_output, json_output, status],
        )

        # ---- Footer -----------------------------------------------------------
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #888; font-size: 0.85em;">
            SimeseUltraMaxPlus+ · Parallel Siamese Network for Print Defect Detection · Step 8 Demo
            </div>
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1)  # serialise inference to avoid OOM
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=THEME,
        css=CSS,
    )
