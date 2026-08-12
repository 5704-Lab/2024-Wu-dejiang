请帮助我完成以下任务：

---

# 项目背景

这是 SimeseUltraMaxPlus+，一个基于并行 Siamese 网络的印刷缺陷检测项目。整体流程分三个阶段：

1. **筛选阶段（Step 3）**：SIFT+RANSAC 宏观配准 → 共享 ResNet-50 Backbone 提取 C3/C4/C5 多尺度特征 → OffsetPredictor 预测 DCN 偏移 → FeatureWarp 对模板特征做形变采样对齐 → CBAM 注意力增强差异特征 → 多尺度学习融合 → 候选框后处理 → 上下文裁剪

2. **分类阶段（Step 5）**：RT-DETR 对裁剪区域做精细分类（4 类：broke/dirty/double/miss）

3. **推理**：Siamese 筛选 → 候选框 → 上下文裁 → pad → RT-DETR → inverse_map_boxes → 全图坐标检测结果

项目根目录：`d:\WuDJ\Desktop\工作室文档\毕业论文\SimeseUltraMaxPlus+`

---

# 你的任务：开发演示软件（Step 8）

构建一个 Gradio Web 应用，让用户上传 query 和 template 图像，运行完整的缺陷检测 pipeline，并可视化每一步的结果。

## 硬性约束

- 不要修改现有 `src/step3_model/` 下的任何模块文件
- 不要修改任何测试文件、配置文件或已有 artifacts
- 不要新建 conda 环境或虚拟环境——直接使用当前 Python 环境
- 不要安装新的 pip 包（gradio 除外，可能需要 `pip install gradio`）
- 不要运行或触发训练
- 不要访问 train/test split，demo 只做推理
- 输出只写入 `artifacts/step3/demo/` 目录
- 不修改 Step 3.2 证据或其他已有 artifacts
- 不自动开始后续步骤

## 关键模块导入路径

所有模块在 `src/step3_model/` 下，通过 `from .xxx import YYY` 相对导入。demo 脚本放在项目根目录或 `src/demo/` 下时需要将 `src/` 加入 sys.path：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from step3_model.backbone import SharedSiameseBackbone, FEATURE_STAGES
from step3_model.offset_predictor import OffsetPredictor, OFFSET_SPEC
from step3_model.feature_warp import FeatureWarp
from step3_model.fusion import MultiScaleFusion, _STAGE_NAMES
from step3_model.proposals import postprocess_heatmap
from step3_model.crop import expand_box, merge_overlapping_boxes, crop_image_tensor, map_boxes_to_crop, inverse_map_boxes
from step3_model.heatmap import compute_diff_maps, upsample_diff_maps, render_heatmap_panel, _imwrite
from step3_model.inference_bridge import pad_image_to_size
from step2_dataset.transforms import pil_to_tensor
from step3_model.alignment import load_image_rgb
```

## 模型架构与接口

### SharedSiameseBackbone
```python
backbone = SharedSiameseBackbone("IMAGENET1K_V2").eval().to(device)
# 输入: query [B,3,800,800] float32 [0,1], template [B,3,800,800]
# 输出: (query_features, template_features) 两个有序 dict
#   {"C3": [B,512,100,100], "C4": [B,1024,50,50], "C5": [B,2048,25,25]}
# 模型内部已包含 ImageNet normalization (mean/std)
```

### OffsetPredictor
```python
predictor = OffsetPredictor(kernel_size=3, hidden_ratio=4, zero_init=True).eval().to(device)
# 注意：zero_init=True 是未训练状态（预训练 baseline）
# 训练后权重通过 predictor.load_state_dict(torch.load("checkpoint_best.pt")["model_state_dict"]) 加载
# 输入: (query_features, template_features) 两个 dict
# 输出: offsets dict {"C3": [B,18,100,100], "C4": [B,18,50,50], "C5": [B,18,25,25]}
```

### FeatureWarp
```python
warp = FeatureWarp(kernel_size=3).eval().to(device)
# 输入: (template_features, offsets) 两个 dict
# 输出: warped dict {"C3": [B,512,100,100], "C4": [B,1024,50,50], "C5": [B,2048,25,25]}
# 零偏移时 warped == template (identity)
```

### MultiScaleFusion
```python
fusion = MultiScaleFusion(hidden=128, reduction=16).eval().to(device)
# 输入: diff_features dict {"C3": warped["C3"]-qf["C3"], "C4": ..., "C5": ...}
# 输出: enhanced [B, 1, 800, 800] 增强热力图
```

### 后处理函数
```python
# postprocess_heatmap() 签名:
boxes = postprocess_heatmap(
    heatmap: np.ndarray,           # [H, W] float32 热力图
    min_area: float = 64,          # 候选框最小面积
    max_aspect_ratio: float = 5.0, # 最大长宽比
    adaptive_block: int = 51,      # 自适应阈值块大小
    adaptive_c: int = -5,          # 自适应阈值常数
    morph_kernel: int = 5,         # 形态学闭运算核大小
) -> np.ndarray  # [K, 4] xyxy boxes

# crop 相关函数:
rect = expand_box(box, k=1.8, image_size=(800,800))  # → (x, y, w, h)
cropped = crop_image_tensor(image, rect)              # → [3, h, w]
padded = pad_image_to_size(cropped, (800,800))        # → [3, 800, 800]

# 热力图可视化:
compute_diff_maps(warped, query_features, eps=1e-8) → dict[str, np.ndarray]
upsample_diff_maps(diff_maps, target_size=(800,800)) → dict[str, np.ndarray]
render_heatmap_panel(query_img, template_img, diff_maps_upsampled, output_path, sample_id)

# 图像 I/O:
load_image_rgb(path) → np.ndarray [H,W,3] uint8 RGB
pil_to_tensor(pil_image) → torch.Tensor [3,H,W] float32 [0,1]
```

## Gradio 界面设计

### 输入区域
- 两个 Image 组件：Query 图上传 + Template 图上传
- 参数区（折叠）：min_area, expand_k, adaptive_block, adaptive_c（默认值即可，高级用户可调）

### 输出区域（Tab 布局）

**Tab 1: 热力图分析**
- 5 列面板图（Query | Template | Diff C3 | Diff C4 | Diff C5），使用 INFERNO colormap
- CBAM 增强热力图（单张，INFERNO colormap）

**Tab 2: 候选框检测**
- 候选框叠加在 query 图上的可视化（红色框，无标签）

**Tab 3: 缺陷检测结果**
- 最终检测结果可视化（带类别标签的彩色框叠加在 query 原图上）
- JSON 结果表格

### 热力图渲染函数（复用已有代码）

```python
import cv2
import numpy as np

def render_single_heatmap(heatmap_2d: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """将 [H,W] 的单通道热力图渲染为 [H,W,3] RGB colormap 图像。"""
    norm = np.clip(heatmap_2d / max(float(heatmap_2d.max()), 1.0), 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
```

### 候选框绘制函数

```python
def draw_boxes(image_rgb: np.ndarray, boxes: np.ndarray, color=(255, 0, 0), thickness=2) -> np.ndarray:
    """在 RGB 图像上绘制候选框。boxes: [K,4] xyxy。"""
    canvas = image_rgb.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return canvas
```

### 推理函数（核心逻辑）

```python
import torch
import numpy as np
from PIL import Image

# 全局加载模型（应用启动时加载一次，避免每次推理重新加载）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = SharedSiameseBackbone("IMAGENET1K_V2").eval().to(device)
for p in backbone.parameters():
    p.requires_grad_(False)

# 尝试加载训练后的 predictor，否则用 zero-init
predictor = OffsetPredictor(kernel_size=3, hidden_ratio=4, zero_init=True).eval().to(device)
# 如果有训练好的权重：
# ckpt = torch.load("artifacts/step3/training/checkpoint_best.pt")
# predictor.load_state_dict(ckpt["model_state_dict"])

warp = FeatureWarp(kernel_size=3).eval().to(device)
fusion = MultiScaleFusion(hidden=128, reduction=16).eval().to(device)

CLASS_NAMES = ("broke", "dirty", "double", "miss")
CLASS_COLORS = {
    0: (42, 120, 214),   # broke: 蓝色
    1: (27, 175, 122),   # dirty: 绿色
    2: (237, 161, 0),    # double: 橙色
    3: (0, 131, 0),      # miss: 深绿
}

def run_full_pipeline(query_pil, template_pil, min_area=64, expand_k=1.8, adaptive_block=51, adaptive_c=-5):
    """
    完整推理 pipeline。
    
    Args:
        query_pil: PIL Image, query 图
        template_pil: PIL Image, template 图
    
    Returns:
        dict with all visualization images and detection results
    """
    # 1. PIL → Tensor → Resize 800×800
    query_pil = query_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    template_pil = template_pil.convert("RGB").resize((800, 800), Image.Resampling.BILINEAR)
    query_rgb = np.array(query_pil)
    template_rgb = np.array(template_pil)
    
    q_tensor = pil_to_tensor(query_pil).unsqueeze(0).to(device)  # [1,3,800,800]
    t_tensor = pil_to_tensor(template_pil).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        # 2. Backbone 提取特征
        qf, tf = backbone(q_tensor, t_tensor)
        
        # 3. OffsetPredictor → offsets
        offsets = predictor(qf, tf)
        
        # 4. FeatureWarp → warped template features
        warped = warp(tf, offsets)
        
        # 5. 差异特征 → MultiScaleFusion → 增强热力图
        diff_features = {k: warped[k] - qf[k] for k in ["C3", "C4", "C5"]}
        enhanced = fusion(diff_features)  # [1, 1, 800, 800]
        
        # 6. 3.5-A 标量热力图（用于可视化对比）
        diff_maps = compute_diff_maps(warped, qf)
        diff_upsampled = upsample_diff_maps(diff_maps)
    
    # 7. 候选框后处理
    enhanced_np = enhanced[0, 0].cpu().numpy()
    candidate_boxes = postprocess_heatmap(
        enhanced_np,
        min_area=min_area,
        max_aspect_ratio=5.0,
        adaptive_block=adaptive_block,
        adaptive_c=adaptive_c,
        morph_kernel=5,
    )
    
    # 8. 渲染热力图面板
    heatmap_panel_path = output_dir / "heatmap_panel.png"
    render_heatmap_panel(query_rgb, template_rgb, diff_upsampled, heatmap_panel_path)
    heatmap_panel_img = cv2.cvtColor(cv2.imread(str(heatmap_panel_path)), cv2.COLOR_BGR2RGB)
    
    # 9. 渲染 CBAM 增强热力图
    enhanced_heatmap_rgb = render_single_heatmap(enhanced_np)
    
    # 10. 绘制候选框叠加图
    candidate_overlay = draw_boxes(query_rgb, candidate_boxes, color=(255, 60, 60), thickness=2)
    
    # 11. 最终检测结果（目前用候选框作为最终结果，等 RT-DETR 训练后用实际分类结果）
    final_overlay = draw_boxes_with_labels(query_rgb, candidate_boxes, labels=["candidate"] * len(candidate_boxes), colors=CLASS_COLORS)
    
    # 12. 结果 JSON
    results_json = [
        {
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "area": float((x2-x1)*(y2-y1)),
        }
        for x1, y1, x2, y2 in candidate_boxes
    ]
    
    return {
        "heatmap_panel": heatmap_panel_img,
        "enhanced_heatmap": enhanced_heatmap_rgb,
        "candidate_overlay": candidate_overlay,
        "final_overlay": final_overlay,
        "candidate_count": len(candidate_boxes),
        "results_json": results_json,
    }
```

## 文件结构

```
src/demo/
  app.py           # Gradio 主入口
  __init__.py      # 空文件

artifacts/step3/demo/    # 运行时产物目录
```

如果检查到 `artifacts/step3/training/checkpoint_best.pt` 存在，自动加载训练后的 OffsetPredictor 权重；否则使用 zero-init。

## 输出要求

1. 创建 `src/demo/app.py`（Gradio 应用主文件）
2. 创建 `src/demo/__init__.py`（空）
3. 确保 demo 可以独立运行：`python src/demo/app.py`
4. 如果 gradio 未安装，先 `pip install gradio`
5. 测试 demo 启动（浏览器打开 http://127.0.0.1:7860），上传两张示例图测试
6. 提交所有新文件并推送 origin/main
7. 不要修改现有模块、配置、测试或 artifacts

## 注意事项

- Backbone 内部已做 ImageNet normalization，输入是 float32 [0,1] RGB
- 所有 pipeline 模块需 `.eval()` 并在 `torch.inference_mode()` 下运行
- 候选框需要在推理时动态计算（不在启动时预计算）
- 热力图渲染使用 OpenCV，注意 BGR/RGB 转换
- 界面响应可能较慢（3-5 秒），在推理按钮上加 loading 状态
- RT-DETR 模型权尚未集成，先用候选框作为最终检测结果（加注释标注未来替换点）
- demo 产物写入 `artifacts/step3/demo/`，该目录不需 gitignore
