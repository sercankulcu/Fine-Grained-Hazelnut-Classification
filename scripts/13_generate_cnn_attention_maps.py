#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate CNN Grad-CAM attention maps for hazelnut classification models.

Default behavior:
    - processes 5 CNN models
    - processes 3 image resolutions: 224, 448, 640
    - processes 5 images per model-resolution pair

"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cnn_training import MODEL_MAP
from src.utils import ensure_dir, list_images


SCRIPT_VERSION = "cnn_gradcam_v2_safe_tensor_hook_2026_05_15"

DEFAULT_MODELS = [
    "ResNet50",
    "DenseNet121",
    "EfficientNetB2",
    "ConvNeXt",
    "MobileNetV3",
]

DEFAULT_IMG_SIZES = [224, 448, 640]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# -----------------------------------------------------------------------------
# Model and Grad-CAM utilities
# -----------------------------------------------------------------------------
def get_target_layer(model: torch.nn.Module, model_name: str):
    """Return a better target layer for Grad-CAM.

    The selected layer should preserve spatial information.
    For MobileNetV3, conv_head may produce weak or almost uniform Grad-CAM maps.
    Therefore, an earlier feature block is preferred.
    """

    if model_name == "ResNet50":
        return model.layer4[-1]

    if model_name == "DenseNet121":
        return model.features.denseblock4

    if model_name == "EfficientNetB2":
        # Better than conv_head for spatial attention.
        return model.blocks[-1]

    if model_name == "MobileNetV3":
        # conv_head often gives poor Grad-CAM maps.
        # Use the last feature block before the head.
        return model.blocks[-1]

    if model_name == "ConvNeXt":
        return model.stages[-1].blocks[-1]

    raise ValueError(f"No target layer defined for model: {model_name}")


class GradCAM:
    """Grad-CAM implementation using tensor hooks.

    This version avoids register_full_backward_hook because MobileNetV3 and
    some timm models may use inplace operations that conflict with backward
    hooks during Grad-CAM.
    """

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Grad-CAM target layer output is not a tensor.")

        self.activations = output.detach().clone()

        def _save_gradient(grad):
            self.gradients = grad.detach().clone()

        output.register_hook(_save_gradient)

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())

        if class_idx is None:
            class_idx = pred_idx

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations or gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=0))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), pred_idx, confidence

    def close(self) -> None:
        self.forward_handle.remove()


def disable_inplace_operations(model: torch.nn.Module) -> None:
    """Disable inplace operations to make Grad-CAM safer."""
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False


def build_model(
    model_name: str,
    num_classes: int,
    weights_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Create a timm model and load saved weights."""
    model = timm.create_model(
        MODEL_MAP[model_name],
        pretrained=False,
        num_classes=num_classes,
    )

    disable_inplace_operations(model)

    state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    clean_state = {}
    for key, value in state.items():
        clean_key = key.replace("module.", "")
        clean_state[clean_key] = value

    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()

    return model


def get_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


# -----------------------------------------------------------------------------
# File selection utilities
# -----------------------------------------------------------------------------
def normalize_name(text: str) -> str:
    return (
        text.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
    )


def find_weights_file(weights: Path, model_name: str, img_size: int) -> Path:
    """Resolve a weight file from either a direct .pth path or a directory."""
    if weights.is_file():
        return weights

    if not weights.exists():
        raise FileNotFoundError(f"Weights path not found: {weights}")

    if not weights.is_dir():
        raise ValueError(f"Weights path must be a .pth file or directory: {weights}")

    pth_files = sorted(weights.rglob("*.pth"))

    if not pth_files:
        raise FileNotFoundError(f"No .pth files found under: {weights}")

    model_key = normalize_name(model_name)
    size_key = str(img_size)

    aliases: Dict[str, List[str]] = {
        "EfficientNetB2": ["effnetb2", "efficientnetb2", "efficientnet_b2"],
        "MobileNetV3": ["mobilenetv3", "mobilenet_v3"],
        "DenseNet121": ["densenet121"],
        "ResNet50": ["resnet50"],
        "ConvNeXt": ["convnext"],
    }

    candidates = []

    for file in pth_files:
        name_key = normalize_name(file.stem)

        model_match = model_key in name_key
        alias_match = any(alias in name_key for alias in aliases.get(model_name, []))
        size_match = size_key in name_key

        if size_match and (model_match or alias_match):
            candidates.append(file)

    if not candidates:
        available = "\n".join(str(p) for p in pth_files[:40])
        raise FileNotFoundError(
            f"No weight file found for model={model_name}, img_size={img_size} under {weights}.\n"
            f"Available .pth files include:\n{available}"
        )

    best_candidates = [p for p in candidates if "best" in p.stem.lower()]
    selected = best_candidates[0] if best_candidates else candidates[0]

    return selected


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_class_names(data_root: Path) -> Optional[List[str]]:
    """Return class names if data_root is an ImageFolder-style dataset."""
    if not data_root.is_dir():
        return None

    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

    class_dirs = [
        p
        for p in class_dirs
        if any(is_image_file(x) for x in p.iterdir() if x.is_file())
    ]

    if not class_dirs:
        return None

    return [p.name for p in class_dirs]


def select_images(data_root: Path, max_images: int, seed: int) -> List[Path]:
    """Select representative images from a dataset root, image folder, or image file."""
    if data_root.is_file():
        if is_image_file(data_root):
            return [data_root]
        raise ValueError(f"Input file is not a supported image: {data_root}")

    if not data_root.exists():
        raise FileNotFoundError(f"Data path not found: {data_root}")

    rng = random.Random(seed)

    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    class_dirs = [
        p
        for p in class_dirs
        if any(is_image_file(x) for x in p.iterdir() if x.is_file())
    ]

    if not class_dirs:
        images = list_images(data_root, recursive=True)
        images = [Path(p) for p in images if Path(p).suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(images)
        return images[:max_images]

    selected: List[Path] = []
    per_class = max(1, int(np.ceil(max_images / len(class_dirs))))

    for class_dir in class_dirs:
        images = [
            p
            for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        rng.shuffle(images)
        selected.extend(images[:per_class])

    rng.shuffle(selected)

    return selected[:max_images]


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def save_overlay(
    image_path: Path,
    cam: np.ndarray,
    out_path: Path,
    model_name: str,
    img_size: int,
    pred_idx: int,
    confidence: float,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    h, w = img.shape[:2]

    cam_tensor = torch.from_numpy(cam).float()[None, None]
    cam_resized = F.interpolate(
        cam_tensor,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap = plt.get_cmap("jet")(cam_resized)[..., :3]
    overlay = np.clip(0.55 * img + 0.45 * heatmap, 0, 1)

    pred_label = str(pred_idx)

    if class_names and 0 <= pred_idx < len(class_names):
        pred_label = class_names[pred_idx]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"{model_name} | {img_size}px | Pred: {pred_label} | Conf: {confidence:.3f}",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def safe_file_stem(text: str) -> str:
    invalid = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for ch in invalid:
        text = text.replace(ch, "_")
    return text


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
def process_setting(
    data: Path,
    weights: Path,
    output: Path,
    model_name: str,
    img_size: int,
    num_classes: int,
    images_per_setting: int,
    seed: int,
    device: torch.device,
) -> None:
    print("=" * 90)
    print(f"[START] Model={model_name} | Image size={img_size} | Images={images_per_setting}")

    weights_path = find_weights_file(weights, model_name, img_size)
    print(f"[WEIGHTS] {weights_path}")

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        weights_path=weights_path,
        device=device,
    )

    target_layer = get_target_layer(model, model_name)
    cam_extractor = GradCAM(model, target_layer)

    transform = get_transform(img_size)
    class_names = get_class_names(data)

    images = select_images(data, images_per_setting, seed)

    if not images:
        print(f"[WARN] No images found under: {data}")
        cam_extractor.close()
        del model
        return

    out_dir = ensure_dir(output / model_name / f"{img_size}px")

    for index, image_path in enumerate(images, start=1):
        try:
            img = Image.open(image_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            cam, pred_idx, confidence = cam_extractor(x)

            safe_name = safe_file_stem(
                f"{index:02d}_{image_path.parent.name}_{image_path.stem}_{model_name}_{img_size}px_gradcam.png"
            )

            out_path = out_dir / safe_name

            save_overlay(
                image_path=image_path,
                cam=cam,
                out_path=out_path,
                model_name=model_name,
                img_size=img_size,
                pred_idx=pred_idx,
                confidence=confidence,
                class_names=class_names,
            )

            print(f"[OK] {index}/{len(images)} saved: {out_path}")

        except Exception as exc:
            print(f"[SKIP] {image_path} -> {exc}")

    cam_extractor.close()

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[DONE] Model={model_name} | Image size={img_size}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CNN Grad-CAM attention maps."
    )

    parser.add_argument(
        "--data",
        default="data/balanced",
        help="Dataset root, image folder, or single image path. Default: data/balanced",
    )

    parser.add_argument(
        "--weights",
        default="outputs/cnn_results/models",
        help="Path to a .pth file or directory containing .pth weights. Default: outputs/cnn_results/models",
    )

    parser.add_argument(
        "--output",
        default="outputs/figures/cnn_attention",
        help="Output directory. Default: outputs/figures/cnn_attention",
    )

    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_MAP.keys()),
        help="Optional single-model mode. If omitted, all default models are processed.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_MAP.keys()),
        help="Optional list of models. If omitted, all default models are processed.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Optional single-resolution mode. If omitted, 224, 448, and 640 are processed.",
    )

    parser.add_argument(
        "--img-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of image sizes. If omitted, 224, 448, and 640 are processed.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=8,
        help="Number of output classes. Default: 8",
    )

    parser.add_argument(
        "--images-per-setting",
        "--max-images",
        dest="images_per_setting",
        type=int,
        default=5,
        help="Number of images per model-resolution pair. Default: 5",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection. Default: 42",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device, e.g. cuda, cpu, cuda:0. Default: auto",
    )

    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.model is not None:
        return [args.model]

    if args.models is not None:
        return args.models

    return DEFAULT_MODELS


def resolve_img_sizes(args: argparse.Namespace) -> List[int]:
    if args.img_size is not None:
        return [args.img_size]

    if args.img_sizes is not None:
        return args.img_sizes

    return DEFAULT_IMG_SIZES


def main() -> None:
    args = parse_args()

    data = Path(args.data)
    weights = Path(args.weights)
    output = Path(args.output)

    if not data.exists():
        raise FileNotFoundError(f"Data path not found: {data}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    models = resolve_models(args)
    img_sizes = resolve_img_sizes(args)

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print("[INFO] CNN Grad-CAM attention map generation")
    print(f"[INFO] Data: {data}")
    print(f"[INFO] Weights: {weights}")
    print(f"[INFO] Output: {output}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Models: {models}")
    print(f"[INFO] Image sizes: {img_sizes}")
    print(f"[INFO] Images per setting: {args.images_per_setting}")

    for model_name in models:
        for img_size in img_sizes:
            process_setting(
                data=data,
                weights=weights,
                output=output,
                model_name=model_name,
                img_size=img_size,
                num_classes=args.num_classes,
                images_per_setting=args.images_per_setting,
                seed=args.seed,
                device=device,
            )

    print("\n[DONE] CNN Grad-CAM attention maps saved to:")
    print(f"       {output.resolve()}")


if __name__ == "__main__":
    main()