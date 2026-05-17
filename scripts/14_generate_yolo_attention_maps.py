#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO classification activation-attention maps.

Default:
    data    = data/balanced
    weights = outputs/yolo_results
    output  = outputs/figures/yolo_attention
    models  = yolov8n-cls yolo11n-cls yolo26n-cls
    sizes   = 224 448 640
    images  = 5 per model-size setting
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO


SCRIPT_VERSION = "activation_attention_v2_no_grad_2026_05_13"

DEFAULT_MODELS = ["yolov8n-cls", "yolo11n-cls", "yolo26n-cls"]
DEFAULT_IMG_SIZES = [224, 448, 640]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(root: str | Path, recursive: bool = True) -> List[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in VALID_EXTENSIONS:
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def select_images(data_dir: str | Path, count: int, seed: int) -> List[Path]:
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    selected: List[Path] = []

    if data_dir.is_dir():
        class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        for class_dir in class_dirs:
            imgs = list_images(class_dir, recursive=False)
            if imgs:
                selected.append(imgs[int(rng.integers(0, len(imgs)))])
            if len(selected) >= count:
                return selected[:count]

    all_imgs = list_images(data_dir, recursive=True)
    remaining = [p for p in all_imgs if p not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_weights(weights: str | Path, model_name: str, img_size: int) -> Path:
    weights = Path(weights)

    if weights.is_file():
        if weights.suffix.lower() != ".pt":
            raise ValueError(f"Weights file must be a .pt file: {weights}")
        return weights

    if not weights.exists():
        raise FileNotFoundError(f"Weights path not found: {weights}")

    candidates = sorted(weights.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under: {weights}")

    model_key = normalize_token(model_name)
    size_key = str(img_size)

    def score(path: Path) -> Tuple[int, str]:
        p = normalize_token(str(path))
        s = 0
        if model_key in p:
            s += 100
        if size_key in p:
            s += 50
        if path.name.lower() == "best.pt":
            s += 20
        if "weights" in [part.lower() for part in path.parts]:
            s += 10
        if path.name.lower() == "last.pt":
            s += 5
        return s, str(path)

    selected = sorted(candidates, key=score, reverse=True)[0]
    print(f"[OK] Auto-selected weights: {selected}")
    return selected


def find_last_conv_layer(module: nn.Module) -> nn.Module:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the YOLO model.")
    return last_conv


class ForwardActivationExtractor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer = find_last_conv_layer(model)
        self.activation = None
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self.activation = output.detach()
        elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            self.activation = output[0].detach()
        else:
            self.activation = None

    def close(self):
        self.handle.remove()

    def make_map(self, x: torch.Tensor) -> np.ndarray:
        self.activation = None
        with torch.inference_mode():
            _ = self.model(x)

        if self.activation is None:
            raise RuntimeError("Activation hook did not capture a tensor.")

        fmap = self.activation[0]  # C,H,W
        att = fmap.abs().mean(dim=0)
        att = att - att.min()
        att = att / (att.max() + 1e-8)
        return att.cpu().numpy()


def preprocess_image(image_path: Path, img_size: int, device: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def predict_with_yolo(yolo: YOLO, image_path: Path) -> Tuple[str, float]:
    result = yolo.predict(str(image_path), verbose=False)[0]
    pred_idx = int(result.probs.top1)
    conf = float(result.probs.top1conf.item())
    names = getattr(yolo, "names", None)
    if isinstance(names, dict):
        pred_name = str(names.get(pred_idx, pred_idx))
    elif isinstance(names, list) and 0 <= pred_idx < len(names):
        pred_name = str(names[pred_idx])
    else:
        pred_name = str(pred_idx)
    return pred_name, conf


def save_figure(
    image_path: Path,
    attention: np.ndarray,
    output_path: Path,
    model_name: str,
    img_size: int,
    pred_name: str,
    conf: float,
    dpi: int,
) -> None:
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]

    att_t = torch.from_numpy(attention).float()[None, None]
    att_r = F.interpolate(att_t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    att_r = np.clip(att_r, 0, 1)

    heatmap = plt.get_cmap("jet")(att_r)[..., :3]
    overlay = np.clip(0.55 * img_np + 0.45 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    im = axes[1].imshow(att_r, cmap="jet")
    axes[1].set_title("Activation map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"{model_name} | {img_size}px | Pred: {pred_name} ({conf:.3f})",
        fontsize=12,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_setting(
    data: Path,
    weights_root: Path,
    output_root: Path,
    model_name: str,
    img_size: int,
    images_per_setting: int,
    seed: int,
    device: str,
    dpi: int,
) -> None:
    print("\n" + "=" * 80)
    print(f"[SETTING] Model={model_name} | Image size={img_size} | Method=forward_activation")
    print("=" * 80)

    weights_path = resolve_weights(weights_root, model_name, img_size)
    print(f"[INFO] Loading YOLO classification model: {weights_path}")

    yolo = YOLO(str(weights_path), task="classify")
    torch_model = yolo.model.to(device).float()
    torch_model.eval()

    extractor = ForwardActivationExtractor(torch_model)
    images = select_images(data, images_per_setting, seed)

    out_dir = ensure_dir(output_root / model_name / f"{img_size}px")

    for i, img_path in enumerate(images, 1):
        try:
            x = preprocess_image(img_path, img_size, device)
            attention = extractor.make_map(x)
            pred_name, conf = predict_with_yolo(yolo, img_path)

            out_name = f"{i:02d}_{img_path.parent.name}_{img_path.stem}_{model_name}_{img_size}px_forward_activation.png"
            out_path = out_dir / out_name

            save_figure(
                image_path=img_path,
                attention=attention,
                output_path=out_path,
                model_name=model_name,
                img_size=img_size,
                pred_name=pred_name,
                conf=conf,
                dpi=dpi,
            )

            print(f"[OK] {i}/{len(images)} saved: {out_path}")

        except Exception as exc:
            print(f"[ERROR] Failed for image {img_path}: {exc}")

    extractor.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO forward-activation attention maps.")
    parser.add_argument("--data", default="data/balanced")
    parser.add_argument("--weights", default="outputs/yolo_results")
    parser.add_argument("--output", default="outputs/figures/yolo_attention")

    parser.add_argument("--model", default=None, choices=DEFAULT_MODELS)
    parser.add_argument("--models", nargs="+", default=None, choices=DEFAULT_MODELS)

    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--img-sizes", nargs="+", type=int, default=None)

    parser.add_argument("--images-per-setting", "--max-images", dest="images_per_setting", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()

    models = [args.model] if args.model else (args.models if args.models else DEFAULT_MODELS)
    img_sizes = [args.img_size] if args.img_size else (args.img_sizes if args.img_sizes else DEFAULT_IMG_SIZES)

    data = Path(args.data)
    weights = Path(args.weights)
    output = Path(args.output)

    print(f"[INFO] Script version: {SCRIPT_VERSION}")
    print("[INFO] YOLO activation-attention generation started.")
    print(f"[INFO] Data: {data}")
    print(f"[INFO] Weights: {weights}")
    print(f"[INFO] Output: {output}")
    print(f"[INFO] Models: {models}")
    print(f"[INFO] Image sizes: {img_sizes}")
    print(f"[INFO] Images per setting: {args.images_per_setting}")
    print("[INFO] Method: forward_activation")
    print(f"[INFO] Device: {args.device}")

    for model_name in models:
        for img_size in img_sizes:
            process_setting(
                data=data,
                weights_root=weights,
                output_root=output,
                model_name=model_name,
                img_size=img_size,
                images_per_setting=args.images_per_setting,
                seed=args.seed,
                device=args.device,
                dpi=args.dpi,
            )

    print("\n[DONE] YOLO activation-attention maps saved to:")
    print(f"       {output.resolve()}")


if __name__ == "__main__":
    main()
