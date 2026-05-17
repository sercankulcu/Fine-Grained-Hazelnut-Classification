from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict

import cv2
import albumentations as A

from .utils import ensure_dir, list_images, class_dirs, set_seed


def build_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.30),
        A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101, p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.25),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=6, val_shift_limit=6, p=0.20),
    ])


def balance_dataset(input_root: str | Path, output_root: str | Path, target_count: int = 1000, seed: int = 42) -> Dict[str, int]:
    set_seed(seed)
    random.seed(seed)
    input_root = Path(input_root)
    output_root = ensure_dir(output_root)
    transform = build_transform()
    summary = {}

    for class_dir in class_dirs(input_root):
        out_class = ensure_dir(output_root / class_dir.name)
        images = list_images(class_dir, recursive=False)
        for src in images:
            shutil.copy2(src, out_class / src.name)
        existing = list_images(out_class, recursive=False)
        needed = max(0, target_count - len(existing))
        if not existing:
            summary[class_dir.name] = 0
            continue
        for i in range(needed):
            src = random.choice(existing)
            img = cv2.imread(str(src))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug_rgb = transform(image=img_rgb)['image']
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
            out_path = out_class / f'aug_{i + 1:05d}_{src.stem}.jpg'
            cv2.imwrite(str(out_path), aug_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        summary[class_dir.name] = len(list_images(out_class, recursive=False))
    return summary
