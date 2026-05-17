from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(root: str | Path, recursive: bool = True) -> List[Path]:
    root = Path(root)
    iterator = root.rglob('*') if recursive else root.glob('*')
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def class_dirs(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir())


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
