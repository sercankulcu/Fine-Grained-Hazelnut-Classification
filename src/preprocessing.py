from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Dict

from PIL import Image

from .utils import ensure_dir, list_images


def md5_file(path: str | Path, block_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            hasher.update(block)
    return hasher.hexdigest()


def copy_without_duplicates(input_root: str | Path, output_root: str | Path) -> Dict[str, int]:
    input_root = Path(input_root)
    output_root = ensure_dir(output_root)
    seen: Dict[str, Path] = {}
    copied = skipped = 0
    for src in list_images(input_root, recursive=True):
        h = md5_file(src)
        if h in seen:
            skipped += 1
            continue
        rel = src.relative_to(input_root)
        dst = output_root / rel
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        seen[h] = dst
        copied += 1
    return {'copied': copied, 'duplicates_skipped': skipped}


def resize_letterbox_image(img: Image.Image, target_size: int = 640, background=(0, 0, 0, 0)) -> Image.Image:
    """Resize to a square canvas while preserving aspect ratio.

    RGBA inputs keep transparency, matching the original behavior.
    RGB inputs are converted to RGBA so transparent rembg crops remain valid.
    """
    img = img.convert('RGBA')
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_size = (int(w * scale), int(h * scale))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', (target_size, target_size), background)
    offset = ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
    canvas.paste(resized, offset, resized)
    return canvas


def resize_tree_letterbox(input_root: str | Path, output_root: str | Path, size: int = 640) -> Dict[str, int]:
    input_root = Path(input_root)
    output_root = ensure_dir(output_root)
    count = 0
    for src in list_images(input_root, recursive=True):
        rel = src.relative_to(input_root)
        dst = (output_root / rel).with_suffix('.png')
        ensure_dir(dst.parent)
        with Image.open(src) as img:
            out = resize_letterbox_image(img, size)
            out.save(dst)
        count += 1
    return {'resized': count}



def crop_foreground_rembg_image(
    img: Image.Image,
    session,
    padding_percent: float = 10.0,
) -> Image.Image:
    """Remove background with rembg and crop the non-transparent foreground.

    This follows the original workflow used in the experiments:
    rembg background removal -> alpha-based tight crop -> PNG output.
    The output is RGBA. Later resize/letterbox keeps the object centered.
    """
    import numpy as np
    from rembg import remove

    rgba = remove(img, session=session, alpha_matting=False)
    arr = np.asarray(rgba)

    if arr.ndim != 3 or arr.shape[2] < 4:
        rgba = rgba.convert('RGBA')
        arr = np.asarray(rgba)

    alpha = arr[:, :, 3]
    coords = np.where(alpha > 0)
    if coords[0].size == 0:
        return rgba.convert('RGBA')

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    h, w = arr.shape[:2]
    pad = int(max(y_max - y_min, x_max - x_min) * padding_percent / 100.0)
    pad = max(pad, 5)

    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad + 1)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad + 1)

    cropped = arr[y_min:y_max, x_min:x_max]
    return Image.fromarray(cropped)


def crop_tree_foreground_rembg(
    input_root: str | Path,
    output_root: str | Path,
    padding_percent: float = 10.0,
    model_name: str = 'u2net',
) -> Dict[str, int]:
    """Apply the original rembg-based crop workflow recursively.

    Class subfolders are preserved. Output files are saved as PNG files named
    crop_<original_stem>.png, matching the original naming style.
    """
    from rembg import new_session

    input_root = Path(input_root)
    output_root = ensure_dir(output_root)

    print(f'[crop] Loading rembg model once: {model_name}', flush=True)
    session = new_session(model_name)
    print('[crop] rembg model is ready.', flush=True)

    count = failed = empty = 0
    for src in list_images(input_root, recursive=True):
        rel_parent = src.parent.relative_to(input_root)
        dst_dir = ensure_dir(output_root / rel_parent)
        dst = dst_dir / f'crop_{src.stem}.png'
        try:
            with Image.open(src) as img:
                out = crop_foreground_rembg_image(
                    img,
                    session=session,
                    padding_percent=padding_percent,
                )
                # Count empty alpha outputs, but still save them for inspection.
                import numpy as np
                arr = np.asarray(out.convert('RGBA'))
                if arr[:, :, 3].max() == 0:
                    empty += 1
                out.save(dst)
            count += 1
            print(f'[crop][OK] {src.relative_to(input_root)} -> {dst.relative_to(output_root)}', flush=True)
        except Exception as exc:
            failed += 1
            print(f'[crop][ERROR] {src}: {exc}', flush=True)
    return {'cropped': count, 'failed': failed, 'empty_alpha_outputs': empty}


def crop_foreground_image(
    img: Image.Image,
    padding_percent: float = 10.0,
    background=(0, 0, 0),
    sat_min: int = 60,
    val_min: int = 40,
) -> Image.Image:
    """Crop the hazelnut foreground and place it on a black RGB background.

    This reproduces the preprocessing style used in the experiments: the object is
    tightly cropped, the background is suppressed, and the remaining image is saved
    as a normal RGB image with a black background.
    """
    import cv2
    import numpy as np
    from .segmentation import segment_hazelnut

    rgb = img.convert('RGB')
    arr_rgb = np.asarray(rgb)
    img_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    mask = segment_hazelnut(img_bgr, sat_min=sat_min, val_min=val_min)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgb

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    h, w = arr_rgb.shape[:2]
    obj_size = max(x_max - x_min + 1, y_max - y_min + 1)
    pad = max(int(obj_size * padding_percent / 100.0), 5)

    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad + 1)
    y_max = min(h, y_max + pad + 1)

    crop_rgb = arr_rgb[y_min:y_max, x_min:x_max].copy()
    crop_mask = mask[y_min:y_max, x_min:x_max]

    bg = np.zeros_like(crop_rgb)
    bg[:, :] = background
    bg[crop_mask > 0] = crop_rgb[crop_mask > 0]
    return Image.fromarray(bg)


def crop_tree_foreground(
    input_root: str | Path,
    output_root: str | Path,
    padding_percent: float = 10.0,
    background=(0, 0, 0),
    sat_min: int = 60,
    val_min: int = 40,
) -> Dict[str, int]:
    input_root = Path(input_root)
    output_root = ensure_dir(output_root)
    count = failed = 0
    for src in list_images(input_root, recursive=True):
        rel = src.relative_to(input_root)
        dst = (output_root / rel).with_suffix('.jpg')
        ensure_dir(dst.parent)
        try:
            with Image.open(src) as img:
                out = crop_foreground_image(
                    img,
                    padding_percent=padding_percent,
                    background=background,
                    sat_min=sat_min,
                    val_min=val_min,
                )
                out.save(dst, quality=95)
            count += 1
            if count % 100 == 0:
                print(f'[crop] processed {count} images...', flush=True)
        except Exception as exc:
            failed += 1
            print(f'[crop][ERROR] {src}: {exc}', flush=True)
    return {'cropped': count, 'failed': failed}
