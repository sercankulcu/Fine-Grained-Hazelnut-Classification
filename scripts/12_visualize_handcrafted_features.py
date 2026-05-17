#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script generates publication-ready visual explanations of the handcrafted
feature extraction pipeline used for hazelnut cultivar classification.

It can process either:
    1. A single image, or
    2. A dataset directory with class subfolders.

For each selected image, the script creates a multi-panel figure containing:
    (a) Original image
    (b) Segmentation mask
    (c) Contour and oriented bounding box (OBB)
    (d) LBP texture map
    (e) HSV color histograms
    (f) Representative shape descriptors
    (g) Representative GLCM texture descriptors
    (h) Masked foreground image

The implementation follows the same HSV-based segmentation logic used in the
handcrafted feature extraction pipeline.

"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
HSV_HIST_BINS = 18


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image_file(path: Path) -> bool:
    """Return True if the path has a supported image extension."""
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    """List image files under a directory."""
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if is_image_file(p))


def get_class_dirs(dataset_dir: Path) -> List[Path]:
    """Return sorted class directories from an ImageFolder-style dataset."""
    return sorted(p for p in dataset_dir.iterdir() if p.is_dir())


def safe_name(text: str) -> str:
    """Create a filesystem-friendly name."""
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# -----------------------------------------------------------------------------
# Segmentation utilities
# -----------------------------------------------------------------------------
def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = np.zeros_like(mask)
    clean[labels == largest_label] = 255
    return clean


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside the foreground object using flood fill."""
    flood_filled = mask.copy()
    cv2.floodFill(flood_filled, None, (0, 0), 255)
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    return mask | flood_filled_inv


def segment_hazelnut(
    img_bgr: np.ndarray,
    saturation_min: int = 60,
    value_min: int = 40,
    kernel_size: int = 11,
    close_iterations: int = 2,
    open_iterations: int = 1,
) -> np.ndarray:
    """
    Segment the hazelnut foreground using HSV saturation and value thresholds.

    Parameters
    ----------
    img_bgr:
        Input image in OpenCV BGR format.
    saturation_min:
        Lower threshold for the HSV saturation channel.
    value_min:
        Lower threshold for the HSV value channel.
    kernel_size:
        Morphological kernel size.
    close_iterations:
        Number of closing iterations.
    open_iterations:
        Number of opening iterations.

    Returns
    -------
    np.ndarray
        Binary foreground mask with values 0 and 255.
    """
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)

    mask_sat = cv2.inRange(saturation, saturation_min, 255)
    mask_val = cv2.inRange(value, value_min, 255)
    mask = cv2.bitwise_and(mask_sat, mask_val)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    mask = keep_largest_component(mask)
    mask = fill_internal_holes(mask)
    return mask


def get_main_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external contour from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def count_holes(mask: np.ndarray) -> int:
    """Count internal holes in a binary mask."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(sum(1 for i in range(len(contours)) if hierarchy[0][i][3] != -1))


# -----------------------------------------------------------------------------
# Feature computations for visualization
# -----------------------------------------------------------------------------
def compute_shape_descriptors(contour: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute representative shape descriptors for display."""
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    angle = float(rect[2])
    width, height = float(width), float(height)
    if width < height:
        width, height = height, width

    aspect_ratio = width / (height + 1e-8)
    circularity = 4.0 * np.pi * area / (perimeter**2 + 1e-8)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    solidity = area / (hull_area + 1e-8)
    convexity = hull_perimeter / (perimeter + 1e-8)
    compactness = perimeter**2 / (4.0 * np.pi * area + 1e-8)
    elongation = ((width - height) ** 2) / (width * height + 1e-8)

    (_, _), radius = cv2.minEnclosingCircle(contour)
    enclosing_circle_ratio = area / (np.pi * radius**2 + 1e-8)

    normalized_area = area / (width * height + 1e-8)
    normalized_perimeter = perimeter / (np.sqrt(area) + 1e-8)

    epsilon = 0.015 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approximation_ratio = len(approx) / (len(contour) + 1e-8)

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    defects = None
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error:
            defects = None

    if defects is not None and len(defects) > 0:
        defect_count = int(len(defects))
        mean_defect_depth = float(np.mean(defects[:, 0, 3]) / 256.0)
    else:
        defect_count = 0
        mean_defect_depth = 0.0

    return {
        "Area": area,
        "Perimeter": perimeter,
        "OBB width": width,
        "OBB height": height,
        "Aspect ratio": aspect_ratio,
        "Circularity": circularity,
        "Solidity": solidity,
        "Convexity": convexity,
        "Compactness": compactness,
        "Elongation": elongation,
        "Enclosing circle ratio": enclosing_circle_ratio,
        "Normalized area": normalized_area,
        "Normalized perimeter": normalized_perimeter,
        "Approx. ratio": approximation_ratio,
        "Defect count": defect_count,
        "Mean defect depth": mean_defect_depth,
        "Hole count": float(count_holes(mask)),
        "OBB angle": angle,
    }


def compute_hu_moments(contour: np.ndarray) -> Dict[str, float]:
    """Compute log-scaled Hu invariant moments."""
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return {f"Hu {i + 1}": float(value) for i, value in enumerate(hu_log)}


def compute_lbp_map(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute LBP texture map and hide the background with NaNs."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    lbp_vis = lbp.astype(np.float32)
    lbp_vis[mask == 0] = np.nan
    return lbp_vis


def compute_lbp_histogram(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute normalized uniform-LBP histogram over the foreground region."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    lbp_pixels = lbp[mask > 0]
    if len(lbp_pixels) == 0:
        return np.zeros(LBP_POINTS + 2, dtype=np.float32)

    hist, _ = np.histogram(
        lbp_pixels.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        density=True,
    )
    return hist.astype(np.float32)


def compute_hsv_statistics(img_bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute HSV mean and standard deviation over the foreground."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        return {
            "H mean": 0.0,
            "S mean": 0.0,
            "V mean": 0.0,
            "H std": 0.0,
            "S std": 0.0,
            "V std": 0.0,
        }
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)
    return {
        "H mean": float(mean[0]),
        "S mean": float(mean[1]),
        "V mean": float(mean[2]),
        "H std": float(std[0]),
        "S std": float(std[1]),
        "V std": float(std[2]),
    }


def compute_hsv_histograms(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    bins: int = HSV_HIST_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute normalized H, S, V histograms and dark-value ratio."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 255]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 255]).flatten()

    h_hist = h_hist / (h_hist.sum() + 1e-8)
    s_hist = s_hist / (s_hist.sum() + 1e-8)
    v_hist = v_hist / (v_hist.sum() + 1e-8)

    low_bins = max(1, int(bins * 0.25))
    dark_value_ratio = float(np.sum(v_hist[:low_bins]))
    return h_hist, s_hist, v_hist, dark_value_ratio


def compute_glcm_summary(img_bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute averaged GLCM descriptors over distances and orientations."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    masked_gray = gray.copy()
    masked_gray[mask == 0] = 0

    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        masked_gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "homogeneity", "energy", "correlation"]
    return {prop.capitalize(): float(np.mean(graycoprops(glcm, prop))) for prop in props}


def make_contour_obb_overlay(img_bgr: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Create RGB image with green contour and red oriented bounding box."""
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 4)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(overlay, [box], 0, (0, 0, 255), 4)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def make_masked_foreground(img_rgb: np.ndarray, mask: np.ndarray, background: str = "white") -> np.ndarray:
    """Return foreground-only RGB image on a white or black background."""
    if background == "black":
        result = np.zeros_like(img_rgb)
    else:
        result = np.full_like(img_rgb, 255)
    result[mask > 0] = img_rgb[mask > 0]
    return result


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def format_value(value: float) -> str:
    """Format descriptor values for text panels."""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if abs(float(value)) >= 100:
        return f"{float(value):.1f}"
    return f"{float(value):.4f}"


def add_text_panel(ax, title: str, values: Dict[str, float], max_items: int = 10) -> None:
    """Render a descriptor dictionary as a text panel."""
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    items = list(values.items())[:max_items]
    text = "\n".join(f"{key}: {format_value(value)}" for key, value in items)
    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )


def plot_hsv_histograms(ax, h_hist: np.ndarray, s_hist: np.ndarray, v_hist: np.ndarray) -> None:
    """Plot HSV histograms in one axis."""
    x = np.arange(len(h_hist))
    ax.plot(x, h_hist, marker="o", linewidth=1.5, label="H")
    ax.plot(x, s_hist, marker="s", linewidth=1.5, label="S")
    ax.plot(x, v_hist, marker="^", linewidth=1.5, label="V")
    ax.set_title("(e) HSV color histograms", fontsize=12, fontweight="bold")
    ax.set_xlabel("Histogram bin")
    ax.set_ylabel("Normalized frequency")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)


def plot_lbp_histogram(ax, hist: np.ndarray) -> None:
    """Plot the LBP histogram."""
    ax.bar(np.arange(len(hist)), hist, width=0.8)
    ax.set_title("(i) LBP histogram", fontsize=12, fontweight="bold")
    ax.set_xlabel("Uniform LBP bin")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------------
def create_feature_visualization(
    image_path: Path,
    output_path: Path,
    dpi: int = 300,
    show: bool = False,
    save_pdf: bool = True,
    background: str = "white",
) -> bool:
    """Create a multi-panel handcrafted feature visualization for one image."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[SKIP] Could not read image: {image_path}")
        return False

    mask = segment_hazelnut(img_bgr)
    contour = get_main_contour(mask)
    if contour is None or len(contour) < 5:
        print(f"[SKIP] Could not find a valid contour: {image_path}")
        return False

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = make_contour_obb_overlay(img_bgr, contour)
    foreground_rgb = make_masked_foreground(img_rgb, mask, background=background)

    lbp_map = compute_lbp_map(img_bgr, mask)
    lbp_hist = compute_lbp_histogram(img_bgr, mask)
    h_hist, s_hist, v_hist, dark_ratio = compute_hsv_histograms(img_bgr, mask)
    hsv_stats = compute_hsv_statistics(img_bgr, mask)
    shape_values = compute_shape_descriptors(contour, mask)
    hu_values = compute_hu_moments(contour)
    glcm_values = compute_glcm_summary(img_bgr, mask)

    # 3 x 4 layout. This is broader than the manuscript-only 2 x 4 layout.
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("(a) Original image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("(b) Segmentation mask", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(overlay_rgb)
    axes[0, 2].set_title("(c) Contour and OBB", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    im = axes[0, 3].imshow(lbp_map, cmap="viridis")
    axes[0, 3].set_title("(d) LBP texture map", fontsize=12, fontweight="bold")
    axes[0, 3].axis("off")
    cbar = fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plot_hsv_histograms(axes[1, 0], h_hist, s_hist, v_hist)

    selected_shape = {
        "Area": shape_values["Area"],
        "Perimeter": shape_values["Perimeter"],
        "Aspect ratio": shape_values["Aspect ratio"],
        "Circularity": shape_values["Circularity"],
        "Solidity": shape_values["Solidity"],
        "Compactness": shape_values["Compactness"],
        "Elongation": shape_values["Elongation"],
        "Defect count": shape_values["Defect count"],
        "Mean defect depth": shape_values["Mean defect depth"],
        "Hole count": shape_values["Hole count"],
    }
    add_text_panel(axes[1, 1], "(f) Shape descriptors", selected_shape, max_items=10)

    add_text_panel(axes[1, 2], "(g) GLCM texture summary", glcm_values, max_items=4)

    axes[1, 3].imshow(foreground_rgb)
    axes[1, 3].set_title("(h) Masked foreground", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")

    plot_lbp_histogram(axes[2, 0], lbp_hist)

    hsv_panel_values = dict(hsv_stats)
    hsv_panel_values["Dark V ratio"] = dark_ratio
    add_text_panel(axes[2, 1], "(j) HSV statistics", hsv_panel_values, max_items=7)

    add_text_panel(axes[2, 2], "(k) Hu invariant moments", hu_values, max_items=7)

    geometry_values = {
        "OBB width": shape_values["OBB width"],
        "OBB height": shape_values["OBB height"],
        "OBB angle": shape_values["OBB angle"],
        "Convexity": shape_values["Convexity"],
        "Encl. circle ratio": shape_values["Enclosing circle ratio"],
        "Norm. area": shape_values["Normalized area"],
        "Norm. perimeter": shape_values["Normalized perimeter"],
        "Approx. ratio": shape_values["Approx. ratio"],
    }
    add_text_panel(axes[2, 3], "(l) Geometry summary", geometry_values, max_items=8)

    fig.suptitle(
        f"Handcrafted feature visualization: {image_path.parent.name} / {image_path.name}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved PNG: {output_path}")

    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[OK] Saved PDF: {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return True


def select_representative_images(
    dataset_dir: Path,
    samples_per_class: int,
    seed: int,
    max_images: Optional[int] = None,
) -> List[Path]:
    """Select representative images from each class folder."""
    rng = random.Random(seed)
    selected: List[Path] = []

    class_dirs = get_class_dirs(dataset_dir)
    if not class_dirs:
        return list_images(dataset_dir, recursive=True)[: max_images or None]

    for class_dir in class_dirs:
        images = list_images(class_dir, recursive=False)
        if not images:
            print(f"[WARN] No images found in class folder: {class_dir}")
            continue

        n = min(samples_per_class, len(images))
        selected.extend(rng.sample(images, n))

    if max_images is not None:
        selected = selected[:max_images]
    return selected


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate broad handcrafted-feature visualization figures for "
            "hazelnut classification images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", type=str, default=None, help="Dataset root or single image path.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for figures.")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process.")

    parser.add_argument("--image", type=str, default=None, help="Single image path. Overrides --input.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset root with class subfolders.")
    parser.add_argument("--output_dir", type=str, default=None, help="Alias for --output.")
    parser.add_argument("--samples_per_class", "--samples-per-class", dest="samples_per_class", type=int, default=1)

    parser.add_argument("--features", type=str, default=None, help="Optional. Accepted for README compatibility; not used.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved PNG figures.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection.")
    parser.add_argument("--no-pdf", action="store_true", help="Do not save PDF copies.")
    parser.add_argument("--background", choices=["white", "black"], default="white", help="Background color for masked foreground panel.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively while saving.")

    args = parser.parse_args(argv)

    if args.output_dir is not None:
        args.output = args.output_dir

    if args.image is None and args.input is None and args.dataset_dir is None:
        parser.error("Provide --input, --image, or --dataset_dir.")

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = ensure_dir(Path(args.output))

    if args.features is not None:
        print(f"[INFO] --features was provided but is not required for this visualization: {args.features}")

    # Single-image mode has priority.
    single_image = args.image or (args.input if args.input and Path(args.input).is_file() else None)
    if single_image is not None:
        image_path = Path(single_image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        class_name = safe_name(image_path.parent.name or "single_image")
        output_path = output_dir / class_name / f"{safe_name(image_path.stem)}_handcrafted_features.png"
        ok = create_feature_visualization(
            image_path=image_path,
            output_path=output_path,
            dpi=args.dpi,
            show=args.show,
            save_pdf=not args.no_pdf,
            background=args.background,
        )
        if not ok:
            sys.exit(1)
        return

    dataset_root = Path(args.dataset_dir or args.input)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    images = select_representative_images(
        dataset_dir=dataset_root,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        max_images=args.max_images,
    )

    if not images:
        print("[ERROR] No valid images found.")
        sys.exit(1)

    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Images selected: {len(images)}")
    print(f"[INFO] Output directory: {output_dir}")

    success_count = 0
    for index, image_path in enumerate(images, start=1):
        class_name = safe_name(image_path.parent.name or "unknown_class")
        output_path = output_dir / class_name / f"{safe_name(image_path.stem)}_handcrafted_features.png"
        print(f"\n[{index}/{len(images)}] Processing: {image_path}")
        if create_feature_visualization(
            image_path=image_path,
            output_path=output_path,
            dpi=args.dpi,
            show=args.show,
            save_pdf=not args.no_pdf,
            background=args.background,
        ):
            success_count += 1

    print("\nDone.")
    print(f"Successful figures: {success_count}/{len(images)}")
    print(f"Figures saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
