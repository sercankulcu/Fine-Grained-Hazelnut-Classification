from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .segmentation import count_holes, draw_contour_and_obb, main_contour, segment_hazelnut
from .utils import ensure_dir, list_images, class_dirs

FEATURE_NAMES = (
    ['area', 'perimeter', 'obb_width', 'obb_height', 'aspect_ratio', 'circularity',
     'solidity', 'convexity', 'compactness', 'elongation', 'enclosing_circle_ratio',
     'norm_area', 'norm_perimeter', 'obb_angle'] +
    [f'hu_{i}' for i in range(1, 8)] +
    ['approx_ratio', 'defect_count', 'mean_defect_depth'] +
    ['mean_h', 'mean_s', 'mean_v', 'std_h', 'std_s', 'std_v'] +
    [f'h_hist_{i}' for i in range(18)] +
    [f's_hist_{i}' for i in range(18)] +
    [f'v_hist_{i}' for i in range(18)] +
    ['dark_v_ratio'] +
    [f'glcm_{p}_{d}_{a}'
     for p in ['contrast', 'homogeneity', 'energy', 'correlation']
     for d in [1, 3, 5]
     for a in ['0', '45', '90', '135']] +
    [f'lbp_{i}' for i in range(26)] +
    ['hole_count']
)


def shape_features(cnt) -> np.ndarray:
    if cnt is None or len(cnt) < 5:
        return np.zeros(21)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1e-6:
        return np.zeros(21)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    angle = rect[2]
    if w < h:
        w, h = h, w
    aspect_ratio = w / (h + 1e-6)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)
    convexity = cv2.arcLength(hull, True) / (perimeter + 1e-6)
    compactness = perimeter ** 2 / (4 * np.pi * area + 1e-6)
    elongation = (w - h) ** 2 / (w * h + 1e-6) if w * h > 0 else 0
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    enclosing_circle_ratio = area / (np.pi * radius ** 2 + 1e-6)
    norm_area = area / (w * h + 1e-6) if w * h > 0 else 0
    norm_perimeter = perimeter / np.sqrt(area + 1e-6)
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return np.hstack([area, perimeter, w, h, aspect_ratio, circularity, solidity, convexity,
                      compactness, elongation, enclosing_circle_ratio, norm_area, norm_perimeter,
                      angle, hu_log])


def contour_complexity(cnt) -> np.ndarray:
    if cnt is None or len(cnt) < 5:
        return np.zeros(3)
    epsilon = 0.015 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx_ratio = len(approx) / len(cnt) if len(cnt) > 0 else 0
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    if defects is not None and len(defects) > 0:
        return np.array([len(approx) / len(cnt), len(defects), np.mean(defects[:, 0, 3]) / 256.0])
    return np.array([approx_ratio, 0, 0])


def color_features(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        return np.zeros(6)
    return np.hstack([np.mean(pixels, axis=0), np.std(pixels, axis=0)])


def color_histogram_features(hsv: np.ndarray, mask: np.ndarray, bins: int = 18) -> np.ndarray:
    if np.sum(mask) == 0:
        return np.zeros(bins * 3 + 1)
    h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 255]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 255]).flatten()
    h_hist /= h_hist.sum() + 1e-8
    s_hist /= s_hist.sum() + 1e-8
    v_hist /= v_hist.sum() + 1e-8
    dark_v_ratio = np.sum(v_hist[:max(1, int(bins * 0.25))])
    return np.hstack([h_hist, s_hist, v_hist, dark_v_ratio])


def glcm_features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.sum(mask) == 0:
        return np.zeros(48)
    masked_gray = gray.copy()
    masked_gray[mask == 0] = 0
    glcm = graycomatrix(masked_gray, distances=[1, 3, 5],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        feats.extend(graycoprops(glcm, prop).flatten())
    return np.array(feats)


def lbp_features(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    pixels = lbp[mask > 0]
    if len(pixels) == 0:
        return np.zeros(n_points + 2)
    hist, _ = np.histogram(pixels.ravel(), bins=np.arange(0, n_points + 3), density=True)
    return hist


def extract_one(image_path: str | Path, debug_path: Optional[str | Path] = None) -> Optional[np.ndarray]:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    mask = segment_hazelnut(img)
    cnt = main_contour(mask)
    if cnt is None:
        return None
    if debug_path is not None:
        ensure_dir(Path(debug_path).parent)
        cv2.imwrite(str(debug_path), draw_contour_and_obb(img, cnt))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.hstack([
        shape_features(cnt), contour_complexity(cnt), color_features(img, mask),
        color_histogram_features(hsv, mask), glcm_features(gray, mask),
        lbp_features(img, mask), np.array([count_holes(mask)])
    ])


def extract_dataset(input_root: str | Path, output_csv: str | Path, debug_dir: Optional[str | Path] = None) -> pd.DataFrame:
    rows, labels, image_names, paths = [], [], [], []
    input_root = Path(input_root)
    for cdir in class_dirs(input_root):
        for img_path in list_images(cdir, recursive=False):
            debug_path = None
            if debug_dir is not None:
                debug_path = Path(debug_dir) / cdir.name / f'{img_path.stem}.jpg'
            feats = extract_one(img_path, debug_path=debug_path)
            if feats is None:
                print(f'[SKIP] {img_path}')
                continue
            rows.append(feats)
            labels.append(cdir.name)
            image_names.append(img_path.name)
            paths.append(str(img_path))
    df = pd.DataFrame(np.array(rows), columns=FEATURE_NAMES)
    df['label'] = labels
    df['image_name'] = image_names
    df['image_path'] = paths
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False, sep=';', encoding='utf-8-sig')
    return df
