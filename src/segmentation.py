from __future__ import annotations

import cv2
import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.zeros_like(mask)
    clean[labels == largest_label] = 255
    return clean


def fill_holes(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    cv2.floodFill(filled, None, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    return mask | filled_inv


def segment_hazelnut(img_bgr: np.ndarray, sat_min: int = 60, val_min: int = 40) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask_sat = cv2.inRange(s, sat_min, 255)
    mask_val = cv2.inRange(v, val_min, 255)
    mask = cv2.bitwise_and(mask_sat, mask_val)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    return mask


def main_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def count_holes(mask: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for i in range(len(contours)) if hierarchy[0][i][3] != -1)


def draw_contour_and_obb(img_bgr: np.ndarray, cnt) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    vis = np.full((h, w, 3), 255, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    vis[mask == 255] = img_bgr[mask == 255]
    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 5)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 5)
    return vis
