#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from ultralytics import YOLO


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_token(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def find_yolo_weight(weights_root: Path, model_name: str, img_size: int) -> Optional[Path]:
    model_key = normalize_token(model_name)
    size_key = str(img_size)
    scored = []
    for path in sorted(weights_root.rglob('*.pt')):
        p = normalize_token(str(path))
        score = 0
        if model_key in p:
            score += 100
        if size_key in p:
            score += 50
        if path.name.lower() == 'best.pt':
            score += 20
        if 'weights' in [part.lower() for part in path.parts]:
            score += 10
        if score >= 150:
            scored.append((score, path))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def metric(results, names: List[str]):
    if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
        for name in names:
            if name in results.results_dict:
                return results.results_dict[name]
    for name in names:
        if hasattr(results, name):
            try:
                return float(getattr(results, name))
            except Exception:
                return getattr(results, name)
    return None


def evaluate_one(weight_path: Path, data_root: Path, img_size: int, batch: int, workers: int, device: Optional[str], project: Path, run_name: str):
    model = YOLO(str(weight_path), task='classify')
    results = model.val(
        data=str(data_root),
        split='test',
        imgsz=img_size,
        batch=batch,
        workers=workers,
        device=device,
        project=str(project),
        name=run_name,
        plots=True,
    )
    row = {
        'Weights': str(weight_path),
        'ImageSize': img_size,
        'Top1_Accuracy': metric(results, ['top1', 'metrics/accuracy_top1', 'accuracy_top1']),
        'Top5_Accuracy': metric(results, ['top5', 'metrics/accuracy_top5', 'accuracy_top5']),
        'Save_Dir': str(getattr(results, 'save_dir', '')),
    }
    if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
        for key, value in results.results_dict.items():
            row[str(key).replace('/', '_')] = value
    return row


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO classification models on the independent test split.')
    parser.add_argument('--data', required=True, help='Split dataset root containing train/val/test.')
    parser.add_argument('--weights', required=True, help='YOLO training output root.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', nargs='+', default=['yolov8n-cls', 'yolo11n-cls', 'yolo26n-cls'])
    parser.add_argument('--img-sizes', nargs='+', type=int, default=[224, 448, 640])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    data_root = Path(args.data)
    if not (data_root / 'test').exists():
        raise FileNotFoundError(f'Test folder not found: {data_root / "test"}')

    weights_root = Path(args.weights)
    output_dir = ensure_dir(Path(args.output))
    runs_dir = ensure_dir(output_dir / 'runs')
    rows = []

    for model_name in args.models:
        for img_size in args.img_sizes:
            weight_path = find_yolo_weight(weights_root, model_name, img_size)
            if weight_path is None:
                print(f'[SKIP] No YOLO weight found for {model_name} {img_size}px')
                continue
            print(f'[TEST] {model_name} | {img_size}px | {weight_path}')
            row = evaluate_one(weight_path, data_root, img_size, args.batch, args.workers, args.device, runs_dir, f'{model_name}_{img_size}_test')
            row['Model'] = model_name
            rows.append(row)
            print(f"       top1={row.get('Top1_Accuracy')} | top5={row.get('Top5_Accuracy')}")

    df = pd.DataFrame(rows)
    report_path = output_dir / 'yolo_test_results.csv'
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    if len(df):
        cols = [c for c in ['Model', 'ImageSize', 'Top1_Accuracy', 'Top5_Accuracy'] if c in df.columns]
        print(df[cols].to_string(index=False))
    print(f'Saved: {report_path}')


if __name__ == '__main__':
    main()
