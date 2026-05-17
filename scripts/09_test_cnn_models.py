#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MODEL_MAP = {
    'ResNet50': 'resnet50',
    'DenseNet121': 'densenet121',
    'EfficientNetB2': 'efficientnet_b2',
    'ConvNeXt': 'convnext_tiny',
    'MobileNetV3': 'mobilenetv3_large_100',
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_name(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def find_weight_file(weights_dir: Path, model_name: str, img_size: int) -> Optional[Path]:
    model_key = normalize_name(model_name)
    size_key = str(img_size)
    candidates = []
    aliases = {
        'EfficientNetB2': ['effnetb2', 'efficientnetb2', 'efficientnet_b2'],
        'MobileNetV3': ['mobilenetv3', 'mobilenet_v3'],
        'DenseNet121': ['densenet121'],
        'ResNet50': ['resnet50'],
        'ConvNeXt': ['convnext'],
    }
    for file in sorted(weights_dir.rglob('*.pth')):
        name_key = normalize_name(file.stem)
        if size_key in name_key and (model_key in name_key or any(a in name_key for a in aliases.get(model_name, []))):
            candidates.append(file)
    best = [p for p in candidates if 'best' in p.stem.lower()]
    return best[0] if best else (candidates[0] if candidates else None)


def load_state(path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        if 'state_dict' in state:
            state = state['state_dict']
        elif 'model' in state:
            state = state['model']
    return {k.replace('module.', ''): v for k, v in state.items()}


def save_cm(cm, class_names: List[str], output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate(model_name, img_size, weight_path, test_dir, output_dir, batch_size, device, workers):
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(str(test_dir), transform=test_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    model = timm.create_model(MODEL_MAP[model_name], pretrained=False, num_classes=len(ds.classes))
    model.load_state_dict(load_state(weight_path, device), strict=True)
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    cm = confusion_matrix(y_true, y_pred)
    cm_path = output_dir / 'confusion_matrices' / f'{model_name}_{img_size}_test_cm.png'
    save_cm(cm, ds.classes, cm_path, f'{model_name} {img_size}px Test Confusion Matrix')

    return {
        'Model': model_name,
        'ImageSize': img_size,
        'Weights': str(weight_path),
        'Test_Images': len(ds),
        'Test_Accuracy': accuracy_score(y_true, y_pred),
        'Macro_Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro_F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Confusion_Matrix': str(cm_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN models on the independent test set.')
    parser.add_argument('--data', required=True, help='Split dataset root containing test folder.')
    parser.add_argument('--weights', required=True, help='Directory containing CNN .pth weights.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--img-sizes', nargs='+', type=int, default=[224, 448, 640])
    parser.add_argument('--models', nargs='+', default=list(MODEL_MAP.keys()))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    data_root = Path(args.data)
    test_dir = data_root / 'test'
    if not test_dir.exists():
        raise FileNotFoundError(f'Test folder not found: {test_dir}')

    weights_dir = Path(args.weights)
    output_dir = ensure_dir(Path(args.output))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    rows = []
    for model_name in args.models:
        for img_size in args.img_sizes:
            weight_path = find_weight_file(weights_dir, model_name, img_size)
            if weight_path is None:
                print(f'[SKIP] No weight found for {model_name} {img_size}px')
                continue
            print(f'[TEST] {model_name} | {img_size}px | {weight_path}')
            row = evaluate(model_name, img_size, weight_path, test_dir, output_dir, args.batch_size, device, args.workers)
            rows.append(row)
            print(f"       acc={row['Test_Accuracy']:.4f} | macro_f1={row['Macro_F1']:.4f}")

    df = pd.DataFrame(rows)
    report_path = output_dir / 'cnn_test_results.csv'
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    if len(df):
        print(df[['Model', 'ImageSize', 'Test_Accuracy', 'Macro_F1']].to_string(index=False))
    print(f'Saved: {report_path}')


if __name__ == '__main__':
    main()
