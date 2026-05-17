from __future__ import annotations

import copy
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm.auto import tqdm

from .utils import ensure_dir, set_seed

MODEL_MAP = {
    'ResNet50': 'resnet50',
    'DenseNet121': 'densenet121',
    'EfficientNetB2': 'efficientnet_b2',
    'ConvNeXt': 'convnext_tiny',
    'MobileNetV3': 'mobilenetv3_large_100',
}


def _make_transforms(img_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((int(img_size), int(img_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def _load_train_val_datasets(data_dir: str | Path, img_size: int):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            'Expected split dataset root with train/ and val/ folders. '
            f'Missing: {train_dir if not train_dir.exists() else val_dir}'
        )

    train_transform, val_transform = _make_transforms(img_size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_transform)

    if train_ds.classes != val_ds.classes:
        raise ValueError(f'Class mismatch: train={train_ds.classes}, val={val_ds.classes}')
    return train_ds, val_ds


def train_cnn_experiments(
    data_dir: str | Path,
    output_dir: str | Path,
    img_sizes=(224, 448, 640),
    models=('ResNet50', 'DenseNet121', 'EfficientNetB2', 'ConvNeXt', 'MobileNetV3'),
    batch_size: int = 16,
    epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Train CNN models using explicit train/val folders.

    Expected structure:
        data_dir/train/<class_name>/*
        data_dir/val/<class_name>/*
        data_dir/test/<class_name>/* 
    """
    set_seed(seed)
    output_dir = Path(output_dir)
    model_dir = ensure_dir(output_dir / 'models')
    report_dir = ensure_dir(output_dir / 'reports')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []

    print('\n' + '=' * 80, flush=True)
    print('CNN TRAINING STARTED', flush=True)
    print('=' * 80, flush=True)
    print(f'Data directory : {data_dir}', flush=True)
    print(f'Output directory: {output_dir}', flush=True)
    print(f'Device         : {device}', flush=True)
    print(f'Image sizes    : {list(img_sizes)}', flush=True)
    print(f'Models         : {list(models)}', flush=True)
    print(f'Batch size     : {batch_size}', flush=True)
    print(f'Max epochs     : {epochs}', flush=True)
    print(f'Patience       : {patience}', flush=True)
    print(f'Learning rate  : {lr}', flush=True)
    print('=' * 80 + '\n', flush=True)

    for img_size_index, img_size in enumerate(img_sizes, start=1):
        print('\n' + '#' * 80, flush=True)
        print(f'[{img_size_index}/{len(img_sizes)}] Preparing explicit train/val dataset for {img_size}x{img_size}', flush=True)
        print('#' * 80, flush=True)

        train_ds, val_ds = _load_train_val_datasets(data_dir, int(img_size))
        n_classes = len(train_ds.classes)
        num_workers = min(4, __import__('os').cpu_count() or 1)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        print(f'Classes        : {train_ds.classes}', flush=True)
        print(f'Train images   : {len(train_ds)}', flush=True)
        print(f'Validation imgs: {len(val_ds)}', flush=True)
        print(f'Train batches  : {len(train_loader)}', flush=True)
        print(f'Val batches    : {len(val_loader)}', flush=True)

        for model_index, model_name in enumerate(models, start=1):
            if model_name not in MODEL_MAP:
                raise ValueError(f'Unknown model: {model_name}. Available models: {list(MODEL_MAP)}')

            print('\n' + '-' * 80, flush=True)
            print(f'Image size {img_size} | Model [{model_index}/{len(models)}]: {model_name}', flush=True)
            print('-' * 80, flush=True)

            model_start = time.time()
            model = timm.create_model(MODEL_MAP[model_name], pretrained=True, num_classes=n_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            best_acc = 0.0
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                model.train()
                train_loss = 0.0
                train_iter = tqdm(train_loader, desc=f'{model_name} {img_size}px epoch {epoch}/{epochs} train', leave=False) if show_progress else train_loader
                for images, labels in train_iter:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                scheduler.step()

                model.eval()
                val_loss, correct, total = 0.0, 0, 0
                val_iter = tqdm(val_loader, desc=f'{model_name} {img_size}px epoch {epoch}/{epochs} val', leave=False) if show_progress else val_loader
                with torch.no_grad():
                    for images, labels in val_iter:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        val_loss += criterion(outputs, labels).item()
                        pred = outputs.argmax(dim=1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                avg_train_loss = train_loss / max(len(train_loader), 1)
                val_acc = correct / max(total, 1)
                avg_val_loss = val_loss / max(len(val_loader), 1)
                epoch_time = time.time() - epoch_start
                rows.append({'Model': model_name, 'ImageSize': img_size, 'Epoch': epoch,
                             'Train_Loss': avg_train_loss, 'Val_Loss': avg_val_loss,
                             'Val_Accuracy': val_acc, 'Epoch_Time_s': epoch_time})

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_path = model_dir / f'{model_name}_{img_size}_best.pth'
                    torch.save(copy.deepcopy(model.state_dict()), best_path)
                    save_msg = f' | saved: {best_path.name}'
                else:
                    save_msg = ''

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f'Epoch {epoch:03d}/{epochs} | train_loss={avg_train_loss:.4f} | '
                      f'val_loss={avg_val_loss:.4f} | val_acc={val_acc*100:.2f}% | '
                      f'best={best_acc*100:.2f}% | patience={patience_counter}/{patience} | '
                      f'time={epoch_time:.1f}s{save_msg}', flush=True)
                pd.DataFrame(rows).to_csv(report_dir / 'cnn_training_history.csv', index=False)

                if patience_counter >= patience:
                    print(f'Early stopping triggered for {model_name} at {img_size}px. Best validation accuracy: {best_acc*100:.2f}%', flush=True)
                    break

            model_time = time.time() - model_start
            print(f'Finished {model_name} at {img_size}px | Best Val Acc: {best_acc*100:.2f}% | Total time: {model_time/60:.2f} min', flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(report_dir / 'cnn_training_history.csv', index=False)
    if len(df) > 0:
        best_summary = df.groupby(['Model', 'ImageSize'], as_index=False)['Val_Accuracy'].max().sort_values(['ImageSize', 'Val_Accuracy'], ascending=[True, False])
        best_summary.to_csv(report_dir / 'cnn_best_summary.csv', index=False)
        print('\nCNN BEST SUMMARY')
        print(best_summary.to_string(index=False), flush=True)
    return df
