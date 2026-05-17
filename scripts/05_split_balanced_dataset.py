#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_files(files: List[Path], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for src in files:
        shutil.copy2(src, out_dir / src.name)


def split_class_images(images: List[Path], train_count: int, val_count: int, test_count: int, seed: int) -> Dict[str, List[Path]]:
    rng = random.Random(seed)
    images = images.copy()
    rng.shuffle(images)
    required = train_count + val_count + test_count
    if len(images) < required:
        raise ValueError(f'Not enough images. Required {required}, found {len(images)}.')
    return {
        'train': images[:train_count],
        'val': images[train_count:train_count + val_count],
        'test': images[train_count + val_count:train_count + val_count + test_count],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test folders.')
    parser.add_argument('--input', required=True, help='Input dataset root, e.g. data/balanced')
    parser.add_argument('--output', required=True, help='Output split dataset root, e.g. data/split')
    parser.add_argument('--train-per-class', type=int, default=600)
    parser.add_argument('--val-per-class', type=int, default=200)
    parser.add_argument('--test-per-class', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clear-output', action='store_true')
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    if not input_root.exists():
        raise FileNotFoundError(f'Input folder not found: {input_root}')
    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)

    class_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f'No class folders found under: {input_root}')

    rows = []
    for class_dir in class_dirs:
        images = list_images(class_dir)
        split = split_class_images(images, args.train_per_class, args.val_per_class, args.test_per_class, args.seed)
        for split_name, files in split.items():
            copy_files(files, output_root / split_name / class_dir.name)
        rows.append((class_dir.name, len(images), len(split['train']), len(split['val']), len(split['test'])))

    print('\nSplit summary')
    print('-' * 80)
    total_train = total_val = total_test = 0
    for cls, available, train_n, val_n, test_n in rows:
        total_train += train_n
        total_val += val_n
        total_test += test_n
        print(f'{cls:<15} available={available:>4} | train={train_n:>4} | val={val_n:>4} | test={test_n:>4}')
    print('-' * 80)
    print(f'{"TOTAL":<15} train={total_train} | val={total_val} | test={total_test}')
    print(f'Saved to: {output_root}')


if __name__ == '__main__':
    main()
