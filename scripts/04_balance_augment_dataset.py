#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.augmentation import balance_dataset


def main():
    parser = argparse.ArgumentParser(description='Apply conservative class-balanced augmentation.')
    parser.add_argument('--input', required=True, help='Input dataset root with class subfolders.')
    parser.add_argument('--output', required=True, help='Output root for balanced dataset.')
    parser.add_argument('--target-count', type=int, default=1000, help='Target image count per class.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    summary = balance_dataset(args.input, args.output, args.target_count, args.seed)
    print(summary)

if __name__ == '__main__':
    main()
