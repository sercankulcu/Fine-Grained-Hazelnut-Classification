#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.handcrafted_features import extract_dataset


def main():
    parser = argparse.ArgumentParser(description='Extract handcrafted shape, color and texture features to CSV.')
    parser.add_argument('--input', required=True, help='Dataset root with class subfolders.')
    parser.add_argument('--output', required=True, help='Output CSV path, e.g., outputs/features/hazelnut_features.csv')
    parser.add_argument('--debug-dir', default=None, help='Optional folder for contour/OBB debug visualizations.')
    args = parser.parse_args()
    df = extract_dataset(args.input, args.output, args.debug_dir)
    print(f'Saved {len(df)} rows to {args.output}')

if __name__ == '__main__':
    main()
