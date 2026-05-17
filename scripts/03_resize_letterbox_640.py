#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import resize_tree_letterbox


def main():
    parser = argparse.ArgumentParser(description='Resize images to a square letterbox canvas while preserving aspect ratio.')
    parser.add_argument('--input', required=True, help='Input dataset root with class subfolders.')
    parser.add_argument('--output', required=True, help='Output root for resized images.')
    parser.add_argument('--size', type=int, default=640, help='Target square size. Default: 640')
    args = parser.parse_args()
    summary = resize_tree_letterbox(args.input, args.output, args.size)
    print(summary)

if __name__ == '__main__':
    main()
