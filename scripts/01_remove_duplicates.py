#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import copy_without_duplicates


def main():
    parser = argparse.ArgumentParser(description='Remove duplicate images using MD5 hashes and copy unique images to a clean folder.')
    parser.add_argument('--input', required=True, help='Input dataset root with class subfolders.')
    parser.add_argument('--output', required=True, help='Output root for unique images.')
    args = parser.parse_args()
    summary = copy_without_duplicates(args.input, args.output)
    print(summary)

if __name__ == '__main__':
    main()
