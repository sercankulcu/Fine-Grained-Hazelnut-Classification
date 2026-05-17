#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import crop_tree_foreground_rembg


def main():
    parser = argparse.ArgumentParser(
        description='Crop hazelnut foregrounds using the original rembg-based workflow.'
    )
    parser.add_argument('--input', required=True, help='Input dataset root with class subfolders.')
    parser.add_argument('--output', required=True, help='Output root for cropped PNG images.')
    parser.add_argument('--padding-percent', type=float, default=10.0, help='Padding around the foreground object.')
    parser.add_argument('--model', default='u2net', help='rembg model name. Default: u2net')
    args = parser.parse_args()

    print('[Step 2] rembg foreground cropping started', flush=True)
    print(f'Input          : {args.input}', flush=True)
    print(f'Output         : {args.output}', flush=True)
    print(f'Padding percent: {args.padding_percent}', flush=True)
    print(f'rembg model    : {args.model}', flush=True)

    summary = crop_tree_foreground_rembg(
        args.input,
        args.output,
        padding_percent=args.padding_percent,
        model_name=args.model,
    )
    print('[Step 2] Finished:', summary, flush=True)


if __name__ == '__main__':
    main()
