#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.cnn_training import MODEL_MAP, train_cnn_experiments


def main():
    parser = argparse.ArgumentParser(description='Train CNN classifiers at multiple image resolutions.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--img-sizes', nargs='+', type=int, default=[224,448,640])
    parser.add_argument('--models', nargs='+', default=list(MODEL_MAP.keys()))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None, help='cuda, cpu, or leave empty for automatic selection')
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm batch progress bars')
    args = parser.parse_args()
    df = train_cnn_experiments(
        args.data,
        args.output,
        args.img_sizes,
        args.models,
        args.batch_size,
        args.epochs,
        args.patience,
        args.lr,
        args.seed,
        device=args.device,
        show_progress=not args.no_progress,
    )
    if len(df) > 0:
        print(df.groupby(['Model','ImageSize'])['Val_Accuracy'].max().reset_index().to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
