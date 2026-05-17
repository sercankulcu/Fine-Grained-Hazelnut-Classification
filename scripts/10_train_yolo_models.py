#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train YOLO classification models at multiple image resolutions.')
    parser.add_argument('--data', required=True, help='Ultralytics classification dataset directory with train/val or class subfolders.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', nargs='+', default=['yolov8n-cls.pt', 'yolo11n-cls.pt', 'yolo26n-cls.pt'])
    parser.add_argument('--img-sizes', nargs='+', type=int, default=[224,448,640])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    from ultralytics import YOLO
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        for size in args.img_sizes:
            run_name = f'{Path(model_name).stem}_{size}'
            model = YOLO(model_name)
            model.train(
                data=args.data,
                imgsz=size,
                batch=args.batch,
                epochs=args.epochs,
                patience=args.patience,
                workers=args.workers,
                project=str(out),
                name=run_name,
                device=args.device,
            )

if __name__ == '__main__':
    main()
