import argparse
import warnings

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='configs/HCM-Net.yaml')
    parser.add_argument('--data', default='datasets/M3FD/data.yaml')
    parser.add_argument('--project', default='runs/M3FD')
    parser.add_argument('--name', default='HCM-Net')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--close-mosaic', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.use_deterministic_algorithms(False)
    warnings.filterwarnings('ignore')
    model = YOLO(args.model)
    model.train(
        data=args.data,
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        amp=False,
        optimizer='SGD',
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
