import argparse
import warnings

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/M3FD/HCM-Net/weights/best.pt')
    parser.add_argument('--data', default='datasets/M3FD/data.yaml')
    parser.add_argument('--project', default='runs/M3FD_eval')
    parser.add_argument('--name', default='HCM-Net')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--split', default='val')
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.use_deterministic_algorithms(False)
    warnings.filterwarnings('ignore')
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == '__main__':
    main()
