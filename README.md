Core Code for HCM-Net

This folder provides the core implementation of HCM-Net, a heterogeneous complementarity mining network for lightweight RGB-T object detection.

Files

RPM.py
Relational Position Miner. This module is used for high-level positional dependency modeling.

SCAA.py
Spatial-Channel Aligned Attention. This module is used for saliency-preserving spatial-channel feature recalibration during RGB-T fusion.

FSDA.py
Frequency-Spatial Detail Aggregator. This file contains the main components used for shallow detail enhancement, including SPDConv, FGM, OmniKernel, and CSPOmniKernel.

train_m3fd.py
An example training script for the M3FD dataset.

Training

Before training, prepare the RGB-T dataset configuration file and the model configuration file.

Example command:

python train_m3fd.py --model configs/HCM-Net.yaml --data datasets/M3FD/data.yaml

Common options:

--model
Path to the model configuration file.

--data
Path to the dataset configuration file.

--project
Directory for saving training runs.

--name
Name of the training run.

--imgsz
Input image size.

--epochs
Number of training epochs.

--batch
Batch size.

Example with custom options:

python train_m3fd.py --model configs/HCM-Net.yaml --data datasets/M3FD/data.yaml --project runs/M3FD --name HCM-Net --imgsz 640 --epochs 400 --batch 16

Notes

The paths in the commands are examples. Please replace them with the actual paths of your model configuration file and dataset configuration file.

The code is intended to be used with an Ultralytics-based RGB-T detection framework that supports YOLOMM and the corresponding model configuration.
