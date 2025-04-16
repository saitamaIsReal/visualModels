# Vision Transformer on CIFAR-10

This project implements a simple Vision Transformer (ViT) using PyTorch and trains it on the CIFAR-10 dataset. The environment is fully reproducible via conda.


## Requirements

- Miniconda or Anaconda
- Git
- Windows or Linux

## Setup

1. Clone the repository:

```bash
git clone https://github.com/saitamaIsReal/visualModels.git
cd visualModels

2. Create the conda environment:
conda env create -f env.yaml
conda activate visual_env

If the environment already exists:
conda env update --file env.yaml --prune 
```

## Usage

Start training:
make train

Run evalutaion:
make test

Export current environment:
make export

Remove model file: 
make clean

Recreate environment from env.yaml:
make env

Files:
vit_model.py – Vision Transformer training script

test_vit.py – Evaluation script

vision_transformer_model.pth – Saved model weights

env.yaml – Conda environment specification

Makefile – Automation for common tasks

Author: Entoni Jombi – LMU Munich, B.Sc. Computer Science