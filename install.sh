#!/bin/bash
set -e  # 出错就退出

# ---- 1. 用 conda 安装大件，避免 pip 冲突 ----
conda install -y -c conda-forge \
  numpy=1.23.5 \
  pandas=1.4.2 \
  pillow=9.0.1 \
  dlib \
  imageio=2.9.0 \
  imgaug=0.4.0 \
  tqdm=4.61.0 \
  scipy=1.10 \
  seaborn=0.11.2 \
  pyyaml=6.0 \
  opencv=4.6 \
  scikit-image=0.19.2 \
  scikit-learn=1.0.2 \
  albumentations=1.1.0 \
  matplotlib \
  setuptools=59.5.0

# ---- 2. 安装 PyTorch 1.12（CUDA 11.3），如果没显卡可换 cpuonly ----
conda install -y -c pytorch \
  pytorch=1.12 torchvision=0.13 torchaudio=0.12 cudatoolkit=11.3

# ---- 3. 其余小件用 pip（禁止它改依赖） ----
pip install efficientnet-pytorch==0.7.1 --no-deps
pip install timm==0.6.12 --no-deps
pip install segmentation-models-pytorch==0.3.2 --no-deps
pip install torchtoolbox==0.1.8.2 --no-deps
pip install tensorboard==2.10.1 --no-deps
pip install loralib --no-deps
pip install einops --no-deps
pip install transformers --no-deps
pip install filterpy --no-deps
pip install simplejson --no-deps
pip install kornia --no-deps
pip install fvcore --no-deps
pip install git+https://github.com/openai/CLIP.git --no-deps
