# CLIP-Based Cross-Domain Deepfake Detection: A Comprehensive Framework

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen) ![CLIP](https://img.shields.io/badge/CLIP-ViT-brightgreen)

<b> Research Focus: Cross-Domain Deepfake Detection using CLIP Visual Representations </b>

基于CLIP视觉表征的帧级跨域人脸伪造检测方法研究



>  🚀 **Key Innovations:**
> 
> 1. **Freq-Adapter**: FFT/DWT高频支路并联融合，增强频域特征表示
> 
> 2. **Boundary Mining**: 高通残差+边缘算子生成软权重W，用于加权特征融合
> 
> 3. **Identity-Invariant**: CLS上接身份判别器+GRL对抗抑制身份信息
> 
> 4. **Cross-Domain Robustness**: 在Celeb-DFv2/DFDC/DFD/FFIW/DSv1等数据集上验证泛化能力



---

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="figures/archi.png" style="max-width:60%;">
</div>

Welcome to *CLIP-Based Cross-Domain Deepfake Detection*, your comprehensive solution for robust deepfake detection across different domains! Here are the key features of our framework:

> ✅ **CLIP Visual Foundation**: Leveraging pre-trained CLIP ViT backbone with frozen parameters for robust visual representations
> 
> ✅ **Frequency Enhancement**: Freq-Adapter module with FFT/DWT parallel branches for enhanced frequency domain features
> 
> ✅ **Boundary-Aware Mining**: Soft weight generation through high-pass residuals and edge operators for adaptive feature fusion
> 
> ✅ **Identity-Invariant Learning**: Adversarial training with GRL to suppress identity-specific information
> 
> ✅ **Cross-Domain Generalization**: Comprehensive evaluation across multiple datasets (Celeb-DFv2, DFDC, DFD, FFIW, DSv1)
> 
> ✅ **Robustness Analysis**: Testing under compression, noise, blur, and resolution variations


---



>  🎯 **Research Methodology:**
> 
> 1. **Data & Preprocessing**: FF++ training; Celeb-DFv2/DFDC/DFD/FFIW/DSv1 cross-domain testing; face detection-alignment-cropping; frame-level sampling and augmentation
> 
> 2. **Baseline Setup**: Load CLIP ViT (frozen backbone), LN-tuning + lightweight binary classification head; L2 normalization to hypersphere
> 
> 3. **Three Key Innovations**: Freq-Adapter, Boundary Mining, Identity-Invariant modules
> 
> 4. **Loss & Training**: CE/ArcMargin + Alignment/Uniformity metrics + frequency consistency + adversarial loss; Slerp feature augmentation
> 
> 5. **Evaluation & Robustness**: Frame AUC, EER, TNR@TPR; stability under compression, noise, blur, resolution changes
> 
> 6. **Ablation & Comparison**: Component-wise removal analysis; comparison with Xception/linear CLIP baselines; parameter count and speed analysis
> 
> 7. **Interpretability & Reproducibility**: t-SNE/Grad-CAM visualization; organized code and configs; one-click reproduction scripts

---

<font size=4><b> Table of Contents </b></font>

- [Methodology](#-methodology)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
  - [Installation](#1-installation)
  - [Data Preparation](#2-data-preparation)
  - [Training](#3-training)
  - [Evaluation](#4-evaluation)
- [Key Components](#-key-components)
- [Results](#-results)
- [Ablation Studies](#-ablation-studies)
- [Citation](#-citation)
- [License](#%EF%B8%8F-license)

---


## 📚 Methodology
<a href="#top">[Back to top]</a>

Our CLIP-based cross-domain deepfake detection framework consists of seven key components:

### 1. Data & Preprocessing
- **Training Dataset**: FaceForensics++ (FF++)
- **Cross-Domain Testing**: Celeb-DFv2, DFDC, DFD, FFIW, DSv1
- **Preprocessing Pipeline**: Face detection → alignment → cropping
- **Frame-Level Processing**: Sampling and data augmentation

### 2. Baseline Architecture
- **Backbone**: CLIP ViT (frozen parameters)
- **Fine-tuning**: LayerNorm tuning + lightweight binary classification head
- **Feature Normalization**: L2 normalization to hypersphere

### 3. Three Key Innovations

#### 🔄 Freq-Adapter
- **FFT Branch**: Fast Fourier Transform for frequency domain analysis
- **DWT Branch**: Discrete Wavelet Transform for multi-scale frequency features
- **Parallel Fusion**: Concatenated frequency features for enhanced representation

#### 🎯 Boundary Mining
- **High-Pass Residuals**: Extract high-frequency components
- **Edge Operators**: Sobel/Canny edge detection
- **Soft Weight Generation**: Adaptive weighting mechanism W for feature fusion

#### 🛡️ Identity-Invariant Learning
- **Identity Discriminator**: Auxiliary classifier on CLS token
- **GRL (Gradient Reversal Layer)**: Adversarial training to suppress identity information
- **Domain Adaptation**: Cross-domain generalization enhancement

### 4. Loss Functions & Training Strategy

|                  | File name                               | Paper                                                                                                                                                                                                                                                                                                                                                         |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Effort          | [effort_detector.py](./training/detectors/effort_detector.py)         | [Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2411.15633) **ICML 2025 Spotlight** ⭐|
| TALL          | [tall_detector.py](./training/detectors/tall_detector.py)         | [TALL: Thumbnail Layout for Deepfake Video Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_TALL_Thumbnail_Layout_for_Deepfake_Video_Detection_ICCV_2023_paper.pdf) ICCV 2023 |
| LSDA          | [lsda_detector.py](./training/detectors/lsda_detector.py)         | [Transcending forgery specificity with latent space augmentation for generalizable deepfake detection](https://arxiv.org/pdf/2311.11278) CVPR 2024 |
| IID          | [iid_detector.py](./training/detectors/iid_detector.py)       | [Implicit Identity Driven Deepfake Face Swapping Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Implicit_Identity_Driven_Deepfake_Face_Swapping_Detection_CVPR_2023_paper.pdf) CVPR 2023                                                                                                                                                                                                                                     |
| SBI          | [sbi_detector.py](./training/detectors/sbi_detector.py)       | [Detecting Deepfakes with Self-Blended Images](https://arxiv.org/pdf/2204.08376) CVPR 2022                                                                                                                                                                                                                                             |
| SLADD              | [sladd_detector.py](./training/detectors/sladd_detector.py)               | [Self-supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection](https://arxiv.org/abs/2203.12208) CVPR 2022                                            |
| FTCN | [ftcn_detector.py](./training/detectors/ftcn_detector.py)                 | [Exploring Temporal Coherence for More General Video Face Forgery Detection](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Exploring_Temporal_Coherence_for_More_General_Video_Face_Forgery_Detection_ICCV_2021_paper.pdf) ICCV 2021                                                                                                 |
| PCL-I2G    | [pcl_xception_detector.py](./training/detectors/pcl_xception_detector.py)                 | [Learning Self-Consistency for Deepfake Detection](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Learning_Self-Consistency_for_Deepfake_Detection_ICCV_2021_paper.pdf) ICCV 2021                                                                                                                                |
| Local-relation             | [lrl_detector.py](./training/detectors/lrl_detector.py)             | [Local Relation Learning for Face Forgery Detection](https://arxiv.org/pdf/2105.02577) AAAI 2021                                                                                                                                               |
| UIA-ViT              | [uia_vit_detector.py](./training/detectors/uia_vit_detector.py)               | [UIA-ViT: Unsupervised Inconsistency-Aware Method based on Vision Transformer for Face Forgery Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650384.pdf) ECCV 2022                                                                                                                                                                                                                                           |
| SIA             | [sia_detector.py](./training/detectors/sia_detector.py)             | [An Information Theoretic Approach for Attention-Driven Face Forgery Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740105.pdf) ECCV 2022                                                                                                                                                                                                     |
| Multi-attention         | [multi_attention_detector.py](./training/detectors/multi_attention_detector.py)     | [Multi-Attentional Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Multi-Attentional_Deepfake_Detection_CVPR_2021_paper.html) CVPR 2021                                                                                                                                                                                               |
| CLIP            | [clip_detector.py](./training/detectors/clip_detector.py)           | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) ICML 2021                                                                                                                                                                                   |
| STIL  | [stil_detector.py](./training/detectors/stil_detector.py)     | [Spatiotemporal Inconsistency Learning for DeepFake Video Detection](https://dl.acm.org/doi/pdf/10.1145/3474085.3475508) ACMMM 2021                                                                                                                                                                                                                                                                                 |
| RFM   | [rfm_detector.py](./training/detectors/rfm_detector.py)       | [Representative Forgery Mining for Fake Face Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Representative_Forgery_Mining_for_Fake_Face_Detection_CVPR_2021_paper.pdf) CVPR 2021                                                                                                                                                                                                                                                                   |
| TimeTransformer    | [timetransformer_detector.py](./training/detectors/timesformer_detector.py)         | [Is space-time attention all you need for video understanding?](https://proceedings.mlr.press/v139/bertasius21a/bertasius21a-supp.pdf) ICML 2021                                                       |
| VideoMAE    | [videomae_detector.py](./training/detectors/videomae_detectors.py)         | [Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training](https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf) NIPS 2022                                                       |
| X-CLIP    | [xclip_detector.py](./training/detectors/xclip_detector.py)         | [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/pdf/2208.02816) ECCV 2022                                                       |


⭐️ **Datasets** (9 datasets): [FaceForensics++](https://github.com/ondyari/FaceForensics), [FaceShifter](https://github.com/ondyari/FaceForensics/tree/master/dataset), [DeepfakeDetection](https://github.com/ondyari/FaceForensics/tree/master/dataset), [Deepfake Detection Challenge (Preview)](https://ai.facebook.com/datasets/dfdc/), [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data), [Celeb-DF-v1](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1), [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics), [DeepForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset), [UADFV](https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform)

## 🏗️ Architecture
<a href="#top">[Back to top]</a>

Our CLIP-based cross-domain deepfake detection framework integrates three key innovations:

### Overall Architecture
```
Input Image → CLIP ViT (Frozen) → Feature Extraction
     ↓
Freq-Adapter (FFT/DWT) → Frequency Features
     ↓
Boundary Mining → Soft Weights W
     ↓
Feature Fusion → L2 Normalization → Binary Classifier
     ↓
Identity Discriminator + GRL → Identity-Invariant Learning
```

### Key Components

#### 🔄 Freq-Adapter Module
- **FFT Branch**: Extracts frequency domain features using Fast Fourier Transform
- **DWT Branch**: Multi-scale frequency analysis via Discrete Wavelet Transform
- **Parallel Processing**: Both branches process input simultaneously
- **Feature Concatenation**: Combined frequency features for enhanced representation

#### 🎯 Boundary Mining Module
- **High-Pass Filtering**: Extracts high-frequency residual components
- **Edge Detection**: Sobel/Canny operators for boundary information
- **Weight Generation**: Soft attention weights W for adaptive feature fusion
- **Adaptive Fusion**: Dynamic weighting based on boundary characteristics

#### 🛡️ Identity-Invariant Module
- **Identity Discriminator**: Auxiliary classifier on CLS token
- **GRL Integration**: Gradient Reversal Layer for adversarial training
- **Domain Adaptation**: Cross-domain generalization enhancement
- **Feature Disentanglement**: Separates identity-specific and forgery-specific features

### 1. Installation
(option 1) You can run the following script to configure the necessary environment:

```
git clone git@github.com:SCLBD/DeepfakeBench.git
cd DeepfakeBench
conda create -n CLIPDeepfake python=3.7.2
conda activate CLIPDeepfake
sh install.sh
```

(option 2) You can also utilize the supplied [`Dockerfile`](./Dockerfile) to set up the entire environment using Docker:

```
docker build -t CLIPDeepfake .
docker run --gpus all -itd -v /path/to/this/repository:/app/ --shm-size 64G CLIPDeepfake
```

### 2. Download Data

<a href="#top">[Back to top]</a>

All datasets used in DeepfakeBench can be downloaded from their own websites or repositories and preprocessed accordingly.
For convenience, we also provide the data we use in our research, including:

| Types                     | Links| Notes|       
| -------------------------|------- | ----------------------- |
| **Rgb-format Datasets**  | [Baidu, Password: ogjn](https://pan.baidu.com/s/1NAMUHcZvsIm7l6hMHeEQjQ?pwd=ogjn), [Google Drive](https://drive.google.com/drive/folders/1N4X3rvx9IhmkEZK-KIk4OxBrQb9BRUcs?usp=drive_link) | Preprocessed data|       
| **Lmdb-format Datasets** | [Baidu, Password: g3gj](https://pan.baidu.com/s/1riMCN5iXTJ2g9fQjtlZswg?pwd=g3gj)| LMDB database for each dataset|       
| **Json Configurations**  | [Baidu, Password: dcwv](https://pan.baidu.com/s/1d7PTV2GK-fpGibcbtnQDqQ?pwd=dcwv), [Google Drive](https://drive.google.com/drive/folders/1ZV3fz5MZZU5BTB5neziN6i8Yv0Z21_LO?usp=drive_link)| Data arrangement|       
     
All the downloaded datasets are already **preprocessed** to cropped faces (32 frames per video) with their masks and landmarks, which can be **directly deployed to evaluate our benchmark**.

The provided datasets are:

| Dataset Name                    | Notes                   |
| ------------------------------- | ----------------------- |
| Celeb-DF-v1                     | -                       |
| Celeb-DF-v2                     | -                       |
| FaceForensics++, DeepfakeDetection, FaceShifter | Only c23      |
| UADFV                           | -                       |
| Deepfake Detection Challenge (Preview) | -                       |
| Deepfake Detection Challenge     |  Only Test Data                       |

🛡️ **Copyright of the above datasets belongs to their original providers.**


Other detailed information about the datasets used in DeepfakeBench is summarized below:


| Dataset | Real Videos | Fake Videos | Total Videos | Rights Cleared | Total Subjects | Synthesis Methods | Perturbations | Original Repository |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FaceForensics++ | 1000 | 4000 | 5000 | NO | N/A | 4 | 2 | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| FaceShifter | 1000 | 1000 | 2000 | NO | N/A | 1 | - | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| DeepfakeDetection | 363 | 3000 | 3363 | YES | 28 | 5 | - | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| Deepfake Detection Challenge (Preview) | 1131 | 4119 | 5250 | YES | 66 | 2 | 3 | [Hyper-link](https://ai.facebook.com/datasets/dfdc/) |
| Deepfake Detection Challenge | 23654 | 104500 | 128154 | YES | 960 | 8 | 19 | [Hyper-link](https://www.kaggle.com/c/deepfake-detection-challenge/data) |
| CelebDF-v1 | 408 | 795 | 1203 | NO | N/A | 1 | - | [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1) |
| CelebDF-v2 | 590 | 5639 | 6229 | NO | 59 | 1 | - | [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics) |
| DeepForensics-1.0 | 50000 | 10000 | 60000 | YES | 100 | 1 | 7 | [Hyper-link](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset) |
| UADFV | 49 | 49 | 98 | NO | 49 | 1 | - | [Hyper-link](https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform) |


Upon downloading the datasets, please ensure to store them in the [`./datasets`](./datasets/) folder, arranging them in accordance with the directory structure outlined below:

```
datasets
├── lmdb
|   ├── FaceForensics++_lmdb
|   |   ├── data.mdb
|   |   ├── lock.mdb
├── rgb
|   ├── FaceForensics++
|   │   ├── original_sequences
|   │   │   ├── youtube
|   │   │   │   ├── c23
|   │   │   │   │   ├── videos
|   │   │   │   │   │   └── *.mp4
|   │   │   │   │   └── frames (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   |   |   |   |   └── masks (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   │   │   │   │   └── landmarks (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   │   │   │   └── c40
|   │   │   │   │   ├── videos
|   │   │   │   │   │   └── *.mp4
|   │   │   │   │   └── frames (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   |   |   |   |   └── masks (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   │   │   │   │   └── landmarks (if you download my processed data)
|   │   │   │   │       └── *.npy
|   │   │   ├── actors
|   │   │   │   ├── c23
|   │   │   │   │   ├── videos
|   │   │   │   │   │   └── *.mp4
|   │   │   │   │   └── frames (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   |   |   |   |   └── masks (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   │   │   │   │   └── landmarks (if you download my processed data)
|   │   │   │   │       └── *.npy
|   │   │   │   └── c40
|   │   │   │   │   ├── videos
|   │   │   │   │   │   └── *.mp4
|   │   │   │   │   └── frames (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   |   |   |   |   └── masks (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   │   │   │   │   └── landmarks (if you download my processed data)
|   │   │   │   │       └── *.npy
|   │   ├── manipulated_sequences
|   │   │   ├── Deepfakes
|   │   │   │   ├── c23
|   │   │   │   │   └── videos
|   │   │   │   │   │   └── *.mp4
|   │   │   │   │   └── frames (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   |   |   |   |   └── masks (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   │   │   │   │   └── landmarks (if you download my processed data)
|   │   │   │   │       └── *.npy
|   │   │   │   └── c40
|   │   │   │   │   ├── videos
|   │   │   │   │   │   └── *.mp4
|   │   │   │   │   └── frames (if you download my processed data)
|   │   │   │   │   │   └── *.png
|   |   |   |   |   └── masks (if you download my processed data)
│   │   │   │   |   │   └── *.png
│   │   │   │   |   └── landmarks (if you download my processed data)
│   │   │   |   │       └── *.npy
│   │   |   ├── Face2Face
│   |   │   │   ├── ...
|   │   │   ├── FaceSwap
|   │   │   │   ├── ...
|   │   │   ├── NeuralTextures
|   │   │   │   ├── ...
|   │   │   ├── FaceShifter
|   │   │   │   ├── ...
|   │   │   └── DeepFakeDetection
|   │   │       ├── ...
Other datasets are similar to the above structure
```

If you choose to store your datasets in a different folder, you may specified the `rgb_dir` or `lmdb_dir` in `training\test_config.yaml` and `training\train_config.yaml`.

The downloaded json configurations should be arranged as:
```
preprocessing
├── dataset_json
|   ├── FaceForensics++.json
```

You may also store your configurations in a different folder by specifying the `dataset_json_folder` in `training\test_config.yaml` and `training\train_config.yaml`.

### 3. Preprocessing (optional)

<a href="#top">[Back to top]</a>

**❗️Note**: If you want to directly utilize the data, including frames, landmarks, masks, and more, that I have provided above, you can skip the pre-processing step. **However, you still need to run the rearrangement script to generate the JSON file** for each dataset for the unified data loading in the training and testing process.

DeepfakeBench follows a sequential workflow for face detection, alignment, and cropping. The processed data, including face images, landmarks, and masks, are saved in separate folders for further analysis.

To start preprocessing your dataset, please follow these steps:

1. Download the [shape_predictor_81_face_landmarks.dat](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat) file. Then, copy the downloaded shape_predictor_81_face_landmarks.dat file into the `./preprocessing/dlib_tools folder`. This file is necessary for Dlib's face detection functionality.

2. Open the [`./preprocessing/config.yaml`](./preprocessing/config.yaml) and locate the line `default: DATASET_YOU_SPECIFY`. Replace `DATASET_YOU_SPECIFY` with the name of the dataset you want to preprocess, such as `FaceForensics++`.

7. Specify the `dataset_root_path` in the config.yaml file. Search for the line that mentions dataset_root_path. By default, it looks like this: ``dataset_root_path: ./datasets``.
Replace `./datasets` with the actual path to the folder where your dataset is arranged. 

Once you have completed these steps, you can proceed with running the following line to do the preprocessing:

```
cd preprocessing

python preprocess.py
```
You may skip the preprocessing step by downloading the provided data.

### 4. Rearrangement
To simplify the handling of different datasets, we propose a unified and convenient way to load them. The function eliminates the need to write separate input/output (I/O) code for each dataset, reducing duplication of effort and easing data management.

After the preprocessing above, you will obtain the processed data (*i.e., frames, landmarks, and masks*) for each dataset you specify. Similarly, you need to set the parameters in `./preprocessing/config.yaml` for each dataset. After that, run the following line:
```
cd preprocessing

python rearrange.py
```
After running the above line, you will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 


### 5. Training (optional)

<a href="#top">[Back to top]</a>

To run the training code, you should first download the pretrained weights for the corresponding **backbones** (These pre-trained weights are from ImageNet). You can download them from [Link](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip). After downloading, you need to put all the weights files into the folder `./training/pretrained`.

Then, you should go to the `./training/config/detector/` folder and then Choose the detector to be trained. For instance, you can adjust the parameters in [`xception.yaml`](./training/config/detector/xception.yaml) to specify the parameters, *e.g.,* training and testing datasets, epoch, frame_num, *etc*.

After setting the parameters, you can run with the following to train the Xception detector:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml
```

You can also adjust the training and testing datasets using the command line, for example:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml  \
--train_dataset "FaceForensics++" \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2"
```

By default, the checkpoints and features will be saved during the training process. If you do not want to save them, run with the following:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml \
--train_dataset "FaceForensics++" \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2" \
--no-save_ckpt \
--no-save_feat
```

For **multi-gpus training** (DDP), please refer to [`train.sh`](./train.sh) file for details.

To train other detectors using the code mentioned above, you can specify the config file accordingly. However, for the Face X-ray detector, an additional step is required before training. To save training time, a pickle file is generated to store the Top-N nearest images for each given image. To generate this file, you should run the [`generate_xray_nearest.py`](./training/dataset/generate_xray_nearest.py) file. Once the pickle file is created, you can train the Face X-ray detector using the same way above. If you want to check/use the files I have already generated, please refer to the [`link`](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.2).


### 6. Evaluation
If you only want to evaluate the detectors to produce the results of the cross-dataset evaluation, you can use the the [`test.py`](./training/test.py) code for evaluation. Here is an example:

```
python3 training/test.py \
--detector_path ./training/config/detector/xception.yaml \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" \
--weights_path ./training/weights/xception_best.pth
```
**Note that we have provided the pre-trained weights for each detector (you can download them from the [`link`](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)).** Make sure to put these weights in the `./training/weights` folder.



## 🏆 Results

<a href="#top">[Back to top]</a>

>  ❗️❗️❗️ **DeepfakeBench-v2 Updates:**
> 
> The below results are cited from our [paper](https://arxiv.org/abs/2307.01426). We have conducted more comprehensive evaluations using the DeepfakeBench-v2, with more datasets used and more detectors implemented. We will update the below table soon.
> 

In our Benchmark, we apply [TensorBoard](https://github.com/tensorflow/tensorboard) to monitor the progress of training models. It provides a visual representation of the training process, allowing users to examine training results conveniently.

To demonstrate the effectiveness of different detectors, we present **partial results** from both within-domain and cross-domain evaluations. The evaluation metric used is the frame-level Area Under the Curve (AUC). In this particular scenario, we train the detectors on the FF++ (c23) dataset and assess their performance on other datasets.

For a comprehensive overview of the results, we strongly recommend referring to our [paper](https://arxiv.org/abs/2307.01426). These resources provide a detailed analysis of the training outcomes and offer a deeper understanding of the methodology and findings.


| Type     | Detector   | Backbone  | FF++\_c23 | FF++\_c40 | FF-DF   | FF-F2F  | FF-FS   | FF-NT   | Avg.     | Top3 | CDFv1   | CDFv2   | DF-1.0  | DFD     | DFDC    | DFDCP   | Fsh     | UADFV   | Avg.    | Top3 |
|----------|------------|-----------|------------|------------|---------|---------|---------|---------|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|------|
| Naive    | Meso4      | MesoNet   | 0.6077     | 0.5920     | 0.6771  | 0.6170  | 0.5946  | 0.5701  | 0.6097   | 0    | 0.7358  | 0.6091  | 0.9113  | 0.5481  | 0.5560  | 0.5994  | 0.5660  | 0.7150  | 0.6551 | 1    |
| Naive    | MesoIncep  | MesoNet   | 0.7583     | 0.7278     | 0.8542  | 0.8087  | 0.7421  | 0.6517  | 0.7571   | 0    | 0.7366  | 0.6966  | 0.9233  | 0.6069  | 0.6226  | 0.7561  | 0.6438  | 0.9049  | 0.7364 | 3    |
| Naive    | CNN-Aug    | ResNet    | 0.8493     | 0.7846     | 0.9048  | 0.8788  | 0.9026  | 0.7313  | 0.8419   | 0    | 0.7420  | 0.7027  | 0.7993  | 0.6464  | 0.6361  | 0.6170  | 0.5985  | 0.8739  | 0.7020 | 0    |
| Naive    | Xception   | Xception  | 0.9637     | 0.8261     | 0.9799  | 0.9785  | 0.9833  | 0.9385  | 0.9450   | 4    | 0.7794  | 0.7365  | 0.8341  | **0.8163**  | 0.7077  | 0.7374  | 0.6249  | 0.9379  | 0.7718 | 2    |
| Naive    | EfficientB4| Efficient | 0.9567     | 0.8150     | 0.9757  | 0.9758  | 0.9797  | 0.9308  | 0.9389   | 0    | 0.7909  | 0.7487  | 0.8330  | 0.8148  | 0.6955  | 0.7283  | 0.6162  | 0.9472  | 0.7718 | 3    |
| Spatial  | Capsule    | Capsule   | 0.8421     | 0.7040     | 0.8669  | 0.8634  | 0.8734  | 0.7804  | 0.8217   | 0    | 0.7909  | 0.7472  | 0.9107  | 0.6841  | 0.6465  | 0.6568  | 0.6465  | 0.9078  | 0.7488 | 2    |
| Spatial  | FWA        | Xception  | 0.8765     | 0.7357     | 0.9210  | 0.9000  | 0.8843  | 0.8120  | 0.8549   | 0    | 0.7897  | 0.6680  | **0.9334**  | 0.7403  | 0.6132  | 0.6375  | 0.5551  | 0.8539  | 0.7239 | 1    |
| Spatial  | Face X-ray      | HRNet     | 0.9592     | 0.7925     | 0.9794  | **0.9872**  | 0.9871  | 0.9290  | 0.9391   | 3    | 0.7093  | 0.6786  | 0.5531  | 0.7655  | 0.6326  | 0.6942  | **0.6553**  | 0.8989  | 0.6985 | 0    |
| Spatial  | FFD        | Xception  | 0.9624     | 0.8237     | 0.9803  | 0.9784  | 0.9853  | 0.9306  | 0.9434   | 1    | 0.7840  | 0.7435  | 0.8609  | 0.8024  | 0.7029  | 0.7426  | 0.6056  | 0.9450  | 0.7733 | 1    |
| Spatial  | CORE       | Xception  | 0.9638     | 0.8194     | 0.9787  | 0.9803  | 0.9823  | 0.9339  | 0.9431   | 2    | 0.7798  | 0.7428  | 0.8475  | 0.8018  | 0.7049  | 0.7341  | 0.6032  | 0.9412  | 0.7694 | 0    |
| Spatial  | Recce      | Designed  | 0.9621     | 0.8190     | 0.9797  | 0.9779  | 0.9785  | 0.9357  | 0.9422   | 1    | 0.7677  | 0.7319  | 0.7985  | 0.8119  | 0.7133  | 0.7419  | 0.6095  | 0.9446  | 0.7649 | 2    |
| Spatial  | UCF        | Xception  | **0.9705** | **0.8399** | **0.9883** | 0.9840  | **0.9896** | **0.9441** | **0.9527** | **6** | 0.7793  | 0.7527  | 0.8241  | 0.8074  | **0.7191**  | **0.7594**  | 0.6462  | **0.9528**  | 0.7801 | **5** |
| Frequency| F3Net      | Xception  | 0.9635     | 0.8271     | 0.9793  | 0.9796  | 0.9844  | 0.9354  | 0.9449   | 1    | 0.7769  | 0.7352  | 0.8431  | 0.7975  | 0.7021  | 0.7354  | 0.5914  | 0.9347  | 0.7645 | 0    |
| Frequency| SPSL       | Xception  | 0.9610     | 0.8174     | 0.9781  | 0.9754  | 0.9829  | 0.9299  | 0.9408   | 0    | **0.8150**  | **0.7650**  | 0.8767  | 0.8122  | 0.7040  | 0.7408  | 0.6437  | 0.9424  | **0.7875** | 3    |
| Frequency| SRM        | Xception  | 0.9576     | 0.8114     | 0.9733  | 0.9696  | 0.9740  | 0.9295  | 0.9359   | 0    | 0.7926  | 0.7552  | 0.8638  | 0.8120  | 0.6995  | 0.7408  | 0.6014  | 0.9427  | 0.7760 | 2    |


In the above table, "Avg." donates the average AUC for within-domain and cross-domain evaluation, and the overall results. "Top3" represents the count of each method ranks within the top-3 across all testing datasets. The best-performing method for each column is highlighted.


Also, we provide all experimental results in [Link (code: qjpd)](https://pan.baidu.com/s/1Mgo5rW08B3ee_8ZBC3EXJA?pwd=qjpd). You can use these results for further analysis using the code in [`./analysis`](`./analysis`) folder.































## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our benchmark useful to your research, please cite it as follows:

```
@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}
```

If interested, you can read our recent works about deepfake detection, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).
```
@inproceedings{UCF_YAN_ICCV2023,
 title={Ucf: Uncovering common features for generalizable deepfake detection},
 author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
 booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
 pages={22412--22423},
 year={2023}
}

@inproceedings{LSDA_YAN_CVPR2024,
  title={Transcending forgery specificity with latent space augmentation for generalizable deepfake detection},
  author={Yan, Zhiyuan and Luo, Yuhao and Lyu, Siwei and Liu, Qingshan and Wu, Baoyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{cheng2024can,
  title={Can We Leave Deepfake Data Behind in Training Deepfake Detector?},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Luo, Yuhao and Wang, Zhongyuan and Li, Chen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{chen2024textit,
  title={X^2-DFD: A framework for eXplainable and eXtendable Deepfake Detection},
  author={Chen, Yize and Yan, Zhiyuan and Lyu, Siwei and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2410.06126},
  year={2024}
}

@article{cheng2024stacking,
  title={Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Hao, Li and Ai, Jiaxin and Zou, Qin and Li, Chen and Wang, Zhongyuan},
  journal={arXiv preprint arXiv:2411.11396},
  year={2024}
}

@article{yan2024effort,
  title={Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection},
  author={Yan, Zhiyuan and Wang, Jiangming and Wang, Zhendong and Jin, Peng and Zhang, Ke-Yue and Chen, Shen and Yao, Taiping and Ding, Shouhong and Wu, Baoyuan and Yuan, Li},
  journal={arXiv preprint arXiv:2411.15633},
  year={2024}
}

```


## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or yanzhiyuan1114@gmail.com. We look forward to collaborating with you in pushing the boundaries of deepfake detection.
