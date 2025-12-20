# GenD 模型完整使用说明

## 完整迁移的功能

✅ **已完整迁移的功能**：
1. **CLIP 编码器** - 支持多种 CLIP 模型（base/large）
2. **LinearProbe 分类头** - 支持普通和归一化两种模式
3. **GenD 自定义损失函数** - 包含 alignment 和 uniformity 损失
4. **参数冻结功能** - 支持冻结特征提取器和选择性解冻层
5. **PEFT/LoRA 支持** - 支持 LoRA 微调
6. **完整的训练/测试接口** - 符合 DeepfakeBench 框架

## 配置参数说明

### 基础配置

```yaml
model_name: gend
backbone: openai/clip-vit-large-patch14  # CLIP 模型名称
num_classes: 2
```

### CLIP 模型选项

- `openai/clip-vit-base-patch16` - 768 维特征
- `openai/clip-vit-base-patch32` - 768 维特征
- `openai/clip-vit-large-patch14` - 1024 维特征（推荐）

### 分类头配置

```yaml
head: linear  # 或 'nlinear' (归一化线性)
```

- `linear`: 标准线性分类头
- `nlinear`: 归一化输入的线性分类头

### 损失函数配置

#### 使用 GenD 自定义损失（推荐）

```yaml
use_gend_loss: true
loss_config:
  ce_labels: 1.0              # Cross-entropy 损失权重
  alignment_labels: 0.1        # Alignment 损失权重
  uniformity: 0.1             # Uniformity 损失权重
  label_smoothing: 0.0        # 标签平滑系数
```

#### 使用标准损失

```yaml
use_gend_loss: false
loss_func: cross_entropy
```

### 参数冻结配置

```yaml
freeze_feature_extractor: true  # 是否冻结特征提取器
unfreeze_layers: []            # 要解冻的层名称列表，例如: ['layer.11', 'layer.10']
```

### PEFT/LoRA 配置

```yaml
peft:
  lora:
    target_modules: ['q_proj', 'v_proj']  # 目标模块
    rank: 8                                # LoRA rank
    alpha: 16                              # LoRA alpha
    dropout: 0.1                           # LoRA dropout
    bias: 'none'                           # bias 类型
    use_rslora: false                      # 是否使用 RSLoRA
    use_dora: false                        # 是否使用 DoRA
```

## 完整配置示例

```yaml
# log dir 
log_dir: ./logs/training/gend

# model setting
pretrained: 'no need to provide this for the gend model'
model_name: gend
backbone_name: vit

# GenD specific settings
backbone: openai/clip-vit-large-patch14
head: linear  # or 'nlinear'
num_classes: 2

# Loss configuration
use_gend_loss: true
loss_config:
  ce_labels: 1.0
  alignment_labels: 0.1
  uniformity: 0.1
  label_smoothing: 0.0

# Parameter freezing
freeze_feature_extractor: true
unfreeze_layers: []

# PEFT (optional)
peft: null
# peft:
#   lora:
#     target_modules: ['q_proj', 'v_proj']
#     rank: 8
#     alpha: 16
#     dropout: 0.1
#     bias: 'none'

# dataset
all_dataset: [FaceForensics++, FF-F2F, FF-DF, FF-FS, FF-NT, FaceShifter, DeepFakeDetection, Celeb-DF-v1, Celeb-DF-v2, DFDCP, DFDC, DeeperForensics-1.0, UADFV]
train_dataset: [FaceForensics++]
test_dataset: [FaceForensics++, Celeb-DF-v2, FaceShifter, DeeperForensics-1.0]

compression: c23
train_batchSize: 32
test_batchSize: 32
workers: 0
frame_num: {'train': 32, 'test': 32}
resolution: 224
with_mask: false
with_landmark: false

# data augmentation
use_data_augmentation: true
data_aug:
  flip_prob: 0.5
  rotate_prob: 0.5
  rotate_limit: [-10, 10]
  blur_prob: 0.5
  blur_limit: [3, 7]
  brightness_prob: 0.5
  brightness_limit: [-0.1, 0.1]
  contrast_limit: [-0.1, 0.1]
  quality_lower: 40
  quality_upper: 100

# mean and std for normalization (CLIP normalization)
mean: [0.48145466, 0.4578275, 0.40821073]
std: [0.26862954, 0.26130258, 0.27577711]

# optimizer config
optimizer:
  type: adam
  adam:
    lr: 0.0002
    beta1: 0.9
    beta2: 0.999
    eps: 0.00000001
    weight_decay: 0.0005
    amsgrad: false
  sgd:
    lr: 0.0002
    momentum: 0.9
    weight_decay: 0.0005

# training config
lr_scheduler: cosine
lr_T_max: 10
lr_eta_min: 0.0000001
nEpochs: 10
start_epoch: 0
save_epoch: 1
rec_iter: 100
logdir: ./logs
manualSeed: 1024
save_ckpt: true
save_feat: true

# metric
metric_scoring: auc

# cuda
ngpu: 1
cuda: true
cudnn: true

save_avg: true
verbose: false  # Set to true to print trainable parameters
```

## 使用方法

### 训练

```bash
python training/train.py --detector_path training/config/detector/gend.yaml
```

### 测试

```bash
python training/test.py --detector_path training/config/detector/gend.yaml --weights_path path/to/checkpoint.pth
```

## 功能对比

| 功能 | 原始 GenD | 迁移后 GenD | 状态 |
|------|----------|------------|------|
| CLIP 编码器 | ✅ | ✅ | 完整 |
| LinearProbe 分类头 | ✅ | ✅ | 完整 |
| Alignment/Uniformity 损失 | ✅ | ✅ | 完整 |
| 参数冻结 | ✅ | ✅ | 完整 |
| PEFT/LoRA | ✅ | ✅ | 完整 |
| DINO 编码器 | ✅ | ❌ | 已移除（仅 CLIP） |
| Perception 编码器 | ✅ | ❌ | 已移除（仅 CLIP） |

## 注意事项

1. **CLIP 模型下载**：首次使用时会自动从 HuggingFace 下载 CLIP 模型
2. **PEFT 依赖**：使用 LoRA 功能需要安装 `peft` 库：`pip install peft`
3. **损失函数**：建议使用 GenD 自定义损失以获得最佳性能
4. **参数冻结**：默认冻结特征提取器，只训练分类头
5. **归一化**：使用 CLIP 的标准归一化参数

## 与原始 GenD 的差异

1. **仅支持 CLIP 编码器**：移除了 DINO 和 Perception 编码器支持
2. **框架适配**：适配到 DeepfakeBench 的 AbstractDetector 接口
3. **配置格式**：使用 YAML 配置文件而非 Python Config 类

