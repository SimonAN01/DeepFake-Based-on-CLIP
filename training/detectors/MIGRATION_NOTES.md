# GenD-main 模型迁移说明

本文档说明从 GenD-main 项目迁移到 DeepfakeBench 项目的模型架构。

## 已迁移的模型

### 1. GenD 模型 (`gend_detector.py`)
- **来源**: `GenD-main/src/model/GenD.py`
- **功能**: 支持多种编码器（CLIP、DINO、PerceptionEncoder）的通用检测器
- **主要组件**:
  - `CLIPEncoder`: CLIP 视觉编码器
  - `DINOEncoder`: DINO 编码器
  - `PerceptionEncoder`: Perception Encoder
  - `LinearProbe`: 线性分类头
- **配置参数**:
  - `backbone`: 编码器类型（如 "openai/clip-vit-large-patch14"）
  - `backbone_args`: 编码器参数（如 `img_size`, `merge_cls_token_with_patches`）
  - `num_classes`: 分类数量（默认 2）
  - `head_args`: 头部参数（如 `normalize_inputs`）
  - `loss_func`: 损失函数（默认 "cross_entropy"）

### 2. ForAda 模型 (`forada_detector.py`)
- **来源**: `GenD-main/src/model/ForAda.py`
- **功能**: 基于 CLIP 和 Adapter 的伪造检测模型
- **依赖组件**:
  - `DS` 模型: `GenD-main/src/model/forada/ds.py`
  - `Adapter`: `GenD-main/src/model/forada/adapters/adapter.py`
  - `RecAttnClip`: `GenD-main/src/model/forada/attn.py`
  - `Layer modules`: `GenD-main/src/model/forada/layer.py`
  - `CLIP model`: `GenD-main/src/model/forada/clip/`
- **注意**: 完整实现需要将 `GenD-main/src/model/forada/` 目录复制到 `DeepfakeBench/training/detectors/utils/forada/`

### 3. FSFM 模型 (`fsfm_detector.py`)
- **来源**: `GenD-main/src/model/FSFM.py`
- **功能**: 基于 Vision Transformer 和 Adapter 的检测模型
- **依赖组件**:
  - `models_vit_fs_adapter.py`: 带 Adapter 的 ViT
  - `models_vit.py`: 标准 ViT
- **注意**: 完整实现需要将 `GenD-main/src/model/fsfm/` 目录复制到 `DeepfakeBench/training/detectors/utils/fsfm/`

### 4. Effort 模型 (`effort_detector.py`)
- **状态**: 已存在于 DeepfakeBench 中
- **来源**: `GenD-main/src/model/Effort.py`
- **功能**: 基于 CLIP 和 SVD Residual 的检测模型

## 迁移步骤

### 完整迁移 ForAda 和 FSFM 模型

1. **复制 ForAda 组件**:
   ```bash
   cp -r GenD-main/src/model/forada DeepfakeBench/training/detectors/utils/
   ```

2. **复制 FSFM 组件**:
   ```bash
   cp -r GenD-main/src/model/fsfm DeepfakeBench/training/detectors/utils/
   ```

3. **更新导入路径**:
   - 确保 `forada_detector.py` 和 `fsfm_detector.py` 中的导入路径正确
   - 可能需要调整相对导入路径

## 使用示例

### GenD 模型配置示例

```yaml
detector:
  name: gend
  backbone: openai/clip-vit-large-patch14
  backbone_args:
    img_size: 224
  num_classes: 2
  head_args:
    normalize_inputs: false
  loss_func: cross_entropy
```

### ForAda 模型配置示例

```yaml
detector:
  name: forada
  forada_config: path/to/forada/config.yaml
  num_classes: 2
  loss_func: cross_entropy
```

### FSFM 模型配置示例

```yaml
detector:
  name: fsfm
  checkpoint: path/to/checkpoint.pth
  use_adapter: true
  num_classes: 2
  loss_func: cross_entropy
```

## 注意事项

1. **依赖项**: 确保安装了所有必要的依赖包：
   - `transformers`
   - `timm`
   - `torch`
   - `torchvision`
   - `numpy`
   - `sklearn`

2. **模型权重**: 
   - GenD 模型会自动从 HuggingFace 下载预训练权重
   - ForAda 和 FSFM 需要手动下载或指定权重路径

3. **兼容性**: 
   - 所有模型都遵循 DeepfakeBench 的 `AbstractDetector` 接口
   - 使用 `@DETECTOR.register_module` 装饰器注册

4. **测试**: 
   - 在正式使用前，请确保所有模型都能正常加载和运行
   - 检查配置文件的格式和参数是否正确

## 文件结构

```
DeepfakeBench/training/detectors/
├── gend_detector.py          # GenD 检测器
├── forada_detector.py         # ForAda 检测器
├── fsfm_detector.py          # FSFM 检测器
├── effort_detector.py         # Effort 检测器（已存在）
├── __init__.py                # 注册所有检测器
└── utils/                     # 工具目录（需要手动复制）
    ├── forada/                # ForAda 组件
    └── fsfm/                  # FSFM 组件
```

## 后续工作

1. 如果需要完整的 ForAda 和 FSFM 功能，请按照上述步骤复制相关组件
2. 测试所有模型在 DeepfakeBench 框架下的训练和推理
3. 根据实际需求调整模型配置和参数

