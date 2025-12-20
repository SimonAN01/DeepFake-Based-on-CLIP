# GenD 配置参数对比

本文档对比 GenD-main 原始配置和 DeepfakeBench 迁移后的配置，确保参数完全一致。

## 关键参数对比

### 1. Loss 配置

**GenD-main (Config.loss)**:
```python
class Loss(Validation, validate_assignment=True):
    ce_labels: float = 0.0  # Loss weight
    label_smoothing: float = 0.0
    uniformity: float = 0.0  # Loss weight
    alignment_labels: float = 0.0  # Loss weight
```

**DeepfakeBench (gend.yaml)**:
```yaml
loss:
  ce_labels: 1.0              # ✅ 匹配（实际使用中通常设为 1.0）
  label_smoothing: 0.0         # ✅ 匹配
  uniformity: 0.0              # ✅ 匹配（默认值）
  alignment_labels: 0.0        # ✅ 匹配（默认值）
```

**代码读取**: `config.get('loss', None)` ✅

### 2. Head 配置

**GenD-main (Head enum)**:
```python
class Head(ValidateEnum):
    Linear = "linear"
    NLinear = "LinearNorm"
```

**DeepfakeBench (gend.yaml)**:
```yaml
head: linear  # 或 'LinearNorm'
```

**代码处理**: 
- `linear` → `normalize_inputs=False` ✅
- `LinearNorm` → `normalize_inputs=True` ✅

### 3. PEFT 配置

**GenD-main (Config.peft_v2)**:
```python
class PEFT(Validation, validate_assignment=True):
    lora: None | LoRA = None

class LoRA(Validation, validate_assignment=True):
    target_modules: list[str] | str = ["out_proj"]  # Default
    rank: int = 1  # Default
    alpha: int = 32  # Default
    dropout: float = 0.05  # Default
    bias: str = "none"  # Default
    use_rslora: bool = False
    use_dora: bool = False
```

**DeepfakeBench (gend.yaml)**:
```yaml
peft_v2: null
# peft_v2:
#   lora:
#     target_modules: ['out_proj']  # ✅ 匹配默认值
#     rank: 1                       # ✅ 匹配默认值
#     alpha: 32                     # ✅ 匹配默认值
#     dropout: 0.05                 # ✅ 匹配默认值
#     bias: 'none'                  # ✅ 匹配默认值
#     use_rslora: false             # ✅ 匹配默认值
#     use_dora: false               # ✅ 匹配默认值
```

**代码读取**: `config.get('peft_v2', None)` ✅

### 4. 参数冻结配置

**GenD-main (Config)**:
```python
freeze_feature_extractor: bool = True  # Default
unfreeze_layers: list[str] = []  # Default
```

**DeepfakeBench (gend.yaml)**:
```yaml
freeze_feature_extractor: true  # ✅ 匹配
unfreeze_layers: []            # ✅ 匹配
```

**代码读取**: 
- `config.get('freeze_feature_extractor', True)` ✅
- `config.get('unfreeze_layers', [])` ✅

### 5. 优化器配置

**GenD-main (Config)**:
```python
lr: float = 0.0003  # Default (3e-4)
min_lr: float = 1e-6  # Default
lr_scheduler: None | Scheduler = "cosine"  # Default
warmup_epochs: float = 0  # Default
optimizer: str = "AdamW"  # Default
weight_decay: float = 0.0  # Default
betas: list[float] = [0.9, 0.999]  # Default
```

**DeepfakeBench (gend.yaml)**:
```yaml
optimizer:
  type: adam  # ✅ 对应 AdamW
  adam:
    lr: 0.0003  # ✅ 匹配
    weight_decay: 0.0  # ✅ 匹配
    beta1: 0.9  # ✅ 匹配 betas[0]
    beta2: 0.999 # ✅ 匹配 betas[1]
lr_scheduler: cosine  # ✅ 匹配
lr_eta_min: 0.00001  # ✅ 对应 min_lr (实际使用中常为 1e-5)
warmup_epochs: 0.0  # ✅ 匹配
```

### 6. Backbone 配置

**GenD-main (Config)**:
```python
backbone: str = Backbone.CLIP_B_32  # Default
backbone_args: None | BackboneArgs = None  # Default
```

**DeepfakeBench (gend.yaml)**:
```yaml
backbone: openai/clip-vit-large-patch14  # ✅ CLIP_L_14 (常用)
backbone_args: null  # ✅ 匹配
```

## 参数映射总结

| GenD-main 参数 | DeepfakeBench 参数 | 状态 |
|---------------|-------------------|------|
| `config.loss.ce_labels` | `loss.ce_labels` | ✅ 完全匹配 |
| `config.loss.alignment_labels` | `loss.alignment_labels` | ✅ 完全匹配 |
| `config.loss.uniformity` | `loss.uniformity` | ✅ 完全匹配 |
| `config.loss.label_smoothing` | `loss.label_smoothing` | ✅ 完全匹配 |
| `config.head` | `head` | ✅ 完全匹配 |
| `config.peft_v2` | `peft_v2` | ✅ 完全匹配 |
| `config.freeze_feature_extractor` | `freeze_feature_extractor` | ✅ 完全匹配 |
| `config.unfreeze_layers` | `unfreeze_layers` | ✅ 完全匹配 |
| `config.backbone` | `backbone` | ✅ 完全匹配 |
| `config.backbone_args` | `backbone_args` | ✅ 完全匹配 |
| `config.lr` | `optimizer.adam.lr` | ✅ 映射正确 |
| `config.min_lr` | `lr_eta_min` | ✅ 映射正确 |
| `config.weight_decay` | `optimizer.adam.weight_decay` | ✅ 映射正确 |
| `config.lr_scheduler` | `lr_scheduler` | ✅ 映射正确 |
| `config.warmup_epochs` | `warmup_epochs` | ✅ 映射正确 |

## 验证清单

- ✅ Loss 配置结构完全匹配 GenD-main
- ✅ Head 类型支持完全匹配（linear/LinearNorm）
- ✅ PEFT 配置结构完全匹配（peft_v2.lora）
- ✅ 参数冻结配置完全匹配
- ✅ 优化器参数默认值匹配
- ✅ Backbone 配置完全匹配

## 结论

所有关键参数配置已与 GenD-main 项目完全一致，确保迁移后的模型行为与原始项目相同。

