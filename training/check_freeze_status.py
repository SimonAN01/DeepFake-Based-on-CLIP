"""
快速检查 CLIP Adapter 模型的参数冻结状态
"""
import sys
import os
sys.path.append('.')

import yaml
import torch
from transformers import CLIPModel
from training.detectors.clip_adapter_detector import CLIPAdapterDetector

# 加载配置
with open('training/config/detector/clip_adapter.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 添加必要的配置项
config['lmdb'] = False
config['rgb_dir'] = './datasets/rgb'
config['lmdb_dir'] = './datasets/lmdb'
config['dataset_json_folder'] = 'preprocessing/dataset_json'
config['label_dict'] = {'FF-real': 0, 'FF-F2F': 1, 'FF-DF': 1, 'FF-FS': 1, 'FF-NT': 1}

print("=" * 80)
print("CLIP Adapter 参数冻结状态检查")
print("=" * 80)
print(f"冻结策略: {config['backbone_config']['mode']}")
print()

# 创建模型
model = CLIPAdapterDetector(config)

# 统计参数
total_params = sum(p.numel() for p in model.backbone.parameters())
trainable_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
frozen_backbone = total_params - trainable_backbone

# 新模块参数
new_modules_params = 0
if model.freq_adapter:
    new_modules_params += sum(p.numel() for p in model.freq_adapter.parameters())
if model.boundary_mining:
    new_modules_params += sum(p.numel() for p in model.boundary_mining.parameters())
if model.head:
    new_modules_params += sum(p.numel() for p in model.head.parameters())

print("📊 参数统计:")
print(f"  Backbone 总参数: {total_params:,}")
print(f"  ✅ Backbone 可训练: {trainable_backbone:,} ({100*trainable_backbone/total_params:.2f}%)")
print(f"  ❌ Backbone 冻结: {frozen_backbone:,} ({100*frozen_backbone/total_params:.2f}%)")
print(f"  ✅ 新模块可训练: {new_modules_params:,} (Freq-Adapter + Boundary + Head)")
print(f"  📦 总可训练参数: {trainable_backbone + new_modules_params:,}")
print()

print("🔍 详细冻结状态:")
print("-" * 80)

# 按模块分组统计
module_stats = {}
for name, param in model.backbone.named_parameters():
    # 提取模块名（例如：encoder.layers.0.self_attn.q_proj）
    parts = name.split('.')
    if len(parts) >= 2:
        module_key = '.'.join(parts[:2])  # 例如：encoder.layers
    else:
        module_key = parts[0]
    
    if module_key not in module_stats:
        module_stats[module_key] = {'trainable': 0, 'frozen': 0}
    
    if param.requires_grad:
        module_stats[module_key]['trainable'] += param.numel()
    else:
        module_stats[module_key]['frozen'] += param.numel()

print("\n✅ 可训练的模块:")
for mod, stats in sorted(module_stats.items()):
    if stats['trainable'] > 0:
        total_mod = stats['trainable'] + stats['frozen']
        print(f"  {mod:40s} {stats['trainable']:>12,} / {total_mod:>12,} ({100*stats['trainable']/total_mod:>5.2f}%)")

print("\n❌ 完全冻结的模块:")
for mod, stats in sorted(module_stats.items()):
    if stats['trainable'] == 0 and stats['frozen'] > 0:
        print(f"  {mod:40s} {stats['frozen']:>12,} params")

print("\n" + "=" * 80)
print("💡 提示:")
print("  - 当前使用 'ln_tuning' 模式，仅 LayerNorm 层可训练")
print("  - 如需解冻更多层，修改 config 中的 mode 和 unfreeze_last_n_layers")
print("  - Freq-Adapter、Boundary Mining、Head 始终可训练")
print("=" * 80)

