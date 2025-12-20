"""
诊断 GenD 模型显存使用情况的脚本
"""
import torch
import torch.nn as nn
from gend_detector import GenDDetector, CLIPEncoder, LinearProbe

def check_model_memory():
    """检查模型显存占用"""
    print("=" * 80)
    print("GenD 模型显存使用诊断")
    print("=" * 80)
    
    # 模拟配置
    config = {
        'backbone': 'openai/clip-vit-large-patch14',
        'head': 'linear',
        'num_classes': 2,
        'freeze_feature_extractor': True,
        'loss': {
            'ce_labels': 1.0,
            'uniformity': 0.5,
            'alignment_labels': 0.1,
        }
    }
    
    # 创建模型
    model = GenDDetector(config=config).cuda()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n📊 模型参数量统计:")
    print(f"  总参数量: {total_params:,} ({total_params * 4 / 1024**3:.2f} GB @ FP32)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params * 4 / 1024**3:.2f} GB @ FP32)")
    print(f"  冻结参数: {frozen_params:,} ({frozen_params * 4 / 1024**3:.2f} GB @ FP32)")
    print(f"  可训练比例: {trainable_params / total_params * 100:.4f}%")
    
    # 测试不同batch size的显存占用
    batch_sizes = [32, 64, 128, 256, 512]
    resolution = 224
    
    print(f"\n🔍 不同 Batch Size 的显存占用测试 (分辨率: {resolution}x{resolution}):")
    print("-" * 80)
    
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 创建输入
        images = torch.randn(bs, 3, resolution, resolution).cuda()
        labels = torch.randint(0, 2, (bs,)).cuda()
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            data_dict = {'image': images, 'label': labels}
            pred_dict = model(data_dict, inference=True)
        
        forward_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        # 训练模式（包含梯度）
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        data_dict = {'image': images, 'label': labels}
        pred_dict = model(data_dict, inference=False)
        losses = model.get_losses(data_dict, pred_dict)
        
        # 反向传播
        losses['overall'].backward()
        
        train_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        # 清理梯度
        model.zero_grad()
        torch.cuda.empty_cache()
        
        print(f"  Batch Size {bs:3d}: 前向 {forward_mem:.2f} GB | 训练 {train_mem:.2f} GB")
    
    # 检查实际batch size
    print(f"\n💡 建议:")
    print(f"  1. 如果显存使用率低，可以尝试:")
    print(f"     - 增加 batch size 到 512 或更大")
    print(f"     - 解冻部分 CLIP 层（设置 unfreeze_layers）")
    print(f"     - 使用更大的 CLIP 模型（如果显存充足）")
    print(f"  2. 当前配置下，batch size 256 的显存占用应该约为 8-10 GB")
    print(f"  3. 如果只有 9 GB，说明:")
    print(f"     - CLIP 编码器被冻结，不存储梯度 ✅")
    print(f"     - 只有分类头在训练 ✅")
    print(f"     - 这是正常的，符合 GenD 的设计 ✅")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    check_model_memory()

