import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, CLIPModel

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train


logger = logging.getLogger(__name__)


def get_clip_visual(model_name: str = "openai/clip-vit-base-patch16"):
    """
    加载 CLIP 视觉编码器，仅返回视觉分支。
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model


class FreqAdapter(nn.Module):
    """
    Freq-Adapter 频域适配分支（对应课题中的高频支路）：
    - 使用固定拉普拉斯高通卷积近似频域高频响应；
    - 再通过轻量 CNN + 全局池化 + 线性层映射到与 CLS 相同的特征维度；
    - 输出作为残差添加到 CLS 特征，实现“频域支路 + 主干融合”。
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 768, hidden_channels: int = 32):
        super().__init__()

        # 固定高通卷积核（Laplacian），模拟频域高频信息
        lap_kernel = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.lap = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=in_channels,
        )
        with torch.no_grad():
            self.lap.weight.copy_(lap_kernel.repeat(in_channels, 1, 1, 1))
        for p in self.lap.parameters():
            p.requires_grad = False

        # 频域特征提取（可学习）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Linear(hidden_channels, feat_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, C, H, W)
        return: (B, D) 频域残差信号，用于与 CLS 特征融合
        """
        # 高通响应（边缘/伪迹增强）
        with torch.no_grad():
            high_freq = self.lap(image)

        # 可学习频域分支
        x = self.conv(high_freq)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, hidden_channels)
        freq_feat = self.proj(x)  # (B, feat_dim)
        return freq_feat


class BoundaryMining(nn.Module):
    """
    Boundary Mining 模块（对应课题中的边界挖掘）：
    - 在特征维度上自适应估计“伪迹边界”权重；
    - 通过残差门控放大边界相关维度。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x)
        return x * (1.0 + w)


class GradientReversalFn(torch.autograd.Function):
    """
    GRL：Gradient Reversal Layer
    - 前向恒等映射；
    - 反向将梯度乘以 -lambda，实现对抗训练。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambd)


class IdentityInvariantHead(nn.Module):
    """
    Identity-Invariant 分支（对应课题中的身份对抗模块）：
    - 主任务：伪造二分类；
    - 副任务：身份预测（通过 GRL 做对抗），减弱 backbone 中的身份信息。
    """

    def __init__(self, feat_dim: int, num_classes: int = 2, id_dim: int = 128, grl_lambda: float = 1.0):
        super().__init__()
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.id_head = nn.Linear(feat_dim, id_dim)
        self.grl_lambda = grl_lambda

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 主任务：直接使用 CLS 特征
        cls_logit = self.cls_head(x)

        # 身份分支：先通过 GRL，再送入 ID 头，实现对抗正则
        id_input = grad_reverse(x, self.grl_lambda)
        id_feat = self.id_head(id_input)
        return {"cls": cls_logit, "id": id_feat}


@DETECTOR.register_module(module_name="clip_adapter")
class CLIPAdapterDetector(AbstractDetector):
    """
    基于 CLIP ViT-B/16 视觉表征的帧级跨域人脸伪造检测器：
    - 只使用视觉编码器，主干大体冻结，仅微调 LayerNorm（LN-tuning）与新头部；
    - 引入 Freq-Adapter 频域支路、Boundary Mining 边界挖掘、Identity-Invariant 身份对抗模块；
    - 在特征空间加入角度约束、对齐/均匀度量、频域一致与身份对抗等正则（通过 config.aux_losses 控制）。
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.config = config

        # 1. Backbone：CLIP 视觉编码器
        self.backbone = self.build_backbone(config)
        self.feat_dim = 768  # ViT-B/16 hidden size

        # 3. 模块定义（在冻结策略之前先定义，确保它们会被标记为可训练）
        modules_cfg = self.config.get("modules", {})
        self.use_freq_adapter = modules_cfg.get("use_freq_adapter", True)
        self.use_boundary_mining = modules_cfg.get("use_boundary_mining", True)
        self.use_identity_invariant = modules_cfg.get("use_identity_invariant", True)

        self.freq_adapter = FreqAdapter(in_channels=3, feat_dim=self.feat_dim) if self.use_freq_adapter else None
        self.boundary_mining = BoundaryMining(self.feat_dim) if self.use_boundary_mining else None

        aux_losses_cfg = self.config.get("aux_losses", {})
        grl_lambda = aux_losses_cfg.get("adv_lambda", 1.0)
        id_dim = aux_losses_cfg.get("id_dim", 128)

        self.head = IdentityInvariantHead(
            self.feat_dim,
            num_classes=2,
            id_dim=id_dim,
            grl_lambda=grl_lambda,
        )

        # 应用冻结策略（确保新模块始终可训练）
        self._apply_ln_tuning()


        # 4. 主损失
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        """
        构建 CLIP 视觉编码器作为 backbone。
        """
        _, backbone = get_clip_visual("openai/clip-vit-base-patch16")
        return backbone

    def build_loss(self, config):
        """
        构建主分类损失函数。
        """
        loss_class = LOSSFUNC[config["loss_func"]]
        loss_func = loss_class()
        return loss_func

    def _apply_ln_tuning(self):
        """
        灵活的冻结策略：
        - ln_tuning: 仅解冻 LayerNorm 层
        - last_n_layers: 解冻最后 N 层 Transformer + 所有 LayerNorm
        - full: 全解冻（用于完整微调）
        """
        mode = self.config.get("backbone_config", {}).get("mode", "ln_tuning")
        unfreeze_n = self.config.get("backbone_config", {}).get("unfreeze_last_n_layers", 0)
        
        if mode == "full":
            # 全解冻：所有参数可训练
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone: Full fine-tuning (all parameters trainable)")
            
        elif mode == "last_n_layers" and unfreeze_n > 0:
            # 解冻最后 N 层 Transformer + 所有 LayerNorm
            total_layers = len(self.backbone.encoder.layers)
            freeze_until = max(0, total_layers - unfreeze_n)
            
            for name, param in self.backbone.named_parameters():
                # 解冻 LayerNorm
                if "layernorm" in name.lower() or "ln_" in name.lower() or "ln" in name.lower():
                    param.requires_grad = True
                # 解冻最后 N 层 Transformer
                elif "encoder.layers" in name:
                    # 提取层号：encoder.layers.11.xxx -> 11
                    try:
                        layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
                        if layer_idx >= freeze_until:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    except:
                        param.requires_grad = False
                # 解冻 pooler（如果有）
                elif "visual_projection" in name or "pooler" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            logger.info(
                f"Backbone: Unfreeze last {unfreeze_n} layers + all LayerNorm "
                f"({trainable}/{total} params trainable, {100*trainable/total:.2f}%)"
            )
            
        else:
            # 默认：LN-tuning（仅 LayerNorm）
            frozen_modules = set()
            trainable_modules = set()
            
            for name, param in self.backbone.named_parameters():
                if "layernorm" in name.lower() or "ln_" in name.lower() or "ln" in name.lower():
                    param.requires_grad = True
                    # 提取模块名（例如：encoder.layers.0.layer_norm1）
                    module_name = ".".join(name.split(".")[:-1]) if "." in name else name
                    trainable_modules.add(module_name)
                else:
                    param.requires_grad = False
                    module_name = ".".join(name.split(".")[:-1]) if "." in name else name
                    frozen_modules.add(module_name)
            
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.backbone.parameters())
            
            logger.info("=" * 80)
            logger.info(f"Backbone Freezing Strategy: LN-tuning only")
            logger.info(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")
            logger.info("-" * 80)
            logger.info("✅ TRAINABLE (LayerNorm layers only):")
            for mod in sorted(trainable_modules):
                logger.info(f"  - {mod}")
            logger.info("-" * 80)
            logger.info("❌ FROZEN (all other parameters):")
            # 只显示主要模块，避免日志过长
            main_frozen = {m for m in frozen_modules if not any(tm in m for tm in trainable_modules)}
            for mod in sorted(list(main_frozen)[:15]):  # 只显示前15个
                logger.info(f"  - {mod}")
            if len(main_frozen) > 15:
                logger.info(f"  ... and {len(main_frozen) - 15} more frozen modules")
            logger.info("=" * 80)
        
        # 确保新添加的模块（Freq-Adapter、Boundary Mining、Head）始终可训练
        for module in [self.freq_adapter, self.boundary_mining, self.head]:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def features(self, data_dict: dict):
        """
        提取 CLS 级别特征：
        - 使用 CLIP vision_model 的 pooler_output；
        - 进入下游模块前做 L2 归一化；
        - 与 Freq-Adapter、Boundary Mining 进行融合。
        """
        # transformers 的 CLIP vision_model 接口：返回 ModelOutput
        out = self.backbone(data_dict["image"])
        feat_raw = out["pooler_output"]  # (B, D)
        feat_raw = F.normalize(feat_raw, p=2, dim=-1)

        feat_after_freq = feat_raw
        # Freq-Adapter：显式利用原始图像的高频信息，生成残差信号
        if self.use_freq_adapter and self.freq_adapter is not None:
            freq_delta = self.freq_adapter(data_dict["image"])
            feat_after_freq = feat_raw + freq_delta

        feat_final = feat_after_freq
        # Boundary Mining：在特征空间做边界挖掘
        if self.use_boundary_mining and self.boundary_mining is not None:
            feat_final = self.boundary_mining(feat_after_freq)

        return {
            "feat": feat_final,
            "feat_raw": feat_raw,
            "feat_freq": feat_after_freq,
        }

    def classifier(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        统一通过 IdentityInvariantHead 输出分类和身份分支。
        """
        if not self.use_identity_invariant:
            # 仅使用分类头
            cls_head = nn.Linear(self.feat_dim, 2).to(features.device)
            cls_logit = cls_head(features)
            return {"cls": cls_logit, "id": None}
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> Dict[str, torch.Tensor]:
        """
        主任务分类损失 + 若干辅助正则：
        - 角度约束：同类特征聚拢到各自类别均值附近；
        - 特征对齐/均匀：整体特征分布居中、不过度塌缩；
        - 频域一致性：约束频域适配后的特征不过度偏离原始 CLS；
        - 身份对抗：为后续接入 ID 标签预留接口（需在 data_dict 中提供相关字段）。
        """
        label = data_dict["label"]
        cls_logit = pred_dict["cls"]
        feat = pred_dict["feat"]
        feat_raw = pred_dict.get("feat_raw", None)
        feat_freq = pred_dict.get("feat_freq", None)
        id_feat = pred_dict.get("id_feat", None)

        # 主分类损失
        loss_overall = self.loss_func(cls_logit, label)
        loss_dict = {"overall": loss_overall}

        aux_cfg = self.config.get("aux_losses", {})

        # 1) 角度约束：样本特征与其类别均值之间的角度尽量小
        if aux_cfg.get("use_angle", False) and feat_raw is not None:
            eps = 1e-6
            angle_loss_val = torch.tensor(0.0, device=feat_raw.device)
            unique_labels = torch.unique(label)
            for cls_id in unique_labels:
                mask = (label == cls_id)
                if mask.sum() < 2:
                    continue
                cls_feat = feat_raw[mask]
                cls_center = cls_feat.mean(dim=0, keepdim=True)  # (1, D)
                cls_center = F.normalize(cls_center, p=2, dim=-1)
                cos = (cls_feat * cls_center).sum(dim=-1).clamp(-1 + eps, 1 - eps)
                theta = torch.acos(cos)
                angle_loss_val = angle_loss_val + theta.mean()
            loss_angle = angle_loss_val / max(len(unique_labels), 1)
            loss_dict["angle_loss"] = loss_angle * aux_cfg.get("angle_weight", 0.1)
            loss_dict["overall"] = loss_dict["overall"] + loss_dict["angle_loss"]

        # 2) 特征对齐/均匀度量：鼓励整体分布居中且不过度塌缩
        if aux_cfg.get("use_alignment", False):
            # 特征均值接近 0
            mean_center = feat.mean(dim=0)
            loss_center = (mean_center ** 2).mean()
            # 特征方差不过小，避免全部塌缩到一点
            var = feat.var(dim=0, unbiased=False)
            loss_var = F.relu(0.1 - var).mean()
            loss_align = loss_center + loss_var
            loss_dict["alignment_loss"] = loss_align * aux_cfg.get("alignment_weight", 0.1)
            loss_dict["overall"] = loss_dict["overall"] + loss_dict["alignment_loss"]

        # 3) 频域一致性：约束频域适配后特征不过度偏离原始 CLS
        if aux_cfg.get("use_freq_consistency", False) and (feat_raw is not None) and (feat_freq is not None):
            loss_freq = F.mse_loss(feat_freq, feat_raw)
            loss_dict["freq_consistency_loss"] = loss_freq * aux_cfg.get("freq_consistency_weight", 0.1)
            loss_dict["overall"] = loss_dict["overall"] + loss_dict["freq_consistency_loss"]

        # 4) 身份对抗正则：
        #    使用同一 video_name 的样本作为“同一身份”的正样对，基于 id_feat 计算 IDLoss。
        if aux_cfg.get("use_adversarial_identity", False) and (id_feat is not None):
            video_names = data_dict.get("video_name", None)
            if video_names is not None:
                # 按 video_name 分组样本索引
                from collections import defaultdict
                groups = defaultdict(list)
                for idx, vname in enumerate(video_names):
                    groups[vname].append(idx)

                pos_pairs = []
                for _, idx_list in groups.items():
                    if len(idx_list) < 2:
                        continue
                    # 简单策略：在该组内构造所有两两正样对
                    for i in range(len(idx_list)):
                        for j in range(i + 1, len(idx_list)):
                            pos_pairs.append((idx_list[i], idx_list[j]))

                if len(pos_pairs) > 0:
                    id_loss_class = LOSSFUNC["id_loss"]
                    id_loss_func = id_loss_class()
                    total_id_loss = 0.0
                    for i, j in pos_pairs:
                        total_id_loss = total_id_loss + id_loss_func(id_feat[i].unsqueeze(0), id_feat[j].unsqueeze(0))
                    total_id_loss = total_id_loss / len(pos_pairs)

                    loss_dict["adversarial_identity_loss"] = total_id_loss * aux_cfg.get(
                        "adversarial_identity_weight", 0.1
                    )
                    loss_dict["overall"] = loss_dict["overall"] + loss_dict["adversarial_identity_loss"]

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> Dict[str, torch.Tensor]:
        label = data_dict["label"]
        cls_logit = pred_dict["cls"]
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), cls_logit.detach())
        metric_batch_dict = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference: bool = False) -> Dict[str, torch.Tensor]:
        feat_dict = self.features(data_dict)
        out = self.classifier(feat_dict["feat"])
        cls_logit = out["cls"]
        prob = torch.softmax(cls_logit, dim=1)[:, 1]
        pred_dict = {
            "cls": cls_logit,
            "prob": prob,
            "feat": feat_dict["feat"],
            "feat_raw": feat_dict.get("feat_raw", None),
            "feat_freq": feat_dict.get("feat_freq", None),
            "id_feat": out.get("id", None),
        }
        return pred_dict


