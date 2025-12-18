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
    频域适配分支（占位实现）：
    - 当前版本在 feature 维度上做一个简单的 MLP 变换，
      后续可以替换成基于 FFT/高通滤波的设计。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        return self.mlp(x)


class BoundaryMining(nn.Module):
    """
    Boundary Mining 模块（占位实现）：
    - 这里通过一个门控权重来放大“疑似伪迹”区域，对应论文中的边界挖掘思想。
    - 后续可以替换为基于高通特征和边缘响应的更细致实现。
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


class IdentityInvariantHead(nn.Module):
    """
    身份不变性分支（占位实现）：
    - 主任务：伪造二分类
    - 副任务：身份预测（目前仅作为接口，占位实现，可后续接入 GRL 做对抗）。
    """

    def __init__(self, feat_dim: int, num_classes: int = 2, id_dim: int = 128):
        super().__init__()
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.id_head = nn.Linear(feat_dim, id_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_logit = self.cls_head(x)
        id_feat = self.id_head(x)
        return {"cls": cls_logit, "id": id_feat}


@DETECTOR.register_module(module_name="clip_adapter")
class CLIPAdapterDetector(AbstractDetector):
    """
    基于 CLIP ViT-B/16 视觉编码器的帧级伪造检测器（课题版）：
    - 只使用视觉分支，不使用文本分支；
    - 支持 Freq-Adapter、Boundary Mining、Identity-Invariant 等模块的占位实现；
    - 采用 LN-tuning 思路：默认仅微调 CLIP 中的 LayerNorm 以及新增头部参数。
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.config = config

        # 1. Backbone：CLIP 视觉编码器
        self.backbone = self.build_backbone(config)
        self.feat_dim = 768  # ViT-B/16 hidden size

        # 冻结大部分参数，仅保留 LayerNorm 参与训练（LN-tuning）
        self._apply_ln_tuning()

        # 2. 适配模块开关
        modules_cfg = self.config.get("modules", {})
        self.use_freq_adapter = modules_cfg.get("use_freq_adapter", True)
        self.use_boundary_mining = modules_cfg.get("use_boundary_mining", True)
        self.use_identity_invariant = modules_cfg.get("use_identity_invariant", True)

        # 3. 模块定义
        self.freq_adapter = FreqAdapter(self.feat_dim) if self.use_freq_adapter else None
        self.boundary_mining = BoundaryMining(self.feat_dim) if self.use_boundary_mining else None
        self.head = IdentityInvariantHead(
            self.feat_dim,
            num_classes=2,
        )

        # 4. 损失
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        """
        构建 CLIP 视觉编码器作为 backbone。
        """
        _, backbone = get_clip_visual("openai/clip-vit-base-patch16")
        return backbone

    def build_loss(self, config):
        """
        构建主损失函数。
        """
        loss_class = LOSSFUNC[config["loss_func"]]
        loss_func = loss_class()
        return loss_func

    def _apply_ln_tuning(self):
        """
        LN-tuning：冻结除 LayerNorm 以外的大部分参数。
        """
        for name, param in self.backbone.named_parameters():
            if "layernorm" in name.lower() or "ln_" in name.lower() or "ln" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

    def features(self, data_dict: dict) -> torch.Tensor:
        """
        提取 CLS 级别特征：
        - 默认使用 backbone 的 pooler_output；
        - 进入下游模块前做 L2 归一化，保证嵌入分布稳定。
        """
        # transformers 的 CLIP vision_model 接口：返回 ModelOutput
        out = self.backbone(data_dict["image"])
        feat = out["pooler_output"]  # (B, D)
        feat = F.normalize(feat, p=2, dim=-1)

        # Freq-Adapter
        if self.use_freq_adapter and self.freq_adapter is not None:
            feat = feat + self.freq_adapter(feat)

        # Boundary Mining
        if self.use_boundary_mining and self.boundary_mining is not None:
            feat = self.boundary_mining(feat)

        return feat

    def classifier(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        统一通过 IdentityInvariantHead 输出分类和身份分支。
        """
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> Dict[str, torch.Tensor]:
        """
        目前只实现主任务分类损失，其它辅助损失按需求逐步接入。
        """
        label = data_dict["label"]
        cls_logit = pred_dict["cls"]
        loss_overall = self.loss_func(cls_logit, label)
        loss_dict = {"overall": loss_overall}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> Dict[str, torch.Tensor]:
        label = data_dict["label"]
        cls_logit = pred_dict["cls"]
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), cls_logit.detach())
        metric_batch_dict = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference: bool = False) -> Dict[str, torch.Tensor]:
        feat = self.features(data_dict)
        out = self.classifier(feat)
        cls_logit = out["cls"]
        prob = torch.softmax(cls_logit, dim=1)[:, 1]
        pred_dict = {
            "cls": cls_logit,
            "prob": prob,
            "feat": feat,
            "id_feat": out.get("id", None),
        }
        return pred_dict


