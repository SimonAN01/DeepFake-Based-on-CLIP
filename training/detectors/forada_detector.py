'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the ForAdaDetector
# Migrated from GenD-main project

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


# ForAda model components - simplified version
# Note: Full implementation requires the complete ForAda model structure from GenD-main
# This is a placeholder that can be extended with the full ForAda implementation

@DETECTOR.register_module(module_name='forada')
class ForAdaDetector(AbstractDetector):
    def __init__(self, config=None, load_param: Union[bool, str] = False):
        super().__init__(config, load_param)
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = self.build_head(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # ForAda uses DS model which combines CLIP and Adapter
        # This is a placeholder - full implementation requires:
        # 1. DS model from GenD-main/src/model/forada/ds.py
        # 2. Adapter from GenD-main/src/model/forada/adapters/adapter.py
        # 3. CLIP model from GenD-main/src/model/forada/clip/
        # 4. Attention modules from GenD-main/src/model/forada/attn.py
        # 5. Layer modules from GenD-main/src/model/forada/layer.py
        
        # For now, we'll create a simplified version
        # In practice, you should copy the entire forada directory structure
        try:
            from training.detectors.utils.forada.ds import DS
            
            # Load config if provided
            if config and 'forada_config' in config:
                with open(config['forada_config'], 'r') as f:
                    forada_config = yaml.safe_load(f)
            else:
                # Default config
                forada_config = {
                    "clip_model_name": "ViT-L/14",
                    "vit_name": "vit_base_patch16_224",
                    "num_quires": 8,
                    "fusion_map": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7},
                    "mlp_dim": 512,
                    "mlp_out_dim": 256,
                    "head_num": 16
                }
            
            model = DS(
                clip_name=forada_config["clip_model_name"],
                adapter_vit_name=forada_config["vit_name"],
                num_quires=forada_config["num_quires"],
                fusion_map=forada_config["fusion_map"],
                mlp_dim=forada_config["mlp_dim"],
                mlp_out_dim=forada_config["mlp_out_dim"],
                head_num=forada_config["head_num"],
            )
            return model
        except ImportError:
            logger.warning("ForAda model components not found. Please copy the forada directory from GenD-main to training/detectors/utils/")
            # Fallback to a simple CLIP-based model
            from transformers import CLIPModel
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            return clip_model.vision_model
    
    def build_head(self, config):
        # ForAda uses PostClipProcess head
        # This should be implemented based on GenD-main/src/model/forada/layer.py
        num_classes = config.get('num_classes', 2) if config else 2
        return nn.Linear(768, num_classes)  # Simplified version
    
    def build_loss(self, config):
        loss_class = LOSSFUNC[config.get('loss_func', 'cross_entropy')] if config else LOSSFUNC['cross_entropy']
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        images = data_dict['image']
        # ForAda model forward pass
        if hasattr(self.backbone, 'forward'):
            # If it's the full DS model
            pred_dict = self.backbone({'image': images}, inference=True)
            return pred_dict.get('logits', pred_dict.get('feat'))
        else:
            # Fallback for simple backbone
            return self.backbone(images).pooler_output
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict

