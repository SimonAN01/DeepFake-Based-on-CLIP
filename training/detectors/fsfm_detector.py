'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the FSFMDetector
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


@DETECTOR.register_module(module_name='fsfm')
class FSFMDetector(AbstractDetector):
    def __init__(self, config=None, load_param: Union[bool, str] = False):
        super().__init__(config, load_param)
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = self.build_head(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # FSFM uses VisionTransformer with adapter
        # This requires models from GenD-main/src/model/fsfm/
        try:
            from training.detectors.utils.fsfm.models_vit_fs_adapter import vit_large_patch16
            from training.detectors.utils.fsfm.models_vit import vit_large_patch16 as vit_large_patch16_no_adapter
            
            checkpoint_path = config.get('checkpoint', '') if config else ''
            
            if 'adapter' in checkpoint_path.lower() or config.get('use_adapter', False):
                model = vit_large_patch16(
                    num_classes=2,
                    drop_path_rate=0.1,
                    global_pool=True
                )
            else:
                model = vit_large_patch16_no_adapter(
                    num_classes=2,
                    drop_path_rate=0.1,
                    global_pool=True
                )
            
            return model
        except ImportError:
            logger.warning("FSFM model components not found. Please copy the fsfm directory from GenD-main to training/detectors/utils/")
            # Fallback to a simple ViT
            import timm
            model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=2)
            return model
    
    def build_head(self, config):
        # FSFM model already includes the head
        return nn.Identity()
    
    def build_loss(self, config):
        loss_class = LOSSFUNC[config.get('loss_func', 'cross_entropy')] if config else LOSSFUNC['cross_entropy']
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        images = data_dict['image']
        # FSFM model forward pass
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(images)
            if isinstance(features, torch.Tensor) and len(features.shape) == 3:
                # If it's sequence output, use global pooling
                features = features.mean(dim=1)
            return features
        else:
            # For models with integrated head
            logits = self.backbone(images)
            return logits
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        if isinstance(self.head, nn.Identity):
            # Head is already integrated in backbone
            return features
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
        images = data_dict['image']
        
        # FSFM model forward
        if hasattr(self.backbone, 'forward'):
            outputs = self.backbone(images)
            # Swap outputs if needed (FSFM outputs [fake, real], we need [real, fake])
            if outputs.shape[-1] == 2:
                outputs = outputs[..., [1, 0]]
            pred = outputs
            features = pred
        else:
            features = self.features(data_dict)
            pred = self.classifier(features)
        
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict

