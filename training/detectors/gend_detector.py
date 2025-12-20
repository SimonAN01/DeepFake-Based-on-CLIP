'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the GenDDetector
# Migrated from GenD-main project - Complete migration with CLIP encoder only

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
from typing import Union, Optional
from collections import defaultdict
from dataclasses import dataclass

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
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


# ==================== Loss Components ====================

def alignment(embeddings: torch.Tensor, labels: torch.Tensor, alpha: float = 2):
    """
    Label-aware Alignment loss.
    Calculates alignment for embeddings of samples with the SAME label within a batch.
    """
    assert embeddings.size(0) == labels.size(0), "Embeddings and labels must have the same size."
    
    n_samples = embeddings.size(0)
    if n_samples < 2:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Create a pairwise label comparison matrix (N x N), exclude self-pairs
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    
    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Get embeddings of positive pairs
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]
    
    # Calculate alignment loss
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x: torch.Tensor, t: float = 2, clip_value: float = 1e-6):
    """
    Calculates the Uniformity loss.
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


@dataclass
class LossInputs:
    logits_labels: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    l2_embeddings: Optional[torch.Tensor] = None


@dataclass
class LossOutputs:
    ce_labels: Optional[float] = None
    uniformity: Optional[float] = None
    alignment_labels: Optional[float] = None
    total: Union[int, torch.Tensor] = 0


class GenDLoss(nn.Module):
    """GenD custom loss with alignment and uniformity"""
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else {}
        # Loss coefficients
        self.ce_labels = self.config.get('ce_labels', 1.0)
        self.alignment_labels = self.config.get('alignment_labels', 0.0)
        self.uniformity = self.config.get('uniformity', 0.0)
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
    
    def forward(self, inputs: LossInputs) -> LossOutputs:
        loss_outputs = LossOutputs()
        
        if inputs.logits_labels is not None:
            if self.ce_labels > 0:
                L = self.ce_labels * F.cross_entropy(
                    inputs.logits_labels, inputs.labels, label_smoothing=self.label_smoothing
                )
                loss_outputs.ce_labels = L.item() if isinstance(L, torch.Tensor) else L
                loss_outputs.total += L
        
        if inputs.l2_embeddings is not None:
            l2_embeddings = inputs.l2_embeddings
            
            # Check that embeddings are normalized
            if not torch.allclose(
                l2_embeddings.norm(p=2, dim=1),
                torch.ones(l2_embeddings.size(0), device=l2_embeddings.device, dtype=l2_embeddings.dtype),
                atol=1e-3
            ):
                logger.warning("Embeddings are not normalized")
            
            if inputs.labels is not None:
                if self.alignment_labels > 0:
                    L = self.alignment_labels * alignment(l2_embeddings, inputs.labels)
                    loss_outputs.alignment_labels = L.item() if isinstance(L, torch.Tensor) else L
                    loss_outputs.total += L
            
            if self.uniformity > 0:
                L = self.uniformity * uniformity(l2_embeddings)
                loss_outputs.uniformity = L.item() if isinstance(L, torch.Tensor) else L
                loss_outputs.total += L
        
        if isinstance(loss_outputs.total, int):
            logger.warning("Total loss is 0. Check if loss coefficients are set correctly.")
        
        if isinstance(loss_outputs.total, torch.Tensor) and loss_outputs.total.isnan():
            logger.warning("Total loss is nan")
            loss_outputs.total = inputs.logits_labels.sum() * 0 if inputs.logits_labels is not None else torch.tensor(0.0)
        
        return loss_outputs


# ==================== Head Components ====================

@dataclass
class HeadOutput:
    logits_labels: Optional[torch.Tensor] = None
    l2_embeddings: Optional[torch.Tensor] = None


class LinearProbe(nn.Module):
    """
    Linear Probe Head for GenD model
    """
    def __init__(self, input_dim, num_classes, normalize_inputs=False, detach_classifier_inputs=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs
        self.detach_classifier_inputs = detach_classifier_inputs
    
    def forward(self, x: torch.Tensor, **kwargs) -> HeadOutput:
        l2_embeddings = F.normalize(x, p=2, dim=1)
        
        if self.normalize_inputs:
            x = l2_embeddings
        
        logits = self.linear(x if not self.detach_classifier_inputs else x.detach())
        
        return HeadOutput(logits_labels=logits, l2_embeddings=l2_embeddings)


# ==================== Encoder Components ====================

class CLIPEncoder(nn.Module):
    """CLIP Encoder for GenD model - Only CLIP encoder is used"""
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        """
        Models:
        1. openai/clip-vit-base-patch16 | 768 features
        2. openai/clip-vit-base-patch32 | 768 features
        3. openai/clip-vit-large-patch14 | 1024 features
        """
        super().__init__()
        
        try:
            self._preprocess = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        clip: CLIPModel = CLIPModel.from_pretrained(model_name)
        
        # take vision model from CLIP, maps image to vision_embed_dim
        self.vision_model = clip.vision_model
        
        self.model_name = model_name
        self.features_dim = self.vision_model.config.hidden_size
        
        # take visual_projection, maps vision_embed_dim to projection_dim
        self.visual_projection = clip.visual_projection
    
    def preprocess(self, image):
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]
    
    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(preprocessed_images).pooler_output
    
    def get_features_dim(self):
        return self.features_dim


# ==================== Main Detector ====================

@DETECTOR.register_module(module_name='gend')
class GenDDetector(AbstractDetector):
    def __init__(self, config=None, load_param: Union[bool, str] = False):
        super().__init__(config, load_param)
        self.config = config if config else {}
        
        # Initialize components
        self.backbone = self.build_backbone(config)
        self.head = self.build_head(config)
        self.loss_func = self.build_loss(config)
        
        # Freeze parameters if needed
        self._freeze_parameters()
        
        # Initialize PEFT if needed
        self._init_peft()
        
        # Print trainable parameters
        if self.config.get('verbose', False):
            self.print_trainable_parameters()
    
    def build_backbone(self, config):
        if config is None:
            config = {}
        
        # Only CLIP encoder is supported
        backbone_name = config.get('backbone', 'openai/clip-vit-large-patch14')
        backbone_lowercase = backbone_name.lower()
        
        if "clip" not in backbone_lowercase:
            logger.warning(f"GenD detector only supports CLIP encoder. Using default: openai/clip-vit-large-patch14")
            backbone_name = 'openai/clip-vit-large-patch14'
        
        self.feature_extractor = CLIPEncoder(backbone_name)
        logger.info(f"Initialized CLIP encoder: {backbone_name}")
        
        return self.feature_extractor
    
    def build_head(self, config):
        if config is None:
            config = {}
        
        features_dim = self.feature_extractor.get_features_dim()
        num_classes = config.get('num_classes', 2)
        
        # Head type: 'linear' or 'LinearNorm' (matching GenD-main Head enum)
        head_type = config.get('head', 'linear')
        # GenD-main uses 'LinearNorm' for normalized linear, we map it to normalize_inputs=True
        normalize_inputs = (head_type.lower() == 'linearnorm' or head_type.lower() == 'nlinear')
        
        head = LinearProbe(features_dim, num_classes, normalize_inputs)
        logger.info(f"Initialized {head_type} head: input_dim={features_dim}, num_classes={num_classes}, normalize_inputs={normalize_inputs}")
        
        return head
    
    def build_loss(self, config):
        if config is None:
            config = {}
        
        # GenD-main uses 'loss' config directly (matching Config.loss structure)
        loss_config = config.get('loss', None)
        
        if loss_config is not None:
            # Use GenD custom loss with alignment/uniformity
            loss_func = GenDLoss(loss_config)
            logger.info(f"Using GenD custom loss: ce_labels={loss_config.get('ce_labels', 0.0)}, "
                       f"alignment={loss_config.get('alignment_labels', 0.0)}, "
                       f"uniformity={loss_config.get('uniformity', 0.0)}")
        else:
            # Fallback to standard loss if 'loss' config not provided
            loss_func_name = config.get('loss_func', 'cross_entropy')
            loss_class = LOSSFUNC.get(loss_func_name, LOSSFUNC['cross_entropy'])
            loss_func = loss_class()
            logger.info(f"Using standard loss: {loss_func_name} (GenD 'loss' config not found)")
        
        return loss_func
    
    def _freeze_parameters(self):
        """Freeze feature extractor parameters if configured"""
        if self.config is None:
            return
        
        freeze_feature_extractor = self.config.get('freeze_feature_extractor', True)
        
        # Freeze feature extractor
        self.feature_extractor.requires_grad_(not freeze_feature_extractor)
        
        # Unfreeze specific layers if specified
        unfreeze_layers = self.config.get('unfreeze_layers', [])
        if len(unfreeze_layers) > 0:
            for name, param in self.named_parameters():
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True
        
        logger.info(f"Feature extractor frozen: {freeze_feature_extractor}")
        if unfreeze_layers:
            logger.info(f"Unfrozen layers: {unfreeze_layers}")
    
    def _init_peft(self):
        """Initialize PEFT (LoRA) if configured (matching GenD-main peft_v2 structure)"""
        if self.config is None:
            return
        
        # GenD-main uses 'peft_v2' not 'peft'
        peft_v2_config = self.config.get('peft_v2', None)
        if peft_v2_config is None:
            return
        
        try:
            from peft import get_peft_model, LoraConfig
            
            lora_config = peft_v2_config.get('lora', None)
            if lora_config is None:
                logger.warning("peft_v2.lora is None, skipping PEFT initialization")
                return
            
            # Use GenD-main defaults if not specified
            target_modules = lora_config.get('target_modules', ['out_proj'])
            if isinstance(target_modules, str):
                target_modules = [target_modules]
            
            peft_lora_config = LoraConfig(
                target_modules=target_modules,
                r=lora_config.get('rank', 1),  # Default in GenD-main: 1
                lora_alpha=lora_config.get('alpha', 32),  # Default in GenD-main: 32
                lora_dropout=lora_config.get('dropout', 0.05),  # Default in GenD-main: 0.05
                bias=lora_config.get('bias', 'none'),  # Default in GenD-main: 'none'
                use_rslora=lora_config.get('use_rslora', False),
                use_dora=lora_config.get('use_dora', False),
            )
            
            backbone = self.feature_extractor
            training_parameters = {name for name, param in backbone.named_parameters() if param.requires_grad}
            
            self.feature_extractor = get_peft_model(self.feature_extractor, peft_lora_config)
            
            # Restore training parameters
            for name, param in backbone.named_parameters():
                if name in training_parameters:
                    param.requires_grad = True
            
            logger.info(f"PEFT (LoRA) initialized: target_modules={target_modules}, rank={lora_config.get('rank', 1)}")
        except ImportError:
            logger.warning("PEFT library not found. Install with: pip install peft")
        except Exception as e:
            logger.warning(f"Failed to initialize PEFT: {e}")
    
    def print_trainable_parameters(self):
        """Print trainable parameters information"""
        trainable_params = []
        total_params = 0
        trainable_count = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_count += param.numel()
                trainable_params.append((name, tuple(param.shape)))
        
        logger.info("\n🔥 Trainable parameters:")
        for name, shape in trainable_params:
            logger.info(f"  - {name} shape = {shape}")
        
        logger.info(f"Total parameters: {total_params:,}, trainable: {trainable_count:,}, "
                   f"%: {trainable_count / total_params * 100:.4f}%")
    
    def features(self, data_dict: dict) -> torch.tensor:
        images = data_dict['image']
        feat = self.backbone(images)
        return feat
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        head_output = self.head(features)
        return head_output.logits_labels
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        l2_embeddings = pred_dict.get('l2_embeddings', None)
        
        # Use GenD custom loss if available
        if isinstance(self.loss_func, GenDLoss):
            loss_inputs = LossInputs(
                logits_labels=pred,
                labels=label,
                l2_embeddings=l2_embeddings
            )
            loss_outputs = self.loss_func(loss_inputs)
            loss_dict = {
                'overall': loss_outputs.total,
                'ce_loss': loss_outputs.ce_labels if loss_outputs.ce_labels is not None else 0.0,
                'alignment_loss': loss_outputs.alignment_labels if loss_outputs.alignment_labels is not None else 0.0,
                'uniformity_loss': loss_outputs.uniformity if loss_outputs.uniformity is not None else 0.0,
            }
        else:
            # Use standard loss
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
        head_output = self.head(features)
        pred = head_output.logits_labels
        prob = torch.softmax(pred, dim=1)[:, 1]
        
        pred_dict = {
            'cls': pred,
            'prob': prob,
            'feat': features,
            'l2_embeddings': head_output.l2_embeddings  # For GenD loss
        }
        return pred_dict
