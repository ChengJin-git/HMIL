import torch
import torch.nn as nn
from typing import Optional, Union
import numpy as np

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss implementation.
    
    This loss function implements the supervised contrastive learning approach
    described in https://arxiv.org/pdf/2004.11362.pdf. It also supports
    unsupervised contrastive loss as used in SimCLR.
    
    The loss encourages features of samples from the same class to be similar
    while pushing features of samples from different classes apart.
    """
    
    def __init__(self,
                 temperature: float = 0.1,
                 contrast_mode: str = 'all',
                 base_temperature: float = 0.1):
        """
        Initialize the supervised contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling logits
            contrast_mode: Mode for contrastive learning ('one' or 'all')
            base_temperature: Base temperature for loss scaling
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self,
                features: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the supervised contrastive loss.
        
        Args:
            features: Hidden vectors of shape [batch_size, n_views, ...]
            labels: Ground truth labels of shape [batch_size]
            mask: Contrastive mask of shape [batch_size, batch_size]
                 mask_{i,j}=1 if sample j has the same class as sample i
                 Can be asymmetric.
        
        Returns:
            A loss scalar.
            
        Raises:
            ValueError: If features dimensions are incorrect or if both labels
                      and mask are provided.
        """
        device = torch.device('cuda' if features.is_cuda else 'cpu')
        
        # Validate input dimensions
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            
        batch_size = features.shape[0]
        
        # Validate and prepare mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            
        # Prepare contrastive features
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Select anchor features based on mode
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')
            
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Prepare mask for contrastive learning
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss