import torch
import torch.nn.functional as F
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import logging

from loss import SupConLoss
from utils import print_metrics, save_checkpoint, gen_mapping_dict


class TrainingMetrics:
    """
    Class to manage and compute training metrics.
    
    This class handles the collection, computation and normalization of various
    metrics during training, including accuracy, AUC scores, and losses.
    """
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.metrics = defaultdict(float)
        self.correct_coarse = 0
        self.total_coarse = 0
        self.correct_fine = 0
        self.total_fine = 0
        self.logits_coarse: List[torch.Tensor] = []
        self.logits_fine: List[torch.Tensor] = []
        self.labels_coarse: List[torch.Tensor] = []
        self.labels_fine: List[torch.Tensor] = []
        
    def update_batch_metrics(self, 
                           logits_coarse: torch.Tensor,
                           logits_fine: torch.Tensor,
                           labels_coarse: torch.Tensor,
                           labels_fine: torch.Tensor) -> None:
        """
        Update metrics with batch results.
        
        Args:
            logits_coarse: Coarse classification logits
            logits_fine: Fine classification logits
            labels_coarse: Coarse classification labels
            labels_fine: Fine classification labels
        """
        self.correct_coarse += torch.argmax(logits_coarse, dim=1).eq(labels_coarse).sum().item()
        self.total_coarse += logits_coarse.shape[0]
        self.correct_fine += torch.argmax(logits_fine, dim=1).eq(labels_fine).sum().item()
        self.total_fine += logits_fine.shape[0]
        
        self.logits_coarse.append(logits_coarse.detach().cpu())
        self.logits_fine.append(logits_fine.detach().cpu())
        self.labels_coarse.append(labels_coarse.detach().cpu())
        self.labels_fine.append(labels_fine.detach().cpu())
        
    def compute_auc_scores(self, n_classes_coarse: int, n_classes_fine: int) -> Tuple[float, float]:
        """
        Compute AUC scores for both coarse and fine classification.
        
        Args:
            n_classes_coarse: Number of coarse classes
            n_classes_fine: Number of fine classes
            
        Returns:
            Tuple of (coarse_auc, fine_auc)
        """
        logits_coarse = torch.cat(self.logits_coarse, dim=0)
        logits_fine = torch.cat(self.logits_fine, dim=0)
        labels_coarse = torch.cat(self.labels_coarse, dim=0)
        labels_fine = torch.cat(self.labels_fine, dim=0)
        
        # Compute coarse AUC
        if n_classes_coarse != 2:
            auc_coarse = roc_auc_score(labels_coarse, F.softmax(logits_coarse).detach().cpu(), multi_class='ovr')
        else:
            auc_coarse = roc_auc_score(labels_coarse, F.softmax(logits_coarse)[:, 1].detach().cpu())
            
        # Compute fine AUC
        if n_classes_fine == 2:
            auc_fine = roc_auc_score(labels_fine, F.softmax(logits_fine)[:, 1].detach().cpu())
        else:
            auc_fine = roc_auc_score(labels_fine, F.softmax(logits_fine).detach().cpu(), multi_class='ovr')
            
        return auc_coarse, auc_fine
        
    def normalize_metrics(self, num_batches: int) -> Dict[str, float]:
        """
        Normalize metrics by number of batches.
        
        Args:
            num_batches: Number of batches processed
            
        Returns:
            Dictionary of normalized metrics
        """
        for k in self.metrics:
            self.metrics[k] /= num_batches
            
        self.metrics['patient_acc_coarse'] = self.correct_coarse / self.total_coarse
        self.metrics['patient_acc_fine'] = self.correct_fine / self.total_fine
        
        return self.metrics


def compute_hierarchical_loss(logits_coarse: torch.Tensor,
                            logits_fine: torch.Tensor,
                            labels_coarse: torch.Tensor,
                            labels_fine: torch.Tensor,
                            cfg: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hierarchical classification losses.
    
    This function computes three types of losses:
    1. Semantic loss for coarse classification
    2. Semantic loss for fine classification
    3. Regularization loss for hierarchical consistency
    
    Args:
        logits_coarse: Coarse classification logits
        logits_fine: Fine classification logits
        labels_coarse: Coarse classification labels
        labels_fine: Fine classification labels
        cfg: Configuration object
        
    Returns:
        Tuple of (semantic_loss_coarse, semantic_loss_fine, regularization_loss)
    """
    # Compute semantic losses
    loss_semantic_coarse = torch.nn.CrossEntropyLoss(reduction='mean')(logits_coarse, labels_coarse)
    loss_semantic_fine = torch.nn.CrossEntropyLoss(reduction='mean')(logits_fine, labels_fine)
    
    # Compute hierarchical regularization loss
    map_dict = gen_mapping_dict(cfg)
    preds_fine = F.softmax(logits_fine, dim=1)
    preds_fine_to_coarse = torch.zeros(preds_fine.shape[0], cfg.n_class[0]).cuda()
    
    for fine_class, coarse_class in map_dict.items():
        preds_fine_to_coarse[:, coarse_class] += preds_fine[:, fine_class]
    
    loss_regularization = torch.nn.CrossEntropyLoss(reduction='mean')(preds_fine_to_coarse, labels_coarse)
    
    return loss_semantic_coarse, loss_semantic_fine, loss_regularization


def compute_attention_matching_loss(batch_attention_coarse: List[torch.Tensor],
                                  batch_attention_fine: List[torch.Tensor],
                                  cfg: Any) -> torch.Tensor:
    """
    Compute hierarchical attention matching loss.
    
    This loss ensures that the attention patterns at fine and coarse levels
    are consistent with the hierarchical structure.
    
    Args:
        batch_attention_coarse: List of coarse attention maps
        batch_attention_fine: List of fine attention maps
        cfg: Configuration object
        
    Returns:
        Attention matching loss
    """
    loss_sim = 0.
    map_dict = gen_mapping_dict(cfg)
    
    for attention_coarse, attention_fine in zip(batch_attention_coarse, batch_attention_fine):
        attention_fine_to_coarse = torch.zeros(cfg.n_class[0], attention_fine.shape[1]).cuda()
        for fine_class, coarse_class in map_dict.items():
            attention_fine_to_coarse[coarse_class, :] += attention_fine[fine_class, :]
        loss_sim += torch.sum((1 - torch.nn.CosineSimilarity(dim=0)(attention_coarse, attention_fine_to_coarse)))/attention_coarse.shape[1]
    
    return loss_sim / len(batch_attention_coarse)


def compute_temperature(epoch: int, num_epochs: int) -> float:
    """
    Compute dynamic temperature for contrastive loss.
    
    Args:
        epoch: Current epoch number
        num_epochs: Total number of epochs
        
    Returns:
        Temperature value
    """
    temp_low, temp_high = 0.1, 1.0
    return (temp_high - temp_low) * (1 + math.cos(2 * math.pi * epoch / num_epochs)) / 2 + temp_low


def compute_contrastive_loss(semantics_features: torch.Tensor,
                           patient_fine_labels: torch.Tensor,
                           epoch: int,
                           cfg: Any) -> Tuple[torch.Tensor, float]:
    """
    Compute contrastive loss with dynamic temperature.
    
    Args:
        semantics_features: Feature vectors for contrastive learning
        patient_fine_labels: Fine-grained labels for supervision
        epoch: Current epoch number
        cfg: Configuration object
        
    Returns:
        Tuple of (contrastive_loss, temperature)
    """
    alpha = max(0.5, 1 - (epoch / cfg.num_epochs)**2)
    tau = compute_temperature(epoch, cfg.num_epochs)
    
    contrastive_loss = SupConLoss(temperature=tau).cuda()
    return contrastive_loss(semantics_features, patient_fine_labels) / len(semantics_features), tau


def train_phase(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                dataloader: torch.utils.data.DataLoader,
                epoch: int,
                cfg: Any,
                writer: torch.utils.tensorboard.SummaryWriter,
                scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau) -> None:
    """
    Training phase for one epoch.
    
    This function implements the training loop for one epoch, including:
    1. Forward pass
    2. Loss computation
    3. Backward pass
    4. Metrics tracking
    5. Logging
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        dataloader: Training data loader
        epoch: Current epoch number
        cfg: Configuration object
        writer: Tensorboard writer
        scheduler: Learning rate scheduler
    """
    model.train()
    metrics = TrainingMetrics()
    logger = logging.getLogger(__name__)
    
    try:
        for cell_images, patient_labels in tqdm(dataloader):
            batch_attention_coarse = []
            batch_attention_fine = []
            batch_semantics = []
            
            batch_logits_coarse = []
            batch_logits_fine = []
            batch_labels_coarse = []
            batch_labels_fine = []
            
            # Process each sample in the batch
            for cell_image, patient_label in zip(cell_images, patient_labels):
                cell_image = cell_image.cuda()
                label_coarse = torch.tensor(patient_label[0]).cuda()
                label_fine = torch.tensor(patient_label[1]).cuda()
                
                # Forward pass
                attention_maps, logits, features = model(cell_image)
                batch_semantics.append([features, label_fine])
                
                # Process outputs
                logits_coarse = logits[0].squeeze()
                logits_fine = logits[1].squeeze()
                
                batch_logits_coarse.append(logits_coarse.unsqueeze(0))
                batch_logits_fine.append(logits_fine.unsqueeze(0))
                batch_labels_coarse.append(label_coarse.unsqueeze(0))
                batch_labels_fine.append(label_fine.unsqueeze(0))
                
                batch_attention_coarse.append(attention_maps[0])
                batch_attention_fine.append(attention_maps[1])
                
                metrics.update_batch_metrics(logits_coarse, logits_fine, label_coarse, label_fine)
            
            # Compute losses
            batch_logits_coarse = torch.cat(batch_logits_coarse, dim=0)
            batch_logits_fine = torch.cat(batch_logits_fine, dim=0)
            batch_labels_coarse = torch.cat(batch_labels_coarse, dim=0)
            batch_labels_fine = torch.cat(batch_labels_fine, dim=0)
            
            loss_semantic_coarse, loss_semantic_fine, loss_regularization = compute_hierarchical_loss(
                batch_logits_coarse, batch_logits_fine, batch_labels_coarse, batch_labels_fine, cfg
            )
            
            loss_attention = compute_attention_matching_loss(batch_attention_coarse, batch_attention_fine, cfg)
            
            # Compute contrastive loss
            semantics_features = torch.stack([sem[0] for sem in batch_semantics])
            patient_fine_labels = torch.stack([sem[1] for sem in batch_semantics])
            loss_contrastive, tau = compute_contrastive_loss(semantics_features, patient_fine_labels, epoch, cfg)
            
            # Combine losses
            alpha = 1 - (epoch / cfg.num_epochs)**2
            loss_classification = loss_semantic_fine + loss_regularization
            loss = alpha * loss_semantic_coarse + loss_classification + (1 - alpha) * loss_contrastive
            
            # Update metrics
            metrics.metrics['loss_semantic_coarse'] += loss_semantic_coarse.data.cpu().numpy()
            metrics.metrics['loss_semantic_fine'] += loss_semantic_fine.data.cpu().numpy()
            metrics.metrics['loss_semantic'] += (loss_semantic_coarse + loss_semantic_fine).data.cpu().numpy()
            metrics.metrics['loss_regularization'] += loss_regularization.data.cpu().numpy()
            metrics.metrics['loss_attention'] += loss_attention.data.cpu().numpy()
            metrics.metrics['loss_contrastive'] += loss_contrastive.data.cpu().numpy()
            metrics.metrics['loss'] += loss.data.cpu().numpy()
            
            # Store alpha and tau for final metrics
            metrics.metrics['alpha'] = alpha
            metrics.metrics['tau'] = tau
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute final metrics
        metrics.metrics['auc_coarse'], metrics.metrics['auc_fine'] = metrics.compute_auc_scores(
            cfg.n_class[0], cfg.n_class[1]
        )
        
        # Normalize and log metrics
        epoch_metrics = metrics.normalize_metrics(len(dataloader))
        print_metrics(epoch_metrics, 'train')
        
        # Print classification reports
        logits_coarse = torch.cat(metrics.logits_coarse, dim=0)
        logits_fine = torch.cat(metrics.logits_fine, dim=0)
        labels_coarse = torch.cat(metrics.labels_coarse, dim=0)
        labels_fine = torch.cat(metrics.labels_fine, dim=0)
        
        logger.info("\n" + classification_report(labels_coarse, torch.argmax(logits_coarse, dim=1), digits=4))
        logger.info("\n" + classification_report(labels_fine, torch.argmax(logits_fine, dim=1), digits=4))
        
        # Log to tensorboard
        for k, v in epoch_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        
        scheduler.step(epoch_metrics['loss'])
        
    except Exception as e:
        logger.error(f"Error in training phase: {e}")
        raise


def validation_phase(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    epoch: int,
                    cfg: Any,
                    writer: torch.utils.tensorboard.SummaryWriter) -> Dict[str, float]:
    """
    Validation phase for one epoch.
    
    This function implements the validation loop for one epoch, including:
    1. Forward pass
    2. Metrics computation
    3. Logging
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        epoch: Current epoch number
        cfg: Configuration object
        writer: Tensorboard writer
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics = TrainingMetrics()
    logger = logging.getLogger(__name__)
    
    try:
        with torch.no_grad():
            for cell_images, patient_labels in tqdm(dataloader):
                batch_attention_coarse = []
                batch_attention_fine = []
                batch_semantics = []
                
                batch_logits_coarse = []
                batch_logits_fine = []
                batch_labels_coarse = []
                batch_labels_fine = []
                
                # Process each sample in the batch
                for cell_image, patient_label in zip(cell_images, patient_labels):
                    cell_image = cell_image.cuda()
                    label_coarse = torch.tensor(patient_label[0]).cuda()
                    label_fine = torch.tensor(patient_label[1]).cuda()
                    
                    # Forward pass
                    attention_maps, logits, features = model(cell_image)
                    batch_semantics.append([features, label_fine])
                    
                    # Process outputs
                    logits_coarse = logits[0].squeeze()
                    logits_fine = logits[1].squeeze()
                    
                    batch_logits_coarse.append(logits_coarse.unsqueeze(0))
                    batch_logits_fine.append(logits_fine.unsqueeze(0))
                    batch_labels_coarse.append(label_coarse.unsqueeze(0))
                    batch_labels_fine.append(label_fine.unsqueeze(0))
                    
                    batch_attention_coarse.append(attention_maps[0])
                    batch_attention_fine.append(attention_maps[1])
                    
                    metrics.update_batch_metrics(logits_coarse, logits_fine, label_coarse, label_fine)
                
                # Compute losses
                batch_logits_coarse = torch.cat(batch_logits_coarse, dim=0)
                batch_logits_fine = torch.cat(batch_logits_fine, dim=0)
                batch_labels_coarse = torch.cat(batch_labels_coarse, dim=0)
                batch_labels_fine = torch.cat(batch_labels_fine, dim=0)
                
                loss_semantic_coarse, loss_semantic_fine, loss_regularization = compute_hierarchical_loss(
                    batch_logits_coarse, batch_logits_fine, batch_labels_coarse, batch_labels_fine, cfg
                )
                
                loss_attention = compute_attention_matching_loss(batch_attention_coarse, batch_attention_fine, cfg)
                
                # Compute contrastive loss
                semantics_features = torch.stack([sem[0] for sem in batch_semantics])
                patient_fine_labels = torch.stack([sem[1] for sem in batch_semantics])
                loss_contrastive, tau = compute_contrastive_loss(semantics_features, patient_fine_labels, epoch, cfg)
                
                # Combine losses
                alpha = 1 - (epoch / cfg.num_epochs)**2
                loss_classification = loss_semantic_fine + loss_regularization
                loss = alpha * loss_semantic_coarse + loss_classification + (1 - alpha) * loss_contrastive
                
                # Update metrics
                metrics.metrics['loss_semantic_coarse'] += loss_semantic_coarse.data.cpu().numpy()
                metrics.metrics['loss_semantic_fine'] += loss_semantic_fine.data.cpu().numpy()
                metrics.metrics['loss_semantic'] += (loss_semantic_coarse + loss_semantic_fine).data.cpu().numpy()
                metrics.metrics['loss_regularization'] += loss_regularization.data.cpu().numpy()
                metrics.metrics['loss_attention'] += loss_attention.data.cpu().numpy()
                metrics.metrics['loss_contrastive'] += loss_contrastive.data.cpu().numpy()
                metrics.metrics['loss'] += loss.data.cpu().numpy()
                
                # Store alpha and tau for final metrics
                metrics.metrics['alpha'] = alpha
                metrics.metrics['tau'] = tau
            
        # Compute final metrics
        metrics.metrics['auc_coarse'], metrics.metrics['auc_fine'] = metrics.compute_auc_scores(
            cfg.n_class[0], cfg.n_class[1]
        )
        
        # Normalize and log metrics
        epoch_metrics = metrics.normalize_metrics(len(dataloader))
        print_metrics(epoch_metrics, 'val')
        
        # Print classification reports
        logits_coarse = torch.cat(metrics.logits_coarse, dim=0)
        logits_fine = torch.cat(metrics.logits_fine, dim=0)
        labels_coarse = torch.cat(metrics.labels_coarse, dim=0)
        labels_fine = torch.cat(metrics.labels_fine, dim=0)
        
        logger.info("\n" + classification_report(labels_coarse, torch.argmax(logits_coarse, dim=1), digits=4))
        logger.info("\n" + classification_report(labels_fine, torch.argmax(logits_fine, dim=1), digits=4))
        
        # Log to tensorboard
        for k, v in epoch_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
            
        return epoch_metrics
        
    except Exception as e:
        logger.error(f"Error in validation phase: {e}")
        raise