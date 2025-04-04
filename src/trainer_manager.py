import os
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf

from trainer import train_phase, validation_phase
from utils import rm_n_mkdir, build_dataset, collate_fn, save_checkpoint
from models import HMIL, ResNetMTL
from dataset import TrainDataset_hierarchy, ValDataset_hierarchy
from trainer import TrainingMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingManager:
    """Training manager class responsible for managing the entire training process"""
    
    def __init__(self, cfg: Any):
        """
        Initialize the training manager
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # Set random seed for reproducibility
        self._set_random_seed()
        
        # Create checkpoint and log directories
        self._setup_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize training state
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.current_epoch = 0
        
        # Build datasets and dataloaders
        self._build_datasets()
        
        # Build model and optimizer
        self._build_model()
        self._build_optimizer()
        
        # Load pretrained model if specified
        if self.cfg.pretrained_path:
            self._load_pretrained_model()
    
    def _set_random_seed(self) -> None:
        """Set random seed to ensure reproducibility"""
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_directories(self) -> None:
        """Create necessary directory structure"""
        self.checkpoint_dir = "./checkpoints/"
        self.log_dir = f'./logs/{self.cfg.model_name}_{self.cfg.data_inst}/'
        self.checkpoint_path = f"{self.checkpoint_dir}/{self.cfg.model_name}_{self.cfg.data_inst}.pth"
        
        rm_n_mkdir(self.checkpoint_dir)
        rm_n_mkdir(self.log_dir)
        
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        logger.info(f"Logs will be saved to: {self.log_dir}")
    
    def _build_datasets(self) -> None:
        """Build datasets and dataloaders"""
        try:
            # Build datasets
            self.dataset_train = TrainDataset_hierarchy(self.cfg.train_splits, self.cfg)
            self.dataset_val = ValDataset_hierarchy(self.cfg.val_splits, self.cfg)
            
            # Build dataloaders
            self.dataloaders = {
                'train': DataLoader(
                    self.dataset_train,
                    batch_size=self.cfg.train_batch_size,
                    shuffle=True,
                    num_workers=self.cfg.num_workers,
                    pin_memory=True
                ),
                'val': DataLoader(
                    self.dataset_val,
                    batch_size=self.cfg.test_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    pin_memory=True
                )
            }
            
            logger.info(f"Training dataset size: {len(self.dataset_train)}")
            logger.info(f"Validation dataset size: {len(self.dataset_val)}")
            
        except Exception as e:
            logger.error(f"Error building datasets: {e}")
            raise
    
    def _build_model(self) -> None:
        """Build and initialize the model"""
        try:
            self.model = ResNetMTL(
                n_class=self.cfg.n_class,
                freeze=self.cfg.freeze,
                pretrained=self.cfg.pretrained
            ).to(self.device)
            
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            
            logger.info("Model built successfully")
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
    
    def _build_optimizer(self) -> None:
        """Build optimizer and learning rate scheduler"""
        try:
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.cfg.lr
            )
            
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
            
            logger.info("Optimizer and scheduler built successfully")
            
        except Exception as e:
            logger.error(f"Error building optimizer: {e}")
            raise
    
    def _load_pretrained_model(self) -> None:
        """Load pretrained model weights"""
        try:
            checkpoint = torch.load(self.cfg.pretrained_path)
            self.model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded pretrained weights from {self.cfg.pretrained_path}")
            
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            raise
    
    def train(self) -> None:
        """
        Execute training process
        
        Includes:
        1. Training loop
        2. Validation evaluation
        3. Model saving
        4. Learning rate adjustment
        """
        try:
            for epoch in range(self.cfg.epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.cfg.epochs}")
                
                # Training phase
                train_metrics = train_phase(
                    self.model,
                    self.dataloaders['train'],
                    self.optimizer,
                    self.device,
                    self.cfg
                )
                
                # Validation phase
                val_metrics = validation_phase(
                    self.model,
                    self.dataloaders['val'],
                    self.device,
                    self.cfg
                )
                
                # Update learning rate
                self.scheduler.step(val_metrics['loss'])
                
                # Log metrics
                self._log_metrics(train_metrics, val_metrics)
                
                # Save best model
                self._save_best_model(val_metrics)
                
                # Save checkpoint
                self._save_checkpoint(val_metrics)
                
                logger.info(f"Completed epoch {epoch + 1}")
                
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """
        Log training and validation metrics
        
        Args:
            train_metrics: Training phase metrics
            val_metrics: Validation phase metrics
        """
        for phase, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'{phase}/{metric_name}', value, self.current_epoch)
                logger.info(f"{phase} {metric_name}: {value:.4f}")
    
    def _save_best_model(self, val_metrics: Dict[str, float]) -> None:
        """
        Save the best model based on validation metrics
        
        Args:
            val_metrics: Validation phase metrics
        """
        if val_metrics['loss'] < self.best_loss:
            self.best_loss = val_metrics['loss']
            self.best_acc = val_metrics['accuracy']
            torch.save(self.model.state_dict(), self.checkpoint_path)
            logger.info(f"Saved best model with loss: {self.best_loss:.4f}")
    
    def _save_checkpoint(self, val_metrics: Dict[str, float]) -> None:
        """
        Save training checkpoint
        
        Args:
            val_metrics: Validation phase metrics
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_acc': self.best_acc,
            'val_metrics': val_metrics
        }
        
        save_checkpoint(
            checkpoint,
            self.checkpoint_path,
            self.current_epoch,
            self.best_acc,
            is_test=False
        )
        logger.info(f"Saved checkpoint for epoch {self.current_epoch + 1}")
    
    def close(self) -> None:
        """Clean up resources"""
        self.writer.close()
        logger.info("Training completed and resources cleaned up")

    def create_dataloaders(self, fold: int) -> dict:
        """
        Create data loaders for training, validation and testing
        
        Args:
            fold: Current training fold
            
        Returns:
            Dictionary containing train/val/test data loaders
        """
        dataset_train, dataset_val, dataset_test = build_dataset(self.cfg, fold)
        
        return {
            'train': DataLoader(
                dataset_train, 
                batch_size=self.cfg.train_batch_size, 
                shuffle=True,
                num_workers=self.cfg.num_workers, 
                pin_memory=True, 
                collate_fn=collate_fn
            ),
            'val': DataLoader(
                dataset_val, 
                batch_size=self.cfg.test_batch_size, 
                shuffle=False,
                num_workers=self.cfg.num_workers, 
                pin_memory=True
            ),
            'test': DataLoader(
                dataset_test, 
                batch_size=self.cfg.test_batch_size, 
                shuffle=False,
                num_workers=self.cfg.num_workers, 
                pin_memory=True
            )
        }
        
    def create_model_and_optimizer(self) -> Tuple[HMIL, optim.Adam]:
        """
        Create model, optimizer
        
        Returns:
            Tuple of (model, optimizer)
        """
        model = HMIL(self.cfg.n_class).cuda()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=self.cfg.lr
        )
        
        if self.cfg.pretrained_path:
            try:
                checkpoint = torch.load(self.cfg.pretrained_path)
                model.load_state_dict(checkpoint, strict=False)
                self.logger.info(f"Loaded pretrained weights from {self.cfg.pretrained_path}")
            except Exception as e:
                self.logger.error(f"Error loading pretrained weights: {e}")
                raise
                
        return model, optimizer
        
    def train_single_fold(self, fold: int, writer: SummaryWriter) -> float:
        """
        Train a single fold
        
        Args:
            fold: Current training fold
            writer: Tensorboard writer
            
        Returns:
            Best validation AUC score
        """
        self.logger.info(f"Starting training for fold {fold}")
        
        dataloaders = self.create_dataloaders(fold)
        model, optimizer = self.create_model_and_optimizer()
        
        best_auc = 0
        
        for epoch in range(self.cfg.num_epochs):
            self.logger.info(f'[training] fold {fold}, epoch {epoch}/{self.cfg.num_epochs}')
            since = time.time()
            
            try:
                train_phase(model, optimizer, dataloaders['train'], epoch, self.cfg, writer)
                best_auc = validation_phase(model, dataloaders['val'], best_auc, epoch, self.cfg, writer, is_test=False)
                
                time_elapsed = time.time() - since
                self.logger.info(f'[log] epoch time: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
                
            except Exception as e:
                self.logger.error(f"Error during training epoch {epoch}: {e}")
                raise
                
        return best_auc
        
    def run_cross_validation(self, folds: List[int]) -> dict:
        """
        Run cross-validation training
        
        Args:
            folds: List of folds to train
            
        Returns:
            Dictionary containing validation and test metrics
        """
        results = {
            'val_auc': [],
            'val_acc': [],
            'test_auc': [],
            'test_acc': []
        }
        
        for fold in folds:
            writer = SummaryWriter(os.path.join(self.cfg.log_folder, self.cfg.dataset_name, f'fold_{fold}'))
            best_auc = self.train_single_fold(fold, writer)
            writer.close()
            
            # Record results
            results['val_auc'].append(best_auc)
            
        return results
        
    def run_single_training(self, fold: int) -> dict:
        """
        Run single training
        
        Args:
            fold: Fold to train
            
        Returns:
            Dictionary containing training results
        """
        writer = SummaryWriter(os.path.join(self.cfg.log_folder, self.cfg.dataset_name))
        best_auc = self.train_single_fold(fold, writer)
        writer.close()
        
        return {'best_auc': best_auc} 