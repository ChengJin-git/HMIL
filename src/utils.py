import os
import shutil
import pickle
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
import numpy as np
from dataset import HierarchicalTrainDataset, HierarchicalValDataset

def rm_n_mkdir(dir_path: str) -> None:
    """
    Remove directory if it exists and create a new one.
    
    Args:
        dir_path: Path to the directory to be recreated
    """
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Load pretrained model weights.
    
    Args:
        model: Model to load weights into
        model_path: Path to the pretrained weights
        
    Returns:
        Model with loaded weights
        
    Raises:
        FileNotFoundError: If model_path does not exist
        RuntimeError: If loading weights fails
    """
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=False)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")

def gen_mapping_dict(cfg: Any) -> Dict[int, int]:
    """
    Generate mapping dictionary from configuration.
    
    Args:
        cfg: Configuration object containing mapping string
        
    Returns:
        Dictionary mapping fine-grained labels to coarse-grained labels
        
    Raises:
        ValueError: If mapping string format is invalid
    """
    try:
        pairs = cfg.mapping.split(", ")
        mapping_dict = dict(pair.split(":") for pair in pairs)
        return {int(k): int(v) for k, v in mapping_dict.items()}
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid mapping format: {e}")

def collate_fn(batch: List[Tuple[torch.Tensor, List[int]]]) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of (cell_images, patient_label) tuples
        
    Returns:
        Tuple of (cell_images list, patient_labels list)
    """
    cell_imgs, patient_labels = [], []
    
    for img, label in batch:
        cell_imgs.append(img)
        patient_labels.append(label)
    
    return cell_imgs, patient_labels
    
def build_dataset(cfg: Any, fold: int) -> Tuple[HierarchicalTrainDataset, HierarchicalValDataset, HierarchicalValDataset]:
    """
    Build train, validation and test datasets for a specific fold.
    
    Args:
        cfg: Configuration object
        fold: Current fold number
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Raises:
        FileNotFoundError: If split files not found
        ValueError: If dataset creation fails
    """
    logger = logging.getLogger(__name__)
    split_root = cfg.splits
    
    try:
        # Load split files
        train_splits = pickle.load(open(os.path.join(split_root, f'train_splits_{fold}.pkl'), 'rb'))
        val_splits = pickle.load(open(os.path.join(split_root, f'val_splits_{fold}.pkl'), 'rb'))
        test_splits = pickle.load(open(os.path.join(split_root, f'test_splits_{fold}.pkl'), 'rb'))
        
        # Create datasets
        dataset_train = HierarchicalTrainDataset(train_splits, cfg)
        dataset_val = HierarchicalValDataset(val_splits, cfg)
        dataset_test = HierarchicalValDataset(test_splits, cfg)
        
        # Log dataset sizes
        logger.info(f'[dataset] Train dataset in fold {fold} size: {len(train_splits)}')
        logger.info(f'[dataset] Val dataset in fold {fold} size: {len(val_splits)}')
        logger.info(f'[dataset] Test dataset in fold {fold} size: {len(test_splits)}')
        
        return dataset_train, dataset_val, dataset_test
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Split files not found: {e}")
    except Exception as e:
        raise ValueError(f"Error building datasets: {e}")

def save_checkpoint(state: Dict[str, Any],
                   save_path: str,
                   epoch: int,
                   best: Optional[float] = None,
                   is_test: bool = False) -> None:
    """
    Save model checkpoint.
    
    Args:
        state: Model state dictionary
        save_path: Directory to save checkpoint
        epoch: Current epoch number
        best: Best metric value
        is_test: Whether this is a test checkpoint
    """
    logger = logging.getLogger(__name__)
    
    if best is not None and not is_test:
        # Remove previous best model files
        for file in os.listdir(save_path):
            if 'model_best' in file:
                os.remove(os.path.join(save_path, file))
                
    if is_test:
        filename = os.path.join(save_path, f"model_last_{best:.4f}_{epoch}.pth.tar")
    else:
        filename = os.path.join(save_path, f"model_best_{best:.4f}_{epoch}.pth.tar")
        logger.info(f"[checkpoint] save best model {filename}")
        
    torch.save(state, filename)
        
def print_metrics(metrics: Dict[str, Union[float, np.ndarray]], phase: str) -> None:
    """
    Print formatted metrics for a given phase.
    
    Args:
        metrics: Dictionary of metric names and values
        phase: Phase name (train/val/test)
    """
    logger = logging.getLogger(__name__)
    outputs = []
    
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics[k] = float(v)
        outputs.append(f"{k}: {metrics[k]:.4f}")
        
    logger.info(f"[metrics] {phase}: {', '.join(outputs)}")