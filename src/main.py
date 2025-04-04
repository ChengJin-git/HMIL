import os
import sys
import warnings
import argparse
import importlib.util
import torch
from omegaconf import OmegaConf

from trainer_manager import TrainingManager

# Setup PyTorch backend
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 
torch.autograd.set_detect_anomaly = True
warnings.filterwarnings('ignore')

def load_config(config_path: str):
    """
    Load configuration file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration object
    """
    try:
        spec = importlib.util.spec_from_file_location("_C", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["_C"] = config_module
        spec.loader.exec_module(config_module)
        cfg = config_module._C
        
        # Convert to OmegaConf configuration object
        cfg = OmegaConf.create(cfg.__dict__)
        return cfg
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {e}")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Hierarchical Multi-Instance Learning Training Script"
    )
    parser.add_argument(
        "--config",
        default="config",
        help="config file path",
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Setup GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[device] device set to: {device}')
        
        # Create training manager
        trainer = TrainingManager(cfg, device)
        
        # Parse training folds
        training_folds = [int(f) for f in cfg.folds.split(',')]
        cross_val_flag = len(training_folds) > 1
        
        if cross_val_flag:
            print(f"[logs] training folds: {training_folds}")
            print(f"[logs] conducting {len(training_folds)}-fold cross-validation")
            results = trainer.run_cross_validation(training_folds)
            print(f"Cross-validation results: {results}")
        else:
            print(f"[logs] training fold: {training_folds}")
            results = trainer.run_single_training(training_folds[0])
            print(f"Training results: {results}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    main()