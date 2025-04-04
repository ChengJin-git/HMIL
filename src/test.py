import os
import math
import argparse
import warnings
import torch
import random
import time
import pickle
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from collections import defaultdict
from tqdm import tqdm
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import importlib.util
import sys
from omegaconf import OmegaConf

from utils import rm_n_mkdir
from loss import print_metrics
from models import ResNetMTL
from dataset import TrainDataset_hierarchy, ValDataset_hierarchy

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 
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

def test_phase(model, dataloader, best_acc, checkpoint_path):
    """
    Test phase for model evaluation.
    
    Args:
        model: PyTorch model
        dataloader: Test data loader
        best_acc: Best accuracy so far
        checkpoint_path: Path to save best model
        
    Returns:
        Best accuracy
    """
    metrics = defaultdict(float)
    epoch_samples = 0
    correct_coarse = 0
    total_coarse = 0
    correct_fine = 0
    total_fine = 0

    logits_coarse_all = []
    logits_fine_all = []
    label_coarse_all = []
    label_fine_all = []
    
    for cell_imgs, patient_label in dataloader:
        cell_imgs = cell_imgs.cuda()
        patient_label_coarse = patient_label[0].cuda()
        patient_label_fine = patient_label[1].cuda()

        with torch.no_grad():
            A_raw, logits = model.test(cell_imgs)
        
        logits_coarse = logits[0]
        logits_fine = logits[1]                       
            
        loss_semantic = torch.nn.CrossEntropyLoss()(logits_coarse, patient_label_coarse)
        loss_semantic += torch.nn.CrossEntropyLoss()(logits_fine, patient_label_fine)
        
        loss = loss_semantic
        logits_out_coarse = logits_coarse
        logits_coarse_all.append(logits_out_coarse)
        label_coarse_all.append(patient_label_coarse)
        correct_coarse += torch.argmax(logits_out_coarse, dim=1).eq(patient_label_coarse).sum().item()
        total_coarse += logits_out_coarse.shape[0]
        
        logits_out_fine = logits_fine
        logits_fine_all.append(logits_out_fine)
        label_fine_all.append(patient_label_fine)
        correct_fine += torch.argmax(logits_out_fine, dim=1).eq(patient_label_fine).sum().item()
        total_fine += logits_out_fine.shape[0]
        
        metrics['loss_semantic'] += loss_semantic.data.cpu().numpy() 
        metrics['loss'] += loss.data.cpu().numpy()

        epoch_samples += 1

    logits_coarse_all = torch.cat(logits_coarse_all, dim=0)
    logits_fine_all = torch.cat(logits_fine_all, dim=0)
    label_coarse_all = torch.cat(label_coarse_all, dim=0)
    label_fine_all = torch.cat(label_fine_all, dim=0)
    metrics['patient_acc_coarse'] = (correct_coarse/total_coarse) * epoch_samples
    metrics['patient_acc_fine'] = (correct_fine/total_fine) * epoch_samples

    print_metrics(metrics, epoch_samples, 'val')

    auc_score_coarse = roc_auc_score(label_coarse_all.cpu(), F.softmax(logits_coarse_all)[:, 1].detach().cpu())
    auc_score_fine = roc_auc_score(label_fine_all.cpu(), F.softmax(logits_fine_all).detach().cpu(), multi_class='ovr')
    
    acc_all = metrics['patient_acc_coarse'] + metrics['patient_acc_fine']
    
    if acc_all > best_acc:
        print(f"saving best model to {checkpoint_path.replace('.pth','.best')}")
        best_acc = acc_all
        torch.save(model.state_dict(), checkpoint_path.replace('.pth','.best'))

    print(classification_report(label_fine_all.cpu(), torch.argmax(logits_fine_all, dim=1).cpu(), digits=4))
    print('auc_coarse:', auc_score_coarse)
    print('auc_fine:', auc_score_fine)
    prob_coarse = torch.concatenate((F.softmax(logits_fine_all)[:, 0].detach().cpu().unsqueeze(1), torch.sum(F.softmax(logits_fine_all)[:, 1:], dim=1).detach().cpu().unsqueeze(1)), dim=1)
    pred_coarse = torch.argmax(prob_coarse, dim=1)
    print(classification_report(label_coarse_all.cpu(), pred_coarse, digits=4))
    print('specificity_coarse:', confusion_matrix(label_coarse_all.cpu(), pred_coarse)[0,0]/(confusion_matrix(label_coarse_all.cpu(), pred_coarse)[0,0]+confusion_matrix(label_coarse_all.cpu(), pred_coarse)[0,1]))
    auc_score_coarse = roc_auc_score(label_coarse_all.cpu(), prob_coarse[:, 1])
    print('auc_coarse:', auc_score_coarse)
    
    return best_acc

def build_model():
    model = ResNetMTL(cfg.n_class, freeze=cfg.freeze, pretrained=cfg.pretrained).cuda()
        
    for name,parameters in model.named_parameters():
        print(name,':',parameters.shape)
    
    return model

def build_dataset():
    train_splits = pickle.load(open(cfg.train_splits, 'rb'))
    val_splits = pickle.load(open(cfg.val_splits, 'rb'))
    test_splits = pickle.load(open(cfg.test_splits, 'rb'))
    dataset_train = TrainDataset_hierarchy(train_splits, cfg)
    print('Train dataset size:', len(train_splits))
    dataset_val = ValDataset_hierarchy(val_splits, cfg)
    print('Val dataset size:', len(val_splits))
    dataset_test = ValDataset_hierarchy(test_splits, cfg)
    print('Test dataset size:', len(test_splits))
    
    return dataset_train, dataset_val, dataset_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, 0,1,2,3"
    )
    parser.add_argument(
        "--data",
        default="fold1",
        help="data to use, string"
    )
    parser.add_argument(
        "--config",
        default="config",
        help="config file path",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    ## check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    
    ## Parameters
    cfg.data_inst = args.data
    checkpoint_folder = "./checkpoints/"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_path = "./checkpoints/%s_%s.pth" % (cfg.model_name, cfg.data_inst)
        
    print("model will be save to %s" % checkpoint_path)
    
    rm_n_mkdir('./logs/%s_%s/' % (cfg.model_name, cfg.data_inst))
    writer = SummaryWriter('./logs/%s_%s/' % (cfg.model_name, cfg.data_inst))
    print("log dir is set to ./logs/%s_%s/" % (cfg.model_name, cfg.data_inst))

    best_loss = 1e10
    best_acc = 0
    
    ## build dataset
    dataset_train, dataset_val, dataset_test = build_dataset()

    dataloaders = {
      'train': DataLoader(dataset_train, batch_size=cfg.train_batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, pin_memory=True),
      'val': DataLoader(dataset_val, batch_size=cfg.test_batch_size, shuffle=False, 
                        num_workers=cfg.num_workers, pin_memory=True),
      'test': DataLoader(dataset_test, batch_size=cfg.test_batch_size, shuffle=False, 
                             num_workers=cfg.num_workers, pin_memory=True)
    }
    
    ## build models
    model = build_model().cuda()
    # model = nn.DataParallel(model)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    
    if cfg.pretrained_path != "":
        checkpoint = torch.load(cfg.pretrained_path)
        print(model.load_state_dict(checkpoint, strict=False))
        model.load_state_dict(checkpoint, strict=False)
        print("load pretrained weights from %s" % cfg.pretrained_path)
    
    best_val_acc = test_phase(model, dataloaders['val'], best_acc, checkpoint_path)
    best_test_acc = test_phase(model, dataloaders['test'], best_acc, checkpoint_path)