from yacs.config import CfgNode as CN

# metadata
_C = CN()
_C.dataset_name = "BRACS"

# dataset
_C.data_root = "/path/to/dataset"
_C.splits = "/path/to/splits"
_C.folds = "0"
_C.label_list = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
_C.mapping = "0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2" # mapping from fine-grained labels to coarse-grained labels

# classification
_C.n_class = [3, 7]

# resume training
_C.pretrained_path = ""
_C.log_folder = "/path/to/logs"

# optimization, training, and testing
_C.lr = 1e-4
_C.num_epochs = 200
_C.train_batch_size = 64
_C.test_batch_size = 1
_C.num_workers = 4

# GPU
_C.gpus = "7" # GPU ID to use