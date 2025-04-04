# HMIL
This repo is the official implementation of "HMIL: Hierarchical Multi-Instance Learning for Fine-Grained WSI Classification".

[[IEEE TMI]](https://ieeexplore.ieee.org/document/10810475) | [[arXiv]](https://arxiv.org/abs/2411.07660) | [[Preprocessing]](https://github.com/ChengJin-git/HMIL?tab=readme-ov-file#Preprocessing) | [[Workflow]](https://github.com/ChengJin-git/HMIL?tab=readme-ov-file#Workflow) | [[Citation]](https://github.com/ChengJin-git/HMIL?tab=readme-ov-file#Citation)

**Abstract**: Fine-grained classification of whole slide images (WSIs) is essential in precision oncology, enabling precise cancer diagnosis and personalized treatment strategies. The core of this task involves distinguishing subtle morphological variations within the same broad category of gigapixel-resolution images, which presents a significant challenge. While the multi-instance learning (MIL) paradigm alleviates the computational burden of WSIs, existing MIL methods often overlook hierarchical label correlations, treating fine-grained classification as a flat multi-class classification task. To overcome these limitations, we introduce a novel hierarchical multi-instance learning (HMIL) framework. By facilitating on the hierarchical alignment of inherent relationships between different hierarchy of labels at instance and bag level, our approach provides a more structured and informative learning process. Specifically, HMIL incorporates a class-wise attention mechanism that aligns hierarchical information at both the instance and bag levels. Furthermore, we introduce supervised contrastive learning to enhance the discriminative capability for fine-grained classification and a curriculum-based dynamic weighting module to adaptively balance the hierarchical feature during training. Extensive experiments on our large-scale cytology cervical cancer (CCC) dataset and two public histology datasets, BRACS and PANDA, demonstrate the state-of-the-art class-wise and overall performance of our HMIL framework. 

## Installation
```bash
pip install -r requirements.txt
```

## Preprocessing
The preprocessing step is to build patches, extract features and split the dataset into k-fold cross-validation folds. This part is based on [ASlide](https://github.com/MrPeterJin/ASlide) and [CLAM](https://github.com/mahmoodlab/CLAM).

### Patch Tiling and Feature Extraction
Build patches and extract features for each whole slide images at a certain resolution, refer to [ASlide](https://github.com/MrPeterJin/ASlide) and [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md?plain=1#L113).

After feature extraction, you will get a FEATURE_DIRECTORY in the following format:

```bash
FEATURE_DIRECTORY/
	├── slide_001.pt
	├── slide_002.pt
	├── slide_003.pt
	└── ```
```

### Dataset Spliting
After obtaining the dataset, adopt a k-fold cross-validation protocol to split the dataset from the instruction of [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md?plain=1#L234). After splitting the dataset, you will get a DATA_SPLIT.csv at the specified DATA_SPLIT_DIRECTORY. Use this file to split the dataset into training and validation sets by
```bash
python src/tools/make_dataset_pkls.py --feature_directory <FEATURE_DIRECTORY> --root <DATA_SPLIT_DIRECTORY> --label_csv <LABEL_CSV> --split_csv_root <SPLIT_CSV_ROOT> --folds <FOLDS>
```

### Training and Validation
Before training and validation, you need to specify the basic settings in the [config file](./src/config/config_example.py). The provided config file is for the BRACS dataset. For other datasets, you need to modify the config file accordingly.

After specifying the config file, you can start training and validation by
```bash
python src/main.py --config <CONFIG_FILE>
```

## Model Structure
The model structure is shown in the following figure:
![](./schematic.jpg)

## Citation
If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```bibtex
@ARTICLE{jin2024hmil,
  author={Jin, Cheng and Luo, Luyang and Lin, Huangjing and Hou, Jun and Chen, Hao},
  journal={IEEE Transactions on Medical Imaging}, 
  title={HMIL: Hierarchical Multi-Instance Learning for Fine-Grained Whole Slide Image Classification}, 
  year={2025},
  volume={44},
  number={4},
  pages={1796-1808},
  doi={10.1109/TMI.2024.3520602}}
```
