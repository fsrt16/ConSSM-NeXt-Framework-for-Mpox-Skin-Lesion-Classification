# ConSSM-NeXt-Framework-for-Mpox-Skin-Lesion-Classification
ConSSM-NeXt: A Contrastive State-Space Enhanced ConvNeXt Framework for Explainable Multi-Class Mpox Skin Lesion Classification
# ConSSM-NeXt: Contrastive State Space Model for Monkeypox Skin Lesion Classification

> **Paper link will be added upon publication.**

***

## Overview

This repository provides the official implementation of **ConSSM-NeXt**, a hybrid deep learning architecture combining Contrastive Learning, Selective State Space Models (SSM), and ConvNeXt-inspired feature extraction for automated Monkeypox skin lesion classification. The model is benchmarked against **33 baseline architectures** spanning first-generation CNNs, second-generation CNNs, and Vision Transformers, achieving state-of-the-art performance on both binary and multi-class settings.

| Metric | ConSSM-NeXt (Proposed) | Best Baseline |
|--------|------------------------|---------------|
| Accuracy | **94.38%** | 72.80% (Noisy Student EfficientNet-L2) |
| F1-Score | **92.57%** | 70.90% (Noisy Student EfficientNet-L2) |
| 95% CI (Acc) | [93.87%, 94.78%] | — |
| 95% CI (F1) | [92.13%, 92.72%] | — |

***

## Datasets

Two publicly available Monkeypox skin image datasets are used in this study.

### MSID — MonkeyPox Skin Images Dataset (4-Class)

Multi-class classification task across the following categories:

- `Chickenpox`
- `Measles`
- `Monkeypox`
- `Normal`

An 80/20 stratified train/test split is applied per class.

### MSLD — Monkeypox Skin Lesion Dataset (Binary)

Binary classification task:

- `Monkeypox`
- `Others`

The official Train/Val/Test split is used, with the Validation set merged into Training following standard practice.

**Directory structure after running `data_prep.py`:**

```
Mpox_Research_Project/
├── Raw_Data/
│   ├── MonkeyPox Skin Images Dataset/
│   └── Monkeypox Skin Lesion Dataset/
└── Processed_Data/
    ├── MSID/
    │   ├── train/{Chickenpox, Measles, Monkeypox, Normal}/
    │   └── test/{Chickenpox, Measles, Monkeypox, Normal}/
    └── MSLD/
        ├── train/{Monkeypox, Others}/
        └── test/{Monkeypox, Others}/
```

***

## Repository Structure

```
.
├── data_prep.py               # Dataset download, organisation, and 80/20 split
├── augmentation_balance.py    # GPU-accelerated augmentation + WeightedRandomSampler
├── models_cnn_transformer.py  # Model registry (20 CNNs + 10 Transformers via timm)
├── run_experiments.py         # Training loop, CosineAnnealingLR, AMP, experiment runner
├── CODE.ipynb                 # Statistical analysis notebook (Friedman, t-test, Wilcoxon)
├── TESTFINAL.txt              # Publication-ready ultimate evaluation suite
├── requirements.txt           # Python dependencies
└── README.md
```

***

## Installation

```bash
git clone https://github.com/<your-repo>/monkeypox-conssm-next.git
cd monkeypox-conssm-next
pip install -r requirements.txt
```

**`requirements.txt`**

```
torch
torchvision
timm
pandas
numpy
seaborn
matplotlib
tqdm
scikit-learn
pillow
```

A CUDA-capable GPU is strongly recommended. The training pipeline uses PyTorch Automatic Mixed Precision (`torch.amp`) and in-RAM dataset caching (`SmartCachedDataset`) for maximum throughput.

***

## Model Architecture — ConSSM-NeXt

ConSSM-NeXt integrates three complementary paradigms:

1. **Selective State Space Modelling (SSM):** Captures long-range sequential dependencies in dermatological feature maps with sub-quadratic complexity.
2. **ConvNeXt-inspired Local Feature Extraction:** Depthwise separable convolutions with inverted bottleneck blocks extract fine-grained texture and morphological cues characteristic of pox lesions.
3. **Contrastive Learning Head:** A projection head trained with a contrastive objective enforces tighter intra-class compactness and larger inter-class separation in the embedding space.

***

## Usage

### Step 1 — Prepare Data

```bash
python data_prep.py
```

Recursively locates class folders, shuffles with `random.seed(42)`, and splits at 80/20.

### Step 2 — Run Experiments

```bash
python run_experiments.py
```

Trains all registered models sequentially. Completed experiments are auto-skipped via `master_experiment_results.csv`. Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Batch Size | 32 |
| Precision | AMP (FP16) |

Results, confusion matrices, and model checkpoints are saved to `plots/` and `checkpoints/`.

### Step 3 — Evaluate

The `TESTFINAL.txt` script contains a publication-ready evaluation suite (`UltimateEvaluator`) covering:

- Core metrics: Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, MCC, Brier Score
- ROC Curve and Precision-Recall Curve with AUC
- Threshold optimisation sweep (0.10 → 0.95)
- Calibration / Reliability diagram (ECE)
- 10-Fold Stratified CV on test set with Shapiro-Wilk normality + one-sample t-test vs 95% baseline
- Error analysis: Top-5 False Positives & False Negatives with image visualisation
- Explainability: Grad-CAM, LIME, SHAP (DeepExplainer), Captum Saliency
- t-SNE feature embedding visualisation
- Computational efficiency: parameter count and inference latency

```python
# Load best checkpoint and run
student.load_state_dict(torch.load("best_student_distilled.pth"))
student.eval()

evaluator = UltimateEvaluator(
    model=student,
    test_loader=test_loader,
    class_names=label_encoder.classes_,
    device=DEVICE
)
evaluator.run_full_evaluation()
```

***

## Baseline Models Benchmarked

### Generation-1 CNNs (6 models)
Inception, MobileNetV2, DenseNet121, ResNet101V2, InceptionResNetV2, NASNetMobile

### Generation-2 CNNs (10 models)
EfficientNetB0, EfficientNetB7, ConvNeXt, PoolFormer, Twins-SVT, HRNet, SE-ResNet50, ResNeXt101, RegNetY-800MF, BiT-R50x1, Noisy Student EfficientNet-L2

### Vision Transformers (15 models)
ViT-Base, ViT-Large, Swin Transformer, Swin V2, DeiT-Small, DeiT-Base, CoaT-Lite Small, PVT, T2T-ViT, BEiT, CvT, ViTAEv2, EfficientFormer-L1, MobileViT, ConvMixer

All baselines are loaded via `timm.create_model()` with ImageNet-pretrained weights.

***

## Augmentation Strategy

GPU-accelerated augmentations applied on-the-fly during training:

| Transform | Parameters |
|-----------|------------|
| RandomHorizontalFlip | p = 0.5 |
| RandomVerticalFlip | p = 0.5 |
| RandomRotation | ±20° |
| ColorJitter | brightness/contrast/saturation = 0.2, hue = 0.1 |
| RandomAffine | translate = (0.1, 0.1), scale = (0.9, 1.1) |

Class imbalance is handled using `WeightedRandomSampler` with inverse frequency weights.

***

## Statistical Analysis

All comparisons reported in the paper are grounded in rigorous statistical testing (see `CODE.ipynb`):

- **Friedman Test:** χ² = 15.00, *p* = 0.0018 — confirms significant differences across groups
- **Paired t-test vs ConSSM-NeXt:** *p* = 0.0000 for all three baseline clusters
- **Wilcoxon Signed-Rank:** *p* = 0.0625 across all pairwise comparisons (5-fold, small *n*)
- **Cluster Mean Accuracy:** Gen1-CNN = 59.30%, Gen2-CNN = 64.75%, Transformer = 67.11%, **Proposed = 94.33%**
- **Cluster Mean F1:** Gen1-CNN = 56.78%, Gen2-CNN = 63.22%, Transformer = 65.41%, **Proposed = 92.43%**

***

## Citation

> Paper link will be added upon publication.

```bibtex
@article{conssmnext2026,
  title   = {ConSSM-NeXt: Contrastive Selective State Space Model for Monkeypox Skin Lesion Classification},
  author  = {Satpathy, Ashita and et al.},
  journal = {TBD},
  year    = {2026},
  note    = {Under Review}
}
```

***

## License

This project is released for academic and research use. Dataset licences apply as per their original sources (MSID and MSLD).
