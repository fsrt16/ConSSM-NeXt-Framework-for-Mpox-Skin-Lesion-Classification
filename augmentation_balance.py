import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import numpy as np
from torchvision import transforms

def get_gpu_augmentations(img_size):
    """
    Returns a Sequential container of GPU-compatible transforms.
    """
    return nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    )

def get_oversampling_sampler(dataset):
    """
    Creates a WeightedRandomSampler to handle class imbalance.
    """
    try:
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'original_dataset'):
            targets = dataset.original_dataset.targets
        else:
            return None

        targets = np.array(targets)
        class_counts = np.bincount(targets)
        class_counts[class_counts == 0] = 1 # Prevent div by zero
        class_weights = 1. / class_counts
        
        sample_weights = class_weights[targets]
        sample_weights = torch.from_numpy(sample_weights).double()
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler
    except Exception as e:
        print(f"⚠️ Sampler Error: {e}")
        return None