import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Import local modules
from dataset_loader import get_dataloaders, get_base_transformations
from augmentation_balance import get_gpu_augmentations, get_oversampling_sampler
from models_cnn_transformer import get_model, RECOMMENDED_IMG_SIZES

# SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR = "plots"
MASTER_RESULTS_FILE = "master_experiment_results.csv"
BATCH_SIZE = 32
NUM_WORKERS = 0  # CRITICAL FIX FOR WINDOWS SPEED

class SmartCachedDataset(Dataset):
    """Caches images in RAM to maximize GPU usage."""
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform
        self.samples = original_dataset.samples
        self.targets = original_dataset.targets
        self.cached_data = [None] * len(original_dataset)
        
        print(f"   🚀 Caching {len(self.samples)} images to RAM...")
        # Use simple loop to avoid tqdm overhead on small data
        for i in range(len(original_dataset)):
            path, label = self.samples[i]
            try:
                with open(path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    self.cached_data[i] = (img, label)
            except:
                pass

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.cached_data[idx]
        if item is None:
            path, label = self.samples[idx]
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                return (self.transform(img), label)
        return item

def save_plots(y_true, y_pred, classes, name):
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(os.path.join(PLOTS_DIR, f"CM_{name}.png"))
        plt.close()
    except:
        pass

def run_single_experiment(data_dir, model_name, epochs=50): # Increased epochs
    dataset_name = os.path.basename(data_dir)
    experiment_name = f"{model_name}_{dataset_name}"
    
    # Skip if already done
    if os.path.exists(MASTER_RESULTS_FILE):
        try:
            df = pd.read_csv(MASTER_RESULTS_FILE)
            if not df[(df['Model'] == model_name) & (df['Dataset'] == dataset_name)].empty:
                print(f"⏩ Skipping {experiment_name} (Already found in CSV)")
                return
        except:
            pass

    print(f"\n🔹 RUNNING: {experiment_name}")
    
    img_size = RECOMMENDED_IMG_SIZES.get(model_name, 224)
    train_raw, test_raw, classes = get_dataloaders(data_dir)
    
    base_tf = get_base_transformations(img_size)
    train_data = SmartCachedDataset(train_raw, transform=base_tf)
    test_data = SmartCachedDataset(test_raw, transform=base_tf)
    
    gpu_aug = get_gpu_augmentations(img_size)
    sampler = get_oversampling_sampler(train_data)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    model = get_model(model_name, len(classes)).to(DEVICE)
    if model is None: return

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) # Slightly higher LR for AdamW
    criterion = nn.CrossEntropyLoss()
    # Cosine Scheduler is better for beating benchmarks
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    y_true_final, y_pred_final = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            imgs = gpu_aug(imgs)
            
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()
                y_true.extend(lbls.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        val_acc = 100 * correct / total
        
        if val_acc > best_acc:
            best_acc = val_acc
            y_true_final, y_pred_final = y_true, y_pred
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{experiment_name}_best.pth"))
            
    # Final Report
    print(f"   🏆 Best Acc: {best_acc:.2f}%")
    save_plots(y_true_final, y_pred_final, classes, experiment_name)
    report = classification_report(y_true_final, y_pred_final, target_names=classes, output_dict=True)
    
    result = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": best_acc,
        "Precision": report['macro avg']['precision'],
        "Recall": report['macro avg']['recall'],
        "F1-Score": report['macro avg']['f1-score'],
        "Kappa": cohen_kappa_score(y_true_final, y_pred_final)
    }
    
    df = pd.DataFrame([result])
    df.to_csv(MASTER_RESULTS_FILE, mode='a', header=not os.path.exists(MASTER_RESULTS_FILE), index=False)