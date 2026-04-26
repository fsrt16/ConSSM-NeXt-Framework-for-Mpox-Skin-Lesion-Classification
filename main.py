import os
import torch
import time
from data_prep import prepare_msid, prepare_msld
from run_experiments import run_single_experiment
from models_cnn_transformer import ALL_20_CNN_MODELS, ALL_10_TRANSFORMER_MODELS

# --- SETUP ---
BASE_DIR = r"D:\S\Mpox_Research_Project"
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed_Data")
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR = "plots"

def setup_directories():
    for d in [CHECKPOINT_DIR, PLOTS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

def main():
    print(f"🚀 Initializing Master Workflow on {torch.cuda.get_device_name(0)}")
    setup_directories()
    
    # 1. Prepare Data
    if not os.path.exists(os.path.join(PROCESSED_DIR, "MSID")):
        prepare_msid()
    if not os.path.exists(os.path.join(PROCESSED_DIR, "MSLD")):
        prepare_msld()

    datasets = [
        os.path.join(PROCESSED_DIR, "MSID"),
        os.path.join(PROCESSED_DIR, "MSLD")
    ]
    
    # 2. Combine ALL Models
    all_models = ALL_20_CNN_MODELS + ALL_10_TRANSFORMER_MODELS
    total_experiments = len(datasets) * len(all_models)
    print(f"\n📊 Total Planned Experiments: {total_experiments}")
    print(f"   - Models: {len(all_models)} (20 CNNs + 10 Transformers)")
    print(f"   - Datasets: {len(datasets)}")
    
    # 3. Execution Loop
    count = 1
    for dataset_path in datasets:
        ds_name = os.path.basename(dataset_path)
        print(f"\n{'='*60}\n  DATASET: {ds_name}\n{'='*60}")
        
        for model_name in all_models:
            print(f"\n[{count}/{total_experiments}] Starting {model_name} on {ds_name}...")
            start_time = time.time()
            
            run_single_experiment(dataset_path, model_name, epochs=20)
            
            elapsed = (time.time() - start_time) / 60
            print(f"⏱️ Model finished in {elapsed:.2f} minutes.")
            count += 1

if __name__ == "__main__":
    main()