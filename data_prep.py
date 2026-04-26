import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"D:\S\Mpox_Research_Project"
RAW_DATA_DIR = os.path.join(BASE_DIR, "Raw_Data")
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed_Data")
MSID_SOURCE = os.path.join(RAW_DATA_DIR, "MonkeyPox Skin Images Dataset")

def find_folder(root_dir, target_name):
    """Recursively finds a folder named target_name."""
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if d.lower() == target_name.lower():
                return os.path.join(root, d)
    return None

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_files(src_dir, dst_dir, files):
    create_dir(dst_dir)
    for f in files:
        src_file = os.path.join(src_dir, f)
        dst_file = os.path.join(dst_dir, f)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)

def prepare_msid():
    print("\n--- Processing MSID (4 Classes) ---")
    dest_root = os.path.join(PROCESSED_DIR, "MSID")
    classes = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
    
    for cls in tqdm(classes, desc="Organizing MSID"):
        cls_path = os.path.join(MSID_SOURCE, cls)
        if not os.path.exists(cls_path):
            print(f"⚠️ Warning: {cls_path} not found.")
            continue
            
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.8)
        copy_files(cls_path, os.path.join(dest_root, "train", cls), images[:split_idx])
        copy_files(cls_path, os.path.join(dest_root, "test", cls), images[split_idx:])
    print("✅ MSID Dataset Ready.")

def prepare_msld():
    print("\n--- Processing MSLD (Binary) ---")
    dest_root = os.path.join(PROCESSED_DIR, "MSLD")
    
    # Auto-detect where the "Fold1" or "Train" folders are hiding
    msld_root = os.path.join(RAW_DATA_DIR, "Monkeypox Skin Lesion Dataset")
    train_src = find_folder(msld_root, "Train")
    test_src = find_folder(msld_root, "Test")
    val_src = find_folder(msld_root, "Val")

    if not train_src:
        print("❌ CRITICAL ERROR: Could not find 'Train' folder in MSLD. Check unzipped structure.")
        return

    # Map sources to destination (Merge Val into Train)
    tasks = [
        (train_src, "train"), 
        (val_src, "train"), 
        (test_src, "test")
    ]
    
    classes = ["Monkeypox", "Others"]

    for src_path, dest_subset in tasks:
        if not src_path: continue
        for cls in classes:
            # Handle case where class folder might be lower/upper case
            cls_src = find_folder(src_path, cls) 
            if cls_src:
                dest = os.path.join(dest_root, dest_subset, cls)
                images = os.listdir(cls_src)
                copy_files(cls_src, dest, images)
            else:
                print(f"⚠️ Could not find class '{cls}' inside {src_path}")

    print("✅ MSLD Dataset Ready.")

if __name__ == "__main__":
    random.seed(42)
    prepare_msid()
    prepare_msld()