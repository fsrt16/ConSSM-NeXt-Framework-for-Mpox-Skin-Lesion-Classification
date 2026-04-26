import os
from torchvision import datasets
from torchvision import transforms

def get_base_transformations(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(data_dir):
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing train/test folders in {data_dir}")
        
    train_dataset = datasets.ImageFolder(root=train_path)
    test_dataset = datasets.ImageFolder(root=test_path)
    classes = train_dataset.classes
    
    print(f"   📂 Classes: {classes}")
    print(f"   📊 Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset, classes