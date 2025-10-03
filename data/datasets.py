import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch

class TLUStatesDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        split_file = os.path.join(root, 'split_zhou_TluStates.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            self.samples = split_data.get(split, [])
            self.class_names = sorted(set([s[1] for s in self.samples]))
        else:
            self._load_from_folders(split)
        
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        
    def _load_from_folders(self, split):
        folders = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        folders = sorted([f for f in folders if not f.startswith('.')])
        self.class_names = folders
        
        all_samples = []
        for class_name in folders:
            class_dir = os.path.join(self.root, class_name)
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for img in images:
                all_samples.append((os.path.join(class_name, img), class_name))
        
        train_ratio = 0.8
        split_idx = int(len(all_samples) * train_ratio)
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        img_full_path = os.path.join(self.root, img_path) if not os.path.isabs(img_path) else img_path
        
        image = Image.open(img_full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[class_name]
        return image, label, class_name
