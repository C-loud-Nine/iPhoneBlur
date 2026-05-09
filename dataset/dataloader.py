import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
from pathlib import Path

class iPhoneBlurDataset(Dataset):
    """
    Exact implementation from notebooks (MIMO-UNet/NAFNet).
    Supports both rglob file discovery and patch-based augmentation.
    """
    def __init__(self, data_dir, patch_size=256, is_train=True):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.is_train = is_train
        
        # Exact rglob loop from notebooks
        blur_files = []
        sharp_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG']:
            blur_files.extend(list(self.data_dir.rglob(f'blur/{ext}')))
            sharp_files.extend(list(self.data_dir.rglob(f'sharp/{ext}')))
            
        self.blur_paths = sorted(blur_files)
        self.sharp_paths = sorted(sharp_files)
        
        assert len(self.blur_paths) == len(self.sharp_paths), \
            f"Mismatch: {len(self.blur_paths)} blur vs {len(self.sharp_paths)} sharp"
            
        print(f"Found {len(self.blur_paths)} image pairs in {self.data_dir.name}")
    
    def __len__(self):
        return len(self.blur_paths)
    
    def __getitem__(self, idx):
        blur = cv2.imread(str(self.blur_paths[idx]), cv2.IMREAD_COLOR)
        sharp = cv2.imread(str(self.sharp_paths[idx]), cv2.IMREAD_COLOR)
        
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
        
        h, w, _ = blur.shape
        ps = self.patch_size
        
        if self.is_train:
            # Random crop + augmentation
            if h > ps and w > ps:
                r = random.randint(0, h - ps)
                c = random.randint(0, w - ps)
                blur = blur[r:r+ps, c:c+ps, :]
                sharp = sharp[r:r+ps, c:c+ps, :]
            else:
                blur = cv2.resize(blur, (ps, ps))
                sharp = cv2.resize(sharp, (ps, ps))
            
            # Random flips
            if random.random() > 0.5:
                blur = np.fliplr(blur).copy()
                sharp = np.fliplr(sharp).copy()
            if random.random() > 0.5:
                blur = np.flipud(blur).copy()
                sharp = np.flipud(sharp).copy()
        else:
            # Center crop for validation
            if h > ps and w > ps:
                r = (h - ps) // 2
                c = (w - ps) // 2
                blur = blur[r:r+ps, c:c+ps, :]
                sharp = sharp[r:r+ps, c:c+ps, :]
        
        # Convert to tensor [0, 1]
        blur = torch.from_numpy(blur).float().permute(2, 0, 1) / 255.0
        sharp = torch.from_numpy(sharp).float().permute(2, 0, 1) / 255.0
        
        return blur, sharp

def get_dataloader(data_dir, patch_size=256, is_train=True, batch_size=4, num_workers=2):
    """
    Convenience function matching notebook usage.
    
    Example:
        train_loader = get_dataloader('./data/iphoneblur/train', is_train=True, batch_size=4)
        val_loader = get_dataloader('./data/iphoneblur/test', is_train=False, batch_size=1)
    """
    dataset = iPhoneBlurDataset(data_dir, patch_size=patch_size, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, 
                     num_workers=num_workers, pin_memory=True)
