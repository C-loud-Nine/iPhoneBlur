# iPhoneBlur: A Difficulty-Stratified Benchmark for Consumer Device Motion Deblurring

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](https://arxiv.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://www.kaggle.com/datasets/shafi09/iphoneblur)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**First severity-stratified motion deblurring benchmark** for deployment-critical evaluation on edge devices. Unlike existing datasets that report only aggregate metrics, iPhoneBlur categorizes 7,400 blur-sharp image pairs into three validated difficulty levels (Easy, Medium, Hard).

[**Paper**](https://arxiv.org/) | [**Dataset**](https://www.kaggle.com/datasets/shafi09/iphoneblur) | [**Supplementary**](https://arxiv.org/)

---

## 🎯 **Key Features**

- **7,400 high-quality blur-sharp pairs** (130% larger than GoPro)
- **Three validated difficulty levels** based on PSNR thresholds (Easy ≥30 dB, Medium 24-30 dB, Hard <24 dB)
- **Motion gradient validation**: Spearman ρ=-0.407 (p<10⁻³⁰⁰), 2.2× motion increase across tiers
- **ISP artifact validation**: 100% suppression (Cohen's d=2.32), confirming synthesis quality
- **Video-level train/test split**: Zero overlap between 39 train and 12 test videos
- **Superior quality**: Mean SSIM 0.85 (vs GoPro 0.81)
- **1920×1080 resolution** JPEG images from iPhone 17 Pro (177-240 fps)

---

## 📊 **Dataset Statistics**

| Metric | Easy | Medium | Hard | Overall |
|--------|------|--------|------|---------|
| **Samples** | 1,277 (17.3%) | 3,895 (52.6%) | 2,228 (30.1%) | 7,400 |
| **PSNR (dB)** | 31.7±1.2 | 26.5±1.7 | 22.6±1.0 | 26.2±3.4 |
| **SSIM** | 0.96±0.01 | 0.87±0.05 | 0.75±0.06 | 0.85±0.09 |
| **Motion (px/fr)** | 6.0±5.0 | 11.3±5.6 | 13.2±4.3 | 11.0±5.7 |
| **Contrast** | 0.264±0.020 | 0.269±0.021 | 0.274±0.022 | 0.270±0.022 |

---

## 🚀 **Quick Start**

### **1. Download Dataset**

```bash
# Option A: Kaggle API
kaggle datasets download -d shafi09/iphoneblur
unzip iphoneblur.zip

# Option B: Manual download
# Visit https://www.kaggle.com/datasets/shafi09/iphoneblur
```

### **2. Install Dependencies**

```bash
pip install torch torchvision numpy opencv-python pillow pandas
```

### **3. Load Dataset**

```python
import pandas as pd
from PIL import Image
import os

# Load metadata
metadata = pd.read_csv('iPhoneBlur/metadata/train_metadata.csv')

# Load a sample
sample = metadata.iloc[0]
blur_img = Image.open(f"iPhoneBlur/train/{sample['video']}/blur/{sample['img_id']}.jpg")
sharp_img = Image.open(f"iPhoneBlur/train/{sample['video']}/sharp/{sample['img_id']}.jpg")

print(f"Difficulty: {sample['difficulty']}")
print(f"PSNR: {sample['psnr']:.2f} dB")
print(f"Motion: {sample['motion']:.2f} px/frame")
```

---

## 📁 **Dataset Structure**

iPhoneBlur/
├── train/                          # Training set (5,714 samples)
│   ├── IMG_2139/
│   │   ├── blur/                   # Blurred images
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── sharp/                  # Sharp ground truth
│   │       ├── 00000.jpg
│   │       ├── 00001.jpg
│   │       └── ...
│   └── [38 more videos]/
├── test/                           # Test set (1,686 samples)
│   └── [12 videos, same structure as train]
└── metadata/
├── complete_metadata.csv       # All 7,400 samples with 20 metadata fields
├── train_metadata.csv          # Training set metadata
├── test_metadata.csv           # Test set metadata
└── dataset_info.json           # Dataset configuration

---

## 📋 **Metadata Fields**

Each sample includes 20 metadata fields:

| Field | Description |
|-------|-------------|
| `img_id` | Unique sample identifier (e.g., "00000") |
| `video` | Source video filename (e.g., "IMG_2139") |
| `img_num` | Frame number in original video |
| `blur_window` | Number of frames averaged (3-21) |
| `blur_duration_ms` | Temporal duration of blur (12.5-120.7 ms) |
| `psnr` | Peak Signal-to-Noise Ratio (dB) |
| `ssim` | Structural Similarity Index (0-1) |
| `lpips` | Learned Perceptual Image Patch Similarity |
| `motion` | Optical flow magnitude (px/frame) |
| `sharpness` | Laplacian variance of sharp image |
| `contrast` | RMS contrast of sharp image |
| `difficulty` | Difficulty level: Easy/Medium/Hard |
| `isp_blur` | ISP artifact energy in blur image |
| `isp_sharp` | ISP artifact energy in sharp image |
| `isp_diff` | ISP suppression (sharp - blur) |
| ... | See paper for full list |

---

## 🧪 **Benchmark Results**

Cross-domain evaluation from GoPro-pretrained models:

| Model | GoPro (in-domain) | iPhoneBlur (zero-shot) | iPhoneBlur (fine-tuned) | Gap Recovery |
|-------|-------------------|------------------------|-------------------------|--------------|
| **NAFNet-64** | 33.7 dB | 26.6 dB | 31.2 dB | **64%** |
| **HINet** | 32.7 dB | 25.8 dB | 30.4 dB | 67% |
| **Restormer** | 32.9 dB | 25.9 dB | 30.6 dB | 67% |
| **MIMO-UNet+** | 32.4 dB | 24.9 dB | 29.7 dB | 63% |
| **Instruct-IR** | 30.4 dB | 22.3 dB | 27.8 dB | 68% |
| **FFTformer** | 33.8 dB | 26.7 dB | 31.3 dB | 65% |

**Key Finding:** 7.2-9.1 dB performance degradation across all architectures, reduced to 2.5-3.1 dB after fine-tuning.

---

## 📈 **Stratified Performance Analysis**

Performance degradation increases monotonically with difficulty:

| Model | Easy → Medium | Medium → Hard | Total Drop (Easy → Hard) |
|-------|---------------|---------------|--------------------------|
| NAFNet-64 | -5.4 dB | -1.8 dB | **-7.2 dB** |
| HINet | -5.6 dB | -1.9 dB | -7.5 dB |
| Restormer | -5.5 dB | -1.9 dB | -7.4 dB |
| MIMO-UNet+ | -6.1 dB | -2.1 dB | -8.2 dB |

---

## 🔬 **Novel Contributions**

### **1. Motion Gradient Validation**
First dataset to validate difficulty stratification through optical flow:
- **Dataset-level**: Spearman ρ=-0.407 (p<10⁻³⁰⁰)
- **Kruskal-Wallis**: H=1265 (p<10⁻²⁷⁵), confirming monotonic motion increase
- **Per-video**: 74% (38/51) show negative motion-PSNR correlation

### **2. ISP Artifact Suppression Validation**
Novel methodology proving synthesis quality:
- **Universal suppression**: 100% of samples (7,400/7,400)
- **Cohen's d**: 2.32 (very large effect, approaching RealBlur-J's 2.24)
- **Physical validation**: Confirms gamma-linearized synthesis reproduces real blur

### **3. Video-Level Train/Test Split**
- 51 source videos: 39 train, 12 test
- Zero sample overlap prevents content memorization
- 78% of videos span all 3 difficulty tiers

---

## 💻 **Code Examples**

### **Evaluation Script**

```python
import torch
from models import NAFNet
from data import iPhoneBlurDataset
from metrics import calculate_psnr, calculate_ssim

# Load model
model = NAFNet(width=64).cuda()
model.load_state_dict(torch.load('nafnet_gopro.pth'))

# Load test set
test_set = iPhoneBlurDataset(
    root='iPhoneBlur/test',
    metadata='iPhoneBlur/metadata/test_metadata.csv'
)

# Evaluate by difficulty
for difficulty in ['Easy', 'Medium', 'Hard']:
    subset = test_set.filter_by_difficulty(difficulty)
    psnr, ssim = evaluate(model, subset)
    print(f"{difficulty}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
```

### **Fine-tuning on iPhoneBlur**

```python
from torch.utils.data import DataLoader

# Load training set
train_set = iPhoneBlurDataset(
    root='iPhoneBlur/train',
    metadata='iPhoneBlur/metadata/train_metadata.csv',
    patch_size=256
)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

# Fine-tune
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(50):
    for blur, sharp in train_loader:
        blur, sharp = blur.cuda(), sharp.cuda()
        pred = model(blur)
        loss = torch.nn.functional.l1_loss(pred, sharp)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🏆 **Use Cases**

### **Academic Research**
- Benchmark new deblurring algorithms
- Study performance degradation across difficulty levels
- Develop difficulty-aware training strategies (curriculum learning)
- Analyze failure modes systematically

### **Industry/Deployment**
- **Confidence-based routing**: Process Easy samples on-device, Hard in cloud
- **Computational budget allocation**: Allocate resources by predicted difficulty
- **Failure prediction**: Predict when restoration will fail before processing
- **Edge device optimization**: Test methods on resource-constrained hardware

---

## 📜 **License**

- **Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Free to use with attribution
- **Code**: [MIT License](https://opensource.org/licenses/MIT) — Free to use, modify, and distribute

**Source Videos**: Available on Kaggle under CC BY 4.0
- [iPhone 17 Pro Videos](https://www.kaggle.com/datasets/shafi09/iphone)
- [iPhone Slow-Motion Videos](https://www.kaggle.com/datasets/shafi09/iphone-slomo)

---

## 🔗 **Links**

- **Dataset**: [Kaggle](https://www.kaggle.com/datasets/shafi09/iphoneblur)
- **Code**: [GitHub](https://github.com/C-loud-Nine/iPhoneBlur)

---

**Data Collection**: All videos captured in Bangladesh, February-April 2026, using iPhone 17 Pro at 177-240 fps.

---

## 📧 **Contact**

**Abdullah Al Shafi**  
Department of Computer Science and Engineering  
Khulna University of Engineering & Technology (KUET)  
Email: `shafi2007055@stud.kuet.ac.bd`

**Supervisor**: Professor Kazi Saeed Alam  
Email: `saeedalam@cse.kuet.ac.bd`


---

**Last Updated**: May 2026  
**Version**: 1.0
