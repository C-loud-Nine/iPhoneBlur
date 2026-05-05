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
