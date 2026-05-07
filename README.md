# iPhoneBlur: A Photorealistic Motion Blur Dataset from iPhone Videos

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](https://arxiv.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://kaggle.com/datasets/shafi09/iphoneblur)
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](LICENSE)

Official implementation and benchmark for **iPhoneBlur**, a motion blur dataset containing 7,400 blur-sharp image pairs extracted from 51 iPhone videos, specifically designed to evaluate image deblurring models under real-world camera ISP conditions.

> **Abdullah Al Shafi**, Kazi Saeed Alam  
> *Khulna University of Engineering & Technology (KUET)*  
> **NeurIPS 2026 Datasets and Benchmarks Track (Under Review)**

---

## 🎯 Key Features

- **7,400 image pairs** (5,714 training, 1,686 test) from iPhone 13/14/15 Pro
- **Photorealistic blur synthesis** via gamma-corrected temporal averaging
- **ISP-validated quality**: Cross-domain PSNR gap of ~6.8 dB confirms benchmark-grade difficulty
- **Stratified difficulty labels**: Easy (280), Medium (953), Hard (453) based on PSNR ranges
- **Extensive metadata**: 20+ physical scene attributes per image
- **Benchmark results**: 5 SOTA models evaluated (NAFNet, MIMO-UNet, HINet, Restormer, InstructIR)

---

## 📊 Dataset Statistics

| Split | Images | Easy | Medium | Hard | Avg PSNR (dB) |
|-------|--------|------|--------|------|---------------|
| Train | 5,714  | -    | -      | -    | 26.12 ± 3.24  |
| Test  | 1,686  | 280  | 953    | 453  | 26.08 ± 3.18  |
| **Total** | **7,400** | **280** | **953** | **453** | **26.11 ± 3.22** |

**Quality Ranges:**
- Easy: PSNR ≥ 30.0 dB
- Medium: 24.0 ≤ PSNR < 30.0 dB
- Hard: PSNR < 24.0 dB

---

## 📥 Downloads

### Dataset

| Source | Size | Format | Link |
|--------|------|--------|------|
| **Kaggle** | 9.08 GB | JPG + CSV | [![Kaggle](https://img.shields.io/badge/Download-Kaggle-20BEFF)](https://kaggle.com/datasets/shafi09/iphoneblur) |

**Quick download:**
```bash
# Install Kaggle API
pip install kaggle

# Download dataset
python dataset/download_dataset.py
```

### Pre-trained Models

All fine-tuned models available on HuggingFace:

| Model | Parameters | PSNR (dB) | SSIM | LPIPS | Size | Download |
|-------|-----------|-----------|------|-------|------|----------|
| **NAFNet** | 67.89M | 26.75 ± 3.94 | 0.9167 | 0.0583 | 272 MB | [Download](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/nafnet_final_weights.pth) |
| **HINet** | 88.7M | 27.12 ± 3.86 | 0.9195 | 0.0561 | 355 MB | [Download](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/hinet_final_weights.pth) |
| **Restormer** | 26.1M | 26.89 ± 3.78 | 0.9178 | 0.0574 | 90 MB | [Download](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/restormer_final_weights.pth) |
| **MIMO-UNet** | 16.1M | 25.45 ± 3.21 | 0.8984 | 0.0698 | 16.5 MB | [Download](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/mimo_final_weights.pth) |
| **InstructIR** | 63.6M | 25.23 ± 3.45 | 0.8967 | 0.0712 | 63.6 MB | [Download](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/instructir_final_weights.pth) |

**All models:** [HuggingFace Repository](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/iphoneblur-benchmark.git
cd iphoneblur-benchmark

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### 2. Download Dataset

```bash
# Option 1: Using our script (recommended)
python dataset/download_dataset.py

# Option 2: Manual download from Kaggle
# Download from: https://kaggle.com/datasets/shafi09/iphoneblur
# Extract to: ./dataset/data/iphoneblur/
```

**Expected structure:**
```
dataset/data/iphoneblur/
├── train/
│   ├── IMG_XXXX/
│   │   ├── blur/
│   │   │   └── *.jpg
│   │   └── sharp/
│   │       └── *.jpg
│   └── ...
├── test/
│   └── (same structure)
└── metadata/
    ├── train_metadata.csv
    ├── test_metadata.csv
    └── complete_metadata.csv
```

### 3. Evaluate Pre-trained Models

```bash
# Evaluate NAFNet (our best model)
python evaluation/eval_nafnet.py

# Evaluate other models
python evaluation/eval_mimo_unet.py
python evaluation/eval_hinet.py
python evaluation/eval_restormer.py
python evaluation/eval_instructir.py
```

**Output:** Results saved to `{model}_results.csv` with per-image PSNR/SSIM/LPIPS.

### 4. Train Your Own Model

```bash
# NAFNet (50 epochs, recommended for testing)
python finetuning/train_nafnet.py --epochs 50 --batch_size 4

# Full training configs from paper
python finetuning/train_nafnet.py --epochs 350 --batch_size 4     # NAFNet
python finetuning/train_mimo_unet.py --epochs 200 --batch_size 4  # MIMO-UNet
python finetuning/train_hinet.py --epochs 300 --batch_size 2      # HINet
python finetuning/train_restormer.py --epochs 250 --batch_size 2  # Restormer
python finetuning/train_instructir.py --epochs 200 --batch_size 2 # InstructIR
```

**Training options:**
```bash
--epochs          Number of training epochs
--batch_size      Batch size (reduce if OOM)
--lr              Learning rate (default: 1e-4)
--pretrained      Path to GoPro pretrained weights (optional)
--checkpoint_dir  Where to save checkpoints
--data_dir        Dataset location (auto-detected by default)
```

---

## 📂 Repository Structure

```
iphoneblur-benchmark/
├── dataset/                    # Dataset utilities
│   ├── download_dataset.py     # Kaggle dataset downloader
│   ├── generate_iphoneblur.py  # Dataset synthesis script
│   ├── dataloader.py           # PyTorch dataloader
│   └── __init__.py
├── models/                     # Model architectures
│   ├── nafnet.py               # NAFNet (67.89M params)
│   ├── mimo_unet.py            # MIMO-UNet (16.1M params)
│   ├── hinet.py                # HINet (88.7M params)
│   ├── restormer.py            # Restormer (26.1M params)
│   ├── instructir.py           # InstructIR (63.6M params)
│   ├── download_weights.py     # HuggingFace weight downloader
│   └── __init__.py
├── evaluation/                 # Evaluation scripts
│   ├── eval_nafnet.py          # Evaluate NAFNet
│   ├── eval_mimo_unet.py       # Evaluate MIMO-UNet
│   ├── eval_hinet.py           # Evaluate HINet
│   ├── eval_restormer.py       # Evaluate Restormer
│   └── eval_instructir.py      # Evaluate InstructIR
├── finetuning/                 # Training scripts
│   ├── train_nafnet.py         # Train NAFNet (50 epochs)
│   ├── train_mimo_unet.py      # Train MIMO-UNet (50 epochs)
│   ├── train_hinet.py          # Train HINet (50 epochs)
│   ├── train_restormer.py      # Train Restormer (30 epochs)
│   └── train_instructir.py     # Train InstructIR (30 epochs)
├── Inferred_Notebooks/         # Kaggle training notebooks
│   ├── nafnet_finetuning.ipynb
│   ├── mimo_unet_finetuning.ipynb
│   └── ...
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📊 Benchmark Results

### Overall Performance (Test Set: 1,686 images)

| Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Parameters | Runtime (ms/img) |
|-------|-------------|--------|---------|------------|------------------|
| **HINet** | **27.12 ± 3.86** | **0.9195** | **0.0561** | 88.7M | 45 |
| **NAFNet** | 26.75 ± 3.94 | 0.9167 | 0.0583 | 67.89M | 38 |
| **Restormer** | 26.89 ± 3.78 | 0.9178 | 0.0574 | 26.1M | 42 |
| **MIMO-UNet** | 25.45 ± 3.21 | 0.8984 | 0.0698 | 16.1M | 28 |
| **InstructIR** | 25.23 ± 3.45 | 0.8967 | 0.0712 | 63.6M | 52 |

### Stratified by Difficulty

| Model | Easy (280) | Medium (953) | Hard (453) |
|-------|-----------|--------------|------------|
| **HINet** | 33.87 ± 1.76 | 26.78 ± 1.82 | 23.89 ± 2.09 |
| **NAFNet** | 33.51 ± 1.82 | 26.42 ± 1.89 | 23.45 ± 2.14 |
| **Restormer** | 33.69 ± 1.79 | 26.56 ± 1.85 | 23.67 ± 2.11 |
| **MIMO-UNet** | 31.98 ± 1.54 | 24.89 ± 1.62 | 22.12 ± 1.89 |
| **InstructIR** | 31.76 ± 1.61 | 24.67 ± 1.71 | 21.98 ± 1.95 |

*All models fine-tuned on iPhoneBlur training set. Runtime measured on NVIDIA RTX 3090.*

---

## 🔬 Dataset Synthesis

Our synthesis pipeline ensures photorealistic blur while maintaining ISP-processed appearance:

### Key Innovation: Gamma-Corrected Temporal Averaging

```python
# Traditional (incorrect): Direct averaging in sRGB space
blur_wrong = np.mean(frames_srgb, axis=0)  # ❌

# Our method: Gamma-corrected averaging
linear_frames = [gamma_decode(f, gamma=2.2) for f in frames_srgb]
blur_correct = gamma_encode(np.mean(linear_frames, axis=0), gamma=2.2)  # ✅
```

**Why this matters:**
- Traditional averaging → 3.2 dB PSNR loss vs. real blur
- Our method → 0.4 dB PSNR loss (8× improvement)
- Preserves ISP characteristics (tone curves, noise, sharpening)

### Synthesis Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Window size | 3-21 frames | Adaptive blur kernel |
| Target PSNR | 21-32.5 dB | Controlled difficulty |
| Frame rate | 60 FPS | Temporal resolution |
| Gamma | 2.2 | sRGB standard |

**Generate custom dataset:**
```bash
python dataset/generate_iphoneblur.py \
    --input_dir /path/to/iphone_videos \
    --output_dir ./data/iphoneblur \
    --min_window 3 \
    --max_window 21 \
    --psnr_floor 21.0 \
    --psnr_ceiling 32.5
```


---

## 🔗 Related Work

- **GoPro Dataset** ([Nah et al., CVPR 2017](https://seungjunnah.github.io/Datasets/gopro.html)) - Synthetic blur from GoPro videos
- **HIDE Dataset** ([Shen et al., ICCV 2019](https://github.com/joanshen0508/HA_deblur)) - Human-aware deblurring
- **RealBlur Dataset** ([Rim et al., ECCV 2020](http://cg.postech.ac.kr/research/realblur/)) - Camera-captured real blur
- **NAFNet** ([Chen et al., ECCV 2022](https://github.com/megvii-research/NAFNet)) - Nonlinear Activation Free Network

**Our contribution:** First iPhone-specific blur dataset with ISP-validated synthesis quality.


---

## 📧 Contact

**Abdullah Al Shafi**  
Department of Computer Science and Engineering  
Khulna University of Engineering & Technology (KUET)  
Email: [abdullah.shafi@example.com](mailto:abdullah.shafi@example.com)

**Supervisor:** Professor Kazi Saeed Alam  
Email: [kazi.alam@kuet.ac.bd](mailto:kazi.alam@kuet.ac.bd)

---

## 📄 License

This project is licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ⚠️ Attribution required

See [LICENSE](LICENSE) for details.


---

**Last Updated:** May 2026  
**Status:** Under NeurIPS 2026 Review  
**Repository:** [github.com/yourusername/iphoneblur-benchmark](https://github.com/yourusername/iphoneblur-benchmark)
