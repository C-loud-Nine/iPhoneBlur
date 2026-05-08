# iPhoneBlur: A Difficulty-Stratified Benchmark for Consumer Device Motion Deblurring

[![arXiv](https://img.shields.io/badge/arXiv-2605.05990-B31B1B.svg)](https://arxiv.org/abs/2605.05990)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)](https://kaggle.com/datasets/shafi09/iphoneblur)
[![Models](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official implementation and benchmark for **iPhoneBlur**, a motion blur dataset containing **7,400 blur-sharp image pairs** extracted from 51 iPhone 17 Pro videos, designed to evaluate image deblurring models under real-world consumer camera ISP conditions.

**Paper:** [arXiv:2605.05990](https://arxiv.org/abs/2605.05990)

---

## 🎯 Key Features

- **7,400 image pairs** (5,714 training, 1,686 test) from **iPhone 17 Pro** videos (177–240 fps)
- **Photorealistic blur synthesis** via gamma-corrected temporal averaging with PSNR-guided adaptive windowing
- **Stratified difficulty labels**: Easy (PSNR ≥30 dB), Medium (24–30 dB), Hard (<24 dB) – validated by optical flow correlation (ρ = -0.41) and user study
- **Rich per-sample metadata**: optical flow, sharpness, contrast, ISP energy, synthesis parameters
- **Benchmark results**: 6 state-of-the-art models fine-tuned on iPhoneBlur, exposing a **7–9 dB Easy-to-Hard performance drop** hidden by aggregate metrics

---

## 📊 Dataset Statistics

| Split | Images | Easy | Medium | Hard | Avg PSNR (dB) |
|-------|--------|------|--------|------|---------------|
| Train | 5,714 | 997 | 2,942 | 1,775 | 26.2 ± 3.4 |
| Test | 1,686 | 280 | 953 | 453 | 26.2 ± 3.4 |
| **Total** | **7,400** | **1,277** | **3,895** | **2,228** | **26.2 ± 3.4** |

*Statistics from Table 1 of the paper.*

---

## 📥 Downloads

### Dataset

| Source | Size | Format | Link |
|--------|------|--------|------|
| **Kaggle** | 9.08 GB | JPG + CSV metadata | [Download](https://kaggle.com/datasets/shafi09/iphoneblur) |

Quick download:

```bash
pip install kaggle
python dataset/download_dataset.py
```

### Fine-tuned Models (Primary Benchmark)

All models fine-tuned on iPhoneBlur training set (5,714 pairs) and evaluated on the test set. Results match **Table 4** of the paper.

| Model | Parameters | PSNR (dB) | SSIM | LPIPS | Download |
|-------|-------------|------------|------|--------|-----------|
| **NAFNet** | 67.9M | **31.2** | 0.945 | 0.049 | [Link](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/nafnet_final_weights.pth) |
| **FFTformer** | 24.7M | **31.0** | 0.943 | 0.050 | [Link](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/fftformer_final_weights.pth) |
| **Restormer** | 26.1M | **30.9** | 0.940 | 0.053 | [Link](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/restormer_final_weights.pth) |
| **HINet** | 26.5M | **30.5** | 0.938 | 0.056 | [Link](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/hinet_final_weights.pth) |
| **InstructIR** | 22.3M | **29.5** | 0.924 | 0.071 | [Link](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/instructir_final_weights.pth) |
| **MIMO-UNet** | 16.8M | **29.3** | 0.920 | 0.075 | [Link](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models/resolve/main/mimo_final_weights.pth) |

**All models:** [HuggingFace Repository](https://huggingface.co/Shafi99/iPhoneBlur_Finetune_Models)

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/C-loud-Nine/iPhoneBlur.git
cd iPhoneBlur
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python dataset/download_dataset.py
```

### 3. Evaluate a Fine-tuned Model

```bash
python evaluation/eval_nafnet.py
```

### 4. Train Your Own Model (from scratch on iPhoneBlur)

```bash
# NAFNet – 60 epochs (as in paper)
python finetuning/train_nafnet.py --epochs 60 --batch_size 8

# Restormer – 30 epochs
python finetuning/train_restormer.py --epochs 30 --batch_size 4

# HINet – 60 epochs
python finetuning/train_hinet.py --epochs 60 --batch_size 4

# MIMO-UNet – 60 epochs
python finetuning/train_mimo_unet.py --epochs 60 --batch_size 4

# InstructIR – 50 epochs
python finetuning/train_instructir.py --epochs 50 --batch_size 2

# FFTformer – 30 epochs
python finetuning/train_fftformer.py --epochs 30 --batch_size 4
```

*All hyperparameters match the supplementary material of the paper.*

---

## 📂 Repository Structure

```text
iPhoneBlur/
├── dataset/                 # download scripts, dataloader, synthesis code
├── models/                  # architectures: NAFNet, HINet, Restormer, MIMO-UNet, InstructIR, FFTformer
├── evaluation/              # evaluation scripts for each model
├── finetuning/              # training scripts with paper hyperparameters
├── Inferred_Notebooks/      # Kaggle notebooks (optional)
├── requirements.txt
└── README.md
```

---

## 📊 Benchmark Results (Fine-tuned on iPhoneBlur)

### Overall Performance (Test Set)

| Model | PSNR (dB) | SSIM | LPIPS |
|-------|------------|------|--------|
| **NAFNet** | **31.2** | 0.945 | 0.049 |
| **FFTformer** | 31.0 | 0.943 | 0.050 |
| **Restormer** | 30.9 | 0.940 | 0.053 |
| **HINet** | 30.5 | 0.938 | 0.056 |
| **InstructIR** | 29.5 | 0.924 | 0.071 |
| **MIMO-UNet** | 29.3 | 0.920 | 0.075 |

### Stratified by Difficulty

| Model | Easy (280) | Medium (953) | Hard (453) |
|-------|--------------|---------------|-------------|
| **NAFNet** | 35.1 | 31.6 | 27.9 |
| **FFTformer** | 35.0 | 31.3 | 27.7 |
| **Restormer** | 34.9 | 31.2 | 27.5 |
| **HINet** | 34.6 | 30.9 | 27.1 |
| **InstructIR** | 34.5 | 29.9 | 25.4 |
| **MIMO-UNet** | 33.9 | 29.7 | 25.7 |

*All values from Table 4 of the paper. The 7–9 dB Easy-to-Hard drop is consistent across architectures.*

---

## 🔬 Dataset Synthesis (Brief)

Our gamma-corrected adaptive temporal averaging preserves the iPhone ISP pipeline:

```python
linear_frames = [gamma_decode(f, gamma=2.2) for f in frames_srgb]
blur = gamma_encode(np.mean(linear_frames, axis=0), gamma=2.2)
```

Window size (3–21 frames) adapts to target PSNR, controlling difficulty. See the paper (Section 3.2) for full details and Algorithm 1.

### Metadata Generation

Each sample includes optical flow (Farnebäck), Laplacian sharpness, RMS contrast, ISP high-frequency energy, and synthesis parameters. Scripts for metadata extraction are provided in `dataset/generate_metadata.py`.

---

## 🔗 Related Work

- **GoPro** (Nah et al., CVPR 2017) – 240 fps action camera, 3,214 pairs
- **HIDE** (Shen et al., ICCV 2019) – human motion, 2,025 test pairs
- **RealBlur** (Rim et al., ECCV 2020) – beam-splitter real blur, 4,738 pairs
- **REDS** (Nah et al., CVPRW 2019) – 240 video sequences

**iPhoneBlur** is the first difficulty-stratified benchmark for consumer-device motion deblurring.

---

## 📧 Contact

**Abdullah Al Shafi** – [abdullah.shafi@example.com](mailto:abdullah.shafi@example.com)  
**Supervisor:** Prof. Kazi Saeed Alam – [kazi.alam@kuet.ac.bd](mailto:kazi.alam@kuet.ac.bd)  
Department of Computer Science & Engineering, KUET, Bangladesh

---

## 📄 License

- **Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [MIT License](https://opensource.org/licenses/MIT)

---

## 📖 Citation

If you use iPhoneBlur in your research, please cite our paper:

```bibtex
@misc{shafi2026iphoneblur,
  title={iPhoneBlur: A Difficulty-Stratified Benchmark for Consumer Device Motion Deblurring},
  author={Abdullah Al Shafi and Kazi Saeed Alam},
  year={2026},
  eprint={2605.05990},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.05990}
}
```

---

**Last Updated:** May 2026  
**Repository:** https://github.com/C-loud-Nine/iPhoneBlur
