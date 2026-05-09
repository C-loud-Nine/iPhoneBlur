"""
Evaluate MIMO-UNet on iPhoneBlur test set
Based on working NAFNet evaluation pattern

Usage: python evaluation/eval_mimo_unet.py
"""

import sys
import subprocess
from pathlib import Path

def check_and_install_packages():
    required = {'torch': 'torch', 'cv2': 'opencv-python', 'numpy': 'numpy', 
                'pandas': 'pandas', 'tqdm': 'tqdm', 'skimage': 'scikit-image', 
                'lpips': 'lpips', 'huggingface_hub': 'huggingface_hub'}
    missing = [pkg for name, pkg in required.items() if not __import__(name, fromlist=['']) or True]
    if missing:
        print("="*70)
        print("📦 INSTALLING PACKAGES")
        print("="*70)
        for pkg in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
                print(f"✅ {pkg}")
            except:
                pass

check_and_install_packages()

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

try:
    import lpips
    LPIPS_AVAILABLE = True
except:
    LPIPS_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# MIMO-UNet Architecture
# ============================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc1_blocks = nn.Sequential(*[BasicBlock(32, 32) for _ in range(num_res)])
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc2_in = nn.Conv2d(3+64, 64, 1)
        self.enc2_blocks = nn.Sequential(*[BasicBlock(64, 64) for _ in range(num_res)])
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3_in = nn.Conv2d(3+128, 128, 1)
        self.enc3_blocks = nn.Sequential(*[BasicBlock(128, 128) for _ in range(num_res)])
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2_blocks = nn.Sequential(*[BasicBlock(64, 64) for _ in range(num_res)])
        self.out2 = nn.Conv2d(64, 3, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1_blocks = nn.Sequential(*[BasicBlock(32, 32) for _ in range(num_res)])
        self.out1 = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_quarter = F.interpolate(x_half, scale_factor=0.5, mode='bilinear', align_corners=False)
        e1 = self.enc1_blocks(self.enc1(x))
        e2_pre = self.down1(e1)
        e2_cat = torch.cat([e2_pre, x_half], dim=1)
        e2 = self.enc2_blocks(self.enc2_in(e2_cat))
        e3_pre = self.down2(e2)
        e3_cat = torch.cat([e3_pre, x_quarter], dim=1)
        e3 = self.enc3_blocks(self.enc3_in(e3_cat))
        d2 = self.dec2_blocks(self.up2(e3) + e2)
        out_half = self.out2(d2)
        d1 = self.dec1_blocks(self.up1(d2) + e1)
        out_full = self.out1(d1)
        return out_full if not self.training else [out_full, out_half]

# ============================================================================
# Helper Functions
# ============================================================================
def find_dataset():
    for path in [Path('./data/iphoneblur'), Path('./dataset/data/iphoneblur'), Path('../dataset/data/iphoneblur')]:
        if path.exists() and (path / 'test').exists():
            return path
    return None

def download_weights():
    from huggingface_hub import hf_hub_download
    print(f"📥 Downloading MIMO-UNet weights from HuggingFace...")
    try:
        weights_path = hf_hub_download(repo_id="Shafi99/iPhoneBlur_Finetune_Models", filename='mimo_final_weights.pth', force_download=False)
        print(f"✅ Downloaded: {weights_path}\n")
        return weights_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_csv', type=str, default='mimo_results.csv')
    args = parser.parse_args()
    
    # Find dataset
    print("="*70)
    print("🔍 DATASET")
    print("="*70)
    data_dir = find_dataset() if args.data_dir is None else Path(args.data_dir)
    if data_dir is None:
        print("❌ Dataset not found!")
        sys.exit(1)
    print(f"✅ Found: {data_dir.absolute()}")
    
    test_dir = data_dir / 'test'
    metadata_path = data_dir / 'metadata' / 'test_metadata.csv'
    blur_files = sorted(list(test_dir.rglob('blur/*.jpg')))
    if not blur_files:
        print("❌ No images!")
        sys.exit(1)
    print(f"✅ {len(blur_files)} images")
    
    has_metadata = metadata_path.exists()
    if has_metadata:
        metadata = pd.read_csv(metadata_path)
        print(f"✅ Metadata: {len(metadata)} samples")
    else:
        metadata = None
    
    # Device
    print(f"\n{'='*70}")
    print("⚙️  DEVICE")
    print("="*70)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    if device == 'cuda':
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("✅ CPU")
    
    # Model
    print(f"\n{'='*70}")
    print("🔧 LOADING MIMO-UNET")
    print("="*70)
    model = MIMOUNet(num_res=8)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Built: {params:.2f}M parameters")
    
    weights_path = download_weights()
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('params', checkpoint)))
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Weights loaded!")
    else:
        print(f"⚠️  Random weights")
    
    model = model.to(device).eval()
    
    # Metrics
    print(f"\n{'='*70}")
    print("📊 METRICS")
    print("="*70)
    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net='alex').eval().to(device)
            print("✅ LPIPS ready")
        except:
            pass
    print("✅ PSNR, SSIM ready")
    
    # Evaluate
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATING {len(blur_files)} IMAGES")
    print("="*70)
    print()
    
    results = []
    running_psnr, running_ssim, running_lpips = 0.0, 0.0, 0.0
    count = 0
    
    with torch.no_grad():
        pbar = tqdm(blur_files, desc="Evaluating", ncols=100)
        for blur_path in pbar:
            sharp_path = Path(str(blur_path).replace('\\blur\\', '\\sharp\\').replace('/blur/', '/sharp/'))
            if not sharp_path.exists():
                continue
            
            blur_img = cv2.cvtColor(cv2.imread(str(blur_path)), cv2.COLOR_BGR2RGB)
            sharp_img = cv2.cvtColor(cv2.imread(str(sharp_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            h, w, _ = blur_img.shape
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            blur_padded = np.pad(blur_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect') if (pad_h or pad_w) else blur_img
            
            blur_t = torch.from_numpy(blur_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            output = model(blur_t.to(device))
            
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if pad_h or pad_w:
                output_np = output_np[:h, :w, :]
            output_np = np.clip(output_np, 0, 1)
            
            psnr_val = calculate_psnr(output_np, sharp_img, data_range=1.0)
            ssim_val = calculate_ssim(output_np, sharp_img, data_range=1.0, channel_axis=2)
            
            lpips_val = None
            if lpips_model:
                out_t = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
                sharp_t = torch.from_numpy(sharp_img).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
                lpips_val = lpips_model(out_t.to(device), sharp_t.to(device)).item()
            
            running_psnr += psnr_val
            running_ssim += ssim_val
            if lpips_val:
                running_lpips += lpips_val
            count += 1
            
            pbar.set_postfix({'PSNR': f"{running_psnr/count:.2f}", 'SSIM': f"{running_ssim/count:.4f}", 'LPIPS': f"{running_lpips/count:.4f}" if lpips_model else "N/A"})
            
            results.append({'img_id': blur_path.stem, 'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val if lpips_val else np.nan})
    
    results_df = pd.DataFrame(results)
    if has_metadata and 'difficulty' in metadata.columns:
        results_df = pd.merge(results_df, metadata[['img_id', 'difficulty']], on='img_id', how='left')
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print("="*70)
    print(f"\n🎯 Overall ({len(results_df)} images):")
    print(f"   PSNR:  {results_df['psnr'].mean():.2f} ± {results_df['psnr'].std():.2f} dB")
    print(f"   SSIM:  {results_df['ssim'].mean():.4f} ± {results_df['ssim'].std():.4f}")
    if lpips_model:
        print(f"   LPIPS: {results_df['lpips'].mean():.4f} ± {results_df['lpips'].std():.4f}")
    
    if 'difficulty' in results_df.columns:
        print(f"\n📈 By Difficulty:")
        for diff in ['Easy', 'Medium', 'Hard']:
            sub = results_df[results_df['difficulty'] == diff]
            if len(sub) > 0:
                print(f"{diff:<10} {len(sub):<8} {sub['psnr'].mean():5.2f} ± {sub['psnr'].std():4.2f}   {sub['ssim'].mean():.4f}")
    
    results_df.to_csv(args.save_csv, index=False)
    print(f"\n💾 Saved: {args.save_csv}")
    print(f"\n{'='*70}")
    print("✅ DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
