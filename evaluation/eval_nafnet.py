"""
Evaluate NAFNet on iPhoneBlur test set
Fixed: CUDA detection + live metrics in progress bar

Usage: python evaluation/eval_nafnet.py
"""

import sys
import subprocess
from pathlib import Path

# Auto-install packages
def check_and_install_packages():
    required = {
        'torch': 'torch',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'skimage': 'scikit-image',
        'lpips': 'lpips',
        'huggingface_hub': 'huggingface_hub'
    }
    
    missing = []
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("=" * 70)
        print("📦 AUTO-INSTALLING PACKAGES")
        print("=" * 70)
        for pkg in missing:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
            print(f"✅ {pkg}")
        print()

check_and_install_packages()

import argparse
import torch
import torch.nn as nn
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
# NAFNet Architecture
# ============================================================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()
        self.norm1 = nn.LayerNorm(c, eps=1e-6)
        self.norm2 = nn.LayerNorm(c, eps=1e-6)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv3(self.sg(self.conv2(self.conv1(x))) * self.sca(self.sg(self.conv2(self.conv1(x)))))
        y = inp + x * self.beta
        x = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv5(self.sg(self.conv4(x)))
        return y + x * self.gamma

class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, inp):
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x); encs.append(x); x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x); x = x + enc_skip; x = decoder(x)
        return self.ending(x) + inp

# ============================================================================
# Helper Functions
# ============================================================================
def find_dataset():
    possible_paths = [
        Path('./data/iphoneblur'),
        Path('./data/iPhoneBlur'),
        Path('./dataset/data/iphoneblur'),
        Path('./dataset/data/iPhoneBlur'),
        Path('../data/iphoneblur'),
        Path('../dataset/data/iphoneblur'),
    ]
    for path in possible_paths:
        if path.exists() and (path / 'test').exists():
            return path
    return None

def download_weights_auto():
    from huggingface_hub import hf_hub_download
    REPO_ID = "Shafi99/iPhoneBlur_Finetune_Models"
    filename = 'nafnet_final_weights.pth'
    
    print(f"📥 Downloading NAFNet weights from HuggingFace...")
    print(f"   Repository: {REPO_ID}")
    print(f"   File: {filename}")
    
    try:
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=filename, force_download=False)
        import os
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"✅ Downloaded ({size_mb:.1f} MB)")
        print(f"   Location: {weights_path}\n")
        return weights_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate NAFNet')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--save_csv', type=str, default='nafnet_results.csv')
    args = parser.parse_args()
    
    # Find dataset
    print("=" * 70)
    print("🔍 LOCATING DATASET")
    print("=" * 70)
    
    if args.data_dir is None:
        data_dir = find_dataset()
        if data_dir is None:
            print("❌ Dataset not found!")
            sys.exit(1)
        print(f"✅ Found: {data_dir.absolute()}")
    else:
        data_dir = Path(args.data_dir)
        print(f"✅ Using: {data_dir.absolute()}")
    
    test_dir = data_dir / 'test'
    metadata_path = data_dir / 'metadata' / 'test_metadata.csv'
    
    # Find images
    blur_files = sorted(list(test_dir.rglob('blur/*.jpg')))
    if not blur_files:
        blur_files = sorted(list(test_dir.rglob('blur/*.JPG')))
    if not blur_files:
        blur_files = sorted(list(test_dir.rglob('blur/*.png')))
    
    if not blur_files:
        print(f"❌ No images in {test_dir}")
        sys.exit(1)
    
    print(f"✅ Found {len(blur_files)} test images")
    
    # Load metadata
    has_metadata = metadata_path.exists()
    if has_metadata:
        metadata = pd.read_csv(metadata_path)
        print(f"✅ Metadata: {len(metadata)} samples")
        if 'difficulty' in metadata.columns:
            for diff in ['Easy', 'Medium', 'Hard']:
                count = len(metadata[metadata['difficulty'] == diff])
                if count > 0:
                    print(f"   {diff}: {count}")
    else:
        print("⚠️  No metadata")
        metadata = None
    
    # Setup device - IMPROVED CUDA DETECTION
    print(f"\n{'=' * 70}")
    print("⚙️  DEVICE SETUP")
    print("=" * 70)
    
    # Check PyTorch CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = args.device
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("\n⚠️  CUDA requested but not available!")
            print("Possible reasons:")
            print("  1. PyTorch installed without CUDA support")
            print("  2. No NVIDIA GPU detected")
            print("  3. CUDA drivers not installed")
            print(f"\nTo fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("\nFalling back to CPU (will be VERY slow)...")
            device = 'cpu'
        else:
            print(f"\n✅ Using CUDA: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
    else:
        print("✅ Using CPU (slow)")
    
    # Load model
    print(f"\n{'=' * 70}")
    print("🔧 LOADING NAFNET")
    print("=" * 70)
    
    model = NAFNet(width=64, enc_blk_nums=[1, 1, 1, 28])
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ NAFNet built: {params:.2f}M parameters")
    
    # Download weights
    weights_path = download_weights_auto()
    
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'params' in checkpoint:
                model.load_state_dict(checkpoint['params'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Weights loaded!")
    else:
        print(f"⚠️  Using random weights")
    
    model = model.to(device)
    model.eval()
    
    # Setup metrics
    print(f"\n{'=' * 70}")
    print("📊 METRICS")
    print("=" * 70)
    
    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net='alex').eval().to(device)
            print("✅ LPIPS ready")
        except:
            print("⚠️  LPIPS failed")
    else:
        print("⚠️  LPIPS not available")
    
    print("✅ PSNR, SSIM ready")
    
    # Evaluate with LIVE METRICS in progress bar
    print(f"\n{'=' * 70}")
    print(f"🎯 EVALUATING {len(blur_files)} IMAGES")
    print("=" * 70)
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
            
            # Load
            blur_img = cv2.imread(str(blur_path))
            sharp_img = cv2.imread(str(sharp_path))
            
            if blur_img is None or sharp_img is None:
                continue
            
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            h, w, _ = blur_img.shape
            
            # Pad
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            blur_padded = np.pad(blur_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect') if (pad_h or pad_w) else blur_img
            
            # Inference
            blur_t = torch.from_numpy(blur_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            output = model(blur_t.to(device))
            
            # Process
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if pad_h or pad_w:
                output_np = output_np[:h, :w, :]
            output_np = np.clip(output_np, 0, 1)
            
            # Metrics
            psnr_val = calculate_psnr(output_np, sharp_img, data_range=1.0)
            ssim_val = calculate_ssim(output_np, sharp_img, data_range=1.0, channel_axis=2)
            
            lpips_val = None
            if lpips_model:
                out_t = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
                sharp_t = torch.from_numpy(sharp_img).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
                lpips_val = lpips_model(out_t.to(device), sharp_t.to(device)).item()
            
            # Update running averages
            running_psnr += psnr_val
            running_ssim += ssim_val
            if lpips_val:
                running_lpips += lpips_val
            count += 1
            
            # Update progress bar with LIVE metrics (like notebook)
            pbar.set_postfix({
                'PSNR': f"{running_psnr/count:.2f}",
                'SSIM': f"{running_ssim/count:.4f}",
                'LPIPS': f"{running_lpips/count:.4f}" if lpips_model else "N/A"
            })
            
            results.append({
                'img_id': blur_path.stem,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'lpips': lpips_val if lpips_val else np.nan
            })
    
    if not results:
        print("❌ No results!")
        sys.exit(1)
    
    # Results
    results_df = pd.DataFrame(results)
    
    if has_metadata and 'difficulty' in metadata.columns:
        results_df = pd.merge(results_df, metadata[['img_id', 'difficulty']], on='img_id', how='left')
    
    print(f"\n{'=' * 70}")
    print("📊 RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Overall ({len(results_df)} images):")
    print(f"   PSNR:  {results_df['psnr'].mean():.2f} ± {results_df['psnr'].std():.2f} dB")
    print(f"   SSIM:  {results_df['ssim'].mean():.4f} ± {results_df['ssim'].std():.4f}")
    if lpips_model:
        print(f"   LPIPS: {results_df['lpips'].mean():.4f} ± {results_df['lpips'].std():.4f}")
    
    if 'difficulty' in results_df.columns:
        print(f"\n📈 By Difficulty:")
        print(f"{'Level':<10} {'N':<8} {'PSNR':<18} {'SSIM':<14} {'LPIPS'}")
        print("-" * 60)
        for diff in ['Easy', 'Medium', 'Hard']:
            sub = results_df[results_df['difficulty'] == diff]
            if len(sub) > 0:
                lpips_str = f"{sub['lpips'].mean():.4f}" if lpips_model else "N/A"
                print(f"{diff:<10} {len(sub):<8} {sub['psnr'].mean():5.2f} ± {sub['psnr'].std():4.2f}   {sub['ssim'].mean():.4f}        {lpips_str}")
    
    results_df.to_csv(args.save_csv, index=False)
    print(f"\n💾 Saved: {Path(args.save_csv).absolute()}")
    print(f"\n{'=' * 70}")
    print("✅ DONE!")
    print("=" * 70)

if __name__ == '__main__':
    main()