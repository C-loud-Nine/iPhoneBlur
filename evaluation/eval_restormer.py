"""
Evaluate Restormer on iPhoneBlur
Uses: models/restormer.py (architecture) + HuggingFace weights (fine-tuned)

Usage: python evaluation/eval_restormer.py
"""
import sys
import subprocess
from pathlib import Path

# Add parent directory to path to import from models/
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_packages():
    """Install required packages"""
    print("="*70)
    print("📦 CHECKING PACKAGES")
    print("="*70)
    
    packages = {
        'torch': 'torch',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'skimage': 'scikit-image',
        'huggingface_hub': 'huggingface_hub',
        'basicsr': 'basicsr'
    }
    
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
                print(f"✅ {package_name}")
            except:
                print(f"⚠️  {package_name} failed")

check_packages()

import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Import build_restormer from models/
from models.restormer import build_restormer

def find_dataset():
    """Find iPhoneBlur dataset"""
    possible_paths = [
        Path('./dataset/data/iphoneblur'),
        Path('../dataset/data/iphoneblur'),
        Path('./data/iphoneblur'),
        Path('./dataset/data/iPhoneBlur'),
    ]
    
    for path in possible_paths:
        if path.exists() and (path / 'test').exists():
            return path
    return None

def download_weights():
    """Download fine-tuned weights from HuggingFace"""
    from huggingface_hub import hf_hub_download
    
    print("\n📥 Downloading Restormer fine-tuned weights from HuggingFace...")
    print("   Repository: Shafi99/iPhoneBlur_Finetune_Models")
    print("   File: restormer_final_weights.pth")
    
    try:
        weights_path = hf_hub_download(
            repo_id="Shafi99/iPhoneBlur_Finetune_Models",
            filename='restormer_final_weights.pth',
            force_download=False
        )
        
        import os
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"✅ Downloaded ({size_mb:.1f} MB)")
        print(f"   Location: {weights_path}\n")
        return weights_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate Restormer on iPhoneBlur')
    parser.add_argument('--data_dir', type=str, default=None, help='Dataset directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--save_csv', type=str, default='restormer_results.csv')
    args = parser.parse_args()
    
    # Find dataset
    print("="*70)
    print("🔍 LOCATING DATASET")
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
    
    print(f"✅ Found {len(blur_files)} test images")
    
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
        print("⚠️  No metadata file")
        metadata = None
    
    # Device setup
    print(f"\n{'='*70}")
    print("⚙️  DEVICE SETUP")
    print("="*70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  CUDA requested but not available!")
        print("Falling back to CPU...")
        device = 'cpu'
    
    if device == 'cuda':
        print(f"\n✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("✅ Using CPU")
    
    # Load model from models/restormer.py
    print(f"\n{'='*70}")
    print("🔧 LOADING RESTORMER")
    print("="*70)
    
    print("Loading architecture from models/restormer.py...")
    model = build_restormer(device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    param_millions = total_params / 1e6
    print(f"✅ Built: {param_millions:.2f}M parameters")
    
    # Download and load fine-tuned weights
    weights_path = download_weights()
    
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print("✅ Fine-tuned weights loaded!")
    else:
        print("⚠️  Using random weights (results will be poor)")
    
    model = model.eval()
    
    # Setup metrics
    print(f"\n{'='*70}")
    print("📊 METRICS")
    print("="*70)
    print("✅ PSNR, SSIM ready")
    
    # Evaluate
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATING {len(blur_files)} IMAGES")
    print("="*70)
    print()
    
    results = []
    running_psnr, running_ssim = 0.0, 0.0
    count = 0
    
    with torch.no_grad():
        pbar = tqdm(blur_files, desc="Evaluating", ncols=100)
        
        for blur_path in pbar:
            sharp_path = Path(str(blur_path).replace('\\blur\\', '\\sharp\\').replace('/blur/', '/sharp/'))
            
            if not sharp_path.exists():
                continue
            
            # Load images
            blur_img = cv2.imread(str(blur_path))
            sharp_img = cv2.imread(str(sharp_path))
            
            if blur_img is None or sharp_img is None:
                continue
            
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            h, w, _ = blur_img.shape
            
            # Padding (8-pixel alignment from notebook)
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            
            if pad_h or pad_w:
                blur_padded = np.pad(blur_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                blur_padded = blur_img
            
            # Inference
            blur_t = torch.from_numpy(blur_padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            output = model(blur_t.to(device))
            
            # Handle list output (some Restormer versions return list)
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Move to CPU and process
            output = output.cpu()
            output_np = output.squeeze(0).permute(1, 2, 0).numpy()
            
            # Remove padding
            if pad_h or pad_w:
                output_np = output_np[:h, :w, :]
            
            output_np = np.clip(output_np, 0, 1)
            
            # Calculate metrics
            psnr_val = psnr_metric(sharp_img, output_np, data_range=1.0)
            ssim_val = ssim_metric(sharp_img, output_np, data_range=1.0, channel_axis=2, win_size=11)
            
            running_psnr += psnr_val
            running_ssim += ssim_val
            count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'PSNR': f"{running_psnr/count:.2f}",
                'SSIM': f"{running_ssim/count:.4f}"
            })
            
            results.append({
                'img_id': blur_path.stem,
                'psnr': psnr_val,
                'ssim': ssim_val
            })
            
            # Memory cleanup
            del blur_t, output, output_np
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    if not results:
        print("❌ No valid results!")
        sys.exit(1)
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    if has_metadata and 'difficulty' in metadata.columns:
        results_df = pd.merge(results_df, metadata[['img_id', 'difficulty']], 
                             on='img_id', how='left')
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print("="*70)
    
    print(f"\n🎯 Overall ({len(results_df)} images):")
    print(f"   PSNR: {results_df['psnr'].mean():.2f} ± {results_df['psnr'].std():.2f} dB")
    print(f"   SSIM: {results_df['ssim'].mean():.4f} ± {results_df['ssim'].std():.4f}")
    
    if 'difficulty' in results_df.columns:
        print(f"\n📈 By Difficulty:")
        print(f"{'Level':<10} {'N':<8} {'PSNR':<18} {'SSIM'}")
        print("-"*50)
        
        for diff in ['Easy', 'Medium', 'Hard']:
            sub = results_df[results_df['difficulty'] == diff]
            if len(sub) > 0:
                print(f"{diff:<10} {len(sub):<8} {sub['psnr'].mean():5.2f} ± {sub['psnr'].std():4.2f}   {sub['ssim'].mean():.4f}")
    
    # Save results
    results_df.to_csv(args.save_csv, index=False)
    print(f"\n💾 Saved: {Path(args.save_csv).absolute()}")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
