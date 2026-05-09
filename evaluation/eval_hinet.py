"""
Evaluate HINet on iPhoneBlur - WORKING VERSION
Fixed: HINet returns list of outputs, need to handle it properly

Usage: python evaluation/eval_hinet.py
"""
import sys, subprocess, os, tempfile, shutil
from pathlib import Path

ORIGINAL_DIR = os.getcwd()

def fix_and_install_basicsr():
    """Install basicsr by fixing the broken setup.py"""
    print("="*70)
    print("🔧 FIXING AND INSTALLING BASICSR")
    print("="*70)
    
    try:
        import basicsr
        print("✅ basicsr already installed")
        return True
    except:
        pass
    
    saved_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    try:
        os.chdir(temp_dir)
        print("Cloning BasicSR...")
        os.system('git clone -q https://github.com/XPixelGroup/BasicSR.git')
        os.chdir('BasicSR')
        
        print("Patching setup.py...")
        with open('setup.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed = content.replace(
            "    return locals()['__version__']",
            "    return locals().get('__version__', '1.4.2')"
        )
        
        with open('setup.py', 'w', encoding='utf-8') as f:
            f.write(fixed)
        
        print("Installing basicsr...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', '.'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ basicsr installed!")
            os.chdir(saved_dir)
            return True
        else:
            print(f"⚠️  Install failed, will use repo directly")
            os.chdir(saved_dir)
            return False
    
    except Exception as e:
        print(f"⚠️  Error: {e}")
        os.chdir(saved_dir)
        return False

def check_packages():
    pkgs = ['torch', 'opencv-python', 'numpy', 'pandas', 'tqdm', 
            'scikit-image', 'lpips', 'huggingface_hub', 'lmdb', 
            'pyyaml', 'tb-nightly', 'yapf']
    
    print("="*70)
    print("📦 INSTALLING PACKAGES")
    print("="*70)
    
    for pkg in pkgs:
        try:
            pkg_import = pkg.replace('-', '_').split('[')[0]
            if pkg_import == 'opencv_python':
                pkg_import = 'cv2'
            elif pkg_import == 'scikit_image':
                pkg_import = 'skimage'
            
            __import__(pkg_import)
        except:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
                print(f"✅ {pkg}")
            except:
                print(f"⚠️  {pkg} failed")

check_packages()
basicsr_ok = fix_and_install_basicsr()

import argparse, torch, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

try:
    import lpips
    LPIPS_OK = True
except:
    LPIPS_OK = False

def setup_hinet(device='cuda'):
    print("⚙️ Setting up HINet...")
    
    os.chdir(ORIGINAL_DIR)
    
    if not os.path.exists('HINet'):
        print("Cloning HINet...")
        os.system('git clone -q https://github.com/megvii-model/HINet.git')
    
    sys.path.insert(0, os.path.abspath('HINet'))
    
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
    
    try:
        from basicsr.models.archs.hinet_arch import HINet
        print("✅ Loaded from basicsr")
        model = HINet(wf=64, hin_position_left=3, hin_position_right=4).to(device)
        return model
    except:
        pass
    
    if not os.path.exists('BasicSR'):
        print("Cloning BasicSR...")
        os.system('git clone -q https://github.com/XPixelGroup/BasicSR.git')
    
    sys.path.insert(0, os.path.abspath('BasicSR'))
    
    try:
        from basicsr.models.archs.hinet_arch import HINet
        print("✅ Loaded from BasicSR repo")
        model = HINet(wf=64, hin_position_left=3, hin_position_right=4).to(device)
        return model
    except Exception as e:
        print(f"❌ Failed to load HINet: {e}")
        sys.exit(1)

def find_data():
    os.chdir(ORIGINAL_DIR)
    
    paths = [
        Path('./dataset/data/iphoneblur'),
        Path('./dataset/data/iPhoneBlur'),
        Path('../dataset/data/iphoneblur'),
        Path('./data/iphoneblur'),
    ]
    
    for p in paths:
        if p.exists() and (p / 'test').exists():
            return p
    
    return None

def download_weights():
    from huggingface_hub import hf_hub_download
    print("📥 Downloading HINet weights...")
    try:
        w = hf_hub_download(
            repo_id="Shafi99/iPhoneBlur_Finetune_Models",
            filename='hinet_final_weights.pth',
            force_download=False
        )
        print(f"✅ Downloaded\n")
        return w
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def main():
    os.chdir(ORIGINAL_DIR)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_csv', default='hinet_results.csv')
    args = parser.parse_args()
    
    # Dataset
    print("="*70)
    print("🔍 DATASET")
    print("="*70)
    
    data = find_data() if not args.data_dir else Path(args.data_dir)
    if not data:
        print("❌ Dataset not found!")
        sys.exit(1)
    
    print(f"✅ {data.absolute()}")
    
    test_dir = data / 'test'
    meta_path = data / 'metadata' / 'test_metadata.csv'
    imgs = sorted(list(test_dir.rglob('blur/*.jpg')))
    
    if not imgs:
        print("❌ No images!")
        sys.exit(1)
    
    print(f"✅ {len(imgs)} images")
    
    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    if meta is not None:
        print(f"✅ Metadata: {len(meta)}")
    
    # Device
    print(f"\n{'='*70}")
    print("⚙️  DEVICE")
    print("="*70)
    dev = 'cpu' if args.device=='cuda' and not torch.cuda.is_available() else args.device
    if dev=='cuda':
        print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("✅ CPU")
    
    # Model
    print(f"\n{'='*70}")
    print("🔧 LOADING HINET")
    print("="*70)
    
    model = setup_hinet(dev)
    params = sum(p.numel() for p in model.parameters())/1e6
    print(f"✅ Built: {params:.2f}M parameters")
    
    wpath = download_weights()
    if wpath:
        ckpt = torch.load(wpath, map_location=dev)
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('params', ckpt)))
        else:
            model.load_state_dict(ckpt)
        print("✅ Weights loaded!")
    
    model = model.eval()
    
    # Metrics
    print(f"\n{'='*70}")
    print("📊 METRICS")
    print("="*70)
    lpips_fn = None
    if LPIPS_OK:
        try:
            lpips_fn = lpips.LPIPS(net='alex').eval().to(dev)
            print("✅ LPIPS")
        except:
            pass
    print("✅ PSNR, SSIM")
    
    # Evaluate
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATING {len(imgs)} IMAGES")
    print("="*70)
    print()
    
    results = []
    r_psnr, r_ssim, r_lpips, cnt = 0.0, 0.0, 0.0, 0
    
    with torch.no_grad():
        pbar = tqdm(imgs, desc="Evaluating", ncols=100)
        for bp in pbar:
            sp = Path(str(bp).replace('\\blur\\','\\sharp\\').replace('/blur/','/sharp/'))
            if not sp.exists():
                continue
            
            blur = cv2.cvtColor(cv2.imread(str(bp)), cv2.COLOR_BGR2RGB)
            sharp = cv2.cvtColor(cv2.imread(str(sp)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            h,w,_ = blur.shape
            
            pad_h = (16-h%16)%16
            pad_w = (16-w%16)%16
            blur_pad = np.pad(blur, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect') if (pad_h or pad_w) else blur
            
            bt = torch.from_numpy(blur_pad).float().permute(2,0,1).unsqueeze(0)/255.0
            out = model(bt.to(dev))
            
            # CRITICAL FIX: HINet returns list [output1, output2]
            # Take the first output (final reconstruction)
            if isinstance(out, list):
                out = out[0]
            
            on = out.squeeze(0).permute(1,2,0).cpu().numpy()
            if pad_h or pad_w:
                on = on[:h,:w,:]
            on = np.clip(on, 0, 1)
            
            pv = psnr(on, sharp, data_range=1.0)
            sv = ssim(on, sharp, data_range=1.0, channel_axis=2)
            
            lv = None
            if lpips_fn:
                ot = torch.from_numpy(on).permute(2,0,1).unsqueeze(0).float()*2.0-1.0
                st = torch.from_numpy(sharp).permute(2,0,1).unsqueeze(0).float()*2.0-1.0
                lv = lpips_fn(ot.to(dev), st.to(dev)).item()
            
            r_psnr += pv; r_ssim += sv
            if lv: r_lpips += lv
            cnt += 1
            
            pbar.set_postfix({'PSNR': f"{r_psnr/cnt:.2f}", 'SSIM': f"{r_ssim/cnt:.4f}", 'LPIPS': f"{r_lpips/cnt:.4f}" if lpips_fn else "N/A"})
            results.append({'img_id': bp.stem, 'psnr': pv, 'ssim': sv, 'lpips': lv if lv else np.nan})
    
    df = pd.DataFrame(results)
    if meta is not None and 'difficulty' in meta.columns:
        df = pd.merge(df, meta[['img_id','difficulty']], on='img_id', how='left')
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print("="*70)
    print(f"\n🎯 Overall ({len(df)}):")
    print(f"   PSNR:  {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
    print(f"   SSIM:  {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
    if lpips_fn:
        print(f"   LPIPS: {df['lpips'].mean():.4f} ± {df['lpips'].std():.4f}")
    
    if 'difficulty' in df.columns:
        print(f"\n📈 By Difficulty:")
        for d in ['Easy','Medium','Hard']:
            sub = df[df['difficulty']==d]
            if len(sub)>0:
                print(f"{d:<10} {len(sub):<8} {sub['psnr'].mean():5.2f} ± {sub['psnr'].std():4.2f}   {sub['ssim'].mean():.4f}")
    
    df.to_csv(args.save_csv, index=False)
    print(f"\n💾 Saved: {args.save_csv}")
    print("\n"+"="*70)
    print("✅ DONE!")
    print("="*70)

if __name__ == '__main__':
    main()
