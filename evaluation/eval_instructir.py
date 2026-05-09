"""
Evaluate InstructIR on iPhoneBlur - STANDALONE
Everything in one file, no models/ directory needed
"""
import sys, subprocess, os, gc
from pathlib import Path

# Check packages
def check_packages():
    pkgs = {'torch':'torch', 'cv2':'opencv-python', 'numpy':'numpy', 'pandas':'pandas', 
            'tqdm':'tqdm', 'skimage':'scikit-image', 'lpips':'lpips', 
            'huggingface_hub':'huggingface_hub', 'yaml':'pyyaml', 'transformers':'transformers',
            'timm':'timm', 'einops':'einops'}
    for name, pkg in pkgs.items():
        try: __import__(name)
        except: subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

check_packages()

import argparse, torch, cv2, numpy as np, pandas as pd, yaml
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

try:
    import lpips
    LPIPS_OK = True
except:
    LPIPS_OK = False

# Config helper
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

# Setup InstructIR
def setup_instructir(device='cuda'):
    print("⚙️ Setting up InstructIR...")
    
    # Clone InstructIR
    if not os.path.exists('InstructIR'):
        print("🔄 Cloning InstructIR repository...")
        subprocess.run(['git', 'clone', '-q', 'https://github.com/mv-lab/InstructIR.git', 'InstructIR'])
    
    # Add to path
    sys.path.insert(0, os.path.abspath('InstructIR'))
    
    # Hotfix
    import torchvision.transforms.functional as TF
    sys.modules['torchvision.transforms.functional_tensor'] = TF
    
    print("   ✅ Repo ready")
    
    # Import from InstructIR repo
    from models.instructir import create_model as create_instructir
    from text.models import LanguageModel, LMHead
    
    print("   ✅ Modules imported")
    
    # Load config
    with open('InstructIR/configs/eval5d.yml', 'r') as f:
        cfg = dict2namespace(yaml.safe_load(f))
    
    print("   ✅ Config loaded")
    
    # Build language model
    lm = LanguageModel(model="TaylorAI/bge-micro-v2")
    lm.eval()
    for param in lm.parameters():
        param.requires_grad = False
    
    lm_head = LMHead(
        embedding_dim=cfg.llm.model_dim,
        hidden_dim=cfg.llm.embd_dim,
        num_classes=cfg.llm.nclasses
    ).to(device)
    
    print("   ✅ Language models ready")
    
    # Build image model
    im = create_instructir(
        input_channels=cfg.model.in_ch,
        width=cfg.model.width,
        enc_blks=cfg.model.enc_blks,
        middle_blk_num=cfg.model.middle_blk_num,
        dec_blks=cfg.model.dec_blks,
        txtdim=cfg.model.textdim
    ).to(device)
    
    params = sum(p.numel() for p in im.parameters()) / 1e6
    print(f"   ✅ Image model ready: {params:.2f}M parameters")
    
    return im, lm_head, lm

def find_data():
    for p in [Path('./dataset/data/iphoneblur'), Path('./data/iphoneblur'), Path('../dataset/data/iphoneblur')]:
        if p.exists() and (p/'test').exists(): return p
    return None

def download_weights():
    from huggingface_hub import hf_hub_download
    print("\n📥 Downloading InstructIR weights...")
    try:
        w = hf_hub_download(repo_id="Shafi99/iPhoneBlur_Finetune_Models", 
                           filename='instructir_final_weights.pth', force_download=False)
        print(f"✅ Downloaded\n")
        return w
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_csv', default='instructir_results.csv')
    args = parser.parse_args()
    
    print("="*70); print("🔍 DATASET"); print("="*70)
    data = find_data() if not args.data_dir else Path(args.data_dir)
    if not data: 
        print("❌ Dataset not found!")
        input("\nPress Enter to exit...")
        sys.exit(1)
    print(f"✅ {data.absolute()}")
    
    test_dir = data / 'test'
    meta_path = data / 'metadata' / 'test_metadata.csv'
    imgs = sorted(list(test_dir.rglob('blur/*.jpg')))
    if not imgs:
        print("❌ No images!")
        input("\nPress Enter to exit...")
        sys.exit(1)
    print(f"✅ {len(imgs)} images")
    
    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    if meta is not None: print(f"✅ Metadata: {len(meta)}")
    
    print(f"\n{'='*70}"); print("⚙️  DEVICE"); print("="*70)
    dev = args.device
    if dev=='cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        dev = 'cpu'
    
    if dev=='cuda': 
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    else: 
        print("✅ Using CPU")
    
    print(f"\n{'='*70}"); print("🔧 LOADING InstructIR"); print("="*70)
    
    try:
        im, lm_head, lm = setup_instructir(dev)
        
        # Create text embedding
        PROMPT = "deblur"
        with torch.no_grad():
            lm_embd = lm(PROMPT)
            text_embd, _ = lm_head(lm_embd.to(dev))
        
        print("✅ Text embedding ready")
        
        # Model function
        def model_fn(x):
            return im(x, text_embd)
        
    except Exception as e:
        print(f"❌ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Download and load weights
    wpath = download_weights()
    if wpath:
        try:
            ckpt = torch.load(wpath, map_location=dev)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('model_state_dict', ckpt.get('params', ckpt))
            else:
                state_dict = ckpt
            im.load_state_dict(state_dict)
            print("✅ Weights loaded!")
        except Exception as e:
            print(f"⚠️  Weight loading failed: {e}")
    
    im = im.to(dev).eval()
    
    print(f"\n{'='*70}"); print("📊 METRICS"); print("="*70)
    lpips_fn = lpips.LPIPS(net='alex').eval().to(dev) if LPIPS_OK else None
    if lpips_fn: print("✅ LPIPS")
    print("✅ PSNR, SSIM")
    
    print(f"\n{'='*70}"); print(f"🎯 EVALUATING {len(imgs)} IMAGES"); print("="*70)
    if dev == 'cpu':
        print(f"⏱️  Estimated time: ~2-3 hours on CPU")
    print()
    
    results = []
    r_psnr, r_ssim, r_lpips, cnt = 0.0, 0.0, 0.0, 0
    error_count = 0
    
    try:
        with torch.no_grad():
            pbar = tqdm(imgs, desc="Evaluating", ncols=100)
            
            for bp in pbar:
                try:
                    sp = Path(str(bp).replace('\\blur\\','\\sharp\\').replace('/blur/','/sharp/'))
                    if not sp.exists(): continue
                    
                    blur = cv2.imread(str(bp))
                    sharp = cv2.imread(str(sp))
                    
                    if blur is None or sharp is None:
                        error_count += 1
                        continue
                    
                    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
                    sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                    h, w, _ = blur.shape
                    
                    # Padding (16-pixel for InstructIR)
                    pad_h = (16-h%16)%16
                    pad_w = (16-w%16)%16
                    blur_pad = np.pad(blur, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect') if (pad_h or pad_w) else blur
                    
                    # Inference
                    bt = torch.from_numpy(blur_pad).float().permute(2,0,1).unsqueeze(0)/255.0
                    out = model_fn(bt.to(dev))
                    
                    # Process output
                    on = out.squeeze(0).permute(1,2,0).cpu().numpy()
                    if pad_h or pad_w: on = on[:h,:w,:]
                    on = np.clip(on, 0, 1)
                    
                    # Metrics
                    pv = calculate_psnr(on, sharp, data_range=1.0)
                    sv = calculate_ssim(on, sharp, data_range=1.0, channel_axis=2, win_size=11)
                    
                    lv = None
                    if lpips_fn:
                        ot = torch.from_numpy(on).permute(2,0,1).unsqueeze(0).float()*2.0-1.0
                        st = torch.from_numpy(sharp).permute(2,0,1).unsqueeze(0).float()*2.0-1.0
                        lv = lpips_fn(ot.to(dev), st.to(dev)).item()
                    
                    r_psnr += pv
                    r_ssim += sv
                    if lv: r_lpips += lv
                    cnt += 1
                    
                    pbar.set_postfix({
                        'PSNR': f"{r_psnr/cnt:.2f}",
                        'SSIM': f"{r_ssim/cnt:.4f}",
                        'LPIPS': f"{r_lpips/cnt:.4f}" if lpips_fn else "N/A"
                    })
                    
                    results.append({
                        'img_id': bp.stem,
                        'psnr': pv,
                        'ssim': sv,
                        'lpips': lv if lv else np.nan
                    })
                    
                    # Cleanup
                    del bt, out, on
                    if dev == 'cuda':
                        torch.cuda.empty_cache()
                
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    error_count += 1
                    continue
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted!")
        if len(results) > 0:
            print(f"Saving {len(results)} partial results...")
        else:
            input("\nPress Enter to exit...")
            sys.exit(0)
    
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    if not results:
        print("\n❌ No valid results!")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print(f"\n✅ Processed {len(results)} images successfully")
    if error_count > 0:
        print(f"⚠️  Skipped {error_count} images due to errors")
    
    df = pd.DataFrame(results)
    if meta is not None and 'difficulty' in meta.columns:
        df = pd.merge(df, meta[['img_id','difficulty']], on='img_id', how='left')
    
    print(f"\n{'='*70}"); print("📊 RESULTS"); print("="*70)
    print(f"\n🎯 Overall ({len(df)} images):")
    print(f"   PSNR:  {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
    print(f"   SSIM:  {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
    if lpips_fn: 
        print(f"   LPIPS: {df['lpips'].mean():.4f} ± {df['lpips'].std():.4f}")
    
    if 'difficulty' in df.columns:
        print(f"\n📈 By Difficulty:")
        print(f"{'Level':<10} {'N':<8} {'PSNR':<18} {'SSIM'}")
        print("-"*50)
        for d in ['Easy','Medium','Hard']:
            sub = df[df['difficulty']==d]
            if len(sub)>0:
                print(f"{d:<10} {len(sub):<8} {sub['psnr'].mean():5.2f} ± {sub['psnr'].std():4.2f}   {sub['ssim'].mean():.4f}")
    
    df.to_csv(args.save_csv, index=False)
    print(f"\n💾 Saved: {Path(args.save_csv).absolute()}")
    print("\n"+"="*70); print("✅ EVALUATION COMPLETE!"); print("="*70)
    
    input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()
