"""
InstructIR - Fixed import collision
"""
import os
import sys
import subprocess
import yaml
import argparse
import torch

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def build_instructir(device='cuda'):
    print("⚙️ Setting up InstructIR...")
    
    # Clone
    if not os.path.exists('InstructIR'):
        print("🔄 Cloning InstructIR repository...")
        subprocess.run(['git', 'clone', '-q', 'https://github.com/mv-lab/InstructIR.git', 'InstructIR'])
    
    # Save current sys.path
    original_path = sys.path.copy()
    
    # Temporarily set sys.path to ONLY InstructIR (avoid import collision)
    instructir_path = os.path.abspath('InstructIR')
    sys.path = [instructir_path] + [p for p in sys.path if 'gitrepo' not in p]
    
    # Hotfix
    import torchvision.transforms.functional as TF
    sys.modules['torchvision.transforms.functional_tensor'] = TF
    
    print("   ✅ Repo ready")
    
    # Import from InstructIR repo (now ONLY InstructIR is in path)
    from models.instructir import create_model as create_instructir
    from text.models import LanguageModel, LMHead
    
    # Restore original path
    sys.path = original_path
    
    print("   ✅ Modules imported")
    
    # Config
    with open('InstructIR/configs/eval5d.yml', 'r') as f:
        cfg = dict2namespace(yaml.safe_load(f))
    
    print("   ✅ Config loaded")
    
    # Build
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
