"""
Restormer Architecture - Efficient Transformer for Image Restoration
Uses Restormer repo (github.com/swz30/Restormer)

NOTE: Clones Restormer repo and imports architecture from it.
"""

import os
import sys
import subprocess

def build_restormer(device='cuda'):
    """
    Build Restormer model.
    Clones Restormer repo and imports architecture.
    
    Args:
        device: 'cuda' or 'cpu'
    
    Returns:
        Restormer model instance
    """
    print("⚙️ Setting up Restormer architecture...")
    
    # Step 1: Clone Restormer repo if not exists
    if not os.path.exists('Restormer'):
        print("   Cloning Restormer repository...")
        result = os.system('git clone -q https://github.com/swz30/Restormer.git')
        if result != 0:
            print("   ❌ Git clone failed!")
            raise RuntimeError("Failed to clone Restormer repository")
        print("   ✅ Cloned successfully")
    
    # Step 2: Add Restormer to Python path
    repo_path = os.path.abspath('Restormer')
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    
    # Step 3: Ensure basicsr is installed
    try:
        import basicsr
    except ImportError:
        print("   Installing basicsr package...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'basicsr', '-q'])
        print("   ✅ basicsr installed")
    
    # Step 4: Import Restormer architecture from the cloned repo
    try:
        from basicsr.models.archs.restormer_arch import Restormer
        print("   ✅ Architecture imported")
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        
        # Fallback: Try direct file import
        import importlib.util
        arch_file = os.path.join(repo_path, 'basicsr', 'models', 'archs', 'restormer_arch.py')
        
        if not os.path.exists(arch_file):
            raise RuntimeError(f"restormer_arch.py not found at {arch_file}")
        
        print("   Attempting direct file import...")
        spec = importlib.util.spec_from_file_location("restormer_arch", arch_file)
        restormer_module = importlib.util.module_from_spec(spec)
        sys.modules['restormer_arch'] = restormer_module
        spec.loader.exec_module(restormer_module)
        Restormer = restormer_module.Restormer
        print("   ✅ Loaded via direct import")
    
    # Step 5: Build model with exact notebook parameters
    import torch
    model = Restormer(
        inp_channels=3, 
        out_channels=3, 
        dim=48, 
        num_blocks=[4, 6, 6, 8], 
        num_refinement_blocks=4, 
        heads=[1, 2, 4, 8], 
        ffn_expansion_factor=2.66, 
        bias=False, 
        LayerNorm_type='WithBias', 
        dual_pixel_task=False
    ).to(device)
    
    print(f"   ✅ Model built on {device}")
    
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing Restormer build...")
    model = build_restormer('cpu')
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n✅ Restormer built successfully!")
    print(f"   Parameters: {params:.2f}M")
    
    # Test forward pass
    import torch
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print("\n✅ Forward pass successful!")
