"""
HINet - Half Instance Normalization Network
Uses basicsr and requires git clone
"""
import os, sys, torch

def build_hinet(device='cuda'):
    print("⚙️ Setting up HINet...")
    if not os.path.exists('HINet'):
        os.system('git clone -q https://github.com/megvii-model/HINet.git')
    sys.path.insert(0, os.path.abspath('HINet'))
    
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
    
    from basicsr.models.archs.hinet_arch import HINet
    model = HINet(wf=64, hin_position_left=3, hin_position_right=4).to(device)
    return model

if __name__ == '__main__':
    model = build_hinet('cpu')
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"HINet: {params:.2f}M parameters")
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print("✅ HINet works!")
