"""
Model architectures and weight loading for iPhoneBlur benchmark.

Available models:
    - NAFNet: Simple baseline architecture
    - MIMOUNet: Multi-scale input/output UNet
    - HINet: Half Instance Normalization Network
    - Restormer: Efficient transformer
    - InstructIR: Instruction-based restoration

Usage:
    from models import NAFNet, download_weights
    
    # Load model
    model = NAFNet(width=64, enc_blk_nums=[1,1,1,28])
    
    # Download and load weights
    weights_path = download_weights('nafnet')
    model.load_state_dict(torch.load(weights_path))
"""

from .nafnet import NAFNet
from .mimo_unet import MIMOUNet
from .hinet import build_hinet
from .restormer import build_restormer
from .instructir import build_instructir
from .download_weights import download_weights, WEIGHT_FILES

__all__ = [
    'NAFNet',
    'MIMOUNet', 
    'build_hinet',
    'build_restormer',
    'build_instructir',
    'download_weights',
    'WEIGHT_FILES'
]
