"""
MIMO-UNet Architecture - Multi-scale Input Multi-scale Output
~16.1M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class MIMOUNet(nn.Module):
    """
    MIMO-UNet: Multi-scale Input Multi-scale Output UNet
    Default: ~16.1M parameters
    """
    def __init__(self, num_res=8):
        super().__init__()
        
        # Scale 1 (Full)
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc1_blocks = nn.Sequential(*[BasicBlock(32, 32) for _ in range(num_res)])
        
        # Scale 2 (Half)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc2_in = nn.Conv2d(3+64, 64, 1)
        self.enc2_blocks = nn.Sequential(*[BasicBlock(64, 64) for _ in range(num_res)])
        
        # Scale 3 (Quarter)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3_in = nn.Conv2d(3+128, 128, 1)
        self.enc3_blocks = nn.Sequential(*[BasicBlock(128, 128) for _ in range(num_res)])

        # Decoder
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

        if self.training:
            return [out_full, out_half]
        else:
            return out_full


if __name__ == '__main__':
    model = MIMOUNet(num_res=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    x = torch.randn(1, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print("✅ MIMOUNet works!")
