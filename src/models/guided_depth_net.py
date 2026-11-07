"""
EdgeGuideNet: Multi-Scale Guided Depth Super-Resolution and Refinement Network.
A deep neural network combining RGB guidance with depth features via Edge Guidance Modules (EGM)
and residual learning to reconstruct razor-sharp depth boundaries without texture-copying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + LeakyReLU with optional residual connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.residual = residual and (in_channels == out_channels and stride == 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.residual:
            out = out + x
        return self.act(out)


class EdgeGuidanceModule(nn.Module):
    """
    Edge Guidance Module (EGM):
    Learns spatial and channel attention between RGB edge features and depth geometry.
    Prevents texture-copying by filtering out RGB features that lack geometric depth correlation.
    """
    def __init__(self, depth_channels: int, rgb_channels: int, out_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(depth_channels + rgb_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.rgb_proj = nn.Conv2d(rgb_channels, out_channels, kernel_size=1)
        self.depth_proj = nn.Conv2d(depth_channels, out_channels, kernel_size=1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, f_depth: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        # Compute spatial edge-gating mask
        gate = self.gate_conv(torch.cat([f_depth, f_rgb], dim=1))
        # Project features
        proj_rgb = self.rgb_proj(f_rgb)
        proj_depth = self.depth_proj(f_depth)
        # Modulated fusion: RGB features only propagate where gate is high
        fused = proj_depth + gate * proj_rgb
        return self.fusion_conv(fused)


class EdgeGuideNet(nn.Module):
    """
    Complete EdgeGuideNet Architecture:
    - RGB Guidance Encoder
    - Depth & Mask Encoder
    - Multi-Scale EGM cross-fusion
    - Residual Refinement Decoder
    - Predicts high-frequency depth residual and boundary confidence map
    """
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        
        # 1. RGB Guidance Encoder (3 channels)
        self.rgb_in = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.rgb_block1 = ConvBlock(c, c)
        self.rgb_down1  = nn.Sequential(nn.Conv2d(c, 2*c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.rgb_block2 = ConvBlock(2*c, 2*c)
        self.rgb_down2  = nn.Sequential(nn.Conv2d(2*c, 4*c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.rgb_block3 = ConvBlock(4*c, 4*c)
        
        # 2. Depth Encoder (2 channels: depth_lr_upsampled, validity_mask)
        self.depth_in = nn.Sequential(
            nn.Conv2d(2, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.depth_block1 = ConvBlock(c, c)
        self.depth_down1  = nn.Sequential(nn.Conv2d(c, 2*c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.depth_block2 = ConvBlock(2*c, 2*c)
        self.depth_down2  = nn.Sequential(nn.Conv2d(2*c, 4*c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.depth_block3 = ConvBlock(4*c, 4*c)
        
        # 3. Edge Guidance Modules at each scale
        self.egm_scale3 = EdgeGuidanceModule(4*c, 4*c, 4*c)
        self.egm_scale2 = EdgeGuidanceModule(2*c, 2*c, 2*c)
        self.egm_scale1 = EdgeGuidanceModule(c, c, c)
        
        # 4. Decoder with Skip Connections
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(4*c, 2*c, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec_block2 = ConvBlock(4*c, 2*c)  # 2*c from up2 + 2*c from egm_scale2
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(2*c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec_block1 = ConvBlock(2*c, c)  # c from up1 + c from egm_scale1
        
        # 5. Output Heads: Residual Depth and Edge Confidence
        self.residual_head = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c // 2, 1, kernel_size=3, padding=1)
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, depth_lr_interp: torch.Tensor, rgb_hr: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            depth_lr_interp: (B, 1, H, W) upsampled initial depth (float32)
            rgb_hr: (B, 3, H, W) high-resolution RGB guidance [0, 1]
            mask: (B, 1, H, W) optional valid depth mask (1 = valid, 0 = missing)
            
        Returns:
            (refined_depth, boundary_confidence)
        """
        if mask is None:
            mask = (depth_lr_interp > 1e-4).float()
            
        depth_input = torch.cat([depth_lr_interp, mask], dim=1)
        
        # RGB Encoder
        f_rgb1 = self.rgb_block1(self.rgb_in(rgb_hr))
        f_rgb2 = self.rgb_block2(self.rgb_down1(f_rgb1))
        f_rgb3 = self.rgb_block3(self.rgb_down2(f_rgb2))
        
        # Depth Encoder
        f_d1 = self.depth_block1(self.depth_in(depth_input))
        f_d2 = self.depth_block2(self.depth_down1(f_d1))
        f_d3 = self.depth_block3(self.depth_down2(f_d2))
        
        # EGM Cross-Guidance
        f_fused3 = self.egm_scale3(f_d3, f_rgb3)
        f_fused2 = self.egm_scale2(f_d2, f_rgb2)
        f_fused1 = self.egm_scale1(f_d1, f_rgb1)
        
        # Decoder
        d_up2 = self.up2(f_fused3)
        d_cat2 = torch.cat([d_up2, f_fused2], dim=1)
        d_dec2 = self.dec_block2(d_cat2)
        
        d_up1 = self.up1(d_dec2)
        d_cat1 = torch.cat([d_up1, f_fused1], dim=1)
        d_dec1 = self.dec_block1(d_cat1)
        
        # Residual prediction
        residual = self.residual_head(d_dec1)
        boundary_conf = self.boundary_head(d_dec1)
        
        # Final reconstructed depth = base interpolated + learned edge residual
        reconstructed_depth = F.relu(depth_lr_interp + residual)
        
        return reconstructed_depth, boundary_conf
