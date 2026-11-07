"""
Edge-Preserving and Boundary-Aware Loss Functions for Depth Super-Resolution.
Combines depth photometric reconstruction, multi-scale gradient matching,
and structural similarity to penalize boundary blurring and texture-copying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthGradientLoss(nn.Module):
    """Computes L1 loss on horizontal and vertical spatial depth gradients."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Spatial differences
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        diff_x = torch.abs(pred_dx - target_dx)
        diff_y = torch.abs(pred_dy - target_dy)
        
        if mask is not None:
            mask_x = mask[:, :, :, :-1] * mask[:, :, :, 1:]
            mask_y = mask[:, :, :-1, :] * mask[:, :, 1:, :]
            loss = (diff_x * mask_x).sum() / (mask_x.sum() + 1e-6) + \
                   (diff_y * mask_y).sum() / (mask_y.sum() + 1e-6)
        else:
            loss = diff_x.mean() + diff_y.mean()
            
        return loss


class CharbonnierLoss(nn.Module):
    """Robust smooth L1 loss: sqrt(x^2 + eps^2)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


class EdgePreservingLoss(nn.Module):
    """
    Combined objective for boundary-preserving depth upsampling:
    L = w_depth * L_depth + w_grad * L_grad + w_bce * L_boundary
    """
    def __init__(self, w_depth: float = 1.0, w_grad: float = 0.5, w_bce: float = 0.2):
        super().__init__()
        self.w_depth = w_depth
        self.w_grad = w_grad
        self.w_bce = w_bce
        
        self.charbonnier = CharbonnierLoss()
        self.grad_loss = DepthGradientLoss()
        
    def forward(self, pred_depth: torch.Tensor, 
                pred_boundary: torch.Tensor, 
                target_depth: torch.Tensor, 
                gt_boundary: torch.Tensor = None, 
                mask: torch.Tensor = None) -> dict:
        
        l_depth = self.charbonnier(pred_depth, target_depth, mask)
        l_grad  = self.grad_loss(pred_depth, target_depth, mask)
        
        total = self.w_depth * l_depth + self.w_grad * l_grad
        
        l_bce = pred_depth.new_zeros(())
        if gt_boundary is not None and pred_boundary is not None:
            bce = F.binary_cross_entropy(
                pred_boundary.clamp(1e-6, 1.0 - 1e-6), gt_boundary, reduction='none'
            )
            if mask is not None:
                l_bce = (bce * mask).sum() / (mask.sum() + 1e-6)
            else:
                l_bce = bce.mean()
            total = total + self.w_bce * l_bce
            
        return {
            'total_loss': total,
            'depth_loss': l_depth,
            'grad_loss': l_grad,
            'boundary_loss': l_bce
        }
