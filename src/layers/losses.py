"""Loss functions for 3D shape completion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Chamfer Distance between two point clouds.
    
    Args:
        pred: Predicted point cloud (B, N, 3).
        target: Target point cloud (B, M, 3).
        
    Returns:
        Chamfer distance.
    """
    # Compute pairwise distances
    dist = torch.cdist(pred, target)  # (B, N, M)
    
    # Chamfer distance: sum of min distances
    dist1 = torch.min(dist, dim=2)[0]  # (B, N)
    dist2 = torch.min(dist, dim=1)[0]  # (B, M)
    
    chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
    return chamfer_dist


def earth_mover_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Earth Mover's Distance (EMD) between two point clouds.
    
    Args:
        pred: Predicted point cloud (B, N, 3).
        target: Target point cloud (B, M, 3).
        
    Returns:
        EMD distance.
    """
    B, N, _ = pred.shape
    _, M, _ = target.shape
    
    # Compute pairwise distances
    dist = torch.cdist(pred, target)  # (B, N, M)
    
    # Use Hungarian algorithm approximation
    # For simplicity, we'll use a greedy assignment
    emd_dist = 0.0
    for b in range(B):
        # Greedy assignment
        remaining_indices = torch.arange(M, device=pred.device)
        total_cost = 0.0
        
        for i in range(N):
            if len(remaining_indices) == 0:
                break
            # Find closest remaining target point
            min_idx = torch.argmin(dist[b, i, remaining_indices])
            target_idx = remaining_indices[min_idx]
            total_cost += dist[b, i, target_idx]
            remaining_indices = remaining_indices[remaining_indices != target_idx]
        
        emd_dist += total_cost / N
    
    return emd_dist / B


class ChamferLoss(nn.Module):
    """Chamfer Distance Loss."""
    
    def __init__(self, reduction: str = "mean"):
        """Initialize Chamfer Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super(ChamferLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pred: Predicted point cloud (B, N, 3).
            target: Target point cloud (B, M, 3).
            
        Returns:
            Chamfer loss.
        """
        loss = chamfer_distance(pred, target)
        
        if self.reduction == "mean":
            return loss
        elif self.reduction == "sum":
            return loss * pred.shape[0]
        else:
            return loss


class EMDLoss(nn.Module):
    """Earth Mover's Distance Loss."""
    
    def __init__(self, reduction: str = "mean"):
        """Initialize EMD Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super(EMDLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pred: Predicted point cloud (B, N, 3).
            target: Target point cloud (B, M, 3).
            
        Returns:
            EMD loss.
        """
        loss = earth_mover_distance(pred, target)
        
        if self.reduction == "mean":
            return loss
        elif self.reduction == "sum":
            return loss * pred.shape[0]
        else:
            return loss


class CombinedLoss(nn.Module):
    """Combined loss function for 3D shape completion."""
    
    def __init__(
        self,
        chamfer_weight: float = 1.0,
        emd_weight: float = 0.1,
        l2_weight: float = 0.1,
        reduction: str = "mean",
    ):
        """Initialize Combined Loss.
        
        Args:
            chamfer_weight: Weight for Chamfer distance loss.
            emd_weight: Weight for EMD loss.
            l2_weight: Weight for L2 loss.
            reduction: Reduction method.
        """
        super(CombinedLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.emd_weight = emd_weight
        self.l2_weight = l2_weight
        
        self.chamfer_loss = ChamferLoss(reduction=reduction)
        self.emd_loss = EMDLoss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass.
        
        Args:
            pred: Predicted point cloud (B, N, 3).
            target: Target point cloud (B, M, 3).
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        chamfer_loss = self.chamfer_loss(pred, target)
        emd_loss = self.emd_loss(pred, target)
        
        # For L2 loss, we need to match the number of points
        if pred.shape[1] == target.shape[1]:
            l2_loss = self.l2_loss(pred, target)
        else:
            # Interpolate target to match pred size
            target_interp = F.interpolate(
                target.permute(0, 2, 1), 
                size=pred.shape[1], 
                mode='linear', 
                align_corners=False
            ).permute(0, 2, 1)
            l2_loss = self.l2_loss(pred, target_interp)
        
        total_loss = (
            self.chamfer_weight * chamfer_loss +
            self.emd_weight * emd_loss +
            self.l2_weight * l2_loss
        )
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "chamfer_loss": chamfer_loss.item(),
            "emd_loss": emd_loss.item(),
            "l2_loss": l2_loss.item(),
        }
        
        return total_loss, loss_dict


class F1Score(nn.Module):
    """F1 Score for point cloud completion."""
    
    def __init__(self, threshold: float = 0.01):
        """Initialize F1 Score.
        
        Args:
            threshold: Distance threshold for considering a point as matched.
        """
        super(F1Score, self).__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pred: Predicted point cloud (B, N, 3).
            target: Target point cloud (B, M, 3).
            
        Returns:
            F1 score.
        """
        # Compute pairwise distances
        dist = torch.cdist(pred, target)  # (B, N, M)
        
        # Precision: fraction of predicted points close to target
        min_dist_pred = torch.min(dist, dim=2)[0]  # (B, N)
        precision = torch.mean((min_dist_pred < self.threshold).float())
        
        # Recall: fraction of target points close to prediction
        min_dist_target = torch.min(dist, dim=1)[0]  # (B, M)
        recall = torch.mean((min_dist_target < self.threshold).float())
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
