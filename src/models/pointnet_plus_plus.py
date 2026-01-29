"""PointNet++ implementation for 3D shape completion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction Layer."""
    
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
    ):
        """Initialize PointNet++ Set Abstraction Layer.
        
        Args:
            npoint: Number of points to sample.
            radius: Radius for ball query.
            nsample: Number of points to sample in each ball.
            in_channel: Input channel dimension.
            mlp: MLP layer dimensions.
            group_all: Whether to group all points.
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            xyz: Input point coordinates (B, N, 3).
            points: Input point features (B, N, C).
            
        Returns:
            Tuple of (new_xyz, new_points).
        """
        B, N, C = xyz.shape
        S = self.npoint
        
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, C)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
            else:
                grouped_points = grouped_xyz
        else:
            # Farthest Point Sampling
            new_xyz = self.farthest_point_sample(xyz, self.npoint)
            # Ball Query
            grouped_xyz, grouped_points = self.ball_query(self.radius, self.nsample, xyz, new_xyz, points)
        
        # PointNet
        if points is not None:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        else:
            grouped_points = grouped_xyz
            
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        
        new_points = torch.max(grouped_points, 2)[0]  # (B, C, npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, C)
        
        return new_xyz, new_points

    def farthest_point_sample(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """Farthest Point Sampling.
        
        Args:
            xyz: Point coordinates (B, N, 3).
            npoint: Number of points to sample.
            
        Returns:
            Sampled point coordinates (B, npoint, 3).
        """
        B, N, C = xyz.shape
        device = xyz.device
        
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        
        return xyz[batch_indices, centroids, :]

    def ball_query(self, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ball Query.
        
        Args:
            radius: Ball radius.
            nsample: Number of points to sample.
            xyz: Original point coordinates (B, N, 3).
            new_xyz: Query point coordinates (B, S, 3).
            points: Original point features (B, N, C).
            
        Returns:
            Tuple of (grouped_xyz, grouped_points).
        """
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        device = xyz.device
        
        # Calculate distances
        dists = torch.cdist(new_xyz, xyz)  # (B, S, N)
        
        # Find points within radius
        idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)
        for b in range(B):
            for s in range(S):
                valid_mask = dists[b, s] <= radius
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) >= nsample:
                    # Randomly sample nsample points
                    selected = torch.randperm(len(valid_indices), device=device)[:nsample]
                    idx[b, s] = valid_indices[selected]
                else:
                    # Pad with closest points
                    sorted_indices = torch.argsort(dists[b, s])
                    idx[b, s, :len(valid_indices)] = sorted_indices[:len(valid_indices)]
                    idx[b, s, len(valid_indices):] = sorted_indices[:nsample-len(valid_indices)]
        
        # Group points
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        grouped_xyz = xyz[batch_indices[:, None, None], idx, :]  # (B, S, nsample, 3)
        grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, C)  # Relative coordinates
        
        if points is not None:
            grouped_points = points[batch_indices[:, None, None], idx, :]  # (B, S, nsample, C)
        else:
            grouped_points = grouped_xyz
            
        return grouped_xyz, grouped_points


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation Layer."""
    
    def __init__(self, in_channel: int, mlp: List[int]):
        """Initialize Feature Propagation Layer.
        
        Args:
            in_channel: Input channel dimension.
            mlp: MLP layer dimensions.
        """
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            xyz1: Query point coordinates (B, N, 3).
            xyz2: Reference point coordinates (B, M, 3).
            points1: Query point features (B, N, C1).
            points2: Reference point features (B, M, C2).
            
        Returns:
            Propagated features (B, N, C_out).
        """
        B, N, C = xyz1.shape
        _, M, _ = xyz2.shape
        
        if M == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Find k nearest neighbors
            dists = torch.cdist(xyz1, xyz2)  # (B, N, M)
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False)  # (B, N, 3)
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            # Interpolate features
            batch_indices = torch.arange(B, dtype=torch.long, device=xyz1.device)
            interpolated_points = torch.sum(
                weight[:, :, :, None] * points2[batch_indices[:, None, None], idx, :], 
                dim=2
            )
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
            
        new_points = new_points.permute(0, 2, 1)  # (B, C, N)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points.permute(0, 2, 1)  # (B, N, C)


class PointNetPlusPlus(nn.Module):
    """PointNet++ for 3D Shape Completion."""
    
    def __init__(
        self,
        num_classes: int = 3,
        normal_channel: bool = False,
        num_points: int = 2048,
        dropout: float = 0.5,
    ):
        """Initialize PointNet++.
        
        Args:
            num_classes: Number of output classes (3 for xyz coordinates).
            normal_channel: Whether to use normal channels.
            num_points: Number of input points.
            dropout: Dropout rate.
        """
        super(PointNetPlusPlus, self).__init__()
        self.num_points = num_points
        in_channel = 6 if normal_channel else 3
        
        # Encoder
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=in_channel, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        
        # Decoder
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])
        
        # Output head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            xyz: Input point cloud (B, N, 3).
            
        Returns:
            Completed point cloud (B, N, 3).
        """
        B, N, C = xyz.shape
        
        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        
        # Output
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.conv3(x)
        
        return x.permute(0, 2, 1)  # (B, N, 3)
