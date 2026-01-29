"""Unit tests for 3D Shape Completion."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.pointnet_plus_plus import PointNetPlusPlus
from src.layers.losses import ChamferLoss, EMDLoss, F1Score, CombinedLoss
from src.data.dataset import SyntheticShapeDataset, PointCloudTransform
from src.utils.device import get_device, set_seed


class TestPointNetPlusPlus:
    """Test PointNet++ model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = PointNetPlusPlus(
            num_classes=3,
            normal_channel=False,
            num_points=1024,
            dropout=0.5
        )
        
        assert model.num_points == 1024
        assert model.conv3.out_channels == 3
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = PointNetPlusPlus(num_points=1024)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        num_points = 1024
        input_tensor = torch.randn(batch_size, num_points, 3)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, num_points, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_gradient_flow(self):
        """Test gradient flow."""
        model = PointNetPlusPlus(num_points=1024)
        model.train()
        
        input_tensor = torch.randn(1, 1024, 3, requires_grad=True)
        output = model(input_tensor)
        
        # Compute dummy loss
        loss = torch.mean(output)
        loss.backward()
        
        # Check gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestLossFunctions:
    """Test loss functions."""
    
    def test_chamfer_loss(self):
        """Test Chamfer loss."""
        loss_fn = ChamferLoss()
        
        pred = torch.randn(2, 100, 3)
        target = torch.randn(2, 100, 3)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_emd_loss(self):
        """Test EMD loss."""
        loss_fn = EMDLoss()
        
        pred = torch.randn(2, 100, 3)
        target = torch.randn(2, 100, 3)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_f1_score(self):
        """Test F1 score."""
        f1_fn = F1Score(threshold=0.01)
        
        pred = torch.randn(2, 100, 3)
        target = torch.randn(2, 100, 3)
        
        f1 = f1_fn(pred, target)
        
        assert 0 <= f1.item() <= 1
        assert not torch.isnan(f1)
        assert not torch.isinf(f1)
    
    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = CombinedLoss(
            chamfer_weight=1.0,
            emd_weight=0.1,
            l2_weight=0.1
        )
        
        pred = torch.randn(2, 100, 3)
        target = torch.randn(2, 100, 3)
        
        loss, loss_dict = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Check loss dictionary
        assert "total_loss" in loss_dict
        assert "chamfer_loss" in loss_dict
        assert "emd_loss" in loss_dict
        assert "l2_loss" in loss_dict


class TestDataset:
    """Test dataset functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset."""
        dataset = SyntheticShapeDataset(
            num_samples=10,
            num_points=512,
            completion_ratio=0.7,
            noise_std=0.01
        )
        
        assert len(dataset) == 10
        
        # Test getting an item
        item = dataset[0]
        
        assert "partial" in item
        assert "complete" in item
        assert item["partial"].shape == (int(512 * 0.7), 3)
        assert item["complete"].shape == (512, 3)
    
    def test_point_cloud_transform(self):
        """Test point cloud transform."""
        transform = PointCloudTransform(
            rotation_range=(0, 2 * np.pi),
            translation_range=(-0.1, 0.1),
            scale_range=(0.9, 1.1),
            jitter_std=0.01
        )
        
        points = torch.randn(100, 3)
        transformed = transform(points)
        
        assert transformed.shape == points.shape
        assert not torch.isnan(transformed).any()
        assert not torch.isinf(transformed).any()


class TestUtils:
    """Test utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again and generate again
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


class TestIntegration:
    """Integration tests."""
    
    def test_training_step(self):
        """Test a single training step."""
        model = PointNetPlusPlus(num_points=512)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = CombinedLoss()
        
        # Create dummy data
        partial_points = torch.randn(2, 512, 3)
        complete_points = torch.randn(2, 512, 3)
        
        # Forward pass
        optimizer.zero_grad()
        predicted_points = model(partial_points)
        loss, loss_dict = criterion(predicted_points, complete_points)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_evaluation_step(self):
        """Test a single evaluation step."""
        model = PointNetPlusPlus(num_points=512)
        model.eval()
        
        criterion = ChamferLoss()
        
        # Create dummy data
        partial_points = torch.randn(1, 512, 3)
        complete_points = torch.randn(1, 512, 3)
        
        with torch.no_grad():
            predicted_points = model(partial_points)
            loss = criterion(predicted_points, complete_points)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    pytest.main([__file__])
