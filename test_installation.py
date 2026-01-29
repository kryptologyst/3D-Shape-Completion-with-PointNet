#!/usr/bin/env python3
"""Test script to verify installation and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.models import PointNetPlusPlus
        from src.layers import ChamferLoss, EMDLoss, CombinedLoss
        from src.data import SyntheticShapeDataset, PointCloudTransform
        from src.utils import get_device, set_seed
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    try:
        from src.models import PointNetPlusPlus
        
        model = PointNetPlusPlus(num_points=512)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation."""
    try:
        from src.data import SyntheticShapeDataset
        
        dataset = SyntheticShapeDataset(num_samples=5, num_points=512)
        item = dataset[0]
        
        assert "partial" in item
        assert "complete" in item
        print(f"✓ Dataset created with {len(dataset)} samples")
        return True
    except Exception as e:
        print(f"✗ Dataset creation error: {e}")
        return False

def test_device_detection():
    """Test device detection."""
    try:
        from src.utils import get_device
        
        device = get_device()
        print(f"✓ Device detected: {device}")
        return True
    except Exception as e:
        print(f"✗ Device detection error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing 3D Shape Completion Installation")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_model_creation,
        test_dataset_creation,
        test_device_detection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Installation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
