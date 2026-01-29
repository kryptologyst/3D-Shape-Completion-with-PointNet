"""Device management and utility functions for 3D Shape Completion."""

import os
import random
from typing import Optional, Union, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    config: Optional[DictConfig] = None,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss value.
        filepath: Path to save checkpoint.
        config: Optional configuration to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if config is not None:
        checkpoint["config"] = config
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model.
        optimizer: Optional optimizer.
        device: Device to load checkpoint on.
        
    Returns:
        Tuple of (epoch, loss).
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"], checkpoint["loss"]


def create_directories(config: DictConfig) -> None:
    """Create necessary directories for the project.
    
    Args:
        config: Configuration object.
    """
    dirs_to_create = [
        config.data_dir,
        config.checkpoint_dir,
        config.log_dir,
        config.asset_dir,
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: PyTorch model.
            
        Returns:
            bool: True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()
