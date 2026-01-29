"""Training script for 3D Shape Completion."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pointnet_plus_plus import PointNetPlusPlus
from src.layers.losses import CombinedLoss, ChamferLoss, EMDLoss, F1Score
from src.data.dataset import create_data_loaders
from src.utils.device import get_device, set_seed, save_checkpoint, load_checkpoint, EarlyStopping


class Trainer:
    """Trainer class for 3D Shape Completion."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = get_device()
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Create directories
        self._create_directories()
        
        # Initialize model, loss, and optimizer
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        
        # Initialize data loaders
        self._setup_data()
        
        # Initialize metrics
        self._setup_metrics()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.logging.get("patience", 10),
            min_delta=1e-6,
            restore_best_weights=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _create_directories(self) -> None:
        """Create necessary directories."""
        dirs = [
            self.config.paths.checkpoint_dir,
            self.config.paths.log_dir,
            self.config.paths.asset_dir,
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_model(self) -> None:
        """Setup model."""
        self.model = PointNetPlusPlus(
            num_classes=self.config.model.num_classes,
            normal_channel=self.config.model.normal_channel,
            num_points=self.config.data.num_points,
            dropout=self.config.model.dropout,
        ).to(self.device)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _setup_loss(self) -> None:
        """Setup loss function."""
        self.criterion = CombinedLoss(
            chamfer_weight=self.config.loss.chamfer_weight,
            emd_weight=self.config.loss.emd_weight,
            l2_weight=self.config.loss.l2_weight,
        )
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        if self.config.training.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=1e-6
            )
        elif self.config.training.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def _setup_data(self) -> None:
        """Setup data loaders."""
        data_config = {
            "train_samples": self.config.data.train_samples,
            "val_samples": self.config.data.val_samples,
            "num_points": self.config.data.num_points,
            "completion_ratio": self.config.data.completion_ratio,
            "noise_std": self.config.data.noise_std,
        }
        
        self.train_loader, self.val_loader = create_data_loaders(
            data_config,
            batch_size=self.config.training.batch_size,
            num_workers=4,
        )
    
    def _setup_metrics(self) -> None:
        """Setup evaluation metrics."""
        self.metrics = {
            "chamfer_distance": ChamferLoss(),
            "emd": EMDLoss(),
            "f1_score": F1Score(threshold=self.config.evaluation.f1_threshold),
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {"chamfer_loss": 0.0, "emd_loss": 0.0, "l2_loss": 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            partial_points = batch["partial"].to(self.device)
            complete_points = batch["complete"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_points = self.model(partial_points)
            
            # Compute loss
            loss, loss_dict = self.criterion(predicted_points, complete_points)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key in loss_components:
                    loss_components[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "chamfer": f"{loss_dict['chamfer_loss']:.6f}",
                "emd": f"{loss_dict['emd_loss']:.6f}",
            })
        
        # Average metrics
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return {
            "train_loss": avg_loss,
            **loss_components,
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {"chamfer_loss": 0.0, "emd_loss": 0.0, "l2_loss": 0.0}
        metric_values = {name: 0.0 for name in self.metrics.keys()}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                partial_points = batch["partial"].to(self.device)
                complete_points = batch["complete"].to(self.device)
                
                # Forward pass
                predicted_points = self.model(partial_points)
                
                # Compute loss
                loss, loss_dict = self.criterion(predicted_points, complete_points)
                
                # Update metrics
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key in loss_components:
                        loss_components[key] += value
                
                # Compute additional metrics
                for name, metric in self.metrics.items():
                    metric_values[name] += metric(predicted_points, complete_points).item()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "chamfer": f"{loss_dict['chamfer_loss']:.6f}",
                })
        
        # Average metrics
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        for key in metric_values:
            metric_values[key] /= len(self.val_loader)
        
        return {
            "val_loss": avg_loss,
            **loss_components,
            **metric_values,
        }
    
    def train(self) -> None:
        """Main training loop."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"Val Chamfer: {val_metrics['chamfer_loss']:.6f}")
            print(f"Val EMD: {val_metrics['emd_loss']:.6f}")
            print(f"Val F1: {val_metrics['f1_score']:.6f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint if best
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                checkpoint_path = os.path.join(
                    self.config.paths.checkpoint_dir,
                    "best_model.pth"
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['val_loss'],
                    checkpoint_path,
                    self.config
                )
                print(f"New best model saved to {checkpoint_path}")
            
            # Early stopping
            if self.early_stopping(val_metrics['val_loss'], self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print("Training completed!")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        epoch, loss = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.device
        )
        self.current_epoch = epoch
        self.best_val_loss = loss
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if specified
    if config.get("checkpoint_path"):
        trainer.load_checkpoint(config.checkpoint_path)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
