"""Evaluation script for 3D Shape Completion."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pointnet_plus_plus import PointNetPlusPlus
from src.layers.losses import ChamferLoss, EMDLoss, F1Score, chamfer_distance, earth_mover_distance
from src.data.dataset import SyntheticShapeDataset, PointCloudTransform
from src.utils.device import get_device, set_seed, load_checkpoint


class Evaluator:
    """Evaluator class for 3D Shape Completion."""
    
    def __init__(self, config: DictConfig, checkpoint_path: str):
        """Initialize evaluator.
        
        Args:
            config: Configuration object.
            checkpoint_path: Path to model checkpoint.
        """
        self.config = config
        self.device = get_device()
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Load model
        self.model = PointNetPlusPlus(
            num_classes=config.model.num_classes,
            normal_channel=config.model.normal_channel,
            num_points=config.data.num_points,
            dropout=config.model.dropout,
        ).to(self.device)
        
        # Load checkpoint
        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()
        
        # Initialize metrics
        self.metrics = {
            "chamfer_distance": ChamferLoss(),
            "emd": EMDLoss(),
            "f1_score": F1Score(threshold=config.evaluation.f1_threshold),
        }
        
        # Create test dataset
        self.test_dataset = SyntheticShapeDataset(
            num_samples=config.data.test_samples,
            num_points=config.data.num_points,
            completion_ratio=config.data.completion_ratio,
            noise_std=config.data.noise_std,
            transform=PointCloudTransform(
                rotation_range=(0, 0),
                translation_range=(0, 0),
                scale_range=(1.0, 1.0),
                jitter_std=0.0,
            ),
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one at a time for visualization
            shuffle=False,
            num_workers=0,
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Returns:
            Dictionary of evaluation metrics.
        """
        print("Evaluating model...")
        
        total_metrics = {name: 0.0 for name in self.metrics.keys()}
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                partial_points = batch["partial"].to(self.device)
                complete_points = batch["complete"].to(self.device)
                
                # Forward pass
                predicted_points = self.model(partial_points)
                
                # Compute metrics
                for name, metric in self.metrics.items():
                    value = metric(predicted_points, complete_points).item()
                    total_metrics[name] += value
                
                num_samples += 1
        
        # Average metrics
        avg_metrics = {name: value / num_samples for name, value in total_metrics.items()}
        
        return avg_metrics
    
    def visualize_results(self, num_samples: int = 5, save_dir: str = "assets") -> None:
        """Visualize results for a few samples.
        
        Args:
            num_samples: Number of samples to visualize.
            save_dir: Directory to save visualizations.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Generating visualizations for {num_samples} samples...")
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i >= num_samples:
                    break
                
                partial_points = batch["partial"].to(self.device)
                complete_points = batch["complete"].to(self.device)
                
                # Forward pass
                predicted_points = self.model(partial_points)
                
                # Convert to numpy
                partial_np = partial_points[0].cpu().numpy()
                complete_np = complete_points[0].cpu().numpy()
                predicted_np = predicted_points[0].cpu().numpy()
                
                # Create point clouds
                partial_pcd = o3d.geometry.PointCloud()
                partial_pcd.points = o3d.utility.Vector3dVector(partial_np)
                partial_pcd.paint_uniform_color([1, 0, 0])  # Red
                
                complete_pcd = o3d.geometry.PointCloud()
                complete_pcd.points = o3d.utility.Vector3dVector(complete_np)
                complete_pcd.paint_uniform_color([0, 1, 0])  # Green
                
                predicted_pcd = o3d.geometry.PointCloud()
                predicted_pcd.points = o3d.utility.Vector3dVector(predicted_np)
                predicted_pcd.paint_uniform_color([0, 0, 1])  # Blue
                
                # Save individual point clouds
                o3d.io.write_point_cloud(
                    os.path.join(save_dir, f"sample_{i}_partial.ply"),
                    partial_pcd
                )
                o3d.io.write_point_cloud(
                    os.path.join(save_dir, f"sample_{i}_complete.ply"),
                    complete_pcd
                )
                o3d.io.write_point_cloud(
                    os.path.join(save_dir, f"sample_{i}_predicted.ply"),
                    predicted_pcd
                )
                
                # Compute metrics for this sample
                chamfer_dist = chamfer_distance(predicted_points, complete_points).item()
                emd_dist = earth_mover_distance(predicted_points, complete_points).item()
                
                print(f"Sample {i}: Chamfer={chamfer_dist:.6f}, EMD={emd_dist:.6f}")
    
    def generate_report(self, metrics: Dict[str, float], save_path: str = "assets/evaluation_report.txt") -> None:
        """Generate evaluation report.
        
        Args:
            metrics: Evaluation metrics.
            save_path: Path to save report.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w") as f:
            f.write("3D Shape Completion Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"  - Model: {self.config.model.name}\n")
            f.write(f"  - Num Points: {self.config.data.num_points}\n")
            f.write(f"  - Completion Ratio: {self.config.data.completion_ratio}\n")
            f.write(f"  - Noise Std: {self.config.data.noise_std}\n\n")
            
            f.write("Evaluation Metrics:\n")
            for name, value in metrics.items():
                f.write(f"  - {name}: {value:.6f}\n")
            
            f.write(f"\nTest Samples: {len(self.test_dataset)}\n")
            f.write(f"Device: {self.device}\n")
        
        print(f"Evaluation report saved to {save_path}")
    
    def benchmark_inference_speed(self, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed.
        
        Args:
            num_runs: Number of runs for benchmarking.
            
        Returns:
            Dictionary with timing statistics.
        """
        print(f"Benchmarking inference speed with {num_runs} runs...")
        
        # Create dummy input
        dummy_input = torch.randn(1, self.config.data.num_points, 3).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        stats = {
            "total_time": total_time,
            "avg_time": avg_time,
            "fps": fps,
            "num_runs": num_runs,
        }
        
        print(f"Inference Speed:")
        print(f"  - Average time per sample: {avg_time:.4f}s")
        print(f"  - FPS: {fps:.2f}")
        
        return stats


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate 3D Shape Completion Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num-visualizations", type=int, default=5, help="Number of visualizations to generate")
    parser.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark")
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 30)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    
    # Generate visualizations
    evaluator.visualize_results(num_samples=args.num_visualizations)
    
    # Generate report
    evaluator.generate_report(metrics)
    
    # Benchmark if requested
    if args.benchmark:
        evaluator.benchmark_inference_speed()


if __name__ == "__main__":
    main()
