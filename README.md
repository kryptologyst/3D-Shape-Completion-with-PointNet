# 3D Shape Completion with PointNet++

A production-ready implementation of 3D shape completion using PointNet++ architecture. This project demonstrates how to complete missing or occluded parts of 3D shapes based on partial point cloud inputs.

## Features

- **PointNet++ Architecture**: State-of-the-art point cloud processing with hierarchical feature learning
- **Multiple Loss Functions**: Chamfer Distance, Earth Mover's Distance, and L2 loss for robust training
- **Synthetic Dataset**: Generates various 3D shapes (spheres, cubes, cylinders, cones, torus) for training
- **Interactive Demo**: Streamlit-based web interface for real-time shape completion
- **Comprehensive Evaluation**: Multiple metrics including Chamfer Distance, EMD, and F1-Score
- **Modern Tech Stack**: PyTorch 2.x, Hydra configs, device fallback (CUDA → MPS → CPU)
- **Production Ready**: Type hints, docstrings, tests, and CI/CD pipeline

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/3D-Shape-Completion-with-PointNet-.git
cd 3D-Shape-Completion-with-PointNet-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up pre-commit hooks (optional):
```bash
pre-commit install
```

### Training

Train the model with default configuration:
```bash
python train.py
```

Train with custom configuration:
```bash
python train.py training.batch_size=32 training.num_epochs=200
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

Run evaluation with visualizations:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --num-visualizations 10
```

### Interactive Demo

Launch the Streamlit demo:
```bash
python demo.py
```

Or run directly:
```bash
streamlit run demo/app.py
```

The demo will be available at `http://localhost:8501`

## Project Structure

```
3d-shape-completion/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── pointnet_plus_plus.py # PointNet++ architecture
│   ├── layers/                   # Custom layers and losses
│   │   └── losses.py             # Loss functions (Chamfer, EMD, F1)
│   ├── data/                     # Data pipeline
│   │   └── dataset.py            # Dataset and data loaders
│   ├── utils/                    # Utilities
│   │   └── device.py             # Device management and utilities
│   ├── train/                    # Training scripts
│   │   └── train.py              # Main training logic
│   └── eval/                     # Evaluation scripts
│       └── evaluate.py           # Evaluation and metrics
├── configs/                      # Configuration files
│   └── config.yaml               # Main configuration
├── demo/                         # Interactive demo
│   └── app.py                    # Streamlit application
├── scripts/                      # Utility scripts
├── tests/                        # Unit tests
├── assets/                       # Generated assets and visualizations
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── data/                         # Dataset storage
├── train.py                      # Main training entry point
├── evaluate.py                   # Main evaluation entry point
├── demo.py                       # Demo launcher
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Configuration

The project uses Hydra for configuration management. Key configuration options:

### Data Configuration
- `data.num_points`: Number of points per point cloud (default: 2048)
- `data.completion_ratio`: Ratio of points to keep (default: 0.7)
- `data.noise_std`: Standard deviation of Gaussian noise (default: 0.01)
- `data.shapes`: List of shape types to generate

### Model Configuration
- `model.num_classes`: Output dimension (3 for xyz coordinates)
- `model.dropout`: Dropout rate (default: 0.5)
- `model.normal_channel`: Whether to use normal channels

### Training Configuration
- `training.batch_size`: Batch size (default: 16)
- `training.num_epochs`: Number of training epochs (default: 100)
- `training.learning_rate`: Learning rate (default: 0.001)
- `training.scheduler`: Learning rate scheduler ("cosine" or "step")

### Loss Configuration
- `loss.chamfer_weight`: Weight for Chamfer distance loss (default: 1.0)
- `loss.emd_weight`: Weight for EMD loss (default: 0.1)
- `loss.l2_weight`: Weight for L2 loss (default: 0.1)

## Model Architecture

The PointNet++ architecture consists of:

1. **Encoder**: Hierarchical feature extraction using set abstraction layers
   - SA1: 512 points, radius 0.2, 32 neighbors
   - SA2: 128 points, radius 0.4, 64 neighbors  
   - SA3: Global feature extraction

2. **Decoder**: Feature propagation layers for upsampling
   - FP3: 256 → 256 features
   - FP2: 256 → 128 features
   - FP1: 128 → 128 features

3. **Output Head**: Final prediction layer
   - Conv layers with batch normalization and dropout
   - Output: 3D coordinates (x, y, z)

## Loss Functions

### Chamfer Distance
Measures the bidirectional distance between predicted and target point clouds:
```
CD(P, Q) = (1/|P|) Σ min ||p - q||² + (1/|Q|) Σ min ||q - p||²
```

### Earth Mover's Distance (EMD)
Computes the minimum cost to transform one point cloud into another:
```
EMD(P, Q) = min Σ ||p - φ(p)||²
```

### F1 Score
Measures precision and recall based on distance threshold:
```
F1 = 2 * Precision * Recall / (Precision + Recall)
```

## Evaluation Metrics

The model is evaluated using:

- **Chamfer Distance**: Lower is better (typical range: 0.001-0.1)
- **Earth Mover's Distance**: Lower is better (typical range: 0.01-0.5)
- **F1 Score**: Higher is better (typical range: 0.1-0.9)

## Dataset

The project uses a synthetic dataset that generates:

- **Sphere**: Uniform sampling on sphere surface
- **Cube**: Points on cube faces with random sampling
- **Cylinder**: Points on cylinder surface
- **Cone**: Points on cone surface with decreasing radius
- **Torus**: Points on torus surface

Each shape is:
- Sampled to a fixed number of points
- Partially occluded based on completion ratio
- Augmented with random transformations and noise

## Performance

### Training Performance
- **Training Time**: ~2-4 hours on GPU (RTX 3080)
- **Memory Usage**: ~4-6 GB VRAM for batch size 16
- **Convergence**: Typically converges in 50-80 epochs

### Inference Performance
- **Speed**: ~50-100 FPS on GPU, ~5-10 FPS on CPU
- **Latency**: ~10-20ms per sample on GPU
- **Memory**: ~1-2 GB VRAM for inference

## Advanced Usage

### Custom Shapes
To add custom shapes, modify the `SyntheticShapeDataset` class in `src/data/dataset.py`:

```python
def _generate_custom_shape(self) -> np.ndarray:
    # Your custom shape generation logic
    return points
```

### Custom Loss Functions
Add new loss functions in `src/layers/losses.py`:

```python
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        # Your custom loss computation
        return loss
```

### Model Architecture Modifications
Modify the PointNet++ architecture in `src/models/pointnet_plus_plus.py`:

```python
class CustomPointNetPlusPlus(PointNetPlusPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Your modifications
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**
   - Increase batch size if memory allows
   - Use multiple GPUs with DataParallel
   - Enable mixed precision training

3. **Poor Convergence**
   - Adjust learning rate
   - Modify loss weights
   - Increase training epochs

### Device Issues

The project automatically detects and uses the best available device:
- CUDA GPU (if available)
- Apple Silicon MPS (if available)
- CPU (fallback)

To force CPU usage, modify the device detection in `src/utils/device.py`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest tests/`
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{3d-shape-completion,
  title={3D Shape Completion with PointNet++},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/3D-Shape-Completion-with-PointNet-}
}
```

## Acknowledgments

- PointNet++ paper: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
- Open3D library for 3D processing
- PyTorch team for the deep learning framework
- Streamlit for the interactive demo interface
# 3D-Shape-Completion-with-PointNet-
