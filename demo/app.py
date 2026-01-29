"""Interactive demo for 3D Shape Completion using Streamlit."""

import streamlit as st
import torch
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pointnet_plus_plus import PointNetPlusPlus
from src.data.dataset import SyntheticShapeDataset, PointCloudTransform
from src.utils.device import get_device, set_seed, load_checkpoint
from src.layers.losses import chamfer_distance, earth_mover_distance, F1Score
from omegaconf import OmegaConf


@st.cache_resource
def load_model(checkpoint_path: str, config_path: str):
    """Load the trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to config file.
        
    Returns:
        Loaded model and config.
    """
    # Load config
    config = OmegaConf.load(config_path)
    
    # Set device
    device = get_device()
    
    # Create model
    model = PointNetPlusPlus(
        num_classes=config.model.num_classes,
        normal_channel=config.model.normal_channel,
        num_points=config.data.num_points,
        dropout=config.model.dropout,
    ).to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    
    return model, config, device


def generate_synthetic_shape(shape_type: str, num_points: int = 2048) -> np.ndarray:
    """Generate a synthetic 3D shape.
    
    Args:
        shape_type: Type of shape to generate.
        num_points: Number of points in the shape.
        
    Returns:
        Point cloud as numpy array (N, 3).
    """
    if shape_type == "sphere":
        # Generate sphere
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        costheta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(costheta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        radius = np.random.uniform(0.8, 1.2, num_points)
        points = np.column_stack([x * radius, y * radius, z * radius])
        
    elif shape_type == "cube":
        # Generate cube
        points = []
        points_per_face = num_points // 6
        
        for face in range(6):
            if face == 0:  # Front face (z = 1)
                x = np.random.uniform(-1, 1, points_per_face)
                y = np.random.uniform(-1, 1, points_per_face)
                z = np.ones(points_per_face)
            elif face == 1:  # Back face (z = -1)
                x = np.random.uniform(-1, 1, points_per_face)
                y = np.random.uniform(-1, 1, points_per_face)
                z = -np.ones(points_per_face)
            elif face == 2:  # Right face (x = 1)
                x = np.ones(points_per_face)
                y = np.random.uniform(-1, 1, points_per_face)
                z = np.random.uniform(-1, 1, points_per_face)
            elif face == 3:  # Left face (x = -1)
                x = -np.ones(points_per_face)
                y = np.random.uniform(-1, 1, points_per_face)
                z = np.random.uniform(-1, 1, points_per_face)
            elif face == 4:  # Top face (y = 1)
                x = np.random.uniform(-1, 1, points_per_face)
                y = np.ones(points_per_face)
                z = np.random.uniform(-1, 1, points_per_face)
            else:  # Bottom face (y = -1)
                x = np.random.uniform(-1, 1, points_per_face)
                y = -np.ones(points_per_face)
                z = np.random.uniform(-1, 1, points_per_face)
            
            points.extend(np.column_stack([x, y, z]))
        
        # Add remaining points randomly
        remaining = num_points - len(points)
        if remaining > 0:
            x = np.random.uniform(-1, 1, remaining)
            y = np.random.uniform(-1, 1, remaining)
            z = np.random.uniform(-1, 1, remaining)
            points.extend(np.column_stack([x, y, z]))
        
        points = np.array(points[:num_points])
        
    elif shape_type == "cylinder":
        # Generate cylinder
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        z = np.random.uniform(-1, 1, num_points)
        
        x = np.cos(theta)
        y = np.sin(theta)
        
        points = np.column_stack([x, y, z])
        
    elif shape_type == "cone":
        # Generate cone
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        z = np.random.uniform(0, 1, num_points)
        
        radius = z  # Radius decreases with height
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        points = np.column_stack([x, y, z])
        
    else:  # torus
        # Generate torus
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        
        R = 1.0  # Major radius
        r = 0.3  # Minor radius
        
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        points = np.column_stack([x, y, z])
    
    return points


def create_partial_shape(complete_points: np.ndarray, completion_ratio: float) -> np.ndarray:
    """Create partial shape by removing points.
    
    Args:
        complete_points: Complete point cloud.
        completion_ratio: Ratio of points to keep.
        
    Returns:
        Partial point cloud.
    """
    num_keep = int(len(complete_points) * completion_ratio)
    indices = np.random.choice(len(complete_points), num_keep, replace=False)
    partial_points = complete_points[indices]
    return partial_points


def plot_point_cloud(points: np.ndarray, title: str, color: str = "blue") -> go.Figure:
    """Create a 3D plot of point cloud.
    
    Args:
        points: Point cloud (N, 3).
        title: Plot title.
        color: Color of points.
        
    Returns:
        Plotly figure.
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=color,
            opacity=0.8
        ),
        name=title
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=600,
        height=500
    )
    
    return fig


def compute_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    """Compute evaluation metrics.
    
    Args:
        predicted: Predicted point cloud.
        target: Target point cloud.
        
    Returns:
        Dictionary of metrics.
    """
    # Convert to tensors
    pred_tensor = torch.tensor(predicted, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    
    # Compute metrics
    chamfer_dist = chamfer_distance(pred_tensor, target_tensor).item()
    emd_dist = earth_mover_distance(pred_tensor, target_tensor).item()
    
    f1_metric = F1Score(threshold=0.01)
    f1_score = f1_metric(pred_tensor, target_tensor).item()
    
    return {
        "Chamfer Distance": chamfer_dist,
        "Earth Mover's Distance": emd_dist,
        "F1 Score": f1_score,
    }


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="3D Shape Completion Demo",
        page_icon="🔮",
        layout="wide"
    )
    
    st.title("🔮 3D Shape Completion Demo")
    st.markdown("Complete missing parts of 3D shapes using PointNet++")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Model loading
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="checkpoints/best_model.pth",
        help="Path to the trained model checkpoint"
    )
    
    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/config.yaml",
        help="Path to the configuration file"
    )
    
    # Shape generation parameters
    st.sidebar.subheader("Shape Generation")
    shape_type = st.sidebar.selectbox(
        "Shape Type",
        ["sphere", "cube", "cylinder", "cone", "torus"],
        help="Type of 3D shape to generate"
    )
    
    completion_ratio = st.sidebar.slider(
        "Completion Ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.7,
        step=0.1,
        help="Ratio of points to keep (lower = more incomplete)"
    )
    
    num_points = st.sidebar.slider(
        "Number of Points",
        min_value=512,
        max_value=4096,
        value=2048,
        step=512,
        help="Number of points in the point cloud"
    )
    
    noise_std = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=0.05,
        value=0.01,
        step=0.005,
        help="Standard deviation of Gaussian noise"
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input (Partial Shape)")
        
        # Generate shape
        if st.button("Generate New Shape", type="primary"):
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Generate complete shape
            complete_points = generate_synthetic_shape(shape_type, num_points)
            
            # Create partial shape
            partial_points = create_partial_shape(complete_points, completion_ratio)
            
            # Add noise
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, partial_points.shape)
                partial_points += noise
            
            # Store in session state
            st.session_state.complete_points = complete_points
            st.session_state.partial_points = partial_points
            st.session_state.shape_type = shape_type
    
    # Check if we have data
    if 'partial_points' not in st.session_state:
        st.info("Click 'Generate New Shape' to start!")
        return
    
    # Load model if not already loaded
    if 'model' not in st.session_state:
        try:
            with st.spinner("Loading model..."):
                model, config, device = load_model(checkpoint_path, config_path)
                st.session_state.model = model
                st.session_state.config = config
                st.session_state.device = device
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Make sure the checkpoint and config paths are correct.")
            return
    
    # Get data from session state
    partial_points = st.session_state.partial_points
    complete_points = st.session_state.complete_points
    
    # Display partial shape
    with col1:
        fig_partial = plot_point_cloud(partial_points, "Partial Shape", "red")
        st.plotly_chart(fig_partial, use_container_width=True)
    
    with col2:
        st.subheader("Output (Completed Shape)")
        
        # Run inference
        if st.button("Complete Shape", type="primary"):
            with st.spinner("Completing shape..."):
                # Prepare input
                input_tensor = torch.tensor(partial_points, dtype=torch.float32).unsqueeze(0)
                input_tensor = input_tensor.to(st.session_state.device)
                
                # Run model
                with torch.no_grad():
                    predicted_points = st.session_state.model(input_tensor)
                    predicted_points = predicted_points[0].cpu().numpy()
                
                # Store prediction
                st.session_state.predicted_points = predicted_points
        
        # Check if we have prediction
        if 'predicted_points' not in st.session_state:
            st.info("Click 'Complete Shape' to see the result!")
            return
        
        predicted_points = st.session_state.predicted_points
        
        # Display predicted shape
        fig_predicted = plot_point_cloud(predicted_points, "Completed Shape", "blue")
        st.plotly_chart(fig_predicted, use_container_width=True)
    
    # Metrics and comparison
    st.subheader("Evaluation Metrics")
    
    if 'predicted_points' in st.session_state:
        metrics = compute_metrics(st.session_state.predicted_points, complete_points)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chamfer Distance", f"{metrics['Chamfer Distance']:.6f}")
        with col2:
            st.metric("Earth Mover's Distance", f"{metrics['Earth Mover's Distance']:.6f}")
        with col3:
            st.metric("F1 Score", f"{metrics['F1 Score']:.6f}")
    
    # Side-by-side comparison
    st.subheader("Side-by-Side Comparison")
    
    if 'predicted_points' in st.session_state:
        # Create subplot
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Partial", "Complete", "Predicted"),
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter3d(
                x=partial_points[:, 0],
                y=partial_points[:, 1],
                z=partial_points[:, 2],
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.8),
                name="Partial"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=complete_points[:, 0],
                y=complete_points[:, 1],
                z=complete_points[:, 2],
                mode='markers',
                marker=dict(size=2, color='green', opacity=0.8),
                name="Complete"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=predicted_points[:, 0],
                y=predicted_points[:, 1],
                z=predicted_points[:, 2],
                mode='markers',
                marker=dict(size=2, color='blue', opacity=0.8),
                name="Predicted"
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            scene=dict(aspectmode='cube'),
            scene2=dict(aspectmode='cube'),
            scene3=dict(aspectmode='cube')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.subheader("Download Results")
    
    if 'predicted_points' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Save as PLY
            partial_pcd = o3d.geometry.PointCloud()
            partial_pcd.points = o3d.utility.Vector3dVector(partial_points)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                o3d.io.write_point_cloud(tmp_file.name, partial_pcd)
                with open(tmp_file.name, 'rb') as f:
                    st.download_button(
                        label="Download Partial Shape",
                        data=f.read(),
                        file_name="partial_shape.ply",
                        mime="application/octet-stream"
                    )
                os.unlink(tmp_file.name)
        
        with col2:
            # Save complete as PLY
            complete_pcd = o3d.geometry.PointCloud()
            complete_pcd.points = o3d.utility.Vector3dVector(complete_points)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                o3d.io.write_point_cloud(tmp_file.name, complete_pcd)
                with open(tmp_file.name, 'rb') as f:
                    st.download_button(
                        label="Download Complete Shape",
                        data=f.read(),
                        file_name="complete_shape.ply",
                        mime="application/octet-stream"
                    )
                os.unlink(tmp_file.name)
        
        with col3:
            # Save predicted as PLY
            predicted_pcd = o3d.geometry.PointCloud()
            predicted_pcd.points = o3d.utility.Vector3dVector(predicted_points)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                o3d.io.write_point_cloud(tmp_file.name, predicted_pcd)
                with open(tmp_file.name, 'rb') as f:
                    st.download_button(
                        label="Download Predicted Shape",
                        data=f.read(),
                        file_name="predicted_shape.ply",
                        mime="application/octet-stream"
                    )
                os.unlink(tmp_file.name)


if __name__ == "__main__":
    main()
