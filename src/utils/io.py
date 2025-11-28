"""
Utility functions for file I/O operations.

Author: Aashik Mathew
"""

import numpy as np
import joblib
import yaml
import json
from pathlib import Path
from typing import Any, Dict


def load_csi_data(data_dir: str) -> tuple:
    """
    Load CSI data and labels from directory.
    
    Args:
        data_dir: Directory containing csi_data.npy and labels.npy
        
    Returns:
        Tuple of (csi_data, labels)
    """
    data_dir = Path(data_dir)
    csi_data = np.load(data_dir / 'csi_data.npy')
    labels = np.load(data_dir / 'labels.npy')
    return csi_data, labels


def save_csi_data(csi_data: np.ndarray, labels: np.ndarray, output_dir: str):
    """
    Save CSI data and labels to directory.
    
    Args:
        csi_data: CSI data array
        labels: Labels array
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'csi_data.npy', csi_data)
    np.save(output_dir / 'labels.npy', labels)


def load_model(model_path: str) -> Any:
    """Load trained model from file."""
    return joblib.load(model_path)


def save_model(model: Any, model_path: str):
    """Save model to file."""
    joblib.dump(model, model_path)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, config_path: str):
    """Save configuration to YAML."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def save_metrics(metrics: Dict, metrics_path: str):
    """Save metrics to JSON file."""
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

