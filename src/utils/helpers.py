"""
Utility functions and helpers
"""
import os
import random
import logging
import numpy as np
import torch
from typing import Dict, Any


def setup_logging(logging_config: Dict[str, Any]):
    """Setup logging configuration"""
    level = getattr(logging, logging_config.get('level', 'INFO').upper())
    format_str = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
        ]
    )


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directory_structure(base_dir: str = "."):
    """Create the necessary directory structure"""
    directories = [
        "data/raw/hausa",
        "data/raw/swahili", 
        "data/raw/english",
        "data/processed",
        "results",
        "models",
        "logs"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_device(device_name: str = "auto"):
    """Get the appropriate device for training"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to file"""
    import yaml
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from file"""
    import yaml
    
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)