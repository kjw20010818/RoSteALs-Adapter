"""
Compatibility helpers for newer PyTorch / PyTorch-Lightning versions.
Import this at the top of modules that need torch.load or PL utilities.
"""
import torch


def safe_torch_load(path, map_location="cpu"):
    """torch.load wrapper that works with PyTorch ≥ 2.6 (weights_only default changed)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have the weights_only argument
        return torch.load(path, map_location=map_location)
