"""Utilities for loading and working with LDS masks."""

import json
from pathlib import Path

import numpy as np
import torch

from layertrak.settings import settings


class LDSMaskManager:
    """Manager for loading and applying LDS masks."""

    def __init__(self, mask_file: Path | None = None, metadata_file: Path | None = None):
        """Initialize mask manager.
        
        Args:
            mask_file: Path to masks .npy file. If None, uses default location.
            metadata_file: Path to masks metadata .json file. If None, uses default location.
        
        Raises:
            FileNotFoundError: If mask files do not exist.
        """
        if mask_file is None:
            mask_file = settings.data_dir / "lds_masks" / "lds_masks.npy"
        if metadata_file is None:
            metadata_file = settings.data_dir / "lds_masks" / "lds_masks_metadata.json"
        
        self.mask_file = Path(mask_file)
        self.metadata_file = Path(metadata_file)
        
        if not self.mask_file.exists():
            raise FileNotFoundError(
                f"Mask file not found: {self.mask_file}\n"
                f"Please generate masks first: python -m layertrak.lds.generate_masks"
            )
        
        self.masks = np.load(self.mask_file)
        
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "num_masks": self.masks.shape[0],
                "train_size": self.masks.shape[1],
            }
    
    def get_mask(self, idx: int) -> np.ndarray:
        """Get a single mask by index.
        
        Args:
            idx: Mask index.
        
        Returns:
            Boolean array representing the mask.
        """
        return self.masks[idx]
    
    def get_subset_indices(self, idx: int) -> np.ndarray:
        """Get indices of samples in a subset defined by a mask.
        
        Args:
            idx: Mask index.
        
        Returns:
            Array of indices where mask is True.
        """
        return np.where(self.masks[idx])[0]
    
    def apply_mask(self, data: np.ndarray, mask_idx: int) -> np.ndarray:
        """Apply a mask to select a subset of data.
        
        Args:
            data: Input data array (first dimension = samples).
            mask_idx: Index of mask to apply.
        
        Returns:
            Subset of data selected by mask.
        """
        return data[self.masks[mask_idx]]
    
    def get_all_masks(self) -> np.ndarray:
        """Get all masks.
        
        Returns:
            All masks array.
        """
        return self.masks
    
    @property
    def num_masks(self) -> int:
        """Get number of masks."""
        return self.masks.shape[0]
    
    @property
    def train_size(self) -> int:
        """Get training set size."""
        return self.masks.shape[1]
    
    @property
    def subset_size(self) -> int:
        """Get size of each subset (average)."""
        return int(self.masks.sum(axis=1).mean())
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LDSMaskManager(num_masks={self.num_masks}, "
            f"train_size={self.train_size}, "
            f"subset_size={self.subset_size})"
        )


def get_mask_manager(mask_file: Path | None = None) -> LDSMaskManager:
    """Get singleton LDS mask manager.
    
    Args:
        mask_file: Path to masks file.
    
    Returns:
        LDS mask manager instance.
    """
    return LDSMaskManager(mask_file=mask_file)
