"""Script to generate universal LDS masks for model training."""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from layertrak.settings import settings


def generate_masks(
    num_masks: int = 40,
    train_size: int = 50000,
    subset_ratio: float = 0.5,
    random_seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Generate universal LDS masks and save to disk.
    
    Universal masks are shared across all models. Each mask represents a random
    50% subset of the training set.
    
    Args:
        num_masks: Total number of masks to generate.
        train_size: Total training set size (CIFAR-10: 50000).
        subset_ratio: Ratio of samples per mask (0.5 = 50%).
        random_seed: Random seed for reproducibility.
        output_dir: Directory to save masks. If None, uses data/lds_masks from project root.
    
    Returns:
        Tuple of (masks array, path to saved file).
    """
    if output_dir is None:
        output_dir = settings.data_dir / "lds_masks"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subset_size = int(train_size * subset_ratio)
    rng = np.random.RandomState(random_seed)
    
    print(f"Generating {num_masks} universal LDS masks...")
    print(f"  Train set size: {train_size}")
    print(f"  Subset size per mask: {subset_size} ({subset_ratio*100:.0f}%)")
    print(f"  Random seed: {random_seed}")
    
    masks = np.zeros((num_masks, train_size), dtype=bool)
    
    for i in tqdm(range(num_masks), desc="Generating masks"):
        selected = rng.choice(train_size, size=subset_size, replace=False)
        masks[i, selected] = True
    
    # Save masks
    mask_file = output_dir / "lds_masks.npy"
    np.save(mask_file, masks)
    
    # Save metadata
    metadata = {
        "num_masks": num_masks,
        "train_size": train_size,
        "subset_ratio": subset_ratio,
        "subset_size": subset_size,
        "random_seed": random_seed,
    }
    metadata_file = output_dir / "lds_masks_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    file_size_mb = mask_file.stat().st_size / 1024 / 1024
    print(f"\n✓ Successfully generated and saved masks!")
    print(f"  Output directory: {output_dir}")
    print(f"  Masks file: {mask_file.name}")
    print(f"  Metadata file: {metadata_file.name}")
    print(f"  Shape: {masks.shape}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Data type: {masks.dtype}")
    
    return masks, mask_file


def load_masks(mask_file: Path | None = None) -> np.ndarray:
    """Load pre-generated LDS masks from disk.
    
    Args:
        mask_file: Path to the masks .npy file. If None, uses default location.
    
    Returns:
        Loaded masks array.
    
    Raises:
        FileNotFoundError: If mask file does not exist.
    """
    if mask_file is None:
        mask_file = settings.data_dir / "lds_masks" / "lds_masks.npy"
    
    mask_file = Path(mask_file)
    
    if not mask_file.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_file}\n"
            f"Please run: python -m layertrak.lds.generate_masks"
        )
    
    masks = np.load(mask_file)
    print(f"✓ Loaded masks from {mask_file}")
    print(f"  Shape: {masks.shape}")
    print(f"  Data type: {masks.dtype}")
    
    return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate universal LDS masks for model training"
    )
    parser.add_argument(
        "--num-masks",
        type=int,
        default=40,
        help="Number of masks to generate (default: 40)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=50000,
        help="Size of training set (default: 50000 for CIFAR-10)",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=0.5,
        help="Ratio of samples per mask (default: 0.5 = 50%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/lds_masks)",
    )
    
    args = parser.parse_args()
    
    generate_masks(
        num_masks=args.num_masks,
        train_size=args.train_size,
        subset_ratio=args.subset_ratio,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )
