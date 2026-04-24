"""Train ensemble of models on LDS-masked subsets of data.

This script trains models (num_architectures x num_masks) where each model
is trained on a random subset of training data defined by an LDS mask.

The training is designed to be resumable: if interrupted, it will skip
models that already have checkpoints.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from layertrak.data import get_cifar10_datasets
from layertrak.lds import get_mask_manager
from layertrak.models.model_factory import ModelName, create_model
from layertrak.models.train import train_model
from layertrak.settings import settings

ARCHITECTURES: list[ModelName] = ["resnet18", "resnet34", "mobilenetv2"]


def train_ensemble(
    num_masks: int = 40,
    architecture: ModelName | None = None,
) -> None:
    """Train ensemble of models on LDS-masked data subsets.

    For each architecture and each mask:
    1. Check if checkpoint already exists (resume capability)
    2. Get subset indices from LDSMaskManager
    3. Create data subset using Subset
    4. Train model from scratch
    5. Save checkpoint to checkpoints/ensemble_models/

    Args:
        num_masks: Number of masks to train on (default: 40).
        architecture: Specific architecture to train (resnet18, resnet34, mobilenetv2).
                     If None, trains all architectures.
    """
    # Initialize
    print("=== Ensemble Training with LDS Masks ===\n")
    
    # Get mask manager
    try:
        manager = get_mask_manager()
        print(f"✓ Loaded mask manager: {manager}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Generate masks first: python -m layertrak.lds.generate_masks")
        return
    
    # Get datasets
    print("Loading training and test datasets...")
    train_dataset, test_dataset = get_cifar10_datasets()
    print(f"  Train set: {len(train_dataset)} samples")
    print(f"  Test set:  {len(test_dataset)} samples")
    
    # Create small validation loader using same subset as TRAK scoring
    # (settings.num_targets = 1000 by default)
    # This provides: sanity check + minimal time overhead (~2s per epoch on RTX 3080)
    small_test_indices = list(range(min(settings.num_targets, len(test_dataset))))
    small_test_dataset = Subset(test_dataset, small_test_indices)
    test_loader = DataLoader(
        small_test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    print(f"  Validation set: {len(small_test_dataset)} samples (same as TRAK targets)")
    
    # Disable W&B logging to avoid spam (120 runs × 30 epochs = massive clutter)
    wandb_enabled_backup = settings.wandb_enabled
    settings.wandb_enabled = False
    print(f"  W&B logging: DISABLED (will be re-enabled after training)")
    
    # Setup output directory
    output_dir = settings.project_root / settings.checkpoints_dir / "ensemble_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoint directory: {output_dir}")
    
    # Determine which architectures to train
    archs_to_train = [architecture] if architecture else ARCHITECTURES
    num_archs = len(archs_to_train)
    total_models = num_archs * num_masks
    
    # Training loop
    trained_count = 0
    skipped_count = 0
    
    print(f"\n{'=' * 70}")
    print(f"Training {total_models} models ({num_archs} arch{'s' if num_archs > 1 else ''} x {num_masks} masks)")
    print(f"{'=' * 70}\n")
    
    for arch in archs_to_train:
        print(f"\n{'=' * 70}")
        print(f"Architecture: {arch.upper()}")
        print(f"{'=' * 70}")
        
        for mask_idx in tqdm(range(num_masks), desc=f"{arch} masks", unit="model"):
            # Define checkpoint path
            checkpoint_name = f"{arch}_mask_{mask_idx}.pt"
            checkpoint_path = output_dir / checkpoint_name
            
            # Skip if already trained
            if checkpoint_path.exists():
                skipped_count += 1
                continue
            
            # Get subset indices from manager
            indices = manager.get_subset_indices(mask_idx)
            subset_size = len(indices)
            
            # Create subset dataset and loader
            subset_dataset = Subset(train_dataset, indices)
            subset_loader = DataLoader(
                subset_dataset,
                batch_size=settings.batch_size,
                shuffle=True,
                num_workers=settings.num_workers,
            )
            
            # Create fresh model
            device = settings.device
            model = create_model(
                arch,
                num_classes=10,
                device=device,
                freeze_early=True,
                pretrained=True,
            )
            
            # Train model on subset
            try:
                train_model(
                    model,
                    subset_loader,
                    test_loader,
                    device=device,
                    run_name=checkpoint_name.replace(".pt", ""),
                    checkpoint_dir=output_dir,
                    save_per_epoch=False,
                )
                trained_count += 1
            except Exception as e:
                print(f"✗ Error training {checkpoint_name}: {e}")
                # Try to remove partial checkpoint if it exists
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                continue
            finally:
                # CRITICAL: Free GPU memory after each model
                # Without this, VRAM will fill up after ~5-10 models
                del model
                torch.cuda.empty_cache()
    
    # Restore W&B setting
    settings.wandb_enabled = wandb_enabled_backup
    
    # Summary
    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")
    print(f"Trained:  {trained_count} models")
    print(f"Skipped:  {skipped_count} models (already exist)")
    print(f"Total:    {trained_count + skipped_count}/{total_models}")
    print(f"\nCheckpoints saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ensemble of models on LDS-masked data subsets"
    )
    parser.add_argument(
        "--num-masks",
        type=int,
        default=40,
        help="Number of masks to train on (default: 40)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["resnet18", "resnet34", "mobilenetv2"],
        default=None,
        help="Specific architecture to train (default: None = train all)",
    )
    args = parser.parse_args()
    
    train_ensemble(num_masks=args.num_masks, architecture=args.architecture)
