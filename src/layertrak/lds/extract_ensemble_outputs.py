"""Extraction of predictions (logits) from trained ensemble models."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from layertrak.data import get_dataloaders
from layertrak.models.model_factory import ModelName, create_model
from layertrak.settings import settings


def extract_outputs(architecture: ModelName, num_masks: int = 40) -> None:
    """Extraction of logits from all ensemble models for a given architecture."""
    device = settings.device

    # 1. Data preparation (load only targets_loader)
    # get_dataloaders returns: train_loader, test_loader, targets_loader
    _, _, targets_loader = get_dataloaders()
    num_targets = settings.num_targets
    num_classes = 10  # CIFAR-10 has 10 classes

    # Prepare empty matrix for results: [40 models, 1000 images, 10 classes]
    all_logits = np.zeros((num_masks, num_targets, num_classes), dtype=np.float32)

    # Paths to folders
    checkpoints_dir = settings.project_root / settings.checkpoints_dir / "ensemble_models"
    output_dir = settings.project_root / "data" / "lds_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Extracting logits for: {architecture} ===")
    print(f"Expected number of models: {num_masks}")
    print(f"Number of test images: {num_targets}")

    # 2. Loop through all ensemble models
    for i in tqdm(range(num_masks), desc=f"Processing {architecture}"):
        ckpt_path = checkpoints_dir / f"{architecture}_mask_{i}.pt"

        if not ckpt_path.exists():
            print(f"\nWARNING: Missing file {ckpt_path}. Skipping model {i}.")
            continue

        # Load model from disk
        model = create_model(architecture, device=device, checkpoint_path=ckpt_path)
        model.eval()  # IMPORTANT: Evaluation mode (disables randomness like Dropout)

        # 3. Pass images through the model
        model_logits = []
        with torch.no_grad():  # Disable gradient computation (faster and less memory)
            for inputs, _ in targets_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                model_logits.append(outputs.cpu().numpy())

        # Save results for i-th model to the main matrix
        all_logits[i] = np.concatenate(model_logits, axis=0)

        # Free GPU memory after each model (protection against OOM error)
        del model
        torch.cuda.empty_cache()

    # 4. Save the ready matrix to disk
    output_path = output_dir / f"{architecture}_ensemble_logits.npy"
    np.save(output_path, all_logits)

    # Also save true labels from targets_loader (necessary for LDS formula)
    # Do this only once, as they are identical for all architectures
    labels_path = output_dir / "target_labels.npy"
    if not labels_path.exists():
        print("Saving original test labels...")
        all_labels = []
        for _, targets in targets_loader:
            all_labels.append(targets.numpy())
        np.save(labels_path, np.concatenate(all_labels, axis=0))

    print(f"\n✓ Done! Saved logits to: {output_path}")
    print(f"  Output matrix shape: {all_logits.shape} (models x images x classes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction of predictions from ensemble models")
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["resnet18", "resnet34", "mobilenetv2"],
        help="Architecture for which to extract results (required)",
    )
    parser.add_argument("--num-masks", type=int, default=40, help="Number of models in ensemble (default 40)")

    args = parser.parse_args()
    extract_outputs(architecture=args.architecture, num_masks=args.num_masks)
