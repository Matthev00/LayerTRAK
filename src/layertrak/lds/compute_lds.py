"""Computing LDS (Linear Datamodeling Score) correlation for all models and layers."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from layertrak.settings import settings


def compute_margins(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute 'Confidence Margins' for each image.

    Margin tells us how confident the model is about the correct answer.
    Formula: Margin = logit(correct_class) - max(logit(other_classes))

    Args:
        logits: Matrix with shape [num_masks, num_images, num_classes]
        labels: Vector of correct labels [num_images]

    Returns:
        Matrix of margins with shape [num_masks, num_images]
    """
    num_masks, num_targets, _ = logits.shape

    # Fast, vectorized numpy operation (no slow for loops)
    m_idx, t_idx = np.indices((num_masks, num_targets))

    # Extract logits only for correct classes
    correct_logits = logits[m_idx, t_idx, labels[None, :]]

    # Clone logits, zero out (set to -inf) the correct ones...
    logits_copy = logits.copy()
    logits_copy[m_idx, t_idx, labels[None, :]] = -np.inf

    # ...to easily find the highest logit from INCORRECT classes
    max_other_logits = np.max(logits_copy, axis=-1)

    return correct_logits - max_other_logits


def get_trak_run_dir(trak_dir: Path | None = None) -> Path:
    """Find the TRAK results folder. If trak_dir is provided, use it; otherwise auto-detect."""
    if trak_dir is not None:
        if not trak_dir.exists():
            raise FileNotFoundError(f"Specified TRAK directory does not exist: {trak_dir}")
        return trak_dir
    
    # Auto-detect logic
    trak_results_dir = settings.project_root / settings.trak_results_dir
    if not trak_results_dir.exists():
        raise FileNotFoundError(f"Missing TRAK results folder: {trak_results_dir}")

    # Check if we have a flat structure with architectures directly
    potential_archs = ["resnet18", "resnet34", "mobilenetv2"]
    arch_dirs = [d for d in trak_results_dir.iterdir() if d.is_dir() and d.name in potential_archs]

    if len(arch_dirs) >= 2:  # If we find at least 2 architectures, it's a flat structure
        return trak_results_dir

    # Otherwise, look for timestamped folders
    runs = sorted([d for d in trak_results_dir.iterdir() if d.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No results found in: {trak_results_dir}")

    return runs[-1]


def compute_lds_for_all(trak_dir: Path | None = None) -> None:
    """Main function to compute the LDS metric."""

    # 1. Load constant, fixed data from disk
    masks_path = settings.project_root / "data" / "lds_masks" / "lds_masks.npy"
    labels_path = settings.project_root / "data" / "lds_outputs" / "target_labels.npy"

    if not masks_path.exists() or not labels_path.exists():
        print("Error: Missing generated masks or labels!")
        print("Make sure you ran `generate_masks.py` and `extract_ensemble_outputs.py`")
        return

    print("Loading shared masks and labels...")
    masks = np.load(masks_path).astype(np.float32)  # Shape: [num_masks, train_size]
    labels = np.load(labels_path)  # Shape: [num_targets]

    trak_run_dir = get_trak_run_dir(trak_dir)
    print(f"Using TRAK results from folder: {trak_run_dir.name}")

    final_results = {}

    # 2. Find all architectures (resnet18, resnet34, mobilenetv2) computed by TRAK
    potential_archs = ["resnet18", "resnet34", "mobilenetv2"]
    child_dirs = [d for d in trak_run_dir.iterdir() if d.is_dir()]
    architectures = [d.name for d in child_dirs if d.name in potential_archs]
    single_arch = False

    if not architectures:
        # The provided folder may already be a single architecture folder.
        architectures = [trak_run_dir.name]
        single_arch = True

    for arch in architectures:
        print(f"\n{'=' * 55}\nAnalysis: {arch.upper()}\n{'=' * 55}")

        logits_path = settings.project_root / "data" / "lds_outputs" / f"{arch}_ensemble_logits.npy"
        if not logits_path.exists():
            print(f"WARNING: Missing extracted logits for {arch}. Skipping this architecture.")
            continue

        print("  Loading responses from your Ensemble models...")
        logits = np.load(logits_path)

        # Protection in case someone trained e.g. only 30 out of 40 models
        num_trained_masks = logits.shape[0]
        current_masks = masks[:num_trained_masks]

        # STEP A: Reality (Computing actual margins from Ensemble)
        actual_margins = compute_margins(logits, labels)  # Shape: [num_masks, num_targets]

        arch_results = {}
        config_root = trak_run_dir if single_arch else trak_run_dir / arch
        configs = [d.name for d in config_root.iterdir() if d.is_dir()]

        # 3. Iterate over each layer configuration (head_only, early, etc.)
        for config in configs:
            scores_path = config_root / config / "scores.npy"
            if not scores_path.exists():
                continue

            # Load TRAK's predictions (Shape: [num_targets, train_size])
            trak_scores = np.load(scores_path)

            # STEP B: Hypothesis (Computing TRAK's margin predictions)
            # Linear LDS metric. If mask = 1 (image used for training), we add its influence score.
            # Matrix multiplication: [num_masks, train_size] @ [train_size, num_targets] = [num_masks, num_targets]
            predicted_margins = current_masks @ trak_scores

            # STEP C: Correlation (Lie Detector Test)
            correlations = []
            num_targets = actual_margins.shape[1]

            # Compute Spearman correlation for each of the 1000 test images
            for t in range(num_targets):
                act = actual_margins[:, t]
                pred = predicted_margins[:, t]

                # Collision: Reality vs Hypothesis
                corr, _ = spearmanr(act, pred)
                if not np.isnan(corr):
                    correlations.append(corr)

            # Final LDS result is simply the average correlation across all images
            lds_score = float(np.mean(correlations))
            arch_results[config] = lds_score

            print(f"  ➜ [{config:^10}]: LDS = {lds_score:.4f}")

        final_results[arch] = arch_results

    # 4. Save analysis results
    output_file = trak_run_dir / "lds_final_scores.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\n✓ Success! Correlation summary saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute LDS scores for TRAK results.")
    parser.add_argument(
        "--trak-dir",
        type=str,
        default=None,
        help="Path to TRAK results directory. If not specified, auto-detect the latest."
    )
    args = parser.parse_args()
    
    trak_dir = Path(args.trak_dir) if args.trak_dir else None
    compute_lds_for_all(trak_dir)
