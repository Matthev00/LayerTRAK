from pathlib import Path

import argparse
import numpy as np
import torch

from layertrak.data import get_dataloaders
from layertrak.experiments.results_io import (
    make_run_dir,
    save_config_artifacts,
    write_overview_csv,
)
from layertrak.experiments.trak import LayerTRAKRunner
from layertrak.models.layers import get_grad_wrt, get_layer_configs
from layertrak.models.model_factory import ModelName, create_model
from layertrak.settings import settings

ARCHITECTURES: list[ModelName] = ["resnet18", "resnet34", "mobilenetv2"]


def _resolve_checkpoint_paths(architectures: list[ModelName], num_last_epochs: int | None = None) -> dict[ModelName, list[Path]]:
    """Return checkpoint paths for each architecture, optionally limited to last N epochs."""
    base_models_dir = settings.project_root / settings.checkpoints_dir / "base_models"
    checkpoint_paths: dict[ModelName, list[Path]] = {}
    
    for arch in architectures:
        # Find all epoch files for this architecture
        epoch_files = sorted(
            [p for p in base_models_dir.glob(f"{arch}_epoch_*.pt")],
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True  # Most recent first
        )
        
        if not epoch_files:
            raise FileNotFoundError(
                f"No checkpoint files found for {arch} in {base_models_dir}.\n"
                "Train base models first with `make train`."
            )
        
        # Select last N epochs or only the last one if None
        if num_last_epochs is not None:
            selected_files = epoch_files[:num_last_epochs]
        else:
            selected_files = [epoch_files[0]]  # Only the most recent epoch
        
        checkpoint_paths[arch] = selected_files
    
    return checkpoint_paths


def run_experiment(architectures: list[ModelName], num_last_epochs: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load trained models and compute TRAK scores for every layer config.

    Args:
        num_last_epochs: Number of last epochs to process. If None, process only the last epoch.
                         If specified, average scores across those epochs.

    Returns:
        Nested dict: results[arch][config_name] = score matrix (averaged if multiple epochs).
    """
    checkpoint_paths = _resolve_checkpoint_paths(architectures, num_last_epochs)
    train_loader, _, targets_loader = get_dataloaders()
    train_set_size = len(train_loader.dataset)  # type: ignore[arg-type]

    from torch.utils.data import DataLoader
    trak_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=settings.batch_size,
        shuffle=False, 
        num_workers=settings.num_workers
    )

    results_root = settings.project_root / settings.trak_results_dir
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(results_root)

    all_scores: dict[str, dict[str, np.ndarray]] = {}
    overview_rows: list[dict[str, object]] = []

    for arch in architectures:
        print(f"\n{'=' * 60}")
        print(f"Architecture: {arch}")
        print(f"{'=' * 60}")

        layer_configs = get_layer_configs(arch)
        
        # Load the first model to compute structural info (grad_wrt, num_params)
        first_checkpoint_path = checkpoint_paths[arch][0]
        print(f"Loading first checkpoint for structural info: {first_checkpoint_path}")
        model = create_model(arch, device=settings.device, checkpoint_path=first_checkpoint_path)
        
        config_info: dict[str, dict] = {}
        for config in layer_configs:
            config_name = config["name"]
            prefixes = config["prefixes"]
            grad_wrt = get_grad_wrt(model, prefixes)
            num_params = sum(p.numel() for name, p in model.named_parameters() if grad_wrt is None or name in grad_wrt)
            config_info[config_name] = {
                "prefixes": prefixes,
                "grad_wrt": grad_wrt,
                "num_params": num_params,
            }

        # Collect scores for each config across epochs
        temp_scores: dict[str, list[np.ndarray]] = {config["name"]: [] for config in layer_configs}

        for checkpoint_path in checkpoint_paths[arch]:
            epoch = checkpoint_path.stem.split('_')[-1]
            print(f"\n--- Epoch: {epoch} ---")
            print(f"Loading checkpoint: {checkpoint_path}")
            model = create_model(arch, device=settings.device, checkpoint_path=checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=settings.device, weights_only=True)

            for config in layer_configs:
                config_name = config["name"]
                info = config_info[config_name]
                prefixes = info["prefixes"]
                grad_wrt = info["grad_wrt"]
                num_params = info["num_params"]
                
                print(f"\n--- {arch}/epoch_{epoch}/{config_name} ---")
                print(f"  Prefixes: {prefixes}")
                print(f"  Params tracked: {num_params:,}")

                artifact_dir = run_dir / arch / f"epoch_{epoch}" / config_name
                save_dir = artifact_dir / "trak_tmp"

                runner = LayerTRAKRunner(
                    model=model,
                    checkpoint=checkpoint,
                    train_set_size=train_set_size,
                    save_dir=save_dir,
                    device=settings.device,
                    grad_wrt=grad_wrt,
                )
                runner.featurize(trak_train_loader)
                scores = runner.score(
                    targets_loader,
                    num_targets=settings.num_targets,
                    exp_name=config_name,
                )
                temp_scores[config_name].append(scores)
                print(f"  Scores shape: {scores.shape}")

                # Note: Individual epoch results are not saved to overview, only averaged

        # Always average scores across epochs (even if only one)
        arch_scores: dict[str, np.ndarray] = {}
        for config_name in temp_scores:
            arch_scores[config_name] = np.mean(temp_scores[config_name], axis=0)

        # Save averaged results
        for config in layer_configs:
            config_name = config["name"]
            scores = arch_scores[config_name]
            info = config_info[config_name]
            prefixes = info["prefixes"]
            num_params = info["num_params"]

            artifact_dir = run_dir / arch / config_name
            overview_rows.append(
                save_config_artifacts(
                    artifact_dir,
                    scores=scores,
                    architecture=arch,
                    config_name=config_name,
                    prefixes=prefixes,
                    num_tracked_params=num_params,
                    num_targets=settings.num_targets,
                    train_set_size=train_set_size,
                    checkpoint_path=first_checkpoint_path,  # Use first checkpoint path for averaged
                )
            )

        all_scores[arch] = arch_scores

    write_overview_csv(run_dir / "overview.csv", overview_rows)

    print(f"\n{'=' * 60}")
    print("All experiments complete.")
    print(f"Results saved to: {run_dir}")
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TRAK experiments on trained models.")
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Specific architecture to run TRAK for (resnet18, resnet34, mobilenetv2). If not specified, run for all."
    )
    parser.add_argument(
        "--num-last-epochs",
        type=int,
        default=None,
        help="Number of last epochs to process for each architecture. If not specified, process only the last epoch."
    )
    args = parser.parse_args()
    
    if args.architecture:
        if args.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {args.architecture}. Available: {ARCHITECTURES}")
        selected_archs = [args.architecture]
    else:
        selected_archs = ARCHITECTURES
    
    run_experiment(architectures=selected_archs, num_last_epochs=args.num_last_epochs)
