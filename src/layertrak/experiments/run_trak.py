from pathlib import Path

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


def _resolve_checkpoint_paths() -> dict[ModelName, Path]:
    """Return checkpoint paths, failing before datasets are loaded if any are missing."""
    checkpoint_dir = settings.project_root / settings.checkpoints_dir
    checkpoint_paths = {arch: checkpoint_dir / f"{arch}.pt" for arch in ARCHITECTURES}
    missing = [f"{arch}: {path}" for arch, path in checkpoint_paths.items() if not path.exists()]
    if missing:
        missing_list = "\n  ".join(missing)
        raise FileNotFoundError(
            "Missing trained model checkpoint(s):\n"
            f"  {missing_list}\n"
            "Train base models first with `make train` or place already trained checkpoints in the checkpoints dir."
        )
    return checkpoint_paths


def run_experiment() -> dict[str, dict[str, np.ndarray]]:
    """Load trained models and compute TRAK scores for every layer config.

    Returns:
        Nested dict: results[arch][config_name] = score matrix.
    """
    checkpoint_paths = _resolve_checkpoint_paths()
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

    for arch in ARCHITECTURES:
        print(f"\n{'=' * 60}")
        print(f"Architecture: {arch}")
        print(f"{'=' * 60}")

        checkpoint_path = checkpoint_paths[arch]
        print(f"Loading checkpoint: {checkpoint_path}")
        model = create_model(arch, device=settings.device, checkpoint_path=checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=settings.device, weights_only=True)

        layer_configs = get_layer_configs(arch)
        arch_scores: dict[str, np.ndarray] = {}

        for config in layer_configs:
            config_name = config["name"]
            prefixes = config["prefixes"]
            grad_wrt = get_grad_wrt(model, prefixes)
            num_params = sum(p.numel() for name, p in model.named_parameters() if grad_wrt is None or name in grad_wrt)
            print(f"\n--- {arch}/{config_name} ---")
            print(f"  Prefixes: {prefixes}")
            print(f"  Params tracked: {num_params:,}")

            artifact_dir = run_dir / arch / config_name
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
            arch_scores[config_name] = scores
            print(f"  Scores shape: {scores.shape}")

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
                    checkpoint_path=checkpoint_path,
                )
            )

        all_scores[arch] = arch_scores

    write_overview_csv(run_dir / "overview.csv", overview_rows)

    print(f"\n{'=' * 60}")
    print("All experiments complete.")
    print(f"Results saved to: {run_dir}")
    return all_scores


if __name__ == "__main__":
    run_experiment()
