from layertrak.data import get_dataloaders
from layertrak.models.model_factory import ModelName, create_model
from layertrak.models.train import train_model
from layertrak.settings import settings

ARCHITECTURES: list[ModelName] = ["resnet18", "resnet34", "mobilenetv2"]


def train_all_base_models() -> None:
    """Train all base model architectures and save checkpoints.

    Skips training if a checkpoint already exists for a given architecture.
    Checkpoints are saved as <arch>.pt inside the configured checkpoints dir.
    """
    train_loader, test_loader, _ = get_dataloaders()
    checkpoint_dir = settings.project_root / settings.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for arch in ARCHITECTURES:
        checkpoint_path = checkpoint_dir / f"{arch}.pt"

        if checkpoint_path.exists() or list(checkpoint_dir.glob(f"{arch}_epoch_*.pt")):
            print(f"[{arch}] Checkpoint already exists, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Training: {arch}  ({settings.num_epochs} epochs)")
        print(f"{'=' * 60}")

        model = create_model(arch, device=settings.device)
        train_model(
            model,
            train_loader,
            test_loader,
            device=settings.device,
            run_name=arch,
            save_per_epoch=True,
        )

    print("\nAll base models trained.")


if __name__ == "__main__":
    train_all_base_models()
