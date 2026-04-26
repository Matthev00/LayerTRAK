from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from layertrak.settings import settings


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: str | None = None,
    num_epochs: int | None = None,
    checkpoint_dir: Path | None = None,
    run_name: str = "train",
    save_per_epoch: bool = False,
) -> dict[str, torch.Tensor]:
    """Train a model on CIFAR-10 and return the final checkpoint.

    Args:
        model: Model to train (already on device).
        train_loader: Training data loader.
        test_loader: Test data loader for evaluation.
        device: Device string. Defaults to settings.device.
        num_epochs: Number of epochs. Defaults to settings.num_epochs.
        checkpoint_dir: Directory to save the checkpoint. Defaults to settings path.
        run_name: Name for wandb run and checkpoint file.
        save_per_epoch: If True, save checkpoint after each epoch. If False, save only final checkpoint.

    Returns:
        The model's state_dict after training.
    """
    device = device or settings.device
    num_epochs = num_epochs or settings.num_epochs
    checkpoint_dir = checkpoint_dir or (settings.project_root / settings.checkpoints_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    if settings.wandb_enabled:
        wandb_run = wandb.init(
            project=settings.wandb_project,
            name=run_name,
            config={
                "num_epochs": num_epochs,
                "batch_size": settings.batch_size,
                "learning_rate": settings.learning_rate,
                "weight_decay": settings.weight_decay,
                "device": device,
            },
        )

    for epoch in range(num_epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        test_loss, test_acc = _evaluate(model, test_loader, criterion, device)

        metrics = {
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        print(
            f"Epoch {epoch + 1}/{num_epochs} — "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log(metrics, step=epoch)

        # Save checkpoint after each epoch if requested
        if save_per_epoch:
            checkpoint = model.state_dict()
            save_path = checkpoint_dir / f"{run_name}_epoch_{epoch + 1:02d}.pt"
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")

    # Save final checkpoint if not saving per epoch
    if not save_per_epoch:
        checkpoint = model.state_dict()
        save_path = checkpoint_dir / f"{run_name}.pt"
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    if wandb_run is not None:
        wandb.finish()

    return checkpoint


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total
