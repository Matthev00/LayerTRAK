from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from layertrak.settings import settings

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, test_transform) for CIFAR-10."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform


def get_cifar10_datasets() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Download and return (train_dataset, test_dataset)."""
    train_transform, test_transform = get_cifar10_transforms()
    data_dir = str(settings.project_root / settings.data_dir)
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, test_loader, targets_loader).

    targets_loader is a subset of test set used for TRAK scoring.
    """
    train_dataset, test_dataset = get_cifar10_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    targets_dataset = Subset(test_dataset, list(range(settings.num_targets)))
    targets_loader = DataLoader(
        targets_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    return train_loader, test_loader, targets_loader
