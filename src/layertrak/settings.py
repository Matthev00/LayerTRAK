from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global project settings loaded from env / .env file."""
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path("data")
    trak_results_dir: Path = Path("trak_results")
    checkpoints_dir: Path = Path("checkpoints")

    device: str = "cpu"
    batch_size: int = 128
    num_workers: int = 0
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    momentum: float = 0.9

    trak_proj_dim: int = 1024
    num_targets: int = 1000

    wandb_project: str = "LayerTRAK"
    wandb_enabled: bool = True


settings = Settings()
