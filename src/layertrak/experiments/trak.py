import copy
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from trak import TRAKer
from trak.score_computers import BasicScoreComputer

from layertrak.settings import settings


class RobustScoreComputer(BasicScoreComputer):
    """Score computer that falls back to pseudo-inverse for singular X^T X."""

    def get_x_xtx_inv(self, grads: torch.Tensor, xtx: torch.Tensor) -> torch.Tensor:
        blocks = torch.split(grads, split_size_or_sections=self.CUDA_MAX_DIM_SIZE, dim=0)

        xtx_float = xtx.to(torch.float32)
        xtx_reg = xtx_float + self.lambda_reg * torch.eye(
            xtx_float.size(dim=0),
            device=xtx_float.device,
            dtype=xtx_float.dtype,
        )
        xtx_reg = torch.nan_to_num(xtx_reg, nan=0.0, posinf=1e6, neginf=-1e6)

        try:
            xtx_inv = torch.linalg.inv(xtx_reg)
        except torch._C._LinAlgError:
            xtx_inv = torch.linalg.pinv(xtx_reg, hermitian=True)

        scale = xtx_inv.abs().mean()
        if torch.isfinite(scale) and scale > 0:
            xtx_inv = xtx_inv / scale

        xtx_inv = xtx_inv.to(self.dtype)
        result = torch.empty(grads.shape[0], xtx_inv.shape[1], dtype=self.dtype, device=self.device)

        for i, block in enumerate(blocks):
            start = i * self.CUDA_MAX_DIM_SIZE
            end = min(grads.shape[0], (i + 1) * self.CUDA_MAX_DIM_SIZE)
            result[start:end] = block.to(self.device) @ xtx_inv

        return result


class LayerTRAKRunner:
    """Wrapper around TRAKer that handles device switching.

    TRAK internally calls pin_memory() which only works with CUDA.
    On MPS, this wrapper moves model and data to CPU automatically.
    On CUDA, everything stays on the original device.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint: dict[str, torch.Tensor],
        train_set_size: int,
        *,
        save_dir: Path | str,
        device: str | None = None,
        proj_dim: int | None = None,
        lambda_reg: float | None = None,
        grad_wrt: list[str] | None = None,
    ):
        """Initialize the TRAK runner.

        Args:
            model: The trained model.
            checkpoint: Model state_dict.
            train_set_size: Number of training samples.
            save_dir: Directory for TRAK intermediate results.
            device: Training device. If MPS, TRAK runs on CPU instead.
            proj_dim: Random projection dimension. Defaults to settings.
            lambda_reg: Diagonal regularization for TRAK's X^T X inverse.
            grad_wrt: List of parameter names to compute gradients for.
        """
        self._save_dir = Path(save_dir)
        self._proj_dim = proj_dim or settings.trak_proj_dim
        self._lambda_reg = settings.trak_lambda_reg if lambda_reg is None else lambda_reg

        source_device = device or settings.device
        self._trak_device = "cpu" if source_device == "mps" else source_device
        self._needs_cpu_cast = source_device == "mps"

        trak_model = copy.deepcopy(model).to(self._trak_device)
        self._checkpoint: dict[str, torch.Tensor] = {k: v.to(self._trak_device) for k, v in checkpoint.items()}
        trak_model.load_state_dict(self._checkpoint)
        trak_model.eval()

        shutil.rmtree(self._save_dir, ignore_errors=True)

        self._traker = TRAKer(
            model=trak_model,
            task="image_classification",
            train_set_size=train_set_size,
            save_dir=str(self._save_dir),
            device=self._trak_device,
            proj_dim=self._proj_dim,
            grad_wrt=grad_wrt,
            lambda_reg=self._lambda_reg,
            score_computer=RobustScoreComputer,
        )

    def _cast_batch(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
        target_device = "cpu" if self._needs_cpu_cast else self._trak_device
        return [x.to(target_device, non_blocking=True) for x in batch]

    def featurize(self, train_loader: DataLoader) -> None:
        """Featurize the entire training set."""
        self._traker.load_checkpoint(self._checkpoint, model_id=0)  # type: ignore[arg-type]
        for batch in tqdm(train_loader, desc="Featurizing", leave=False):
            batch = self._cast_batch(batch)
            self._traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        self._traker.finalize_features()

    def score(
        self,
        targets_loader: DataLoader,
        num_targets: int,
        *,
        exp_name: str = "default",
    ) -> np.ndarray:
        """Score target samples against the featurized training set.

        Args:
            targets_loader: DataLoader over target samples.
            num_targets: Number of target samples.
            exp_name: Experiment name for TRAK bookkeeping.

        Returns:
            Score matrix of shape (num_targets, train_set_size).
        """
        self._traker.start_scoring_checkpoint(
            exp_name=exp_name,
            checkpoint=self._checkpoint,  # type: ignore[arg-type]
            model_id=0,
            num_targets=num_targets,
        )
        for batch in tqdm(targets_loader, desc="Scoring", leave=False):
            batch = self._cast_batch(batch)
            self._traker.score(batch=batch, num_samples=batch[0].shape[0])
        return np.asarray(self._traker.finalize_scores(exp_name=exp_name))
