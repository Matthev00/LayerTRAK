import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from layertrak.settings import settings


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plotting script."""
    parser = argparse.ArgumentParser(
        description="Generate plots for a saved LayerTRAK experiment run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific run directory. Defaults to the newest directory in trak_results.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=25,
        help="Maximum number of target rows shown in score heatmaps.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=100,
        help="Maximum number of training columns shown in score heatmaps.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=80,
        help="Number of bins used in score histograms.",
    )
    return parser.parse_args()


def _resolve_run_dir(run_dir: Path | None) -> Path:
    """Return the selected run directory or the newest available run."""
    if run_dir is not None:
        return run_dir

    results_root = settings.project_root / settings.trak_results_dir
    candidates = sorted(path for path in results_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {results_root}")
    return candidates[-1]


def _load_overview(path: Path) -> list[dict[str, str]]:
    """Load the run overview CSV file."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_summary(path: Path) -> dict[str, object]:
    """Load the summary metadata for a single configuration."""
    return json.loads(path.read_text(encoding="utf-8"))


def _make_output_dir(run_dir: Path) -> Path:
    """Create and return the directory used for generated plots."""
    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _plot_overview_metric(rows: list[dict[str, str]], metric: str, output_dir: Path) -> None:
    """Plot one summary metric across architectures and layer configs."""
    architectures = sorted({row["architecture"] for row in rows})
    configs = sorted({row["config_name"] for row in rows})
    x = np.arange(len(architectures))
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, config_name in enumerate(configs):
        values = []
        for arch in architectures:
            matching = next(row for row in rows if row["architecture"] == arch and row["config_name"] == config_name)
            values.append(float(matching[metric]))
        ax.bar(x + idx * width - (len(configs) - 1) * width / 2, values, width=width, label=config_name)

    ax.set_title(f"{metric} by architecture and layer config")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"overview_{metric}.png", dpi=180)
    plt.close(fig)


def _plot_num_params(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Plot the number of tracked parameters for each configuration."""
    architectures = sorted({row["architecture"] for row in rows})
    configs = sorted({row["config_name"] for row in rows})
    x = np.arange(len(architectures))
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, config_name in enumerate(configs):
        values = []
        for arch in architectures:
            matching = next(row for row in rows if row["architecture"] == arch and row["config_name"] == config_name)
            values.append(float(matching["num_tracked_params"]))
        ax.bar(x + idx * width - (len(configs) - 1) * width / 2, values, width=width, label=config_name)

    ax.set_title("Tracked parameters by architecture and layer config")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylabel("num_tracked_params")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "overview_num_tracked_params.png", dpi=180)
    plt.close(fig)


def _plot_histogram(scores: np.ndarray, output_path: Path, bins: int, title: str) -> None:
    """Plot the distribution of TRAK scores for one configuration."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores.ravel(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel("TRAK score")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(scores: np.ndarray, output_path: Path, title: str, max_targets: int, max_train_samples: int) -> None:
    """Plot a cropped heatmap of the score matrix."""
    cropped = scores[:max_targets, :max_train_samples]
    vmax = float(np.max(np.abs(cropped)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax > 0 else None

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(cropped, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set_title(title)
    ax.set_xlabel("train sample index")
    ax.set_ylabel("target index")
    fig.colorbar(image, ax=ax, label="TRAK score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_topk_curve(scores: np.ndarray, output_path: Path, title: str) -> None:
    """Plot top-score trends across target samples."""
    top1_scores = np.max(scores, axis=1)
    top10_mean = np.mean(np.sort(scores, axis=1)[:, -10:], axis=1)
    target_indices = np.arange(scores.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(target_indices, top1_scores, label="top1 score", linewidth=1.5)
    ax.plot(target_indices, top10_mean, label="mean top10 score", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("target index")
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_config_artifacts(
    run_dir: Path,
    output_dir: Path,
    max_targets: int,
    max_train_samples: int,
    hist_bins: int,
) -> None:
    """Generate per-configuration plots for all saved experiment artifacts."""
    for summary_path in sorted(run_dir.glob("*/*/summary.json")):
        config_dir = summary_path.parent
        summary = _load_summary(summary_path)
        architecture = str(summary["architecture"])
        config_name = str(summary["config_name"])
        scores = np.load(config_dir / "scores.npy")

        config_output_dir = output_dir / architecture / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)

        title_prefix = f"{architecture} / {config_name}"
        _plot_histogram(
            scores,
            config_output_dir / "score_histogram.png",
            bins=hist_bins,
            title=f"{title_prefix} score distribution",
        )
        _plot_heatmap(
            scores,
            config_output_dir / "score_heatmap.png",
            title=f"{title_prefix} score heatmap",
            max_targets=max_targets,
            max_train_samples=max_train_samples,
        )
        _plot_topk_curve(
            scores,
            config_output_dir / "topk_curve.png",
            title=f"{title_prefix} top influence by target",
        )


def generate_plots(
    *,
    run_dir: Path | None = None,
    max_targets: int = 25,
    max_train_samples: int = 100,
    hist_bins: int = 80,
) -> Path:
    """Generate all plots for one saved LayerTRAK run."""
    resolved_run_dir = _resolve_run_dir(run_dir)
    overview_path = resolved_run_dir / "overview.csv"
    if not overview_path.exists():
        raise FileNotFoundError(f"Missing overview.csv in {resolved_run_dir}")

    rows = _load_overview(overview_path)
    output_dir = _make_output_dir(resolved_run_dir)

    _plot_num_params(rows, output_dir)
    for metric in ("mean", "std", "p95", "p99"):
        _plot_overview_metric(rows, metric, output_dir)

    _plot_config_artifacts(
        resolved_run_dir,
        output_dir,
        max_targets=max_targets,
        max_train_samples=max_train_samples,
        hist_bins=hist_bins,
    )
    return output_dir


def main() -> None:
    """Run the plotting script from the command line."""
    args = _parse_args()
    output_dir = generate_plots(
        run_dir=args.run_dir,
        max_targets=args.max_targets,
        max_train_samples=args.max_train_samples,
        hist_bins=args.hist_bins,
    )
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
