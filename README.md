# LayerTRAK

Layer-selective data attribution with [TRAK](https://github.com/MadryLab/trak) on CIFAR-10.

- **Documentation (PDF):** [docs/final_doc.pdf](docs/final_doc.pdf)
- **Literature analysis:** [docs/literature_analysis.md](docs/literature_analysis.md)
- **Github repository:** <https://github.com/Matthev00/LayerTRAK>

LayerTRAK investigates how restricting TRAK to a chosen subset of a network's parameters affects the quality of training-data attribution. Experiments run on **CIFAR-10** across three architectures - **ResNet-18**, **ResNet-34**, and **MobileNetV2** - each evaluated under five layer configurations: `head_only`, `late`, `mid_late`, `early`, and `full_model`. Attribution quality is measured with **LDS (Linear Datamodeling Score)**.

## Layer configurations

| Config | ResNet-18 / ResNet-34 | MobileNetV2 |
|---|---|---|
| `head_only` | `fc` | `classifier` |
| `late` | `layer4` + `fc` | `features[14:19]` + `classifier` |
| `mid_late` | `layer3` | `features[7:14]` |
| `early` | `conv1` + `layer1` | `features[0:7]` |
| `full_model` | all parameters | all parameters |

## Project structure

```
src/layertrak/
├── settings.py          # Pydantic-Settings config (reads from .env)
├── data.py              # CIFAR-10 data pipeline
├── experiments/         # TRAK runner (LayerTRAKRunner) + training scripts
├── models/              # model_factory, ResNet/MobileNetV2 wrappers, layers.py
├── lds/                 # mask generation, ensemble training helpers, LDS computation
└── visualization/       # plot_trak_results - all figures from the paper
```

Key files:
- `experiments/trak.py` - `LayerTRAKRunner` wraps `TRAKer` and exposes the `grad_wrt` parameter that controls which layers are traced.
- `models/layers.py` - defines `RESNET_LAYER_CONFIGS` / `MOBILENETV2_LAYER_CONFIGS` and `get_grad_wrt()` which maps a config name to the exact parameter list.
- `lds/compute_lds.py` - computes Spearman rank correlation between predicted and actual model margins.

## Quickstart

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv), CUDA GPU recommended.

```bash
make install          # create venv and install dependencies
make train            # fine-tune ResNet-18, ResNet-34, MobileNetV2 on CIFAR-10
make masks            # generate 40 random subset masks for LDS
make ensemble-train-all   # train 40 auxiliary models per architecture
make extract-ensemble-all # extract logits from the ensemble
make experiment       # run TRAK for all architectures × all layer configs
make compute-lds      # compute LDS scores
make visualize-trak   # generate all plots
```

## Configuration

All settings live in `src/layertrak/settings.py` and can be overridden via a `.env` file or environment variables:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` / `cpu` | Training device |
| `NUM_EPOCHS` | `30` | Base model training epochs |
| `BATCH_SIZE` | `128` | Batch size |
| `TRAK_PROJ_DIM` | `1024` | Random projection dimension |
| `TRAK_LAMBDA_REG` | `1e-3` | TRAK regularization λ |
| `NUM_TARGETS` | `1000` | Test samples evaluated for TRAK scores |

## Results summary

`head_only` achieves the highest LDS across all three architectures. Adding earlier layers consistently degrades attribution quality.

| Architecture | head_only | late | full_model | mid_late | early |
|---|---|---|---|---|---|
| ResNet-18 | **0.0863** | 0.0582 | 0.0445 | 0.0329 | 0.0138 |
| ResNet-34 | **0.0910** | 0.0567 | 0.0342 | 0.0209 | 0.0026 |
| MobileNetV2 | **0.0953** | 0.0703 | 0.0368 | 0.0316 | 0.0064 |
