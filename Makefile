.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running ty"
	@uv run ty check src/
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry src

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: train
train: ## Train all base models and save checkpoints
	@echo "🚀 Training base models"
	@uv run python -m layertrak.experiments.train_base_models

.PHONY: masks
NUM_MASKS ?= 40
TRAIN_SIZE ?= 50000
SUBSET_RATIO ?= 0.5
SEED ?= 42
OUTPUT_DIR ?=

masks: ## Generate universal LDS masks for ensemble training
	@echo "🚀 Generating LDS masks"
	@uv run python -m layertrak.lds.generate_masks \
		--num-masks $(NUM_MASKS) \
		--train-size $(TRAIN_SIZE) \
		--subset-ratio $(SUBSET_RATIO) \
		--seed $(SEED) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

.PHONY: ensemble-train
ensemble-train: ## Train ensemble of LDS-masked models (usage: make ensemble-train ARCH=resnet18 NUM_MASKS=20)
	@echo "🚀 Training ensemble models"
	@uv run python -m layertrak.experiments.train_ensemble $(if $(ARCH),--architecture $(ARCH)) $(if $(NUM_MASKS),--num-masks $(NUM_MASKS))

.PHONY: ensemble-train-all
ensemble-train-all: ## Train all architectures sequentially (resnet18 → resnet34 → mobilenetv2)
	@echo "🚀 Training ensemble: all architectures sequentially (NUM_MASKS=$(NUM_MASKS))"
	@make ensemble-train ARCH=resnet18 NUM_MASKS=$(NUM_MASKS) && \
	 make ensemble-train ARCH=resnet34 NUM_MASKS=$(NUM_MASKS) && \
	 make ensemble-train ARCH=mobilenetv2 NUM_MASKS=$(NUM_MASKS)
	@echo "✓ All ensemble training complete!"

.PHONY: extract-ensemble
ARCH ?= resnet18
NUM_MASKS ?= 40
extract-ensemble: ## Extract logits from ensemble models (usage: make extract-ensemble ARCH=resnet18 NUM_MASKS=40)
	@echo "🚀 Extracting ensemble model outputs"
	@uv run python -m layertrak.lds.extract_ensemble_outputs --architecture $(ARCH) --num-masks $(NUM_MASKS)

.PHONY: extract-ensemble-all
NUM_MASKS ?= 40
extract-ensemble-all: ## Extract logits from all ensemble architectures (resnet18 → resnet34 → mobilenetv2)
	@echo "🚀 Extracting ensemble outputs: all architectures sequentially (NUM_MASKS=$(NUM_MASKS))"
	@make extract-ensemble ARCH=resnet18 NUM_MASKS=$(NUM_MASKS) && \
	 make extract-ensemble ARCH=resnet34 NUM_MASKS=$(NUM_MASKS) && \
	 make extract-ensemble ARCH=mobilenetv2 NUM_MASKS=$(NUM_MASKS)
	@echo "✓ All ensemble extraction complete!"

.PHONY: compute-lds
TRAK_DIR ?=
compute-lds: ## Compute LDS scores for all architectures and layer configurations (usage: make compute-lds TRAK_DIR=trak_results/2026-...)
	@echo "🚀 Computing LDS scores"
	@uv run python -m layertrak.lds.compute_lds $(if $(TRAK_DIR),--trak-dir $(TRAK_DIR),)

.PHONY: experiment
NUM_EPOCHS ?=
experiment: ## Run TRAK experiment for all models and layer configs (usage: make experiment NUM_EPOCHS=5; use experiment-arch for single arch)
	@echo "🚀 Running TRAK experiment $(if $(NUM_EPOCHS),with $(NUM_EPOCHS) last epochs,with last epoch only)"
	@uv run python -m layertrak.experiments.run_trak $(if $(NUM_EPOCHS),--num-last-epochs $(NUM_EPOCHS),)

.PHONY: experiment-arch
ARCH ?= resnet18
NUM_EPOCHS ?=
experiment-arch: ## Run TRAK experiment for a specific architecture (usage: make experiment-arch ARCH=resnet18 NUM_EPOCHS=5)
	@echo "🚀 Running TRAK experiment for $(ARCH) $(if $(NUM_EPOCHS),with $(NUM_EPOCHS) last epochs,with last epoch only)"
	@uv run python -m layertrak.experiments.run_trak --architecture $(ARCH) $(if $(NUM_EPOCHS),--num-last-epochs $(NUM_EPOCHS),)

.PHONY: visualize-trak
visualize-trak: ## Generate plots for TRAK results. Optional: RUN_DIR=trak_results/2026-...
	@echo "🚀 Visualizing TRAK results"
	@uv run python -m layertrak.visualization.plot_trak_results $(if $(RUN_DIR),--run-dir $(RUN_DIR),)

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
