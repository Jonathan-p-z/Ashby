VENV    := .venv
PYTHON  := $(VENV)/bin/python
MATURIN := $(VENV)/bin/maturin
PIP     := $(VENV)/bin/pip

.PHONY: build train eval pretrain benchmark run docs setup clean help

help:
	@echo ""
	@echo "  Ashby — reinforcement learning across different worlds"
	@echo ""
	@echo "  setup      create .venv and install all dependencies (run once)"
	@echo "  build      compile the Rust bridge (PyO3 -> Python)"
	@echo "  train      train Ashby on all environments (original pipeline)"
	@echo "  eval       quick retrain + evaluation table"
	@echo "  pretrain   sequential pretraining for transfer learning, saves weights"
	@echo "  benchmark  scratch vs transfer comparison, generates transfer_benchmark.png"
	@echo "  run        build + pretrain + benchmark (transfer learning demo)"
	@echo "  docs       verify all documentation files exist and are non-empty"
	@echo "  clean      remove Rust artifacts and Python cache"
	@echo ""

setup:
	@echo "Setting up Ashby environment..."
	@python3 -m venv $(VENV)
	@$(PIP) install --quiet maturin numpy matplotlib
	@$(PIP) install --quiet torch --index-url https://download.pytorch.org/whl/cpu
	@echo "Setup complete. Run 'make run' to build and train."

build:
	@echo "Building Ashby bridge..."
	@$(MATURIN) develop --manifest-path bridge/Cargo.toml

train:
	@echo "Training Ashby..."
	@MPLBACKEND=Agg $(PYTHON) mind/train.py

eval:
	@echo "Evaluating Ashby..."
	@MPLBACKEND=Agg $(PYTHON) mind/eval.py

pretrain:
	@echo "Pretraining Ashby (sequential transfer)..."
	@MPLBACKEND=Agg $(PYTHON) mind/pretrain.py

benchmark:
	@echo "Running transfer vs scratch benchmark..."
	@MPLBACKEND=Agg $(PYTHON) mind/benchmark.py

run: build pretrain benchmark

docs:
	@echo "Checking documentation files..."
	@for f in README.md CONTRIBUTING.md docs/environments.md docs/transfer_learning.md docs/architecture.md docs/results.md; do \
		if [ ! -s "$$f" ]; then \
			echo "  MISSING or empty: $$f"; \
			exit 1; \
		fi; \
		echo "  ok: $$f"; \
	done
	@echo "All documentation files present."

clean:
	@echo "Cleaning up..."
	@cargo clean
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	@find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "Done."
