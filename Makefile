# Windows sets OS=Windows_NT by default; nothing sets it on Linux/macOS
ifeq ($(OS),Windows_NT)
    PYTHON  := .venv\Scripts\python.exe
    PIP     := .venv\Scripts\pip.exe
    MATURIN := .venv\Scripts\maturin.exe
    RM      := rmdir /s /q
    SEP     := \\
    SET_MPL := set "MPLBACKEND=Agg" &&
else
    PYTHON  := .venv/bin/python
    PIP     := .venv/bin/pip
    MATURIN := .venv/bin/maturin
    RM      := rm -rf
    SEP     := /
    SET_MPL := MPLBACKEND=Agg
endif

.PHONY: build train eval pretrain benchmark run docs setup clean help vizdoom-train vizdoom-watch vizdoom-clean vizdoom-defend vizdoom-defend-watch vizdoom-transfer vizdoom-live rl-record rl-train-bc rl-train-rl rl-watch rl-clean

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
	@echo "  vizdoom-train  train the ViZDoom DQN agent (headless, 1000 episodes)"
	@echo "  vizdoom-watch  load weights and watch the agent play (visual window)"
	@echo "  vizdoom-clean  delete saved ViZDoom weights"
	@echo ""
	@echo "  vizdoom-defend        train defend_the_center (headless, 2000 episodes)"
	@echo "  vizdoom-defend-watch  watch the defend agent play (visual window)"
	@echo "  vizdoom-transfer      transfer benchmark: basic -> defend"
	@echo "  vizdoom-live          watch the agent learn live in a Doom window (3000 episodes)"
	@echo ""
	@echo "  rl-record      record mode -- you play Rocket League (rlgym_sim), Ashby captures"
	@echo "  rl-train-bc    behavioral cloning from recorded sessions -> rl_imitation.pth"
	@echo "  rl-train-rl    autonomous RL vs a scripted bot (10000 episodes) -> rl_policy.pth"
	@echo "  rl-watch       watch the trained policy play 1v1"
	@echo "  rl-clean       delete captured recording sessions"
	@echo ""

setup:
	@echo "Setting up Ashby environment..."
	@python -m venv .venv
	@$(PIP) install --quiet maturin numpy matplotlib vizdoom
	@$(PIP) install --quiet torch --index-url https://download.pytorch.org/whl/cpu
	@$(PIP) install --quiet rlgym rlgym-rocket-league RocketSim pygame-ce
	@$(PIP) install --quiet git+https://github.com/AechPro/rocket-league-gym-sim
	@echo "Setup complete. Run 'make run' to build and train."

build:
	@echo "Building Ashby bridge..."
	@$(MATURIN) develop --manifest-path bridge/Cargo.toml

train:
	@echo "Training Ashby..."
	@$(SET_MPL) $(PYTHON) mind/train.py

eval:
	@echo "Evaluating Ashby..."
	@$(SET_MPL) $(PYTHON) mind/eval.py

pretrain:
	@echo "Pretraining Ashby (sequential transfer)..."
	@$(SET_MPL) $(PYTHON) mind/pretrain.py

benchmark:
	@echo "Running transfer vs scratch benchmark..."
	@$(SET_MPL) $(PYTHON) mind/benchmark.py

run: build pretrain benchmark

# Python is always available after setup — use it instead of shell-specific stat commands
docs:
	@echo "Checking documentation files..."
	@$(PYTHON) -c "import sys,os; files=['README.md','CONTRIBUTING.md','docs/environments.md','docs/transfer_learning.md','docs/architecture.md','docs/results.md']; [print('  ok:',f) if os.path.isfile(f) and os.path.getsize(f)>0 else sys.exit('  MISSING or empty: '+f) for f in files]; print('All documentation files present.')"

vizdoom-train:
	@echo "Training ViZDoom agent (headless, 1000 episodes)..."
	@$(SET_MPL) $(PYTHON) mind/vizdoom_train.py

vizdoom-watch:
	@echo "Launching ViZDoom watch mode..."
	@$(PYTHON) mind/vizdoom_watch.py

vizdoom-defend:
	@echo "Training ViZDoom defend_the_center agent (headless, 2000 episodes)..."
	@$(SET_MPL) $(PYTHON) mind/vizdoom_defend.py

vizdoom-defend-watch:
	@echo "Launching ViZDoom defend_the_center watch mode..."
	@$(PYTHON) mind/vizdoom_defend_watch.py

vizdoom-transfer:
	@echo "Running transfer benchmark: basic -> defend_the_center..."
	@$(SET_MPL) $(PYTHON) mind/vizdoom_transfer.py

vizdoom-live:
	@echo "Launching ViZDoom live training (window open, 3000 episodes)..."
	@$(PYTHON) mind/vizdoom_live.py

vizdoom-clean:
	@echo "Cleaning ViZDoom weights..."
	@$(PYTHON) -c "import os; p='mind/weights/vizdoom_basic.pth'; os.remove(p) if os.path.exists(p) else print('  nothing to clean')"

rl-record:
	@echo "Recording mode -- play now, Ashby is capturing (Ctrl+C to stop and save)..."
	@$(PYTHON) mind/rl/record.py

rl-train-bc:
	@echo "Behavioral cloning from recorded sessions..."
	@$(SET_MPL) $(PYTHON) mind/rl/imitation.py

rl-train-rl:
	@echo "Autonomous RL training vs a scripted bot (10000 episodes)..."
	@$(SET_MPL) $(PYTHON) mind/rl/rl_train.py

rl-watch:
	@echo "Launching RLGym watch mode..."
	@$(PYTHON) mind/rl/watch.py

rl-clean:
	@echo "Cleaning captured RLGym sessions..."
	@$(PYTHON) -c "import os,glob; files=glob.glob('mind/rl/data/session_*.pkl'); [os.remove(f) for f in files]; print(f'  removed {len(files)} session file(s)') if files else print('  nothing to clean')"

clean:
	@echo "Cleaning up..."
	@cargo clean
ifeq ($(OS),Windows_NT)
	@$(PYTHON) -c "import os,shutil; [shutil.rmtree(os.path.join(r,d),ignore_errors=True) for r,ds,_ in os.walk('.') for d in ds if d=='__pycache__']"
else
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	@find . -name "*.pyc" -delete 2>/dev/null; true
endif
	@echo "Done."
