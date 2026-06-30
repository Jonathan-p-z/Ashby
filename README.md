# Ashby

![Rust](https://img.shields.io/badge/Rust-1.78+-orange?logo=rust)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

A reinforcement learning agent that learns to navigate five radically different environments — then transfers that knowledge so it learns new ones faster.

Named after [William Ross Ashby](https://en.wikipedia.org/wiki/W._Ross_Ashby) and his homeostatic principle: a system that adapts to maintain viability across changing conditions.

---

## What it does

Five environments, five mechanics, one agent. Ashby trains a DQN whose hidden layer learns generic navigation features — spatial reasoning, threat avoidance, multi-objective planning. Those features transfer: fine-tuning on MazeWorld with frozen base layers converges **8x faster** than training from scratch.

The negative results are just as interesting. Transfer hurts on environments with fixed random maps (FrozenLake, PredatorGrid) because the frozen input layer encodes features for the pretrain instance, not the benchmark instance. That's not a bug — it's an honest finding about when transfer learning works and when it doesn't.

---

## Pipeline

```
  Rust environments (games/)
  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐
  │  GridWorld   │  │  FrozenLake  │  │ PredatorGrid  │  │  MazeWorld   │  │ ResourceHunter  │
  │  5×5, det.   │  │  5×5, stoch. │  │  5×5, advers. │  │  7×7, proc.  │  │  7×7, multi-obj │
  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘  └────────┬────────┘
         └─────────────────┴─────────────────────┘               └─────────────────────┘
                                        │ PyO3 bridge (bridge/)
                                        ▼
                              Python agents (mind/)
                         ┌────────────────────────────┐
                         │  QLearningAgent  (tabular)  │
                         │  DQNAgent        (neural)   │
                         │  TransferAgent   (transfer) │
                         └─────────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                   ▼
             training_results.png  weights/*.pth  transfer_benchmark.png
```

---

## Environments

| Name | Mechanic | State | Agent | Result |
|---|---|---|---|---|
| GridWorld | Reach fixed target, 5×5 grid | 4D (positions) | Tabular Q | 100%, ep 101 |
| FrozenLake | Navigate ice + holes, stochastic slip | 5D (pos + tile type) | Tabular Q | 97%, ep ~391 |
| PredatorGrid | Reach center, evade greedy predator | 6D (agent + target + predator) | Tabular Q | 96%, ep ~484 |
| MazeWorld | New random 7×7 maze every episode | 53D (pos + full grid) | DQN | 91%, ep ~883 |
| ResourceHunter | Collect 5 resources before starving | 18D (pos + hunger + 5 resources) | DQN | converges ~ep 4000 |

All environments share the same interface: `reset() -> State`, `step(action) -> (State, Reward, done)`.

---

## Transfer learning results

Pretrain one agent sequentially on all five environments, then benchmark: scratch vs. fine-tuning with frozen base layers.

| Environment | Scratch | Transfer | Gain |
|---|---|---|---|
| GridWorld | 101 | 101 | 0 |
| FrozenLake | 681 | 904 | -223 |
| PredatorGrid | 578 | 699 | -121 |
| **MazeWorld** | **811** | **101** | **+710** |
| ResourceHunter | >2500 | >2500 | 0 |

**MazeWorld (+710 episodes, 8x speedup)** — this is the cleanest result. Every episode generates a new random maze, so pretrained features are genuinely distribution-general. With input+hidden layers frozen, only the 64→4 output head retrains. That's a trivial optimization problem when the feature extractor is already good.

**FrozenLake and PredatorGrid (-223, -121)** — the frozen input layer learned to encode a specific map instance during pretraining. The benchmark runs on a different random instance. Freezing the wrong encoder is worse than starting fresh. The negative results aren't hidden — they tell you something real about the limits of transfer learning on fixed-instance environments.

**GridWorld (0)** — too simple. 101 episodes is already fast enough that there's nothing to gain.

**ResourceHunter (0)** — needs ~5000 episodes in the full pipeline. Pretrain allocates 4000 and the task is hard enough that neither agent converges in 2500.

See [docs/transfer_learning.md](docs/transfer_learning.md) for the full analysis.

---

## Getting started

**Prerequisites:** Rust (stable), Python 3.12, `cargo`

```bash
git clone <repo>
cd Ashby

# Create .venv and install all Python deps (once)
make setup

# Full transfer learning demo: build → pretrain → benchmark
make run

# Or just the original training pipeline
make train

# Eval table (fast, no plots)
make eval
```

**What you get after `make run`:**
- `mind/weights/` — pretrained checkpoints per environment
- `mind/transfer_benchmark.png` — scratch vs transfer learning curves side by side

**Individual targets:**
```bash
make build      # compile Rust bridge (needed before any Python script)
make pretrain   # sequential pretraining, saves weights
make benchmark  # scratch vs transfer comparison
make train      # original full training pipeline
make eval       # quick evaluation table
make help       # list all targets
```

---

## File structure

```
Ashby/
├── Cargo.toml                  # Rust workspace
├── Makefile                    # all build and training commands
│
├── core/                       # shared Rust types
│   └── src/
│       ├── environment.rs      # Environment trait (the contract all envs fulfill)
│       └── types.rs            # State, Action, Reward type aliases
│
├── games/                      # the five environments (Rust)
│   └── src/
│       ├── grid.rs             # GridWorld — 5×5 deterministic navigation
│       ├── frozen_lake.rs      # FrozenLake — stochastic ice, fixed holes
│       ├── predator_grid.rs    # PredatorGrid — adversarial greedy predator
│       ├── maze_world.rs       # MazeWorld — procedural maze, new layout every ep
│       └── resource_hunter.rs  # ResourceHunter — multi-objective under time pressure
│
├── bridge/                     # PyO3 layer
│   └── src/lib.rs              # exposes all envs to Python via maturin
│
├── mind/                       # Python — agents and training
│   ├── agent.py                # QLearningAgent — tabular Q-Learning
│   ├── dqn_agent.py            # DQNAgent — Deep Q-Network with target network
│   ├── transfer_agent.py       # TransferAgent — save/load/freeze for transfer
│   ├── memory.py               # ReplayBuffer — experience replay
│   ├── train.py                # full training pipeline, generates training_results.png
│   ├── eval.py                 # evaluation table across all environments
│   ├── pretrain.py             # sequential pretraining, saves mind/weights/*.pth
│   ├── benchmark.py            # scratch vs transfer benchmark, generates transfer_benchmark.png
│   └── weights/                # pretrained checkpoints (created by make pretrain)
│       ├── gridworld.pth
│       ├── frozen_lake.pth
│       ├── predator_grid.pth
│       ├── maze_world.pth
│       └── resource_hunter.pth
│
└── docs/
    ├── environments.md         # per-environment design decisions and mechanics
    ├── transfer_learning.md    # how transfer works here, why the results look like they do
    ├── architecture.md         # Rust trait, PyO3 bridge, DQN, TransferAgent internals
    └── results.md              # all plots, tables, and what the curves tell you
```

---

## References

- W. Ross Ashby, *Design for a Brain* (1952) — homeostasis, adaptive behavior, essential variables
- R. Bellman, *Dynamic Programming* (1957) — the Bellman equation underlying Q-learning
- V. Mnih et al., *Human-level control through deep reinforcement learning* (2015, Nature) — DQN, target network, experience replay
- A. Lazaric, *Transfer in Reinforcement Learning: a Framework and a Survey* (2012) — when and why transfer works in RL
