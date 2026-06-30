"""Transfer learning benchmark — does pretraining make Ashby learn faster?

For each environment, two agents race to 75% success:
  Scratch   — random weights, all layers trainable, no prior knowledge
  Transfer  — pretrained weights, base layers frozen, only output head fine-tunes

Key question: does the hidden layer's accumulated navigation knowledge
give a meaningful head start over learning everything from zero?
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ashby
from transfer_agent import TransferAgent

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
CONVERGENCE_THRESHOLD = 0.75
CONVERGENCE_WINDOW = 100
MAX_STEPS = 300

# Each env uses its own pretrained checkpoint — full state + hidden layer transfer
# when dimensionality matches.
ENV_CONFIGS = [
    {
        "name": "GridWorld",
        "env_fn": ashby.PyGridWorld,
        "weights": "gridworld.pth",
        "n_episodes": 600,
        "epsilon_decay": 0.995,
    },
    {
        "name": "FrozenLake",
        "env_fn": ashby.PyFrozenLake,
        "weights": "frozen_lake.pth",
        "n_episodes": 1000,
        "epsilon_decay": 0.997,
    },
    {
        "name": "PredatorGrid",
        "env_fn": ashby.PyPredatorGrid,
        "weights": "predator_grid.pth",
        "n_episodes": 1500,
        "epsilon_decay": 0.997,
    },
    {
        "name": "MazeWorld",
        "env_fn": ashby.PyMazeWorld,
        "weights": "maze_world.pth",
        "n_episodes": 2000,
        "epsilon_decay": 0.997,
    },
    {
        "name": "ResourceHunter",
        "env_fn": ashby.PyResourceHunter,
        "weights": "resource_hunter.pth",
        "n_episodes": 2500,
        "epsilon_decay": 0.999,
    },
]


def run_agent(env, agent: TransferAgent, n_episodes: int) -> tuple[list, int | None]:
    """Train agent, return episode success curve and first convergence episode."""
    successes = []
    converged_at = None

    for ep in range(n_episodes):
        state = env.reset()
        terminal_reward = 0.0
        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                terminal_reward = reward
                break
        agent.decay_epsilon()
        successes.append(int(terminal_reward > 0))

        if converged_at is None and ep >= CONVERGENCE_WINDOW:
            if np.mean(successes[-CONVERGENCE_WINDOW:]) >= CONVERGENCE_THRESHOLD:
                converged_at = ep + 1

    return successes, converged_at


def plot_benchmark(results: list, output_path: str):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n))
    fig.suptitle("Ashby — Transfer Learning: Scratch vs Pretrained", fontsize=13)
    if n == 1:
        axes = [axes]

    window = 60

    for i, r in enumerate(results):
        ax = axes[i]
        n_ep = r['n_episodes']

        def smooth(s):
            arr = np.array([float(x) for x in s])
            if len(arr) < window:
                return arr
            return np.convolve(arr, np.ones(window) / window, mode='valid')

        sc_smooth = smooth(r['scratch_successes'])
        tr_smooth = smooth(r['transfer_successes'])
        x_sc = range(window - 1, len(r['scratch_successes']))
        x_tr = range(window - 1, len(r['transfer_successes']))

        ax.plot(list(x_sc), sc_smooth, color='steelblue', linewidth=1.8, label='Scratch')
        ax.plot(list(x_tr), tr_smooth, color='darkorange', linewidth=1.8, label='Transfer')
        ax.axhline(CONVERGENCE_THRESHOLD, color='gray', linewidth=0.8,
                   linestyle=':', label=f'75% threshold')

        if r['scratch_conv']:
            ax.axvline(r['scratch_conv'], color='steelblue', linewidth=1.2,
                       linestyle='--', alpha=0.6, label=f"scratch: ep {r['scratch_conv']}")
        if r['transfer_conv']:
            ax.axvline(r['transfer_conv'], color='darkorange', linewidth=1.2,
                       linestyle='--', alpha=0.6, label=f"transfer: ep {r['transfer_conv']}")

        gain = (r['scratch_conv'] or n_ep) - (r['transfer_conv'] or n_ep)
        frozen_str = r['layers_frozen']
        title = f"{r['name']}  [frozen: {frozen_str}]"
        if gain > 0:
            title += f"  —  transfer {gain} episodes faster"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Success rate')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Episode')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved -> {output_path}")


def run_benchmark():
    print("\n" + "=" * 65)
    print("  ASHBY — TRANSFER vs SCRATCH BENCHMARK")
    print(f"  Metric: episodes to reach {CONVERGENCE_THRESHOLD:.0%} success (window={CONVERGENCE_WINDOW})")
    print("=" * 65)

    results = []

    for cfg in ENV_CONFIGS:
        name = cfg["name"]
        weights_path = os.path.join(WEIGHTS_DIR, cfg["weights"])

        if not os.path.exists(weights_path):
            print(f"\n  [{name}] weights not found: {weights_path}")
            print(f"  Run 'make pretrain' first.")
            sys.exit(1)

        print(f"\n  {name}  (max {cfg['n_episodes']} episodes each)")

        env = cfg["env_fn"]()
        state_size = env.state_shape()[0]
        action_size = len(env.action_space())

        # Both agents train on the same environment instance for a fair comparison.
        # (Critical for FrozenLake where the map is fixed at construction time.)

        print(f"    scratch ... ", end="", flush=True)
        scratch = TransferAgent(
            state_size=state_size,
            action_size=action_size,
            epsilon_decay=cfg["epsilon_decay"],
            target_update_freq=200,
        )
        scratch_s, scratch_conv = run_agent(env, scratch, cfg["n_episodes"])
        sc_str = str(scratch_conv) if scratch_conv else f">{cfg['n_episodes']}"
        print(f"converged ep {sc_str}")

        print(f"    transfer ... ", end="", flush=True)
        transfer = TransferAgent(
            state_size=state_size,
            action_size=action_size,
            epsilon_decay=cfg["epsilon_decay"],
            target_update_freq=200,
        )
        load_info = transfer.load_weights(weights_path)
        transfer.freeze_base_layers()
        layers_str = " + ".join(load_info['layers_transferred'])
        transfer_s, transfer_conv = run_agent(env, transfer, cfg["n_episodes"])
        tc_str = str(transfer_conv) if transfer_conv else f">{cfg['n_episodes']}"
        print(f"converged ep {tc_str}  [frozen: {layers_str}]")

        results.append({
            "name": name,
            "scratch_conv": scratch_conv,
            "transfer_conv": transfer_conv,
            "scratch_successes": scratch_s,
            "transfer_successes": transfer_s,
            "layers_frozen": layers_str,
            "n_episodes": cfg["n_episodes"],
        })

    # --- Comparison table ---
    col_w = 17
    print(f"\n{'=' * 68}")
    print(
        f"  {'Environment':<{col_w}} {'Scratch':>10}  {'Transfer':>10}  "
        f"{'Gain':>8}  Frozen layers"
    )
    print(f"  {'-' * 65}")

    speedups = []
    for r in results:
        n_ep = r['n_episodes']
        sc = r['scratch_conv'] or n_ep
        tc = r['transfer_conv'] or n_ep
        gain = sc - tc
        speedups.append(gain)

        sc_str = str(r['scratch_conv']) if r['scratch_conv'] else f">{n_ep}"
        tc_str = str(r['transfer_conv']) if r['transfer_conv'] else f">{n_ep}"
        gain_str = f"+{gain}" if gain > 0 else str(gain)
        print(
            f"  {r['name']:<{col_w}} {sc_str:>10}  {tc_str:>10}  "
            f"{gain_str:>8}  {r['layers_frozen']}"
        )

    print(f"  {'-' * 65}")
    print("  Positive gain = transfer converged faster (fewer episodes needed).")

    best_idx = int(np.argmax(speedups))
    best = results[best_idx]
    best_gain = speedups[best_idx]

    if best_gain > 0:
        print(
            f"\n  Best speedup: {best['name']} — "
            f"transfer needed {best_gain} fewer episodes "
            f"({best['layers_frozen']} frozen)."
        )
    else:
        print(
            "\n  No speedup detected. Possible causes: pretrain too short, "
            "catastrophic forgetting, or epsilon schedule too slow to exploit "
            "pretrained features."
        )

    # --- Plot ---
    out = os.path.join(os.path.dirname(__file__), "transfer_benchmark.png")
    plot_benchmark(results, out)
    print(f"\n{'=' * 68}\n")


if __name__ == "__main__":
    run_benchmark()
