"""Sequential pretraining — Ashby learns to navigate across 5 environments one after another.

Each environment adds a layer of knowledge: spatial navigation, stochastic uncertainty,
adversarial threat, layout generalization, multi-objective survival. The hidden layer
accumulates these experiences into reusable feature representations.
"""

import os
import numpy as np

import ashby
from transfer_agent import TransferAgent

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
CONVERGENCE_THRESHOLD = 0.75
CONVERGENCE_WINDOW = 100

ENV_SEQUENCE = [
    {
        "name": "GridWorld",
        "env_fn": ashby.PyGridWorld,
        "filename": "gridworld.pth",
        "n_episodes": 800,
        "epsilon_decay": 0.995,
        "report_every": 200,
        "story": "learning to navigate toward a fixed target",
    },
    {
        "name": "FrozenLake",
        "env_fn": ashby.PyFrozenLake,
        "filename": "frozen_lake.pth",
        "n_episodes": 1500,
        "epsilon_decay": 0.997,
        "report_every": 300,
        "story": "adapting navigation to probabilistic movement (ice tiles)",
    },
    {
        "name": "PredatorGrid",
        "env_fn": ashby.PyPredatorGrid,
        "filename": "predator_grid.pth",
        "n_episodes": 2000,
        "epsilon_decay": 0.997,
        "report_every": 400,
        "story": "learning to reach the goal while evading a pursuer",
    },
    {
        "name": "MazeWorld",
        "env_fn": ashby.PyMazeWorld,
        "filename": "maze_world.pth",
        "n_episodes": 2500,
        "epsilon_decay": 0.997,
        "report_every": 500,
        "story": "generalizing navigation across procedurally generated 7x7 mazes",
    },
    {
        "name": "ResourceHunter",
        "env_fn": ashby.PyResourceHunter,
        "filename": "resource_hunter.pth",
        "n_episodes": 4000,
        "epsilon_decay": 0.999,
        "report_every": 800,
        "story": "collecting 5 resources under hunger pressure — multi-objective reasoning",
    },
]


def train_env(env, agent: TransferAgent, name: str, n_episodes: int,
              report_every: int, max_steps: int = 300) -> dict:
    successes = []
    converged_at = None

    for ep in range(n_episodes):
        state = env.reset()
        terminal_reward = 0.0

        for _ in range(max_steps):
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

        if (ep + 1) % report_every == 0:
            rate = np.mean(successes[-report_every:])
            phase = "exploring" if agent.is_exploring else "exploiting"
            print(
                f"    ep {ep + 1:4d}  |  success {rate:.0%}  |  "
                f"epsilon={agent.epsilon:.3f} ({phase})"
            )

    final_rate = np.mean(successes[-CONVERGENCE_WINDOW:])
    conv_str = f"episode {converged_at}" if converged_at else "not yet converged"
    print(f"    -> final success {final_rate:.0%}, converged: {conv_str}")
    return {"converged_at": converged_at, "final_success": final_rate}


def run_pretrain():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  ASHBY — SEQUENTIAL PRETRAINING")
    print("  Building shared navigation features across 5 environments.")
    print("=" * 65)

    prev_checkpoint = None

    for i, cfg in enumerate(ENV_SEQUENCE):
        env = cfg["env_fn"]()
        state_size = env.state_shape()[0]
        action_size = len(env.action_space())
        save_path = os.path.join(WEIGHTS_DIR, cfg["filename"])

        print(f"\n[{i + 1}/5] {cfg['name']}  (state={state_size}D, actions={action_size})")
        print(f"      Task: {cfg['story']}")

        agent = TransferAgent(
            state_size=state_size,
            action_size=action_size,
            epsilon_decay=cfg["epsilon_decay"],
            target_update_freq=200,
        )

        if prev_checkpoint is not None:
            info = agent.load_weights(prev_checkpoint)
            prev_name = os.path.basename(prev_checkpoint).replace(".pth", "")
            layers = " + ".join(info['layers_transferred'])
            prev_state = info['saved_state_size']
            print(
                f"      Carrying knowledge from {prev_name} "
                f"(state {prev_state}D -> {state_size}D, transferred: {layers})"
            )
        else:
            print("      First environment — no prior knowledge, starting from random weights.")

        snapshot = agent.weight_snapshot()

        result = train_env(
            env, agent,
            name=cfg["name"],
            n_episodes=cfg["n_episodes"],
            report_every=cfg["report_every"],
        )

        deltas = agent.weight_delta(snapshot)
        print("      Layer evolution during training:")
        layer_labels = {
            'input':  'input  (state->128)',
            'hidden': 'hidden (128->64) ',
            'output': 'output (64->act) ',
        }
        for layer, delta in deltas.items():
            label = layer_labels[layer]
            bar = "#" * min(int(delta * 20), 30)
            print(f"        {label}  delta={delta:.4f}  {bar}")

        agent.save_weights(save_path)
        print(f"      Weights saved -> {save_path}")

        prev_checkpoint = save_path

    print(f"\n{'=' * 65}")
    print("  Sequential pretraining complete.")
    print("  The hidden layer has now experienced:")
    for cfg in ENV_SEQUENCE:
        print(f"    - {cfg['name']}: {cfg['story']}")
    print(f"\n  Weights saved in: {WEIGHTS_DIR}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_pretrain()
