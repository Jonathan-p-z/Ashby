"""Transfer benchmark: basic -> defend_the_center.

Trains two agents on defend_the_center from scratch and compares convergence.
Agent A starts with random weights. Agent B starts with conv+fc weights from
basic.pth but has the output head reinitialized — the visual features transfer,
but the Q-value estimates are reset so defend's policy is learned from scratch.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import vizdoom as vzd

sys.path.insert(0, os.path.dirname(__file__))
from vizdoom_agent import VizDoomDQNAgent
from vizdoom_defend import DEFEND_ACTIONS, _make_game

BASIC_WEIGHTS  = os.path.join(os.path.dirname(__file__), "weights", "vizdoom_basic.pth")
RESULTS_PATH   = os.path.join(os.path.dirname(__file__), "vizdoom_transfer.png")
N_EPISODES     = 1000
FRAME_SKIP     = 4
KILL_THRESHOLD = 2.0   # avg kills/ep we consider "converged enough to compare"
SMOOTH_WINDOW  = 100


def _make_transfer_agent() -> VizDoomDQNAgent:
    """Load basic weights into a defend agent, reset the output head.

    The conv stack and first dense layer learned to parse vizdoom frames —
    where enemies are, walls, open space. Those features are environment-
    agnostic enough to transfer. The output head encodes specific Q-values
    for basic's reward structure, which doesn't match defend's, so we reset
    it and let it relearn.
    """
    agent = VizDoomDQNAgent()
    state_dict = torch.load(BASIC_WEIGHTS, map_location="cpu", weights_only=True)
    agent.policy_net.load_state_dict(state_dict)

    # fc[2] is Linear(512, n_actions) — the output head
    nn.init.xavier_uniform_(agent.policy_net.fc[2].weight)
    nn.init.zeros_(agent.policy_net.fc[2].bias)

    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    return agent


def _run(game: vzd.DoomGame, agent: VizDoomDQNAgent, n_episodes: int, label: str) -> list:
    """Run one training session, return kills-per-episode list."""
    kills_per_ep = []

    for episode in range(1, n_episodes + 1):
        game.new_episode()
        frame         = game.get_state().screen_buffer
        episode_kills = 0

        while not game.is_episode_finished():
            action_idx = agent.act(frame)
            prev_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

            game.make_action(DEFEND_ACTIONS[action_idx], FRAME_SKIP)
            done       = game.is_episode_finished()
            curr_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            health     = int(game.get_game_variable(vzd.GameVariable.HEALTH))

            kill_happened  = curr_kills > prev_kills
            died           = done and health <= 0
            reward         = 1.0 if kill_happened else (-1.0 if died else 0.0)
            episode_kills += int(kill_happened)

            next_frame = frame if done else game.get_state().screen_buffer
            agent.remember(frame, action_idx, reward, next_frame, done)
            agent.learn()
            frame = next_frame

        agent.decay_epsilon()
        kills_per_ep.append(episode_kills)

        if episode % 200 == 0 or episode == 1:
            window   = kills_per_ep[-SMOOTH_WINDOW:]
            kill_avg = np.mean(window)
            print(f"  [{label}] episode {episode:4d} -- avg kills {kill_avg:.2f} (e={agent.epsilon:.2f})")

    return kills_per_ep


def _convergence_ep(kills: list, threshold: float) -> str:
    """Episode number where rolling avg first crosses threshold, or 'not reached'."""
    w = SMOOTH_WINDOW
    for i in range(w, len(kills) + 1):
        if np.mean(kills[i - w:i]) >= threshold:
            return str(i)
    return "not reached"


def benchmark():
    if not os.path.exists(BASIC_WEIGHTS):
        print("No basic weights found -- run 'make vizdoom-train' first")
        sys.exit(0)

    print("=" * 65)
    print("  Transfer benchmark: basic -> defend_the_center")
    print(f"  {N_EPISODES} episodes per agent, threshold: {KILL_THRESHOLD} kills/ep avg")
    print("=" * 65)

    # --- Agent A: scratch ---
    print("\n[A] Training from scratch...")
    game_a   = _make_game(visible=False)
    agent_a  = VizDoomDQNAgent()
    kills_a  = _run(game_a, agent_a, N_EPISODES, "scratch ")
    game_a.close()

    # --- Agent B: transfer ---
    print("\n[B] Training with basic pretrain (output head reset)...")
    game_b   = _make_game(visible=False)
    agent_b  = _make_transfer_agent()
    kills_b  = _run(game_b, agent_b, N_EPISODES, "transfer")
    game_b.close()

    # --- Summary table ---
    conv_a = _convergence_ep(kills_a, KILL_THRESHOLD)
    conv_b = _convergence_ep(kills_b, KILL_THRESHOLD)
    final_a = np.mean(kills_a[-SMOOTH_WINDOW:])
    final_b = np.mean(kills_b[-SMOOTH_WINDOW:])

    print("\n" + "=" * 55)
    print("  Results -- defend_the_center transfer benchmark")
    print("=" * 55)
    print(f"  {'Agent':<12} {'Converged at':>14} {'Final avg kills':>16}")
    print(f"  {'-'*12} {'-'*14} {'-'*16}")
    print(f"  {'Scratch':<12} {conv_a:>14} {final_a:>15.2f}")
    print(f"  {'Transfer':<12} {conv_b:>14} {final_b:>15.2f}")
    print("=" * 55)

    if conv_a != "not reached" and conv_b != "not reached":
        gain = int(conv_a) - int(conv_b)
        if gain > 0:
            print(f"\n  Transfer converged {gain} episodes faster.")
        elif gain < 0:
            print(f"\n  Scratch converged {-gain} episodes faster (transfer hurt).")
        else:
            print("\n  Both converged at the same episode.")
    else:
        print(f"\n  Threshold {KILL_THRESHOLD} kills/ep not reached by both agents.")
        print(f"  Final kills -- scratch: {final_a:.2f} | transfer: {final_b:.2f}")

    _plot(kills_a, kills_b)
    print(f"\nPlot saved -> {RESULTS_PATH}")


def _plot(kills_a: list, kills_b: list):
    n = len(kills_a)
    w = min(SMOOTH_WINDOW, n)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Transfer benchmark: basic -> defend_the_center", fontsize=13)

    for kills, color, label in [
        (kills_a, "steelblue", "scratch"),
        (kills_b, "tomato",    "transfer (basic pretrain)"),
    ]:
        ax.plot(kills, alpha=0.15, color=color)
        if w > 1:
            smooth = np.convolve(kills, np.ones(w) / w, mode="valid")
            ax.plot(range(w, n + 1), smooth, color=color, label=label, linewidth=2)

    if KILL_THRESHOLD > 0:
        ax.axhline(KILL_THRESHOLD, linestyle="--", color="gray", alpha=0.6,
                   label=f"threshold: {KILL_THRESHOLD} kills/ep")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Kills/episode ({w}-ep rolling avg)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH, dpi=120)


if __name__ == "__main__":
    benchmark()
