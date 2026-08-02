"""Autonomous RL -- Ashby keeps playing after the human stops watching.

Starts from rl_imitation.pth (if it exists) so the agent isn't relearning "which pedal is
throttle" from scratch, then plays 1v1 against the scripted bot in env.py, fine-tuning the
actor/critic pair from rl_agent.py on whatever reward its own play generates. No human in
the loop from here on -- this is the part that's supposed to eventually get better than
the demonstrations it started from.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from env import make_env, scripted_bot_action, WEIGHTS_DIR
from rl_agent import RLAgent

IMITATION_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_imitation.pth")
POLICY_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_policy.pth")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_training_results.png")

N_EPISODES = 10_000
CHECKPOINT_EVERY = 500
REPORT_EVERY = 100
WINDOW = 100  # rolling window for the narrative stats


def _narrative(episode: int, touch_rate: float, goal_diff_avg: float, epsilon: float) -> str:
    """Grounded in what we can actually measure -- touch rate and goal differential
    against the scripted bot -- rather than claiming behaviors (aerials, dribbles) we
    have no detector for."""
    if episode <= 100:
        phase = "mostly random motion, any touch is a happy accident"
    elif touch_rate < 0.3:
        phase = "starting to close the distance to the ball more often"
    elif touch_rate < 0.6:
        phase = "getting a touch in most episodes now"
    elif goal_diff_avg <= 0.0:
        phase = "touching the ball reliably, not converting to goals yet"
    elif goal_diff_avg < 0.5:
        phase = "trading goals roughly evenly with the scripted bot"
    else:
        phase = "outperforming the beginner bot -- goal difference solidly positive"
    return (
        f"Episode {episode:5d} -- touch rate {touch_rate:4.0%} | "
        f"goal diff {goal_diff_avg:+.2f} | {phase} (e={epsilon:.2f})"
    )


def train():
    print("=" * 70)
    print("  Ashby x RLGym -- autonomous RL vs scripted bot")
    print(f"  {N_EPISODES} episodes, checkpoint every {CHECKPOINT_EVERY}")
    print("=" * 70)

    env = make_env(spawn_opponents=True)

    init_weights = IMITATION_WEIGHTS if os.path.exists(IMITATION_WEIGHTS) else None
    agent = RLAgent(imitation_weights=init_weights)

    rewards_per_ep, touches_per_ep, goal_diffs_per_ep = [], [], []

    for episode in range(1, N_EPISODES + 1):
        obs, info = env.reset(return_info=True)
        blue0, orange0 = info["state"].blue_score, info["state"].orange_score

        done = False
        ep_reward = 0.0
        touched = False

        while not done:
            action_agent = agent.act(np.asarray(obs[0]))
            action_bot = scripted_bot_action(np.asarray(obs[1]))

            next_obs, rewards, done, info = env.step(np.stack([action_agent, action_bot]))

            agent.remember(np.asarray(obs[0]), action_agent, rewards[0], np.asarray(next_obs[0]), done)
            agent.learn()

            touched = touched or info["state"].players[0].ball_touched
            ep_reward += rewards[0]
            obs = next_obs

        agent.decay_epsilon()

        blue1, orange1 = info["state"].blue_score, info["state"].orange_score
        goal_diff = (blue1 - blue0) - (orange1 - orange0)

        rewards_per_ep.append(ep_reward)
        touches_per_ep.append(float(touched))
        goal_diffs_per_ep.append(goal_diff)

        if episode == 1 or episode % REPORT_EVERY == 0:
            touch_rate = float(np.mean(touches_per_ep[-WINDOW:]))
            goal_diff_avg = float(np.mean(goal_diffs_per_ep[-WINDOW:]))
            print(_narrative(episode, touch_rate, goal_diff_avg, agent.epsilon))

        if episode % CHECKPOINT_EVERY == 0:
            agent.save_weights(POLICY_WEIGHTS)
            print(f"  -> checkpoint saved ({episode}/{N_EPISODES}) -> {POLICY_WEIGHTS}")

    env.close()
    agent.save_weights(POLICY_WEIGHTS)
    print(f"\nFinal policy saved -> {POLICY_WEIGHTS}")

    _plot(rewards_per_ep, touches_per_ep, goal_diffs_per_ep)
    print(f"Plot saved         -> {RESULTS_PATH}")


def _plot(rewards: list, touches: list, goal_diffs: list):
    n = len(rewards)
    w = min(WINDOW, n)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle("Ashby x RLGym -- autonomous RL vs scripted bot", fontsize=13)

    ax1.plot(rewards, alpha=0.2, color="steelblue")
    if w > 1:
        ax1.plot(range(w, n + 1), np.convolve(rewards, np.ones(w) / w, mode="valid"), color="steelblue")
    ax1.set_ylabel("Episode reward")
    ax1.set_title("Reward over time")
    ax1.grid(alpha=0.3)

    if w > 1:
        ax2.plot(range(w, n + 1), np.convolve(touches, np.ones(w) / w, mode="valid"), color="tomato")
    ax2.set_ylabel(f"Touch rate ({w}-ep avg)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Ball touch rate")
    ax2.grid(alpha=0.3)

    if w > 1:
        ax3.plot(range(w, n + 1), np.convolve(goal_diffs, np.ones(w) / w, mode="valid"), color="seagreen")
    ax3.axhline(0, linestyle="--", color="gray", alpha=0.5)
    ax3.set_ylabel(f"Goal diff ({w}-ep avg)")
    ax3.set_xlabel("Episode")
    ax3.set_title("Goal differential vs scripted bot")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH, dpi=120)
    plt.close(fig)


def smoke_test():
    """--test: a handful of real env steps with a fresh agent, no full training run.
    Proves the 1v1 env, the scripted bot, and the actor-critic learning step all
    actually fit together before committing to 10,000 episodes."""
    print("Building 1v1 env and agent...")
    env = make_env(spawn_opponents=True)
    init_weights = IMITATION_WEIGHTS if os.path.exists(IMITATION_WEIGHTS) else None
    agent = RLAgent(imitation_weights=init_weights, batch_size=8)

    obs, info = env.reset(return_info=True)
    assert len(obs) == 2, f"expected 2 players (1v1), got {len(obs)}"

    for step in range(20):
        action_agent = agent.act(np.asarray(obs[0]))
        action_bot = scripted_bot_action(np.asarray(obs[1]))
        next_obs, rewards, done, info = env.step(np.stack([action_agent, action_bot]))

        agent.remember(np.asarray(obs[0]), action_agent, rewards[0], np.asarray(next_obs[0]), done)
        result = agent.learn()

        obs = next_obs
        if done:
            obs, info = env.reset(return_info=True)

    env.close()
    print(f"  ran 20 steps of a 1v1 episode, last learn() result: {result}")
    print("rl_train.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLGym -- autonomous RL training")
    parser.add_argument("--test", action="store_true", help="smoke-test the 1v1 env + agent without a full run")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        train()
