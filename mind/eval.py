"""Standalone evaluation script.

Trains a fresh agent on each environment (quick run) then evaluates it in
pure exploitation mode and prints a comparison table.
Intentionally self-contained — no dependency on saved model state.
"""

import os
import numpy as np

import ashby
from agent import QLearningAgent

TRAIN_EPISODES = {"GridWorld": 1000, "FrozenLake": 2000}
EVAL_EPISODES = 300
MAX_STEPS = 200
CONVERGENCE_WINDOW = 100
CONVERGENCE_THRESHOLD = 0.75


def quick_train(env, n_episodes: int, epsilon_decay: float = 0.995) -> tuple[QLearningAgent, int | None]:
    """Train silently. Returns (agent, episode_of_convergence)."""
    agent = QLearningAgent(n_actions=len(env.action_space()), epsilon_decay=epsilon_decay)
    successes = []
    converged_at = None

    for ep in range(n_episodes):
        state = env.reset()
        reached_goal = False
        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                reached_goal = reward > 0
                break
        agent.decay_epsilon()
        successes.append(int(reached_goal))
        if converged_at is None and ep >= CONVERGENCE_WINDOW:
            if np.mean(successes[-CONVERGENCE_WINDOW:]) >= CONVERGENCE_THRESHOLD:
                converged_at = ep + 1

    return agent, converged_at


def evaluate(agent: QLearningAgent, env, n_episodes: int = EVAL_EPISODES) -> dict:
    """Exploit only — no random actions — to measure the learned policy."""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    successes = 0
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        total = 0.0
        for _ in range(MAX_STEPS):
            action = agent.best_action(state)
            state, reward, done = env.step(action)
            total += reward
            if done:
                if reward > 0:
                    successes += 1
                break
        rewards.append(total)

    agent.epsilon = saved_epsilon
    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
    }


def run_eval():
    configs = [
        ("GridWorld", ashby.PyGridWorld(), 0.995),
        ("FrozenLake", ashby.PyFrozenLake(), 0.997),
    ]

    print("\nEvaluating Ashby — training then testing each environment...\n")
    print(f"  {'Environment':<14}  {'Train eps':>9}  {'Converged':>10}  {'Success':>8}  {'Avg reward':>11}")
    print(f"  {'-'*58}")

    for name, env, eps_decay in configs:
        n_ep = TRAIN_EPISODES[name]
        print(f"  Training {name}...", end="", flush=True)
        agent, conv = quick_train(env, n_ep, epsilon_decay=eps_decay)
        stats = evaluate(agent, env)
        conv_str = str(conv) if conv else "—"
        print(
            f"\r  {name:<14}  {n_ep:>9}  {conv_str:>10}  "
            f"{stats['success_rate']:>7.0%}  {stats['mean_reward']:>11.3f}"
        )

    print(f"  {'-'*58}")
    print("\n  Note: FrozenLake uses a randomly generated map — results vary between runs.\n")


if __name__ == "__main__":
    run_eval()
