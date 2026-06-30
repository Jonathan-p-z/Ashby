import os
import numpy as np
import matplotlib.pyplot as plt

import ashby
from agent import QLearningAgent
from dqn_agent import DQNAgent

CONVERGENCE_THRESHOLD = 0.75
CONVERGENCE_WINDOW = 100


def train_environment(
    env,
    name: str,
    n_episodes: int,
    agent,          # QLearningAgent or DQNAgent — created by the caller
    agent_type: str,
    max_steps: int = 200,
    report_every: int = 200,
    failure_label: str = "caught",
) -> dict:
    rewards = []
    successes = []
    caught = []
    converged_at = None

    print(f"\n{'='*60}")
    print(f"  {name}  [{agent_type}]  —  {n_episodes} episodes")
    print(f"{'='*60}")

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        terminal_reward = 0.0

        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                terminal_reward = reward
                break

        agent.decay_epsilon()
        reached_goal = terminal_reward > 0
        got_caught = terminal_reward < 0

        rewards.append(total_reward)
        successes.append(int(reached_goal))
        caught.append(int(got_caught))

        if converged_at is None and episode >= CONVERGENCE_WINDOW:
            if np.mean(successes[-CONVERGENCE_WINDOW:]) >= CONVERGENCE_THRESHOLD:
                converged_at = episode + 1

        if (episode + 1) % report_every == 0:
            w_rewards = rewards[-report_every:]
            w_success = np.mean(successes[-report_every:])
            w_caught = np.mean(caught[-report_every:])
            phase = "still wandering" if agent.is_exploring else "mostly exploiting"

            fail_str = f"  |  {failure_label} {w_caught:.0%}" if w_caught > 0.01 else ""
            print(
                f"  ep {episode + 1:4d}  |  "
                f"avg reward {np.mean(w_rewards):+.3f}  |  "
                f"success {w_success:.0%}{fail_str}  |  "
                f"ε={agent.epsilon:.3f} ({phase})"
            )

    final_success = np.mean(successes[-CONVERGENCE_WINDOW:])
    final_caught = np.mean(caught[-CONVERGENCE_WINDOW:])
    conv_str = f"episode {converged_at}" if converged_at else "never fully converged"

    print(f"\n  → final success rate: {final_success:.0%}  |  converged: {conv_str}")
    if final_caught > 0.01:
        print(f"  → {failure_label} rate (last {CONVERGENCE_WINDOW} eps): {final_caught:.0%}")
    if agent_type == "DQN":
        q_size_str = f"{agent._steps} gradient steps taken"
    else:
        q_size_str = f"{len(agent.q_table)} (state, action) pairs visited"
    print(f"  → {q_size_str}")

    return {
        "name": name,
        "agent_type": agent_type,
        "rewards": rewards,
        "successes": successes,
        "caught": caught,
        "converged_at": converged_at,
        "final_success": final_success,
        "final_caught": final_caught,
    }


def _nearest_uncollected_dist(state: list) -> float:
    """Manhattan distance from agent to closest uncollected resource, read from state."""
    ar, ac = state[0] * 6, state[1] * 6
    min_dist = float("inf")
    for i in range(5):
        if state[3 + i * 3 + 2] < 0.5:  # uncollected
            rr = state[3 + i * 3] * 6
            rc = state[3 + i * 3 + 1] * 6
            dist = abs(ar - rr) + abs(ac - rc)
            if dist < min_dist:
                min_dist = dist
    return min_dist if min_dist < float("inf") else 0.0


def analyze_hunger_urgency(env, agent, n_episodes: int = 200) -> dict:
    """Does the agent rush toward the nearest resource when hunger is critical?

    Compares how often the agent closes distance to the nearest resource
    at hunger < 20 vs. hunger >= 20. A positive gap means the learned
    policy becomes more directional under survival pressure.
    """
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    urgent_toward = []
    normal_toward = []

    for _ in range(n_episodes):
        state = list(env.reset())
        for _ in range(250):
            hunger = state[2] * 50.0
            dist_before = _nearest_uncollected_dist(state)

            action = agent.best_action(state)
            next_state, _, done = env.step(action)
            next_state = list(next_state)

            # Skip collection steps — the distance drop is trivially caused by eating
            newly_collected = any(
                next_state[3 + i * 3 + 2] > state[3 + i * 3 + 2] for i in range(5)
            )
            if not newly_collected and dist_before > 0:
                dist_after = _nearest_uncollected_dist(next_state)
                moved_closer = int(dist_after < dist_before)
                if hunger < 20.0:
                    urgent_toward.append(moved_closer)
                else:
                    normal_toward.append(moved_closer)

            state = next_state
            if done:
                break

    agent.epsilon = saved_eps
    urgent_rate = float(np.mean(urgent_toward)) if urgent_toward else 0.0
    normal_rate = float(np.mean(normal_toward)) if normal_toward else 0.0
    return {
        "urgent_rate": urgent_rate,
        "normal_rate": normal_rate,
        "n_urgent_steps": len(urgent_toward),
        "n_normal_steps": len(normal_toward),
    }


def plot_comparison(results: list):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    fig.suptitle("Ashby — training comparison", fontsize=14)

    if n == 1:
        axes = [axes]

    window = 40
    colors = ["steelblue", "darkorange", "crimson", "mediumorchid", "seagreen"]

    for i, r in enumerate(results):
        color = colors[i % len(colors)]
        rewards = r["rewards"]
        successes = [float(s) for s in r["successes"]]
        caught = [float(c) for c in r["caught"]]
        name = r["name"]
        atype = r["agent_type"]

        ax_r, ax_s = axes[i]

        smoothed_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax_r.plot(rewards, alpha=0.15, color=color)
        ax_r.plot(range(window - 1, len(rewards)), smoothed_r, color=color,
                  label=f"{window}-ep avg")
        ax_r.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_r.set_title(f"{name} [{atype}] — reward")
        ax_r.set_ylabel("Total reward")
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.3)

        smoothed_s = np.convolve(successes, np.ones(window) / window, mode="valid")
        ax_s.plot(successes, alpha=0.1, color=color)
        ax_s.plot(range(window - 1, len(successes)), smoothed_s, color=color,
                  label="success rate")

        if max(caught) > 0:
            fail_label = "starved rate" if name == "ResourceHunter" else "caught/fail rate"
            smoothed_c = np.convolve(caught, np.ones(window) / window, mode="valid")
            ax_s.plot(range(window - 1, len(caught)), smoothed_c,
                      color="black", linestyle="--", linewidth=1, alpha=0.6,
                      label=fail_label)

        ax_s.axhline(CONVERGENCE_THRESHOLD, color="gray", linewidth=0.8,
                     linestyle=":", label=f"threshold ({CONVERGENCE_THRESHOLD:.0%})")
        ax_s.set_title(f"{name} [{atype}] — outcomes")
        ax_s.set_ylabel("Rate")
        ax_s.set_ylim(-0.05, 1.1)
        ax_s.legend(fontsize=8)
        ax_s.grid(True, alpha=0.3)

        if r["converged_at"]:
            for ax in (ax_r, ax_s):
                ax.axvline(r["converged_at"], color="green", linewidth=1.0,
                           linestyle=":", alpha=0.7)

    for ax in axes[-1]:
        ax.set_xlabel("Episode")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "training_results.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


def run_training():
    print("\nAshby is waking up...\n")

    results = []

    gw_env = ashby.PyGridWorld()
    results.append(train_environment(
        env=gw_env, name="GridWorld", n_episodes=1000,
        agent=QLearningAgent(n_actions=len(gw_env.action_space()), epsilon_decay=0.995),
        agent_type="Tabular", report_every=200,
    ))

    fl_env = ashby.PyFrozenLake()
    results.append(train_environment(
        env=fl_env, name="FrozenLake", n_episodes=2000,
        agent=QLearningAgent(n_actions=len(fl_env.action_space()), epsilon_decay=0.997),
        agent_type="Tabular", report_every=400,
    ))

    pg_env = ashby.PyPredatorGrid()
    results.append(train_environment(
        env=pg_env, name="PredatorGrid", n_episodes=3000,
        agent=QLearningAgent(n_actions=len(pg_env.action_space()), epsilon_decay=0.997),
        agent_type="Tabular", report_every=500,
    ))

    mw_env = ashby.PyMazeWorld()
    state_size = mw_env.state_shape()[0]  # 53: agent_pos + target_pos + 49 grid cells
    print(f"\n  [MazeWorld] state is {state_size}D — new maze layout every episode.")
    print(f"  [MazeWorld] Tabular Q-learning would need a unique entry per layout.")
    print(f"  [MazeWorld] DQN must generalize from the grid map it sees in the state.\n")
    results.append(train_environment(
        env=mw_env, name="MazeWorld", n_episodes=3000,
        agent=DQNAgent(
            state_size=state_size,
            n_actions=len(mw_env.action_space()),
            epsilon_decay=0.997,
            target_update_freq=200,
        ),
        agent_type="DQN",
        max_steps=300,
        report_every=500,
    ))

    rh_env = ashby.PyResourceHunter()
    state_size = rh_env.state_shape()[0]  # 18: agent(2) + hunger(1) + 5×resource(3)
    print(f"\n  [ResourceHunter] state is {state_size}D — positions and hunger re-randomize every episode.")
    print(f"  [ResourceHunter] Must collect all 5 resources before hunger hits zero.")
    print(f"  [ResourceHunter] DQN must learn to prioritize the nearest resource under survival pressure.\n")
    rh_agent = DQNAgent(
        state_size=state_size,
        n_actions=len(rh_env.action_space()),
        # 0.999 → epsilon reaches minimum around ep 3000 (vs ep 1000 with 0.997).
        # Collecting 5 sequential targets needs much more exploration time than 1 target.
        epsilon_decay=0.999,
        target_update_freq=200,
    )
    rh_result = train_environment(
        env=rh_env, name="ResourceHunter", n_episodes=5000,
        agent=rh_agent,
        agent_type="DQN",
        max_steps=300,
        report_every=500,
        failure_label="starved",
    )
    results.append(rh_result)

    print(f"\n  [ResourceHunter] Analyzing hunger urgency behavior...")
    urgency = analyze_hunger_urgency(rh_env, rh_agent, n_episodes=200)
    if urgency["n_urgent_steps"] > 50:
        diff = urgency["urgent_rate"] - urgency["normal_rate"]
        verdict = "YES — rushes toward nearest resource" if diff > 0.05 else "no significant urgency bias detected"
        print(
            f"  [ResourceHunter] Approach rate: hunger<20 → {urgency['urgent_rate']:.0%}  "
            f"vs  hunger≥20 → {urgency['normal_rate']:.0%}  ({urgency['n_urgent_steps']} urgent steps)"
        )
        print(f"  [ResourceHunter] Hunger management verdict: {verdict}")
    else:
        print(f"  [ResourceHunter] Agent rarely reaches critical hunger — it eats proactively.")

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    parts = []
    for r in results:
        conv = r["converged_at"] or "N/A"
        line = (
            f"{r['name']} : {r['final_success']:.0%} ({r['agent_type']}) "
            f"en {conv} épisodes"
        )
        if r["name"] == "ResourceHunter":
            line += f", starved rate : {r['final_caught']:.0%}"
        elif r["final_caught"] > 0 or r["name"] == "PredatorGrid":
            line += f", caught rate : {r['final_caught']:.0%}"
        print(f"  {line}")
        parts.append(line)

    print(f"\n→ {' / '.join(parts)}")

    # Confirm DQN generalization explicitly.
    mw = next(r for r in results if r["name"] == "MazeWorld")
    trend_early = np.mean(mw["successes"][:200])
    trend_late = np.mean(mw["successes"][-200:])
    print(
        f"\n  MazeWorld generalization: success {trend_early:.0%} (ep 1-200)"
        f" → {trend_late:.0%} (last 200) on unseen maze layouts."
    )

    plot_comparison(results)


if __name__ == "__main__":
    run_training()
