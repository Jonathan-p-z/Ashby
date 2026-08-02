"""Visual mode -- watch the trained policy play 1v1 against the scripted bot.

Runs the actor in pure exploitation (no epsilon, no learning) and prints live stats
every step. Renders through rlviser_py if it's installed; otherwise falls back to a
text-only readout rather than failing outright, since a render backend has nothing to
do with whether the policy itself works.
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import make_env, scripted_bot_action, SECONDS_PER_STEP, WEIGHTS_DIR
from rl_agent import RLAgent

POLICY_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_policy.pth")
DEFAULT_EPISODES = 5


def _try_enable_render(env) -> bool:
    try:
        env.render()  # prev_state is None pre-reset -- this only probes for rlviser_py
        return True
    except ImportError:
        return False


def watch(episodes: int = DEFAULT_EPISODES):
    if not os.path.exists(POLICY_WEIGHTS):
        raise FileNotFoundError(
            f"No policy weights at {POLICY_WEIGHTS} -- run `make rl-train-rl` first "
            "(or `make rl-train-bc` if you haven't even done imitation learning yet)."
        )

    env = make_env(spawn_opponents=True)
    agent = RLAgent(imitation_weights=None)
    agent.load_weights(POLICY_WEIGHTS)
    print(f"Loaded policy from {POLICY_WEIGHTS}")

    can_render = _try_enable_render(env)
    if can_render:
        print("Rendering via rlviser_py.\n")
    else:
        print("rlviser_py not installed -- running headless with live text stats only.")
        print("  (pip install rlviser-py for an actual visual window)\n")

    for ep in range(1, episodes + 1):
        obs, info = env.reset(return_info=True)
        blue0, orange0 = info["state"].blue_score, info["state"].orange_score
        last_boost = info["state"].players[0].boost_amount

        touches = 0
        boost_used = 0.0
        steps = 0
        done = False

        while not done:
            action_agent = agent.best_action(np.asarray(obs[0]))
            action_bot = scripted_bot_action(np.asarray(obs[1]))

            obs, rewards, done, info = env.step(np.stack([action_agent, action_bot]))
            steps += 1

            player = info["state"].players[0]
            if player.ball_touched:
                touches += 1
            if player.boost_amount < last_boost:
                boost_used += last_boost - player.boost_amount
            last_boost = player.boost_amount

            if can_render:
                env.render()

            survival_s = steps * SECONDS_PER_STEP
            print(
                f"\rEp {ep}/{episodes} | t={survival_s:5.1f}s | touches={touches:3d} | "
                f"boost used={boost_used * 100:5.1f} | score {info['state'].blue_score}-{info['state'].orange_score}",
                end="", flush=True,
            )

        blue1, orange1 = info["state"].blue_score, info["state"].orange_score
        print(
            f"\n  Episode {ep} result: agent {blue1 - blue0} - {orange1 - orange0} bot | "
            f"touches={touches} | boost used={boost_used * 100:.0f} | survived {steps * SECONDS_PER_STEP:.1f}s\n"
        )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLGym -- watch the trained policy play")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="number of episodes to watch")
    args = parser.parse_args()

    watch(episodes=args.episodes)
