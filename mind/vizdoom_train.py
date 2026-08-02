import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vizdoom as vzd

sys.path.insert(0, os.path.dirname(__file__))
from vizdoom_agent import VizDoomDQNAgent, ACTIONS

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "vizdoom_basic.pth")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "vizdoom_results.png")
N_EPISODES   = 1000
FRAME_SKIP   = 4
REPORT_EVERY = 100


def _make_game() -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(False)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.init()
    return game


def _narrative(episode: int, kill_rate: float, epsilon: float) -> str:
    if episode <= 10:
        phase = "agent fires randomly, learning what buttons do"
    elif kill_rate < 0.05:
        phase = "still spraying and praying, monster barely notices"
    elif kill_rate < 0.25:
        phase = "starting to track the monster, aim coming together"
    elif kill_rate < 0.50:
        phase = "timing and positioning improving, kills aren't flukes"
    elif kill_rate < 0.75:
        phase = "consistent kills, plays are starting to look intentional"
    else:
        phase = "reliable killer -- clean movement, clean shots"
    return f"Episode {episode:4d} -- kill rate {kill_rate:4.0%} -- {phase} (e={epsilon:.2f})"


def train():
    print("=" * 65)
    print("  Ashby x ViZDoom -- basic scenario")
    print("  3 actions: move left / move right / attack")
    print("  reward: +1 kill, 0 otherwise")
    print(f"  training: {N_EPISODES} episodes, frame skip {FRAME_SKIP}")
    print("=" * 65)

    game  = _make_game()
    agent = VizDoomDQNAgent()

    rewards_per_ep = []
    kills_per_ep   = []

    for episode in range(1, N_EPISODES + 1):
        game.new_episode()
        frame          = game.get_state().screen_buffer
        episode_reward = 0.0
        episode_kills  = 0

        while not game.is_episode_finished():
            action_idx   = agent.act(frame)
            prev_kills   = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

            game.make_action(ACTIONS[action_idx], FRAME_SKIP)
            done       = game.is_episode_finished()
            curr_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

            kill_happened   = curr_kills > prev_kills
            reward          = 1.0 if kill_happened else 0.0
            episode_kills  += int(kill_happened)
            episode_reward += reward

            next_frame = frame if done else game.get_state().screen_buffer
            agent.remember(frame, action_idx, reward, next_frame, done)
            agent.learn()
            frame = next_frame

        agent.decay_epsilon()
        rewards_per_ep.append(episode_reward)
        kills_per_ep.append(episode_kills)

        if episode == 1 or episode % REPORT_EVERY == 0:
            window    = kills_per_ep[-100:]
            kill_rate = float(np.mean([min(k, 1) for k in window]))
            print(_narrative(episode, kill_rate, agent.epsilon))
            agent.save_weights(WEIGHTS_PATH)

    game.close()
    agent.save_weights(WEIGHTS_PATH)
    print(f"\nWeights saved -> {WEIGHTS_PATH}")

    _plot(rewards_per_ep, kills_per_ep)
    print(f"Plot saved    -> {RESULTS_PATH}")


def _plot(rewards: list, kills: list):
    n = len(rewards)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Ashby x ViZDoom -- basic scenario", fontsize=14)

    ax1.plot(rewards, alpha=0.25, color="steelblue", label="raw")
    smooth_w = min(50, n)
    if smooth_w > 1:
        smooth_r = np.convolve(rewards, np.ones(smooth_w) / smooth_w, mode="valid")
        ax1.plot(range(smooth_w, n + 1), smooth_r, color="steelblue", label=f"{smooth_w}-ep avg")
    ax1.set_ylabel("Episode reward")
    ax1.set_title("Reward over time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    kill_indicators = [min(k, 1) for k in kills]
    kill_w = min(100, n)
    if kill_w > 1:
        smooth_k = np.convolve(kill_indicators, np.ones(kill_w) / kill_w, mode="valid")
        ax2.plot(range(kill_w, n + 1), smooth_k, color="tomato")
    ax2.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="50% baseline")
    ax2.set_ylabel(f"Kill rate ({kill_w}-ep rolling avg)")
    ax2.set_xlabel("Episode")
    ax2.set_title("Kill rate over time")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH, dpi=120)


if __name__ == "__main__":
    train()
