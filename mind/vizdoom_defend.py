import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vizdoom as vzd

sys.path.insert(0, os.path.dirname(__file__))
from vizdoom_agent import VizDoomDQNAgent, ACTIONS

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "vizdoom_defend.pth")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "vizdoom_defend_results.png")
N_EPISODES   = 2000
FRAME_SKIP   = 4
REPORT_EVERY = 200

# Same 3-action layout as basic — TURN replaces MOVE, agent learns to rotate and aim
DEFEND_ACTIONS = [
    [1, 0, 0],  # TURN_LEFT
    [0, 1, 0],  # TURN_RIGHT
    [0, 0, 1],  # ATTACK
]


def _make_game(visible: bool = False) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(visible)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(
        vzd.ScreenResolution.RES_640X480 if visible else vzd.ScreenResolution.RES_160X120
    )
    # KILLCOUNT isn't in the default config — add it so we can reward kills inline
    game.set_available_game_variables([
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.HEALTH,
    ])
    game.init()
    return game


def _narrative(episode: int, kill_avg: float, survival_avg: float, epsilon: float, attack_rate: float = 1.0) -> str:
    if episode <= 20:
        phase = f"spamming ATTACK ({attack_rate:.0%} of steps) -- penalty hasn't landed yet"
    elif kill_avg < 0.5:
        phase = f"can't keep up -- trigger-happy at {attack_rate:.0%}, mostly misses"
    elif kill_avg < 2.0:
        if attack_rate > 0.5:
            phase = f"landing shots but still wasteful -- {attack_rate:.0%} attack rate"
        else:
            phase = f"patience kicking in ({attack_rate:.0%} attacks) -- shots connect more"
    elif kill_avg < 4.0:
        phase = f"rotating cleanly, {attack_rate:.0%} attack rate -- shoots when it counts"
    else:
        phase = f"efficient -- {kill_avg:.1f} kills at {attack_rate:.0%} attack rate, survival is real"
    return (
        f"Episode {episode:5d} -- kills/ep {kill_avg:.1f}"
        f" | survive {survival_avg:.0f} steps | {phase} (e={epsilon:.2f})"
    )


def train(agent: VizDoomDQNAgent = None, n_episodes: int = None) -> dict:
    """Train on defend_the_center.

    Returns metrics dict so vizdoom_transfer.py can call this without subprocess.
    When called standalone (__main__), agent=None and defaults kick in.
    """
    if agent is None:
        agent = VizDoomDQNAgent()
    if n_episodes is None:
        n_episodes = N_EPISODES

    print("=" * 65)
    print("  Ashby x ViZDoom -- defend_the_center")
    print("  3 actions: turn left / turn right / attack")
    print("  reward: +1 kill | -0.1 miss | -0.5 death | +0.02 patience")
    print(f"  training: {n_episodes} episodes, frame skip {FRAME_SKIP}")
    print("=" * 65)

    game = _make_game(visible=False)

    kills_per_ep   = []
    steps_per_ep   = []
    rewards_per_ep = []

    for episode in range(1, n_episodes + 1):
        game.new_episode()
        frame            = game.get_state().screen_buffer
        episode_kills    = 0
        episode_attacks  = 0
        episode_steps    = 0
        episode_reward   = 0.0

        while not game.is_episode_finished():
            action_idx = agent.act(frame)
            prev_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

            game.make_action(DEFEND_ACTIONS[action_idx], FRAME_SKIP)
            done       = game.is_episode_finished()
            curr_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            health     = int(game.get_game_variable(vzd.GameVariable.HEALTH))

            kill_happened   = curr_kills > prev_kills
            shot_fired      = action_idx == 2  # DEFEND_ACTIONS[2] = ATTACK
            died            = done and health <= 0

            if kill_happened:
                reward = 1.0
            elif died:
                reward = -0.5
            elif shot_fired:
                reward = -0.1  # misfire penalty — forces aim discipline before committing
            else:
                reward = 0.02  # patience bonus per step survived without shooting

            episode_kills   += int(kill_happened)
            episode_attacks += int(shot_fired)
            episode_steps   += 1
            episode_reward  += reward

            next_frame = frame if done else game.get_state().screen_buffer
            agent.remember(frame, action_idx, reward, next_frame, done)
            agent.learn()
            frame = next_frame

        agent.decay_epsilon()
        kills_per_ep.append(episode_kills)
        steps_per_ep.append(episode_steps)
        rewards_per_ep.append(episode_reward)

        if episode == 1 or episode % REPORT_EVERY == 0:
            window       = kills_per_ep[-100:]
            kill_avg     = float(np.mean(window))
            survival_avg = float(np.mean(steps_per_ep[-100:]))
            attack_rate  = episode_attacks / max(episode_steps, 1)
            print(_narrative(episode, kill_avg, survival_avg, agent.epsilon, attack_rate))
            agent.save_weights(WEIGHTS_PATH)

    game.close()
    agent.save_weights(WEIGHTS_PATH)
    print(f"\nWeights saved -> {WEIGHTS_PATH}")

    _plot(kills_per_ep, steps_per_ep)
    print(f"Plot saved    -> {RESULTS_PATH}")

    return {"kills": kills_per_ep, "steps": steps_per_ep, "rewards": rewards_per_ep}


def _plot(kills: list, steps: list):
    n = len(kills)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Ashby x ViZDoom -- defend_the_center", fontsize=14)

    w = min(100, n)
    if w > 1:
        smooth_k = np.convolve(kills, np.ones(w) / w, mode="valid")
        ax1.plot(range(w, n + 1), smooth_k, color="tomato")
    ax1.plot(kills, alpha=0.2, color="tomato")
    ax1.set_ylabel(f"Kills/episode ({w}-ep avg)")
    ax1.set_title("Kills over time")
    ax1.grid(alpha=0.3)

    if w > 1:
        smooth_s = np.convolve(steps, np.ones(w) / w, mode="valid")
        ax2.plot(range(w, n + 1), smooth_s, color="steelblue")
    ax2.plot(steps, alpha=0.2, color="steelblue")
    ax2.set_ylabel(f"Steps survived ({w}-ep avg)")
    ax2.set_xlabel("Episode")
    ax2.set_title("Survival over time")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH, dpi=120)


if __name__ == "__main__":
    train()
