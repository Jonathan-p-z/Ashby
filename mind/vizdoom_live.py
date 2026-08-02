import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import vizdoom as vzd

sys.path.insert(0, os.path.dirname(__file__))
from vizdoom_agent import VizDoomDQNAgent

# 5 actions instead of 3 — combined turn+attack lets the agent shoot while
# rotating, which is how a real player handles threats coming from the side.
LIVE_ACTIONS = [
    [True,  False, False],  # turn left
    [False, True,  False],  # turn right
    [False, False, True ],  # attack
    [True,  False, True ],  # turn left + attack
    [False, True,  True ],  # turn right + attack
]
_SHOOTS = {2, 3, 4}  # action indices that include ATTACK

WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), "weights", "vizdoom_live.pth")
RESULTS_PATH  = os.path.join(os.path.dirname(__file__), "vizdoom_live_results.png")
N_EPISODES    = 10000
FRAME_SKIP    = 4
SUMMARY_EVERY = 50


def _make_game() -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    # Display at 640×480 so the window is actually watchable.
    # The screen_buffer returned at this resolution gets resized to (1, 120, 160)
    # inside _preprocess before hitting the CNN — display and CNN input are independent.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    # KILLCOUNT isn't in the default config — same fix as vizdoom_defend.py
    game.set_available_game_variables([
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.HEALTH,
    ])
    game.init()

    return game


def _phase(episode: int) -> str:
    # Thresholds scaled from the 3000-episode version (÷4) to match
    # the same total gradient steps at frame_skip=1 vs frame_skip=4.
    if episode <= 50:
        return "chaos"
    elif episode <= 200:
        return "emergence"
    elif episode <= 500:
        return "consolidation"
    else:
        return "exploitation"


def _step_sleep(episode: int) -> float:
    """Slow down only the very early chaos phase.

    frame_skip=1 means each make_action() advances exactly 1 game tick, so
    the agent already acts at Doom's native 35 Hz rhythm. CNN backward passes
    (~15ms each) cap us around 67 actions/sec in practice — readable without
    help. We only add sleep in the first 50 episodes where random spinning
    happens too fast to follow.
    """
    if episode <= 50:
        return 0.02   # chaos: slow to ~30 actions/sec so the randomness is legible
    else:
        return 0.0    # CNN-limited from here, frame_skip=1 already provides fine resolution


def _narrative(episode: int, kill_avg: float, surv_avg: float, epsilon: float, attack_rate: float = 1.0) -> str:
    phase = _phase(episode)
    if phase == "chaos":
        desc = f"spamming ATTACK {attack_rate:.0%} of steps -- penalty hasn't landed yet"
    elif phase == "emergence":
        if attack_rate > 0.6:
            desc = f"still trigger-happy ({attack_rate:.0%} attacks) but kills are up"
        elif attack_rate > 0.3:
            desc = f"learning to wait -- attack rate dropping to {attack_rate:.0%}, aim improving"
        else:
            desc = f"real patience emerging -- only {attack_rate:.0%} attacks, shoots when it counts"
    elif phase == "consolidation":
        if kill_avg < 3.0:
            desc = f"controlled bursts ({attack_rate:.0%} atk rate), center held"
        elif kill_avg < 5.0:
            desc = f"deliberate shots paying off -- {kill_avg:.1f} kills at {attack_rate:.0%} attack rate"
        else:
            desc = f"efficient -- {kill_avg:.1f} kills, {attack_rate:.0%} attacks, barely wastes ammo"
    else:
        desc = (f"plays clean -- {kill_avg:.1f} kills at {attack_rate:.0%} attack rate"
                if kill_avg >= 5.0 else f"settling into exploitation, {attack_rate:.0%} attack rate")
    return (
        f"Episode {episode} [{phase}] -- avg {kill_avg:.1f} kills | "
        f"survive {surv_avg:.0f} steps | {desc} (e={epsilon:.2f})"
    )


def train():
    print("=" * 70)
    print("  Ashby x ViZDoom -- defend_the_center LIVE training")
    print("  Window stays open for all episodes. Watch it learn.")
    print("  frame_skip=1 -- one decision per game tick, 4x finer reflexes than basic training")
    print("  chaos (1-50) | emergence (50-200) | consolidation (200-500) | exploitation (500+)")
    print("=" * 70)
    print()

    game  = _make_game()
    agent = VizDoomDQNAgent(n_actions=len(LIVE_ACTIONS), input_shape=(1, 120, 160))

    device_name = torch.cuda.get_device_name(0) if agent.device.type == "cuda" else "CPU"
    print(f"Device: {agent.device} ({device_name})")
    print(f"Resolution: 160x120 -- CNN input: (1, 120, 160)")

    if os.path.exists(WEIGHTS_PATH):
        try:
            agent.load_weights(WEIGHTS_PATH)
            agent.policy_net.train()
            agent.epsilon = agent.epsilon_end
            print(f"Resuming from {WEIGHTS_PATH} (e={agent.epsilon:.2f})")
        except RuntimeError:
            # Old weights from a different action space (e.g. 3-action network) —
            # shape mismatch on the output layer. Delete and restart clean.
            os.remove(WEIGHTS_PATH)
            print("Old weights incompatible (action space changed) -- starting fresh")
    else:
        print("Old weights removed -- starting fresh with realistic reward shaping")

    kills_per_ep = []
    steps_per_ep = []

    for episode in range(1, N_EPISODES + 1):
        game.new_episode()
        frame            = game.get_state().screen_buffer
        episode_kills    = 0
        episode_attacks  = 0
        episode_steps    = 0
        phase            = _phase(episode)
        sleep_s          = _step_sleep(episode)

        while not game.is_episode_finished():
            action_idx = agent.act(frame)
            prev_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

            game.make_action(LIVE_ACTIONS[action_idx], FRAME_SKIP)
            done       = game.is_episode_finished()
            curr_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            health     = int(game.get_game_variable(vzd.GameVariable.HEALTH))

            kill_happened = curr_kills > prev_kills
            shot_fired    = action_idx in _SHOOTS  # actions 2/3/4 all include ATTACK
            died          = done and health <= 0

            if kill_happened:
                reward = 1.0
            elif died:
                reward = -0.5
            elif shot_fired:
                reward = -0.02  # light misfire penalty — discourages spam without blocking aggression
            else:
                reward = 0.0

            episode_kills   += int(kill_happened)
            episode_attacks += int(shot_fired)
            episode_steps   += 1

            next_frame = frame if done else game.get_state().screen_buffer
            agent.remember(frame, action_idx, reward, next_frame, done)
            agent.learn()
            frame = next_frame

            if sleep_s > 0:
                time.sleep(sleep_s)

        agent.decay_epsilon()
        kills_per_ep.append(episode_kills)
        steps_per_ep.append(episode_steps)

        window      = kills_per_ep[-50:]
        kill_avg    = float(np.mean(window))
        kill_rate   = float(np.mean([min(k, 1) for k in window]))
        surv_avg    = float(np.mean(steps_per_ep[-50:]))
        attack_rate = episode_attacks / max(episode_steps, 1)

        # \r rewrites the same terminal line each episode — clean live readout
        print(
            f"\rEp {episode:4d}/{N_EPISODES} [{phase:<13}] | "
            f"kills: {episode_kills:2d} | "
            f"atk: {attack_rate:.0%} | "
            f"e={agent.epsilon:.2f} | "
            f"avg: {kill_avg:.1f}  ",
            end="", flush=True
        )

        if episode % SUMMARY_EVERY == 0:
            print()  # commit the \r line to history
            print(f"\n  >>> {_narrative(episode, kill_avg, surv_avg, agent.epsilon, attack_rate)}\n")
            agent.save_weights(WEIGHTS_PATH)
            time.sleep(2.0)  # pause so the user can read the summary before the next burst

    game.close()
    agent.save_weights(WEIGHTS_PATH)
    print(f"\n\nWeights saved -> {WEIGHTS_PATH}")
    _plot(kills_per_ep, steps_per_ep)
    print(f"Plot saved    -> {RESULTS_PATH}")


def _plot(kills: list, steps: list):
    n = len(kills)
    w = min(100, n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Ashby x ViZDoom -- defend_the_center live training", fontsize=13)

    # Phase backgrounds make it easy to correlate curve shape with training phase
    bands = [
        (0,   50,  "red",    "chaos"),
        (50,  200, "orange", "emergence"),
        (200, 500, "gold",   "consolidation"),
        (500, n,   "green",  "exploitation"),
    ]
    for ax in (ax1, ax2):
        for start, end, color, label in bands:
            if start < n:
                ax.axvspan(start, min(end, n), alpha=0.07, color=color, label=label)

    ax1.plot(kills, alpha=0.2, color="tomato")
    if w > 1:
        ax1.plot(range(w, n + 1), np.convolve(kills, np.ones(w) / w, mode="valid"),
                 color="tomato", linewidth=2)
    ax1.set_ylabel("Kills/episode")
    ax1.set_title("Kills over time")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, alpha=0.2, color="steelblue")
    if w > 1:
        ax2.plot(range(w, n + 1), np.convolve(steps, np.ones(w) / w, mode="valid"),
                 color="steelblue", linewidth=2)
    ax2.set_ylabel("Steps survived")
    ax2.set_xlabel("Episode")
    ax2.set_title("Survival over time")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH, dpi=120)


if __name__ == "__main__":
    train()
