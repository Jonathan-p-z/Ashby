import os
import sys
import vizdoom as vzd

sys.path.insert(0, os.path.dirname(__file__))
from vizdoom_agent import VizDoomDQNAgent, ACTIONS

WEIGHTS_PATH      = os.path.join(os.path.dirname(__file__), "weights", "vizdoom_basic.pth")
N_WATCH_EPISODES  = 10
FRAME_SKIP        = 4


def _make_game() -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    return game


def watch():
    if not os.path.exists(WEIGHTS_PATH):
        print("No weights found -- run 'make vizdoom-train' first")
        sys.exit(0)

    agent = VizDoomDQNAgent()
    agent.load_weights(WEIGHTS_PATH)
    agent.epsilon = 0.0  # pure exploitation, no random moves

    game = _make_game()

    print("=" * 50)
    print("  Ashby x ViZDoom -- watch mode")
    print("  epsilon = 0.0 -- pure exploitation")
    print("=" * 50)

    total_kills = 0

    for ep in range(1, N_WATCH_EPISODES + 1):
        game.new_episode()
        frame = game.get_state().screen_buffer
        ep_kills = 0

        while not game.is_episode_finished():
            action_idx = agent.best_action(frame)
            prev_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

            game.make_action(ACTIONS[action_idx], FRAME_SKIP)

            if not game.is_episode_finished():
                curr_kills  = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                ep_kills   += max(0, curr_kills - prev_kills)
                frame       = game.get_state().screen_buffer

        vizdoom_score = game.get_total_reward()
        result        = "KILL    " if ep_kills > 0 else "timeout "
        total_kills  += ep_kills
        print(f"  Episode {ep:2d} -- {result} | kills {ep_kills} | vizdoom score {vizdoom_score:6.0f}")

    game.close()
    print(f"\n  {total_kills}/{N_WATCH_EPISODES} episodes with a kill")


if __name__ == "__main__":
    watch()
