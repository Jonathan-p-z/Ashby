import os
import sys
import vizdoom as vzd

sys.path.insert(0, os.path.dirname(__file__))
from vizdoom_agent import VizDoomDQNAgent
from vizdoom_defend import DEFEND_ACTIONS, _make_game

WEIGHTS_PATH     = os.path.join(os.path.dirname(__file__), "weights", "vizdoom_defend.pth")
N_WATCH_EPISODES = 10
FRAME_SKIP       = 4


def watch():
    if not os.path.exists(WEIGHTS_PATH):
        print("No weights found -- run 'make vizdoom-defend' first")
        sys.exit(0)

    agent = VizDoomDQNAgent()
    agent.load_weights(WEIGHTS_PATH)
    agent.epsilon = 0.0

    game = _make_game(visible=True)

    print("=" * 50)
    print("  Ashby x ViZDoom -- defend_the_center watch")
    print("  epsilon = 0.0 -- pure exploitation")
    print("=" * 50)

    total_kills = 0

    for ep in range(1, N_WATCH_EPISODES + 1):
        game.new_episode()
        frame    = game.get_state().screen_buffer
        ep_kills = 0
        ep_steps = 0

        while not game.is_episode_finished():
            action_idx = agent.best_action(frame)
            prev_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))

            game.make_action(DEFEND_ACTIONS[action_idx], FRAME_SKIP)

            if not game.is_episode_finished():
                curr_kills  = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                ep_kills   += max(0, curr_kills - prev_kills)
                ep_steps   += 1
                frame       = game.get_state().screen_buffer

        health      = int(game.get_game_variable(vzd.GameVariable.HEALTH))
        outcome     = "DEAD   " if health <= 0 else "timeout"
        total_kills += ep_kills
        print(f"  Episode {ep:2d} -- {outcome} | kills {ep_kills:2d} | survived {ep_steps * FRAME_SKIP} frames")

    game.close()
    print(f"\n  Total kills across {N_WATCH_EPISODES} episodes: {total_kills}")


if __name__ == "__main__":
    watch()
