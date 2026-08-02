"""Recording mode -- you play 1v1 against Ashby, controller in hand.

Drives your side of a live rlgym_sim match with a physical DualSense while Ashby
plays the other side (its trained policy if one exists yet, the scripted bot
otherwise), and logs every (state, action) pair *for your car only* to disk.
Ashby's own actions never get captured -- the whole point is learning from the
human, not from whatever the bot happened to do that frame.

That log is the entire training signal for imitation.py -- there's no reward
function here, no notion of "good" or "bad", just "this is what a human did in
this situation". Garbage in, garbage out applies harder here than almost anywhere
else in the project, so play like you mean it.

Ctrl+C (or closing the render window) stops and saves whatever was captured so
far -- no need to finish a "clean" session.
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))
from env import make_env, scripted_bot_action, DATA_DIR, WEIGHTS_DIR
from rl_agent import RLAgent
from render import MatchRenderer

# --- Gamepad mapping (DualSense, via pygame/SDL) -----------------------------------
# SDL usually enumerates the DualSense with the same axis order as a standard XInput
# pad, but PS-pad quirks (USB vs Bluetooth, DS4Windows/HID mode) can shift indices.
# If steering feels backwards or an input doesn't respond, run `record.py --calibrate`
# and read the live axis/button indices off your own pad instead of guessing.
AXIS_STEER = 0        # left stick X
AXIS_THROTTLE = 1     # left stick Y -- pushed forward (up) reads negative
AXIS_YAW = 2          # right stick X -- aerial rotation, unused on the ground
AXIS_PITCH = 3        # right stick Y
AXIS_L2 = 4           # left trigger  -> boost
AXIS_R2 = 5           # right trigger -> handbrake
BTN_CROSS = 0         # jump
BTN_L1 = 4            # roll left  -- not in the brief, but the 8D action needs
BTN_R1 = 5            # roll right    *something* here, and the bumpers are free

DEADZONE = 0.08
TRIGGER_PRESS = 0.5   # L2/R2 rest at -1, full pull at +1 -- treat past halfway as "held"

POLICY_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_policy.pth")


def _axis(joystick, index: int, default: float = 0.0) -> float:
    if index >= joystick.get_numaxes():
        return default
    return joystick.get_axis(index)


def _apply_deadzone(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v


def read_gamepad_action(joystick) -> np.ndarray:
    """Reads the current pad state into the [throttle, steer, pitch, yaw, roll,
    jump, boost, handbrake] layout RocketSim's controls expect."""
    pygame.event.pump()

    steer = _apply_deadzone(_axis(joystick, AXIS_STEER))
    throttle = _apply_deadzone(-_axis(joystick, AXIS_THROTTLE))
    pitch = _apply_deadzone(-_axis(joystick, AXIS_PITCH))
    yaw = _apply_deadzone(_axis(joystick, AXIS_YAW))

    roll_left = joystick.get_button(BTN_L1) if joystick.get_numbuttons() > BTN_L1 else 0
    roll_right = joystick.get_button(BTN_R1) if joystick.get_numbuttons() > BTN_R1 else 0
    roll = float(roll_right) - float(roll_left)

    jump = 1.0 if joystick.get_button(BTN_CROSS) else 0.0

    # Triggers rest at -1 and read +1 fully pressed -- rescale to 0..1, then treat
    # boost/handbrake as on/off past the halfway point rather than analog.
    l2 = (_axis(joystick, AXIS_L2, -1.0) + 1.0) / 2.0
    r2 = (_axis(joystick, AXIS_R2, -1.0) + 1.0) / 2.0
    boost = 1.0 if l2 > TRIGGER_PRESS else 0.0
    handbrake = 1.0 if r2 > TRIGGER_PRESS else 0.0

    return np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)


def calibrate():
    """Prints live axis/button values so you can fix the constants above for your pad."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Plug in the DualSense and try again.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Calibrating: {joystick.get_name()}")
    print(f"  axes: {joystick.get_numaxes()}, buttons: {joystick.get_numbuttons()}")
    print("Move sticks / triggers, press buttons. Ctrl+C to stop.\n")

    try:
        while True:
            pygame.event.pump()
            axes = [round(joystick.get_axis(i), 2) for i in range(joystick.get_numaxes())]
            buttons = [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]
            print(f"\raxes={axes}  pressed={buttons}          ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")


def _connect_gamepad():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError(
            "No gamepad detected. record.py drives your side of the 1v1 with a physical "
            "DualSense -- plug one in, or run --calibrate to verify it's seen before a "
            "real session."
        )
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    name = joystick.get_name().lower()
    if "dualsense" not in name and "wireless controller" not in name:
        print(
            f"  note: connected pad reports as '{joystick.get_name()}' -- the mapping "
            "above is tuned for a DualSense, adjust with --calibrate if inputs feel off."
        )
    return joystick


def _make_opponent():
    """Ashby's side of the 1v1 -- its trained policy once one exists, the scripted
    bot before that. Either way the human always drives player 0, Ashby player 1."""
    if os.path.exists(POLICY_WEIGHTS):
        agent = RLAgent(imitation_weights=None)
        agent.load_weights(POLICY_WEIGHTS)
        print(f"Ashby is playing its trained policy ({POLICY_WEIGHTS}).")
        return agent.best_action
    print("No trained policy yet -- Ashby is playing the scripted bot instead.")
    return scripted_bot_action


def _save_session(states: list, actions: list, episodes: int, goals_for: int, goals_against: int):
    if not states:
        print("Nothing captured -- skipping save.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DATA_DIR, f"session_{timestamp}.pkl")

    with open(path, "wb") as f:
        pickle.dump({
            "states": np.stack(states).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32),
            "episodes": episodes,
            "goals": goals_for,
            "goals_against": goals_against,
        }, f)

    print(
        f"\nSaved {len(states)} frames across {episodes} episode(s) -> {path}"
        f"  ({goals_for}-{goals_against} vs Ashby)"
    )


def record():
    joystick = _connect_gamepad()
    print(f"Gamepad connected: {joystick.get_name()}")

    opponent_action = _make_opponent()
    env = make_env(spawn_opponents=True)
    renderer = MatchRenderer()
    print("Connected to rlgym_sim. You're blue, Ashby's orange.")
    print("Recording starts now -- Ctrl+C (or close the window) to stop and save.\n")

    states, actions = [], []
    episode = 0
    goals_for = 0
    goals_against = 0
    frame_in_episode = 0
    last_touch_state = False

    try:
        while True:
            episode += 1
            frame_in_episode = 0
            obs = env.reset()
            done = False

            while not done:
                action_human = read_gamepad_action(joystick)
                action_ashby = opponent_action(np.asarray(obs[1]))

                # Only the human's (state, action) pair is training data -- Ashby's
                # action for this frame gets fed into the sim and then thrown away.
                states.append(np.asarray(obs[0], dtype=np.float32))
                actions.append(action_human)

                obs, reward, done, info = env.step(np.stack([action_human, action_ashby]))
                frame_in_episode += 1
                time.sleep(1 / 30)  # 30 fps -- otherwise the sim outruns a human's reaction time

                if not renderer.update(info["state"]):
                    raise KeyboardInterrupt  # window closed -- same clean-stop path as Ctrl+C

                player = info["state"].players[0]
                if player.ball_touched and not last_touch_state:
                    print(f"\r  ep {episode:3d}  frame {frame_in_episode:4d}  -- ball touch!            ")
                last_touch_state = player.ball_touched

                if info["state"].blue_score > goals_for:
                    goals_for = info["state"].blue_score
                    print(f"\r  ep {episode:3d}  frame {frame_in_episode:4d}  -- GOAL! ({goals_for}-{goals_against})   ")
                if info["state"].orange_score > goals_against:
                    goals_against = info["state"].orange_score
                    print(f"\r  ep {episode:3d}  frame {frame_in_episode:4d}  -- Ashby scores. ({goals_for}-{goals_against})   ")

                print(f"\r  ep {episode:3d}  frame {frame_in_episode:4d}  captured={len(states):5d}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        renderer.close()
        env.close()
        _save_session(states, actions, episode, goals_for, goals_against)


def smoke_test():
    """--test: verify every moving part short of an actual human session -- gamepad
    detection, the 1v1 env, a few sim steps driven by real controller input, and the
    render window -- without needing a full recording run."""
    print("Checking gamepad...")
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected -- plug in the DualSense and try again.")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"  gamepad: ok ({joystick.get_name()})")

    print("Building 1v1 env (you vs Ashby)...")
    opponent_action = _make_opponent()
    env = make_env(spawn_opponents=True)
    obs = env.reset()
    assert len(obs) == 2, f"expected 2 players (1v1), got {len(obs)}"
    print("  env: ok (2 players, blue=you, orange=Ashby)")

    print("Opening render window...")
    renderer = MatchRenderer()
    print("  render: ok (window opened)")

    print("Running sim steps with live gamepad input...")
    for _ in range(10):
        action_human = read_gamepad_action(joystick)
        action_ashby = opponent_action(np.asarray(obs[1]))
        obs, reward, done, info = env.step(np.stack([action_human, action_ashby]))
        renderer.update(info["state"])
        if done:
            obs = env.reset()
    print("  sim loop: ok (10 steps, gamepad input -> env.step -> render)")

    renderer.close()
    env.close()
    print("record.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLGym -- human vs Ashby recording mode")
    parser.add_argument("--test", action="store_true", help="smoke-test gamepad/env/render without a real session")
    parser.add_argument("--calibrate", action="store_true", help="print live gamepad axis/button values")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    elif args.calibrate:
        calibrate()
    else:
        record()
