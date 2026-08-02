"""Recording mode -- you play, Ashby watches.

Drives a single car through rlgym_sim with a physical gamepad and logs every
(state, action) pair to disk. That log is the entire training signal for imitation.py --
there's no reward function here, no notion of "good" or "bad", just "this is what a human
did in this situation". Garbage in, garbage out applies harder here than almost anywhere
else in the project, so play like you mean it.

Ctrl+C stops and saves whatever was captured so far -- no need to finish a "clean" session.
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
from env import make_env, DATA_DIR

# --- Gamepad mapping ---------------------------------------------------------------
# Tuned for a standard XInput Xbox controller on Windows. SDL's raw Joystick axis
# order shifts across OS/driver combinations, so if steering feels backwards or the
# throttle doesn't respond, run `record.py --calibrate` and read the live axis/button
# indices off your own pad instead of guessing.
AXIS_STEER = 0       # left stick X
AXIS_YAW = 2          # right stick X -- aerial rotation, unused on the ground
AXIS_PITCH = 3        # right stick Y
AXIS_LTRIGGER = 4     # left trigger  -> brake / reverse
AXIS_RTRIGGER = 5     # right trigger -> throttle
BTN_JUMP = 0          # A
BTN_BOOST = 1         # B
BTN_HANDBRAKE = 2     # X
BTN_ROLL_LEFT = 4     # LB
BTN_ROLL_RIGHT = 5    # RB

DEADZONE = 0.08


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

    # Triggers rest at -1 and read +1 fully pressed -- rescale to 0..1 before combining.
    rt = (_axis(joystick, AXIS_RTRIGGER, -1.0) + 1.0) / 2.0
    lt = (_axis(joystick, AXIS_LTRIGGER, -1.0) + 1.0) / 2.0
    throttle = float(np.clip(rt - lt, -1.0, 1.0))

    steer = _apply_deadzone(_axis(joystick, AXIS_STEER))
    pitch = _apply_deadzone(_axis(joystick, AXIS_PITCH))
    yaw = _apply_deadzone(_axis(joystick, AXIS_YAW))

    roll_left = joystick.get_button(BTN_ROLL_LEFT) if joystick.get_numbuttons() > BTN_ROLL_LEFT else 0
    roll_right = joystick.get_button(BTN_ROLL_RIGHT) if joystick.get_numbuttons() > BTN_ROLL_RIGHT else 0
    roll = float(roll_right) - float(roll_left)

    jump = 1.0 if joystick.get_button(BTN_JUMP) else 0.0
    boost = 1.0 if joystick.get_button(BTN_BOOST) else 0.0
    handbrake = 1.0 if joystick.get_button(BTN_HANDBRAKE) else 0.0

    return np.array([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)


def calibrate():
    """Prints live axis/button values so you can fix the constants above for your pad."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Plug one in and try again.")
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
            "No gamepad detected. record.py drives the sim car with a physical controller "
            "-- plug one in, or run --calibrate to verify it's seen before a real session."
        )
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick


def _save_session(states: list, actions: list, episodes: int, goals: int):
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
            "goals": goals,
        }, f)

    print(f"\nSaved {len(states)} frames across {episodes} episode(s) -> {path}")


def record():
    joystick = _connect_gamepad()
    print(f"Gamepad connected: {joystick.get_name()}")

    env = make_env(spawn_opponents=False)
    print("Connected to rlgym_sim. Recording starts now -- Ctrl+C to stop and save.\n")

    states, actions = [], []
    episode = 0
    goals = 0
    frame_in_episode = 0
    last_touch_state = False

    try:
        while True:
            episode += 1
            frame_in_episode = 0
            obs = env.reset()
            done = False

            while not done:
                action = read_gamepad_action(joystick)

                states.append(np.asarray(obs, dtype=np.float32))
                actions.append(action)

                obs, reward, done, info = env.step(action.reshape(1, -1))
                frame_in_episode += 1

                player = info["state"].players[0]
                if player.ball_touched and not last_touch_state:
                    print(f"\r  ep {episode:3d}  frame {frame_in_episode:4d}  -- ball touch!            ")
                last_touch_state = player.ball_touched

                if info["state"].blue_score > goals:
                    goals = info["state"].blue_score
                    print(f"\r  ep {episode:3d}  frame {frame_in_episode:4d}  -- GOAL! ({goals} total)   ")

                print(f"\r  ep {episode:3d}  frame {frame_in_episode:4d}  captured={len(states):5d}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        env.close()
        _save_session(states, actions, episode, goals)


def smoke_test():
    """--test: verify the gamepad-free parts of the pipeline without needing hardware
    or a real recording session. This is what CI / `make rl-record` sanity checks run."""
    print("Checking rlgym_sim connection...")
    env = make_env(spawn_opponents=False)
    obs = env.reset()
    assert np.shape(obs) == (18,), f"expected 18D state, got {np.shape(obs)}"

    fake_action = np.zeros((1, 8), dtype=np.float32)
    obs, reward, done, info = env.step(fake_action)
    assert "state" in info
    env.close()
    print("  rlgym_sim env: ok (18D state, 8D action, info['state'] populated)")

    pygame.init()
    pygame.joystick.init()
    n_pads = pygame.joystick.get_count()
    print(f"  pygame joystick backend: ok ({n_pads} gamepad(s) detected)")

    print("record.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLGym -- human recording mode")
    parser.add_argument("--test", action="store_true", help="smoke-test imports/env without a real session")
    parser.add_argument("--calibrate", action="store_true", help="print live gamepad axis/button values")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    elif args.calibrate:
        calibrate()
    else:
        record()
