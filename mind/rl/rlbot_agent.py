"""RLBot v2 wrapper -- puts Ashby's trained network in an actual Rocket League match,
not just RocketSim.

Same network, same 18D state / 8D action layout rl_agent.py was trained on (env.py) --
the only job here is translating between RLBot's GamePacket/ControllerState and that
exact format every tick. RocketSim (what env.py trains against) mirrors the real game's
physics and coordinate conventions, so reusing rlgym_sim's own euler_to_rotation /
rotation_to_quaternion functions on RLBot's Euler rotator reconstructs the same
quaternion the training data used, no separate math to get subtly wrong.

No training happens here -- if Ashby plays badly in a real match, that's a rl_train.py
problem, not a wrapper problem.
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import POS_COEF, VEL_COEF, WEIGHTS_DIR
from rl_agent import RLAgent

from rlgym_sim.utils import common_values
from rlgym_sim.utils import math as rl_math

from rlbot import flat
from rlbot.managers import Bot

POLICY_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_policy.pth")
AGENT_ID = "yaiito/ashby"

# 180-degree rotation about the vertical axis, applied elementwise to any world-frame
# vector (position, velocity, or a rotation matrix row) -- AshbyObsBuilder's orange-team
# view, where "forward" always means "toward the opponent's goal".
MIRROR = np.array([-1.0, -1.0, 1.0], dtype=np.float32)


def _vec3(v: flat.Vector3) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def _quaternion(rotator: flat.Rotator, invert: bool) -> np.ndarray:
    """RLBot hands over Euler angles; env.py trained on quaternions."""
    pyr = np.array([rotator.pitch, rotator.yaw, rotator.roll], dtype=np.float32)
    rot_mtx = rl_math.euler_to_rotation(pyr)
    if invert:
        rot_mtx = rot_mtx * MIRROR[:, None]  # negate the x/y rows, leave z alone
    return rl_math.rotation_to_quaternion(rot_mtx)


def build_obs(packet: flat.GamePacket, index: int) -> np.ndarray:
    """Reconstructs AshbyObsBuilder's 18D state (env.py) from a live GamePacket."""
    player = packet.players[index]
    ball = packet.balls[0]
    invert = player.team == common_values.ORANGE_TEAM

    car_pos = _vec3(player.physics.location)
    car_vel = _vec3(player.physics.velocity)
    ball_pos = _vec3(ball.physics.location)
    ball_vel = _vec3(ball.physics.velocity)
    quat = _quaternion(player.physics.rotation, invert)

    if invert:
        car_pos = car_pos * MIRROR
        car_vel = car_vel * MIRROR
        ball_pos = ball_pos * MIRROR
        ball_vel = ball_vel * MIRROR

    boost = player.boost / 100.0  # RLBot reports 0-100, training data is 0-1
    on_ground = 1.0 if player.air_state == flat.AirState.OnGround else 0.0

    return np.concatenate([
        car_pos * POS_COEF,
        car_vel * VEL_COEF,
        quat,
        ball_pos * POS_COEF,
        ball_vel * VEL_COEF,
        [boost, on_ground],
    ]).astype(np.float32)


def to_controller(action: np.ndarray) -> flat.ControllerState:
    """[throttle, steer, pitch, yaw, roll, jump, boost, handbrake], clipped/thresholded
    the same way rlgym_sim's ContinuousAction does at training time -- the network
    should see the same effective action space here as it did in the sim."""
    continuous = np.clip(action[:5], -1.0, 1.0)
    jump, boost, handbrake = action[5:] > 0
    return flat.ControllerState(
        throttle=float(continuous[0]),
        steer=float(continuous[1]),
        pitch=float(continuous[2]),
        yaw=float(continuous[3]),
        roll=float(continuous[4]),
        jump=bool(jump),
        boost=bool(boost),
        handbrake=bool(handbrake),
    )


class AshbyBot(Bot):
    def initialize(self):
        if not os.path.exists(POLICY_WEIGHTS):
            raise FileNotFoundError(
                f"No policy weights at {POLICY_WEIGHTS} -- run `make rl-train-rl` "
                "(or at least `make rl-train-bc`) before sending Ashby into a real match."
            )
        self.agent = RLAgent(imitation_weights=None)
        self.agent.load_weights(POLICY_WEIGHTS)
        self.logger.info(f"Loaded policy from {POLICY_WEIGHTS}")

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        obs = build_obs(packet, self.index)
        action = self.agent.best_action(obs)
        return to_controller(action)


def smoke_test():
    """--test: runs the GamePacket -> obs -> ControllerState pipeline against a
    synthetic packet. No live RLBotServer or running game needed -- those can only be
    exercised for real once Ashby is actually pointed at a match from the RLBot GUI.
    Falls back to a fresh (untrained) policy if rl_policy.pth doesn't exist yet -- this
    is testing the plumbing, not the play quality."""
    print("Building a synthetic GamePacket...")
    packet = flat.GamePacket(
        players=[
            flat.PlayerInfo(
                physics=flat.Physics(
                    location=flat.Vector3(100.0, -200.0, 17.0),
                    rotation=flat.Rotator(0.0, 1.2, 0.0),
                    velocity=flat.Vector3(300.0, 0.0, 0.0),
                ),
                boost=75.0,
                team=0,
                air_state=flat.AirState.OnGround,
            ),
            flat.PlayerInfo(
                physics=flat.Physics(location=flat.Vector3(-100.0, 200.0, 17.0)),
                boost=50.0,
                team=1,
            ),
        ],
        balls=[flat.BallInfo(physics=flat.Physics(location=flat.Vector3(0.0, 0.0, 93.0)))],
    )

    obs = build_obs(packet, index=0)
    assert obs.shape == (18,), f"expected 18D state, got {obs.shape}"
    print(f"  build_obs: ok ({obs.shape[0]}D state)")

    print("Loading policy and running one forward pass...")
    agent = RLAgent(imitation_weights=None)
    if os.path.exists(POLICY_WEIGHTS):
        agent.load_weights(POLICY_WEIGHTS)
    action = agent.best_action(obs)
    controller = to_controller(action)
    assert isinstance(controller, flat.ControllerState)
    print(f"  policy + controller conversion: ok (throttle={controller.throttle:.2f}, steer={controller.steer:.2f})")

    print("rlbot_agent.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLBot -- live match wrapper")
    parser.add_argument("--test", action="store_true", help="smoke-test the obs/action pipeline on a synthetic packet")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        AshbyBot(AGENT_ID).run()
