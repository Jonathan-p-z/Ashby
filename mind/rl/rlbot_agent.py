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
import torch

sys.path.insert(0, os.path.dirname(__file__))
from env import POS_COEF, VEL_COEF, WEIGHTS_DIR, STATE_SIZE, ACTION_SIZE
from rl_agent import RLAgent
from ppo_train import POLICY_LAYER_SIZES as PPO_POLICY_LAYER_SIZES

from rlgym_sim.utils import common_values
from rlgym_sim.utils import math as rl_math

from rlbot import flat
from rlbot.managers import Bot

POLICY_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_policy.pth")
PPO_CHECKPOINTS_DIR = os.path.join(WEIGHTS_DIR, "ppo")
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
    """Reconstructs AshbyObsBuilder's 18D state (env.py) from a live GamePacket.

    Between rounds (kickoff countdown, goal replay, before the match fully spins up)
    RLBot can hand over a packet with no ball or no players yet -- returning a zeroed
    state for that one tick beats crashing get_output over a transient empty packet.
    """
    if not packet.balls or index >= len(packet.players):
        return np.zeros(STATE_SIZE, dtype=np.float32)

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


def _latest_ppo_checkpoint(checkpoints_dir: str) -> str | None:
    """rlgym-ppo names each checkpoint folder after its cumulative timestep count
    (see ppo_train.py's checkpoints_save_folder) -- the highest number is the most
    recently saved one. Returns the path to that checkpoint's policy weights, or
    None if there's no PPO checkpoint yet.
    """
    if not os.path.isdir(checkpoints_dir):
        return None
    numbered = [
        d for d in os.listdir(checkpoints_dir)
        if d.isdigit() and os.path.isdir(os.path.join(checkpoints_dir, d))
    ]
    if not numbered:
        return None
    latest = max(numbered, key=int)
    policy_path = os.path.join(checkpoints_dir, latest, "PPO_POLICY.pt")
    return policy_path if os.path.exists(policy_path) else None


class _PPOPolicy:
    """Wraps rlgym-ppo's ContinuousPolicy behind the same .best_action(obs)
    interface RLAgent exposes, so AshbyBot.get_output doesn't need to know which
    training pipeline actually produced the loaded weights.

    Can't just load a PPO_POLICY.pt state dict into RLAgent's ImitationNet actor --
    ContinuousPolicy ends in Linear(256, ACTION_SIZE*2) + Tanh (mean and std
    concatenated, see ppo_train.py's _load_imitation_into_policy for the full
    story), not ImitationNet's bare Linear(256, ACTION_SIZE). Reconstructing the
    real ContinuousPolicy class and taking its deterministic mean output is the
    only faithful way to reproduce what the PPO-trained policy actually does.
    """

    def __init__(self, checkpoint_path: str):
        from rlgym_ppo.ppo import ContinuousPolicy

        self.policy = ContinuousPolicy(STATE_SIZE, ACTION_SIZE * 2, PPO_POLICY_LAYER_SIZES, "cpu")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def best_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.policy.get_action(state_t, deterministic=True)
        return mean[0].numpy()


class AshbyBot(Bot):
    def initialize(self):
        ppo_checkpoint = _latest_ppo_checkpoint(PPO_CHECKPOINTS_DIR)
        if ppo_checkpoint:
            self.agent = _PPOPolicy(ppo_checkpoint)
            self.logger.info(f"Loaded PPO policy from {ppo_checkpoint}")
        elif os.path.exists(POLICY_WEIGHTS):
            self.agent = RLAgent(imitation_weights=None)
            self.agent.load_weights(POLICY_WEIGHTS)
            self.logger.info(f"No PPO checkpoint under {PPO_CHECKPOINTS_DIR} -- loaded {POLICY_WEIGHTS} instead")
        else:
            raise FileNotFoundError(
                f"No PPO checkpoint under {PPO_CHECKPOINTS_DIR} and no policy weights at "
                f"{POLICY_WEIGHTS} either -- run `make rl-train-ppo` (or at least "
                "`make rl-train-bc`) before sending Ashby into a real match."
            )

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        # Between rounds (kickoff countdown, goal replay, before the match fully spins
        # up) RLBot can hand over a packet with no ball or no players yet -- skip the
        # network entirely and hand back neutral input rather than feeding it a state
        # build_obs had to zero-fill anyway.
        if not packet.balls or len(packet.players) <= self.index:
            return flat.ControllerState()

        obs = build_obs(packet, self.index)
        action = self.agent.best_action(obs)
        return to_controller(action)


def smoke_test():
    """--test: runs the GamePacket -> obs -> ControllerState pipeline against a
    synthetic packet. No live RLBotServer or running game needed -- those can only be
    exercised for real once Ashby is actually pointed at a match from the RLBot GUI.
    Falls back to a fresh (untrained) policy if neither a PPO checkpoint nor
    rl_policy.pth exists yet -- this is testing the plumbing, not the play quality."""
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
    ppo_checkpoint = _latest_ppo_checkpoint(PPO_CHECKPOINTS_DIR)
    if ppo_checkpoint:
        agent = _PPOPolicy(ppo_checkpoint)
        print(f"  source: PPO checkpoint at {ppo_checkpoint}")
    else:
        agent = RLAgent(imitation_weights=None)
        if os.path.exists(POLICY_WEIGHTS):
            agent.load_weights(POLICY_WEIGHTS)
            print(f"  source: {POLICY_WEIGHTS}")
        else:
            print("  source: none found -- untrained network, testing plumbing only")
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
