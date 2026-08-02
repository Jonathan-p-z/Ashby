"""Shared rlgym_sim wiring — obs builder, reward shaping, env factory, scripted bot.

Every other file in this package (record, imitation, rl_agent, rl_train, watch) needs the
exact same 18D state layout and the exact same action ordering, or a model trained on one
script's observations silently misreads another's. Centralizing it here means there's only
one place that can get that wrong instead of five.
"""

import os
import numpy as np
import torch
import gym.spaces

import rlgym_sim
from rlgym_sim.utils import common_values
from rlgym_sim.utils import math as rl_math
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

def _ensure_rocketsim_assets():
    """RocketSim needs its arena collision meshes on disk before it can build an Arena.

    It looks for a "collision_meshes/soccar/*.cmf" folder relative to the cwd by default,
    which breaks the moment you run a script from anywhere else. rlgym-rocket-league (the
    real-game RLGym v2 package, also on the install list) happens to ship the exact same
    mesh files under its own package directory, so we point RocketSim at those instead of
    asking for a separate download.
    """
    if os.environ.get("RS_COLLISION_MESHES"):
        return  # user already pointed RocketSim at their own copy

    mesh_dir = None
    try:
        import rlgym as _rlgym_v2  # namespace package -- no __file__, use __path__ instead
        candidate = os.path.join(list(_rlgym_v2.__path__)[0], "rocket_league", "sim", "collision_meshes")
        if os.path.isdir(candidate):
            mesh_dir = candidate
    except ImportError:
        pass

    if mesh_dir is None:
        raise RuntimeError(
            "RocketSim can't find its collision meshes. `pip install rlgym-rocket-league` "
            "ships them (used here), or set RS_COLLISION_MESHES to a folder containing a "
            "'soccar' subdirectory of .cmf files."
        )

    import RocketSim as rsim
    rsim.init(mesh_dir)


_ensure_rocketsim_assets()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_SIZE = 18
ACTION_SIZE = 8

# RocketSim steps physics at 120Hz. tick_skip=8 means one env.step() = 8 physics ticks,
# i.e. 15 decisions/sec -- fast enough to react, slow enough that training isn't
# bottlenecked on Python overhead between every single tick.
PHYSICS_TPS = 120.0
TICK_SKIP = 8
SECONDS_PER_STEP = TICK_SKIP / PHYSICS_TPS

# ~33 real-game seconds per episode. Long enough for a possession or two, short enough
# that a bad episode doesn't waste an eternity of simulated time before we learn from it.
EPISODE_TIMEOUT_STEPS = 500

# Same normalization coefficients as rlgym's own DefaultObs -- not exact bounds (the ball
# can outrun this), just enough to keep everything roughly in [-2, 2] for the network.
POS_COEF = 1 / 2300.0
VEL_COEF = 1 / 2300.0

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class AshbyObsBuilder(ObsBuilder):
    """Builds the 18D state Ashby actually trains on.

    3 (car pos) + 3 (car vel) + 4 (car rotation, quaternion) + 3 (ball pos) + 3 (ball vel)
    + 1 (boost) + 1 (on_ground) = 18. Rotation is a quaternion rather than raw pitch/yaw/roll
    on purpose -- Euler angles wrap around at +-180 degrees, which means two nearly-identical
    orientations can land on opposite ends of the input range. A network shouldn't have to
    learn that discontinuity is meaningless.

    Orange-team players get the inverted car/ball data, same trick rlgym's own DefaultObs
    uses, so "forward" always means "toward the opponent's goal" no matter which side of
    the field the agent spawns on. One network, either team.
    """

    def get_obs_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, shape=(STATE_SIZE,))

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        if player.team_num == common_values.ORANGE_TEAM:
            car = player.inverted_car_data
            ball = state.inverted_ball
        else:
            car = player.car_data
            ball = state.ball

        return np.concatenate([
            car.position * POS_COEF,
            car.linear_velocity * VEL_COEF,
            car.quaternion,
            ball.position * POS_COEF,
            ball.linear_velocity * VEL_COEF,
            [player.boost_amount, float(player.on_ground)],
        ]).astype(np.float32)


class AshbyReward(RewardFunction):
    """Reward shaping tuned to push the agent toward the ball early, not just toward goals.

    Goals are the sparsest possible signal -- an agent that only gets rewarded for scoring
    can wander for thousands of episodes without ever stumbling into one. Touch and boost
    rewards give it something to climb every few seconds while it's still bad at the game;
    the idle penalty exists so "stand still in a corner" isn't a stable local optimum.
    """

    GOAL_REWARD = 1.0
    CONCEDED_PENALTY = -1.0
    TOUCH_REWARD = 0.1
    BOOST_REWARD = 0.05
    IDLE_PENALTY_PER_SEC = -0.01

    def __init__(self):
        super().__init__()
        self._last_boost = {}
        self._blue_score = 0
        self._orange_score = 0
        self._blue_delta = 0
        self._orange_delta = 0

    def reset(self, initial_state: GameState):
        self._blue_score = initial_state.blue_score
        self._orange_score = initial_state.orange_score
        self._last_boost = {p.car_id: p.boost_amount for p in initial_state.players}

    def pre_step(self, state: GameState):
        # Goals affect both players' rewards in the same tick -- compute the delta once
        # here instead of re-deriving it inside get_reward for every player.
        self._blue_delta = state.blue_score - self._blue_score
        self._orange_delta = state.orange_score - self._orange_score
        self._blue_score = state.blue_score
        self._orange_score = state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        if player.team_num == common_values.BLUE_TEAM:
            reward += self.GOAL_REWARD * self._blue_delta + self.CONCEDED_PENALTY * self._orange_delta
        else:
            reward += self.GOAL_REWARD * self._orange_delta + self.CONCEDED_PENALTY * self._blue_delta

        if player.ball_touched:
            reward += self.TOUCH_REWARD
        else:
            reward += self.IDLE_PENALTY_PER_SEC * SECONDS_PER_STEP

        # A pickup is the only way boost goes up in this sim -- no regen, so any positive
        # delta since last step means a pad was grabbed.
        prev_boost = self._last_boost.get(player.car_id, player.boost_amount)
        if player.boost_amount > prev_boost:
            reward += self.BOOST_REWARD
        self._last_boost[player.car_id] = player.boost_amount

        return reward


def make_env(spawn_opponents: bool = False):
    """Builds an rlgym_sim Gym env wired to Ashby's obs/reward/action conventions.

    spawn_opponents=False -> one car, empty field. Used for recording and imitation
    replay, where we only care about a single human-controlled car.

    spawn_opponents=True -> two cars, 1v1. Used for autonomous RL, where index 0 is
    always the learning agent and index 1 is whatever opponent rl_train/watch decide
    to drive (currently the scripted bot below).
    """
    return rlgym_sim.make(
        tick_skip=TICK_SKIP,
        spawn_opponents=spawn_opponents,
        team_size=1,
        terminal_conditions=(TimeoutCondition(EPISODE_TIMEOUT_STEPS), GoalScoredCondition()),
        reward_fn=AshbyReward(),
        obs_builder=AshbyObsBuilder(),
        action_parser=ContinuousAction(),
    )


def scripted_bot_action(obs: np.ndarray) -> np.ndarray:
    """A rule-based opponent: drive at the ball, boost when it's clear to do so.

    RocketSim is pure physics -- it doesn't ship Psyonix's bot AI, so there's no "real"
    bot to train against. This is a stand-in just competent enough to give the RL agent
    something that contests the ball, without needing a second trained network before
    the first one exists. No aerials, no jumps -- ground game only, on purpose.

    Takes the same self-centric 18D obs the learning agent sees (car pos/vel/rot at
    [0:10], ball pos/vel at [10:16], boost at [16]) so it works identically for whichever
    car it's driving, blue or orange.
    """
    car_pos = obs[0:3]
    quat = obs[6:10]
    ball_pos = obs[10:13]
    boost = obs[16]

    forward = rl_math.quat_to_rot_mtx(quat)[:, 0]
    forward[2] = 0.0
    forward_norm = np.linalg.norm(forward)
    forward = forward / forward_norm if forward_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    to_ball = ball_pos - car_pos
    to_ball[2] = 0.0
    dist = np.linalg.norm(to_ball)
    direction = to_ball / dist if dist > 1e-3 else forward

    # Signed angle between where the car is facing and where the ball is, via the
    # z-component of the cross product -- positive means "steer right".
    cross_z = forward[0] * direction[1] - forward[1] * direction[0]
    facing_ball = float(np.dot(forward, direction))

    steer = float(np.clip(cross_z * 3.0, -1.0, 1.0))
    throttle = 1.0 if facing_ball > -0.5 else -0.3  # back out of a bad angle instead of stalling
    boost_input = 1.0 if (dist > 800 and boost > 0.3 and facing_ball > 0.8) else 0.0

    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    action[0] = throttle
    action[1] = steer
    action[6] = boost_input
    return action
