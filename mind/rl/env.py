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
    """Dense reward inspired by Lucy-SKG (the bot that beat Nexto) -- shaped densely
    enough that useful signal exists on nearly every tick instead of only the rare
    tick where a goal actually happens. Every component below is a proxy guessing at
    what leads to a goal (closing distance, lining up a shot, sending the ball
    goalward); GoalScored is the only one that's ground truth, which is why it
    dwarfs everything else in magnitude.
    """

    OFFENSIVE_POTENTIAL_WEIGHT = 0.4
    VELOCITY_BALL_TO_GOAL_WEIGHT = 0.5
    TOUCH_ACCEL_WEIGHT = 0.3
    DIST_WEIGHTED_ALIGN_WEIGHT = 0.2
    GOAL_SCORED_REWARD = 10.0
    CONCEDED_PENALTY = -10.0
    TOUCH_BALL_REWARD = 0.1
    BOOST_PICKUP_REWARD = 0.01
    BOOST_WASTE_PENALTY = -0.02

    BOOST_WASTE_RANGE = 500.0     # uu -- burning boost farther than this from the ball is "wasted"
    OFFENSIVE_DIST_DECAY = 2000.0  # uu -- PlayerToBallDistance's exponential falloff
    ALIGN_DIST_DECAY = 1000.0      # uu -- DistanceWeightedAlignment's exponential falloff

    def __init__(self):
        super().__init__()
        self._last_boost = {}
        self._blue_score = 0
        self._orange_score = 0
        self._blue_delta = 0
        self._orange_delta = 0
        self._ball_vel_before = np.zeros(3)
        self._prev_ball_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self._blue_score = initial_state.blue_score
        self._orange_score = initial_state.orange_score
        self._last_boost = {p.car_id: p.boost_amount for p in initial_state.players}
        self._prev_ball_vel = initial_state.ball.linear_velocity.copy()

    def pre_step(self, state: GameState):
        # Goals affect both players' rewards in the same tick -- compute the delta once
        # here instead of re-deriving it inside get_reward for every player.
        self._blue_delta = state.blue_score - self._blue_score
        self._orange_delta = state.orange_score - self._orange_score
        self._blue_score = state.blue_score
        self._orange_score = state.orange_score

        # Snapshot the ball's velocity as it stood BEFORE this tick's physics ran, so
        # TouchBallToGoalAcceleration can measure what a touch actually did to it --
        # "ball_touched is True" alone doesn't distinguish a full-power strike from a
        # graze that barely nudges it, and only one of those deserves the reward.
        self._ball_vel_before = self._prev_ball_vel
        self._prev_ball_vel = state.ball.linear_velocity.copy()

    @staticmethod
    def _goal_pos(player: PlayerData) -> np.ndarray:
        return np.array(
            common_values.ORANGE_GOAL_BACK if player.team_num == common_values.BLUE_TEAM
            else common_values.BLUE_GOAL_BACK
        )

    @staticmethod
    def _krc(*components: float) -> float:
        """Kinesthetic Reward Combination, the trick Lucy-SKG uses to fold several
        normalized sub-rewards into one: multiply them (as a sign-preserving geometric
        mean) instead of summing them, so a high score requires speed, alignment AND
        proximity all at once. A plain sum lets "flying toward the ball at full speed
        from completely the wrong angle" average out into a mediocre-but-positive
        number -- a product doesn't let any one bad component hide behind the others.
        """
        sign = 1.0
        magnitude = 1.0
        for c in components:
            if c < 0:
                sign *= -1.0
            magnitude *= abs(c)
        return sign * magnitude ** (1.0 / len(components))

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        car_pos = player.car_data.position
        car_vel = player.car_data.linear_velocity
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        goal_pos = self._goal_pos(player)

        to_ball = ball_pos - car_pos
        dist_to_ball = float(np.linalg.norm(to_ball))
        ball_to_goal = goal_pos - ball_pos
        ball_to_goal_dist = float(np.linalg.norm(ball_to_goal))

        # VelocityPlayerToBall -- how much of the car's speed is actually pointed at
        # the ball, not just how fast it's going. Standing still scores 0, driving flat
        # out in the wrong direction scores negative, same as a human coach would judge it.
        vel_to_ball = (
            float(np.dot(car_vel, to_ball / dist_to_ball)) / common_values.CAR_MAX_SPEED
            if dist_to_ball > 1e-3 else 0.0
        )

        # AlignBallToGoal -- is the agent on the correct side of the ball to send it
        # toward the net if it touches it right now? Raw speed and distance can't tell
        # good positioning from bad positioning on their own -- chasing the ball from
        # the wrong side just clears it for the opponent.
        align = float(rl_math.cosine_similarity(ball_to_goal, to_ball)) if dist_to_ball > 1e-3 else 0.0

        # PlayerToBallDistance -- exponential rather than linear falloff so the reward
        # landscape is steep right around the ball, where positioning actually decides
        # who gets there first, and nearly flat across the empty half of the field,
        # where 3000uu vs 3500uu away barely matters.
        dist_reward = float(np.exp(-dist_to_ball / self.OFFENSIVE_DIST_DECAY))

        # OffensivePotential -- see _krc's docstring for why this is a product of the
        # three components above instead of a sum.
        offensive_potential = self._krc(vel_to_ball, align, dist_reward)

        # VelocityBallToGoal -- the ball's own velocity component toward the opponent's
        # net. This is the closest thing to "are we creating a scoring chance right
        # now" that's measurable every tick instead of only at the goal itself.
        vel_ball_to_goal = (
            float(np.dot(ball_vel, ball_to_goal / ball_to_goal_dist)) / common_values.BALL_MAX_SPEED
            if ball_to_goal_dist > 1e-3 else 0.0
        )

        # TouchBallToGoalAcceleration -- rewards the CHANGE a touch caused in the ball's
        # velocity toward goal, not the touch itself. A touch that stops the ball dead
        # or deflects it sideways scores ~0 here even though ball_touched is True; one
        # that sends it rocketing goalward gets the full weight. This is what separates
        # "made contact" from "made a good play".
        if player.ball_touched and ball_to_goal_dist > 1e-3:
            delta_v = ball_vel - self._ball_vel_before
            touch_accel = float(np.dot(delta_v, ball_to_goal / ball_to_goal_dist)) / common_values.BALL_MAX_SPEED
        else:
            touch_accel = 0.0

        # DistanceWeightedAlignment -- the same alignment signal as above, scaled down
        # the further the agent is from the ball. Being perfectly lined up for a shot
        # from halfway across the map isn't worth much yet; being lined up while
        # actually within striking range is.
        dist_weighted_align = align * float(np.exp(-dist_to_ball / self.ALIGN_DIST_DECAY))

        reward = (
            self.OFFENSIVE_POTENTIAL_WEIGHT * offensive_potential
            + self.VELOCITY_BALL_TO_GOAL_WEIGHT * vel_ball_to_goal
            + self.TOUCH_ACCEL_WEIGHT * touch_accel
            + self.DIST_WEIGHTED_ALIGN_WEIGHT * dist_weighted_align
        )

        # GoalScored -- the only reward here that matches the actual win condition
        # instead of guessing at it, so it gets by far the largest magnitude in the
        # whole function on purpose.
        if player.team_num == common_values.BLUE_TEAM:
            reward += self.GOAL_SCORED_REWARD * self._blue_delta + self.CONCEDED_PENALTY * self._orange_delta
        else:
            reward += self.GOAL_SCORED_REWARD * self._orange_delta + self.CONCEDED_PENALTY * self._blue_delta

        # TouchBall -- a small flat bonus for making contact at all, on top of whatever
        # TouchBallToGoalAcceleration decided that contact was worth. Early in training,
        # when the agent barely reaches the ball, this is what teaches "go touch it"
        # before it's coordinated enough to make the touch itself count for much.
        if player.ball_touched:
            reward += self.TOUCH_BALL_REWARD

        # BoostManagement -- boost is a finite resource shared with nobody but future-you;
        # the only time spending it is clearly correct is when it's closing distance to
        # the ball. Boosting in a corner 4000uu from the play burns a resource a real
        # attack needed. Picking boost up is unconditionally good, so it's cheap to reward.
        prev_boost = self._last_boost.get(player.car_id, player.boost_amount)
        if player.boost_amount > prev_boost:
            reward += self.BOOST_PICKUP_REWARD
        elif player.boost_amount < prev_boost and dist_to_ball > self.BOOST_WASTE_RANGE:
            reward += self.BOOST_WASTE_PENALTY
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
