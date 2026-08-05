"""Turns downloaded ballchasing.com replays into a behavioral-cloning dataset.

boxcars_py only hands back the raw network-replication stream -- an ordered
list of "this actor's attribute changed to X" deltas keyed by a numeric
actor_id, not a ready per-tick gamestate. Building one (state, action) row per
car per frame means replaying that stream ourselves: track which actor_id is
the ball vs. which is a car, carry the last-known value of every attribute
forward between updates (Rocket League only replicates *changes*, not every
attribute every frame), and walk the Car -> PlayerReplicationInfo -> Team
actor chain to know which side each car is on.

Note on the dependency: the boxcars-py wheel that pip resolves for this
Python/OS combo is 0.1.0, its very first release, which serializes the
*undecoded* network-compression internals (raw quantized ints) instead of
real floats -- useless for anything numeric. What's actually installed here
is a wheel rebuilt from github.com/SaltieRL/boxcars-py via `maturin build
--release`, which pins a current `boxcars` Rust crate that decodes
RigidBody/Quaternion attributes into real coordinates before they ever reach
Python. If parse_replays.py ever starts producing wildly-scaled numbers again
(location values in the tens of thousands, rotations with no `w` component),
check `pip show boxcars-py` first -- that's what a regression to 0.1.0 looks
like.

Two things replays fundamentally cannot tell us, independent of parser
version:
  - on_ground isn't a replicated field. Approximated from ride height
    (ON_GROUND_Z) since replays never carry a wheel-contact flag.
  - pitch/yaw/roll are analog stick input during aerial control, and replays
    only carry the *result* of that input (the car's physical rotation), never
    the raw stick values -- there's no way to recover them, so they're always
    0.0 in the extracted actions. Throttle, steer, jump, boost and handbrake
    ARE real recorded values.
"""

import os
import glob
import pickle
import argparse

import numpy as np
import boxcars_py

import sys
sys.path.insert(0, os.path.dirname(__file__))
from env import STATE_SIZE, ACTION_SIZE, POS_COEF, VEL_COEF, common_values

REPLAYS_DIR = os.path.join(os.path.dirname(__file__), "replays")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_replays.pkl")

BALL_CLASS = "Archetypes.Ball.Ball_Default"
CAR_CLASS = "Archetypes.Car.Car_Default"
BOOST_COMPONENT_CLASS = "Archetypes.CarComponents.CarComponent_Boost"
JUMP_COMPONENT_CLASS = "Archetypes.CarComponents.CarComponent_Jump"

RB_STATE_ATTR = "TAGame.RBActor_TA:ReplicatedRBState"
THROTTLE_ATTR = "TAGame.Vehicle_TA:ReplicatedThrottle"
STEER_ATTR = "TAGame.Vehicle_TA:ReplicatedSteer"
HANDBRAKE_ATTR = "TAGame.Vehicle_TA:bReplicatedHandbrake"
BOOST_AMOUNT_ATTR = "TAGame.CarComponent_Boost_TA:ReplicatedBoostAmount"
COMPONENT_ACTIVE_ATTR = "TAGame.CarComponent_TA:ReplicatedActive"
COMPONENT_VEHICLE_ATTR = "TAGame.CarComponent_TA:Vehicle"
PAWN_PRI_ATTR = "Engine.Pawn:PlayerReplicationInfo"
PRI_TEAM_ATTR = "Engine.PlayerReplicationInfo:Team"

# uu -- a resting car sits at z~17. Nothing in a replay ever says "wheels are
# touching the ground" directly, so this is a height cutoff standing in for
# that missing signal, not a measured value.
ON_GROUND_Z = 30.0


def _vec3(d: dict) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float32)


def _quat_xyzw_to_wxyz(d: dict) -> np.ndarray:
    # boxcars/Unreal networks quaternions as (x, y, z, w); RocketSim (and
    # therefore AshbyObsBuilder) expects (w, x, y, z) -- same rotation, just
    # reordered.
    return np.array([d["w"], d["x"], d["y"], d["z"]], dtype=np.float32)


def _decode_control_byte(b: int) -> float:
    # Throttle/steer are networked as a byte around a neutral 128, not a
    # float -- same rescale ContinuousAction expects everywhere else.
    return float(np.clip((b - 128) / 128.0, -1.0, 1.0))


def _decode_bool_attr(attr: dict) -> bool:
    # bReplicatedHandbrake always comes through as {"Boolean": bool}, but
    # TAGame.CarComponent_TA:ReplicatedActive (boost/jump/dodge "is held")
    # is replicated as {"Byte": N} in every real replay checked -- a classic
    # Unreal toggle-counter encoding, not a flag: the value increments by 1
    # (with wraparound) on every press AND every release, so it climbs
    # steadily across a whole match and is almost never exactly 0 again
    # after the first press. Treating "nonzero" as "active" was wrong --
    # confirmed by checking real per-actor sequences (e.g. 3,4,5,...,14,1,2)
    # -- the actual held/released state is the counter's PARITY, odd=active.
    if "Byte" in attr:
        return bool(attr["Byte"] % 2)
    return bool(next(iter(attr.values())))


def _invert_vec(v: np.ndarray) -> np.ndarray:
    return np.array([-v[0], -v[1], v[2]], dtype=np.float32)


def _invert_quat(q: np.ndarray) -> np.ndarray:
    # Mirrors AshbyObsBuilder's inverted_car_data trick so "forward" always
    # points at the opponent's goal, regardless of team. Verified against
    # env.py's own quat_to_rot_mtx: rotating the resulting rotation matrix by
    # diag(-1, -1, 1) (a 180deg turn about world Z) matches quat_to_rot_mtx on
    # (-z, -y, x, w) to floating-point precision.
    w, x, y, z = q
    return np.array([-z, -y, x, w], dtype=np.float32)


def _new_car_state() -> dict:
    return {"pos": None, "quat_wxyz": None, "lin_vel": None,
            "throttle": 0.0, "steer": 0.0, "handbrake": 0.0}


class _ReplayActors:
    """Walks one replay's network frames, keeping just enough running state
    per actor to emit a (state, action) row per car per frame. Everything
    here mirrors how the game actually replicates it: attributes persist
    until the next update, and a car's team/components/controller are only
    known once the handful of reference attributes that point at it have
    shown up at least once.
    """

    def __init__(self, objects: list):
        self.objects = objects
        self.actor_class: dict[int, str] = {}
        self.ball_id: int | None = None
        self.ball_state = {"pos": None, "lin_vel": None}
        self.cars: dict[int, dict] = {}
        self.car_boost_actor: dict[int, int] = {}
        self.car_jump_actor: dict[int, int] = {}
        self.boost_amounts: dict[int, float] = {}
        self.component_active: dict[int, bool] = {}
        self.car_to_pri: dict[int, int] = {}
        self.pri_to_team_actor: dict[int, int] = {}

    def handle_new_actors(self, new_actors: list):
        for na in new_actors:
            actor_id = na["actor_id"]
            cls = self.objects[na["object_id"]]
            self.actor_class[actor_id] = cls
            if cls == BALL_CLASS:
                self.ball_id = actor_id
            elif cls == CAR_CLASS:
                self.cars[actor_id] = _new_car_state()

    def handle_deleted_actors(self, deleted_actors: list):
        for actor_id in deleted_actors:
            self.actor_class.pop(actor_id, None)
            self.cars.pop(actor_id, None)

    def handle_updated_actors(self, updated_actors: list):
        for u in updated_actors:
            name = self.objects[u["object_id"]]
            actor_id = u["actor_id"]
            attr = u["attribute"]

            if name == RB_STATE_ATTR:
                self._handle_rigid_body(actor_id, attr["RigidBody"])
            elif name == THROTTLE_ATTR and actor_id in self.cars:
                self.cars[actor_id]["throttle"] = _decode_control_byte(attr["Byte"])
            elif name == STEER_ATTR and actor_id in self.cars:
                self.cars[actor_id]["steer"] = _decode_control_byte(attr["Byte"])
            elif name == HANDBRAKE_ATTR and actor_id in self.cars:
                self.cars[actor_id]["handbrake"] = 1.0 if _decode_bool_attr(attr) else 0.0
            elif name == BOOST_AMOUNT_ATTR:
                self.boost_amounts[actor_id] = attr["Byte"] / 255.0
            elif name == COMPONENT_ACTIVE_ATTR:
                self.component_active[actor_id] = _decode_bool_attr(attr)
            elif name == COMPONENT_VEHICLE_ATTR:
                self._handle_component_vehicle_link(actor_id, attr["ActiveActor"]["actor"])
            elif name == PAWN_PRI_ATTR:
                self.car_to_pri[actor_id] = attr["ActiveActor"]["actor"]
            elif name == PRI_TEAM_ATTR:
                self.pri_to_team_actor[actor_id] = attr["ActiveActor"]["actor"]

    def _handle_rigid_body(self, actor_id: int, rb: dict):
        pos = _vec3(rb["location"])
        lin_vel = _vec3(rb["linear_velocity"]) if rb["linear_velocity"] else np.zeros(3, dtype=np.float32)
        if actor_id == self.ball_id:
            self.ball_state["pos"] = pos
            self.ball_state["lin_vel"] = lin_vel
        elif actor_id in self.cars:
            car = self.cars[actor_id]
            car["pos"] = pos
            car["lin_vel"] = lin_vel
            if rb["rotation"] is not None:
                car["quat_wxyz"] = _quat_xyzw_to_wxyz(rb["rotation"])

    def _handle_component_vehicle_link(self, component_id: int, car_id: int):
        cls = self.actor_class.get(component_id)
        if cls == BOOST_COMPONENT_CLASS:
            self.car_boost_actor[car_id] = component_id
        elif cls == JUMP_COMPONENT_CLASS:
            self.car_jump_actor[car_id] = component_id

    def _team_of(self, car_id: int) -> int | None:
        pri = self.car_to_pri.get(car_id)
        team_actor = self.pri_to_team_actor.get(pri) if pri is not None else None
        cls = self.actor_class.get(team_actor, "") if team_actor is not None else ""
        if "Team0" in cls:
            return common_values.BLUE_TEAM
        if "Team1" in cls:
            return common_values.ORANGE_TEAM
        return None

    def sample_frame(self) -> list:
        """One (state, action) pair per car that has everything it needs
        resolved so far -- position/rotation/velocity, team, and the ball's
        own state. Cars still missing a piece (e.g. team not linked up yet)
        are silently skipped for this frame and picked up once it resolves.
        """
        if self.ball_state["pos"] is None or self.ball_state["lin_vel"] is None:
            return []

        rows = []
        for car_id, car in self.cars.items():
            if car["pos"] is None or car["quat_wxyz"] is None or car["lin_vel"] is None:
                continue
            team = self._team_of(car_id)
            if team is None:
                continue

            if team == common_values.ORANGE_TEAM:
                car_pos = _invert_vec(car["pos"])
                car_vel = _invert_vec(car["lin_vel"])
                car_quat = _invert_quat(car["quat_wxyz"])
                ball_pos = _invert_vec(self.ball_state["pos"])
                ball_vel = _invert_vec(self.ball_state["lin_vel"])
            else:
                car_pos, car_vel, car_quat = car["pos"], car["lin_vel"], car["quat_wxyz"]
                ball_pos, ball_vel = self.ball_state["pos"], self.ball_state["lin_vel"]

            boost_actor = self.car_boost_actor.get(car_id)
            boost_amount = self.boost_amounts.get(boost_actor, 0.0)
            boost_held = self.component_active.get(boost_actor, False)
            jump_actor = self.car_jump_actor.get(car_id)
            jump_held = self.component_active.get(jump_actor, False)
            on_ground = 1.0 if car_pos[2] < ON_GROUND_Z else 0.0

            state = np.concatenate([
                car_pos * POS_COEF,
                car_vel * VEL_COEF,
                car_quat,
                ball_pos * POS_COEF,
                ball_vel * VEL_COEF,
                [boost_amount, on_ground],
            ]).astype(np.float32)

            action = np.array([
                car["throttle"], car["steer"],
                0.0, 0.0, 0.0,  # pitch, yaw, roll -- not recoverable, see module docstring
                1.0 if jump_held else 0.0,
                1.0 if boost_held else 0.0,
                car["handbrake"],
            ], dtype=np.float32)

            rows.append((state, action))
        return rows


def _extract_from_parsed(replay: dict) -> tuple:
    objects = replay["objects"]
    tracker = _ReplayActors(objects)

    states, actions = [], []
    for frame in replay["network_frames"]["frames"]:
        tracker.handle_new_actors(frame["new_actors"])
        tracker.handle_updated_actors(frame["updated_actors"])
        tracker.handle_deleted_actors(frame["deleted_actors"])

        for state, action in tracker.sample_frame():
            states.append(state)
            actions.append(action)

    return states, actions


def _extract_replay(path: str) -> tuple:
    with open(path, "rb") as f:
        data = f.read()
    replay = boxcars_py.parse_replay(data)
    return _extract_from_parsed(replay)


def parse_all():
    paths = sorted(glob.glob(os.path.join(REPLAYS_DIR, "*.replay")))
    if not paths:
        raise FileNotFoundError(
            f"No .replay files in {REPLAYS_DIR} -- run `make rl-download` first."
        )

    all_states, all_actions = [], []
    parsed = 0
    for i, path in enumerate(paths, 1):
        try:
            states, actions = _extract_replay(path)
        except Exception as e:
            print(f"\n  skipping {os.path.basename(path)}: {e}")
            continue
        if not states:
            print(f"\n  skipping {os.path.basename(path)}: no usable frames extracted")
            continue
        all_states.extend(states)
        all_actions.extend(actions)
        parsed += 1
        print(f"\r{i}/{len(paths)} replays parsed ({len(all_states)} frames so far)", end="", flush=True)

    if not all_states:
        raise RuntimeError("No frames extracted from any replay -- nothing to save.")

    states_arr = np.stack(all_states).astype(np.float32)
    actions_arr = np.stack(all_actions).astype(np.float32)

    with open(DATASET_PATH, "wb") as f:
        pickle.dump({"states": states_arr, "actions": actions_arr, "n_replays": parsed}, f)

    print(f"\n{len(states_arr)} frames extracted from {parsed} replay(s) -> {DATASET_PATH}")


def _synthetic_replay() -> dict:
    """A minimal hand-built replay dict shaped exactly like boxcars_py's real
    output, covering one frame per team so both the plain and inverted
    (orange) code paths run. Lets --test exercise the actual extraction logic
    without needing a real .replay file or a live download.
    """
    objects = [
        BALL_CLASS,                         # 0
        CAR_CLASS,                          # 1
        BOOST_COMPONENT_CLASS,              # 2
        JUMP_COMPONENT_CLASS,               # 3
        "Archetypes.Teams.Team0",           # 4
        "Archetypes.Teams.Team1",           # 5
        "TAGame.Default__PRI_TA",           # 6
        RB_STATE_ATTR,                      # 7
        THROTTLE_ATTR,                      # 8
        STEER_ATTR,                         # 9
        HANDBRAKE_ATTR,                     # 10
        BOOST_AMOUNT_ATTR,                  # 11
        COMPONENT_ACTIVE_ATTR,              # 12
        COMPONENT_VEHICLE_ATTR,             # 13
        PAWN_PRI_ATTR,                      # 14
        PRI_TEAM_ATTR,                      # 15
    ]
    idx = {name: i for i, name in enumerate(objects)}

    def rb_update(actor_id, pos, vel, quat=None):
        rb = {"location": {"x": pos[0], "y": pos[1], "z": pos[2]},
              "linear_velocity": {"x": vel[0], "y": vel[1], "z": vel[2]},
              "angular_velocity": None, "sleeping": False,
              "rotation": None if quat is None else
                          {"w": quat[0], "x": quat[1], "y": quat[2], "z": quat[3]}}
        return {"actor_id": actor_id, "object_id": idx[RB_STATE_ATTR],
                "attribute": {"RigidBody": rb}, "stream_id": 0}

    def active_actor(actor_id, object_name, target):
        return {"actor_id": actor_id, "object_id": idx[object_name],
                "attribute": {"ActiveActor": {"active": True, "actor": target}}, "stream_id": 0}

    def byte(actor_id, object_name, value):
        return {"actor_id": actor_id, "object_id": idx[object_name],
                "attribute": {"Byte": value}, "stream_id": 0}

    def boolean(actor_id, object_name, value):
        return {"actor_id": actor_id, "object_id": idx[object_name],
                "attribute": {"Boolean": value}, "stream_id": 0}

    # actor ids: 0=ball, blue car=1/pri=10/boost=11/jump=12, orange car=2/pri=20/boost=21/jump=22
    new_actors_frame_0 = [
        {"actor_id": 0, "object_id": idx[BALL_CLASS], "name_id": 0, "initial_trajectory": {}},
        {"actor_id": 1, "object_id": idx[CAR_CLASS], "name_id": 1, "initial_trajectory": {}},
        {"actor_id": 11, "object_id": idx[BOOST_COMPONENT_CLASS], "name_id": 2, "initial_trajectory": {}},
        {"actor_id": 12, "object_id": idx[JUMP_COMPONENT_CLASS], "name_id": 3, "initial_trajectory": {}},
        {"actor_id": 10, "object_id": idx["TAGame.Default__PRI_TA"], "name_id": 4, "initial_trajectory": {}},
        {"actor_id": 100, "object_id": idx["Archetypes.Teams.Team0"], "name_id": 5, "initial_trajectory": {}},
        {"actor_id": 2, "object_id": idx[CAR_CLASS], "name_id": 6, "initial_trajectory": {}},
        {"actor_id": 21, "object_id": idx[BOOST_COMPONENT_CLASS], "name_id": 7, "initial_trajectory": {}},
        {"actor_id": 22, "object_id": idx[JUMP_COMPONENT_CLASS], "name_id": 8, "initial_trajectory": {}},
        {"actor_id": 20, "object_id": idx["TAGame.Default__PRI_TA"], "name_id": 9, "initial_trajectory": {}},
        {"actor_id": 101, "object_id": idx["Archetypes.Teams.Team1"], "name_id": 10, "initial_trajectory": {}},
    ]

    updated_actors_frame_0 = [
        rb_update(0, (0.0, 0.0, 92.75), (0.0, 0.0, 0.0)),
        rb_update(1, (100.0, 200.0, 17.0), (500.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
        rb_update(2, (-300.0, -400.0, 17.0), (-500.0, 100.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
        byte(1, THROTTLE_ATTR, 255),
        byte(1, STEER_ATTR, 200),
        boolean(1, HANDBRAKE_ATTR, False),
        byte(11, BOOST_AMOUNT_ATTR, 128),
        # Real replays encode this as a Byte activation counter, not a Boolean --
        # exercise that path here (nonzero = active), Boolean for orange below.
        byte(11, COMPONENT_ACTIVE_ATTR, 3),
        active_actor(11, COMPONENT_VEHICLE_ATTR, 1),
        byte(12, COMPONENT_ACTIVE_ATTR, 0),
        active_actor(12, COMPONENT_VEHICLE_ATTR, 1),
        active_actor(1, PAWN_PRI_ATTR, 10),
        active_actor(10, PRI_TEAM_ATTR, 100),
        byte(2, THROTTLE_ATTR, 0),
        byte(2, STEER_ATTR, 128),
        boolean(2, HANDBRAKE_ATTR, True),
        byte(21, BOOST_AMOUNT_ATTR, 255),
        boolean(21, COMPONENT_ACTIVE_ATTR, False),
        active_actor(21, COMPONENT_VEHICLE_ATTR, 2),
        boolean(22, COMPONENT_ACTIVE_ATTR, True),
        active_actor(22, COMPONENT_VEHICLE_ATTR, 2),
        active_actor(2, PAWN_PRI_ATTR, 20),
        active_actor(20, PRI_TEAM_ATTR, 101),
    ]

    return {
        "objects": objects,
        "network_frames": {"frames": [
            {"new_actors": new_actors_frame_0, "updated_actors": updated_actors_frame_0,
             "deleted_actors": [], "delta": 0.0, "time": 0.0},
        ]},
    }


def smoke_test():
    """--test: run the real extraction pipeline over a hand-built synthetic
    replay (see _synthetic_replay) instead of a downloaded one, so this
    passes without needing a .replay file on disk or a live API key.
    """
    print("Extracting a synthetic 1-frame, 2-car replay...")
    states, actions = _extract_from_parsed(_synthetic_replay())

    assert len(states) == 2, f"expected 2 rows (one per car), got {len(states)}"
    assert len(actions) == 2
    for s, a in zip(states, actions):
        assert s.shape == (STATE_SIZE,), f"state shape {s.shape}, expected ({STATE_SIZE},)"
        assert a.shape == (ACTION_SIZE,), f"action shape {a.shape}, expected ({ACTION_SIZE},)"
    print(f"  shapes ok: {len(states)} rows of state{states[0].shape} / action{actions[0].shape}")

    blue_state, blue_action = states[0], actions[0]
    orange_state, orange_action = states[1], actions[1]

    # Blue car isn't inverted -- its raw x position (100.0 * POS_COEF) should
    # come straight through.
    assert abs(blue_state[0] - 100.0 * POS_COEF) < 1e-5, "blue car position wasn't left uninverted"
    # Orange car IS inverted -- its raw x position (-300.0) should flip sign.
    assert abs(orange_state[0] - 300.0 * POS_COEF) < 1e-5, "orange car x wasn't inverted"

    assert abs(blue_action[0] - (255 - 128) / 128.0) < 1e-5, "blue throttle byte didn't decode correctly"
    assert abs(blue_action[1] - (200 - 128) / 128.0) < 1e-5, "blue steer byte didn't decode correctly"
    assert blue_action[5] == 0.0 and blue_action[6] == 1.0, "blue jump/boost active flags swapped"
    assert orange_action[7] == 1.0, "orange handbrake should be on"
    assert list(blue_action[2:5]) == [0.0, 0.0, 0.0], "pitch/yaw/roll must always be 0 (not recoverable)"

    print("  decoded values ok: throttle/steer bytes, boost/jump flags, team inversion")
    print("parse_replays.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ballchasing .replay files into a BC dataset")
    parser.add_argument("--test", action="store_true", help="smoke-test the extraction pipeline on synthetic data")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        parse_all()
