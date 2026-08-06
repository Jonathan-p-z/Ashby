"""Launches a real-game 1v1: you vs Ashby, via RLBot v2.

Nothing here trains anything -- this is a real Rocket League window, launched and
refereed by RLBotServer, controller in hand. It's the real-match equivalent of
record.py's RocketSim session: same "you're blue, Ashby's orange" convention, except
this time it's whatever rlbot_agent.py currently loads (the latest PPO checkpoint,
falling back to rl_policy.pth -- see rlbot_agent.py) in an actual game rather than
RocketSim's physics approximation.
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from rlbot import flat
from rlbot.config import load_player_config, get_human
from rlbot.managers import MatchManager

RLBOT_TOML = os.path.join(os.path.dirname(__file__), "rlbot.toml")
STATS_INTERVAL = 1.0  # seconds between live stat-line refreshes


def _build_match_config(launcher: flat.Launcher) -> flat.MatchConfiguration:
    """You on blue, Ashby on orange -- matches record.py's convention ("You're blue,
    Ashby's orange"). Unlike record.py, this isn't arbitrary here: a human seat has no
    run_command to launch, so team assignment is the only place that distinction shows
    up in this file, and keeping it consistent with record.py avoids second-guessing
    which side is which mid-match.
    """
    human = get_human(team=0)
    ashby = load_player_config(RLBOT_TOML, team=1, name_override="Ashby")

    return flat.MatchConfiguration(
        launcher=launcher,
        auto_start_agents=True,   # RLBotServer spawns rlbot_agent.py itself via rlbot.toml
        wait_for_agents=True,     # ...and waits for it to connect before kicking off
        game_mode=flat.GameMode.Soccar,
        player_configurations=[human, ashby],
        mutators=flat.MutatorSettings(),  # all defaults -- standard 5-minute soccar
        existing_match_behavior=flat.ExistingMatchBehavior.Restart,
    )


class _TouchCounter:
    """Tracks total ball touches per team from GamePacket.players[i].latest_touch.

    RLBot doesn't expose a running touch count directly, only the timestamp of each
    player's most recent touch -- so "a new touch happened" is detected as "that
    timestamp moved forward from what we last saw for this team".
    """

    def __init__(self):
        self._last_seen = {}
        self.counts = {}

    def update(self, packet: flat.GamePacket):
        for player in packet.players:
            touch = player.latest_touch
            seconds = touch.game_seconds if touch is not None else 0.0
            team = player.team
            if seconds > 0.0 and seconds != self._last_seen.get(team, 0.0):
                self._last_seen[team] = seconds
                self.counts[team] = self.counts.get(team, 0) + 1


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _team_score(packet: flat.GamePacket, team_index: int) -> int:
    return next((t.score for t in packet.teams if t.team_index == team_index), 0)


def run(launcher: flat.Launcher):
    manager = MatchManager()
    config = _build_match_config(launcher)

    print("Starting RLBotServer and launching the match (you vs Ashby)...")
    manager.start_match(config)
    print("Match started -- you're blue, Ashby's orange. Good luck.\n")

    touches = _TouchCounter()
    last_score = (0, 0)
    last_print = 0.0
    packet = None

    try:
        while True:
            packet = manager.packet
            if packet is None:
                time.sleep(0.1)
                continue

            touches.update(packet)

            score = (_team_score(packet, 0), _team_score(packet, 1))
            if score != last_score:
                last_score = score
                print(f"\n  GOAL! You {score[0]} - {score[1]} Ashby\n")

            now = time.monotonic()
            if now - last_print >= STATS_INTERVAL:
                last_print = now
                duration = _format_duration(packet.match_info.seconds_elapsed)
                print(
                    f"\r  {duration}  |  You {score[0]} - {score[1]} Ashby  "
                    f"|  touches: You {touches.counts.get(0, 0)} / Ashby {touches.counts.get(1, 0)}",
                    end="", flush=True,
                )

            if packet.match_info.match_phase == flat.MatchPhase.Ended:
                print("\n\nMatch ended.")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        score = (_team_score(packet, 0), _team_score(packet, 1)) if packet is not None else (0, 0)
        duration = _format_duration(packet.match_info.seconds_elapsed) if packet is not None else "00:00"
        print(f"\nFinal score: You {score[0]} - {score[1]} Ashby  ({duration})")
        manager.disconnect()


def smoke_test():
    """--test: builds the real match config and exercises the stats-tracking logic on
    synthetic packets -- no RLBotServer or Rocket League needed, those can only be
    exercised for real by actually running this script without --test.
    """
    print("Building match config (you vs Ashby)...")
    config = _build_match_config(flat.Launcher.Epic)
    assert len(config.player_configurations) == 2, "expected exactly 2 players"
    config.pack()  # forces flatbuffers to validate every field, not just construct the object
    print("  match config: ok (packs cleanly, 2 players)")

    print("Checking touch counter and score/duration formatting on synthetic packets...")
    touches = _TouchCounter()

    def _packet(blue_touch_s, orange_touch_s, blue_score, orange_score, elapsed):
        return flat.GamePacket(
            players=[
                flat.PlayerInfo(team=0, latest_touch=flat.Touch(game_seconds=blue_touch_s)),
                flat.PlayerInfo(team=1, latest_touch=flat.Touch(game_seconds=orange_touch_s)),
            ],
            teams=[flat.TeamInfo(team_index=0, score=blue_score), flat.TeamInfo(team_index=1, score=orange_score)],
            match_info=flat.MatchInfo(seconds_elapsed=elapsed, match_phase=flat.MatchPhase.Active),
        )

    touches.update(_packet(0.0, 0.0, 0, 0, 0.0))
    touches.update(_packet(1.2, 0.0, 0, 0, 1.5))
    touches.update(_packet(1.2, 3.4, 0, 0, 3.7))   # blue's touch unchanged -- shouldn't recount
    touches.update(_packet(5.0, 3.4, 1, 0, 6.0))    # blue touches again -- should count
    assert touches.counts.get(0, 0) == 2, f"expected 2 blue touches, got {touches.counts.get(0, 0)}"
    assert touches.counts.get(1, 0) == 1, f"expected 1 orange touch, got {touches.counts.get(1, 0)}"
    print("  touch counter: ok (repeated timestamps don't double-count)")

    assert _format_duration(75) == "01:15", "duration formatting is wrong"
    print("  duration formatting: ok")

    last_packet = _packet(5.0, 3.4, 1, 0, 6.0)
    assert _team_score(last_packet, 0) == 1 and _team_score(last_packet, 1) == 0
    print("  team score lookup: ok")

    print("run_match.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="You vs Ashby -- live RLBot v2 match")
    parser.add_argument("--launcher", choices=["steam", "epic"], default="epic", help="which Rocket League launcher to use (default: epic)")
    parser.add_argument("--test", action="store_true", help="smoke-test match config + stats tracking, no real match")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        launcher = flat.Launcher.Steam if args.launcher == "steam" else flat.Launcher.Epic
        run(launcher)
