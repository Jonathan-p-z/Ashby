"""Rebuilds rl_train.py's training-progress plot from a saved terminal log instead of
from live training state.

Useful after the fact: you ran a real training session, redirected its console output
to a file, and only later wanted the graph -- or the run is long gone and the log is
all that's left. rl_train.py's own _plot() works from in-memory per-episode lists that
don't survive the process; this works from whatever got printed to the screen.

Important limitation, not a bug: rl_train.py only prints a narrative line every
REPORT_EVERY (100) episodes, e.g.

    Episode   100 -- touch rate  12% | goal diff -0.30 | starting to close ...

and touch_rate/goal_diff in that line are ALREADY a rolling average (over WINDOW=100
episodes) at the moment they were printed -- the raw per-episode values were never
printed anywhere. So this script can't recompute a fresh 50-episode-per-episode
rolling average the way a live run can; it re-smooths whatever samples actually exist
in the log, which is coarser than that. Same idea, best available data.

Usage: python mind/rl/parse_logs.py logs.txt
"""

import os
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # a script whose only job is to save a PNG has no business trying to open a GUI window
import matplotlib.pyplot as plt

# Duplicated from rl_train.py rather than imported -- importing that module would drag
# in torch/rlgym_sim/RocketSim just to read one path constant, which is pointless for a
# script that never touches the environment.
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "rl_training_results.png")

PLOT_WINDOW = 50

# Matches rl_train.py's _narrative() output exactly, e.g.:
#   "Episode   100 -- touch rate  12% | goal diff -0.30 | starting to close the distance..."
NARRATIVE_RE = re.compile(
    r"Episode\s+(\d+)\s+--\s+touch rate\s+(\d+)%\s+\|\s+goal diff\s+([+-]?\d+\.\d+)"
)


def parse_log(path: str) -> tuple:
    """Pulls (episode, touch_rate, goal_diff) triples out of every narrative line in
    the log. Both values are themselves rolling averages already -- see the module
    docstring for why that's the ceiling on what's recoverable from log text."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    episodes, touch_rates, goal_diffs = [], [], []
    for match in NARRATIVE_RE.finditer(text):
        episodes.append(int(match.group(1)))
        touch_rates.append(int(match.group(2)) / 100.0)
        goal_diffs.append(float(match.group(3)))

    if not episodes:
        raise ValueError(
            f"No narrative lines found in {path} -- expected lines like "
            "'Episode   100 -- touch rate  12% | goal diff -0.30 | ...' "
            "produced by rl_train.py's own console output."
        )

    return episodes, touch_rates, goal_diffs


def plot(episodes: list, touch_rates: list, goal_diffs: list, output_path: str):
    """Same two-subplot layout as rl_train.py's own _plot() -- touch rate and goal
    diff are what's actually legible at a glance about whether a run went anywhere.
    """
    n = len(episodes)
    w = min(PLOT_WINDOW, n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle("Ashby x RLGym -- training progress (reconstructed from log)", fontsize=13)

    # convolve operates on sample index, not episode number -- episodes[w - 1:] maps
    # each 'valid' output point back to the real episode it corresponds to, which
    # matters here since log samples are REPORT_EVERY episodes apart, not 1.
    x = episodes[w - 1:] if w > 1 else []

    if w > 1:
        ax1.plot(x, np.convolve(touch_rates, np.ones(w) / w, mode="valid"), color="tomato")
    ax1.set_ylabel(f"Touch rate ({w}-sample avg)")
    ax1.set_ylim(0, 1)
    ax1.set_title("Ball touch rate")
    ax1.grid(alpha=0.3)

    if w > 1:
        ax2.plot(x, np.convolve(goal_diffs, np.ones(w) / w, mode="valid"), color="seagreen")
    ax2.axhline(0, linestyle="--", color="gray", alpha=0.5)
    ax2.set_ylabel(f"Goal diff ({w}-sample avg)")
    ax2.set_xlabel("Episode")
    ax2.set_title("Goal differential vs self-play pool")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)

    print(f"Parsed {n} narrative line(s) from the log (episodes {episodes[0]}-{episodes[-1]})")
    print(f"Plot saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLGym -- rebuild the training plot from a saved log")
    parser.add_argument("log_file", help="path to a text file of rl_train.py's console output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"where to save the plot (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    episodes, touch_rates, goal_diffs = parse_log(args.log_file)
    plot(episodes, touch_rates, goal_diffs, args.output)
