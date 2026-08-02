"""Autonomous RL -- Ashby keeps playing after the human stops watching.

Starts from rl_imitation.pth (if it exists) so the agent isn't relearning "which pedal is
throttle" from scratch, then plays 1v1 self-play, fine-tuning the actor/critic pair from
rl_agent.py on whatever reward its own play generates. The opponent every worker faces is
a random past snapshot of the same policy pulled from mind/weights/pool/, not a fixed
scripted bot -- a static opponent stops teaching anything the moment the agent gets better
than it, while a pool of its own recent history keeps raising the bar as training goes.

NUM_WORKERS separate processes each run their own RocketSim instance on CPU, playing out
1v1s in real time and shipping (state, action, reward, next_state, done) tuples back to
this process over a Queue. This process is the only one that ever touches the GPU -- it
owns the replay buffer, runs the actual gradient steps on the 4060, and every
WEIGHT_SYNC_EVERY learner steps pushes the updated actor back out to the workers so their
rollouts keep tracking the improving policy instead of freezing at whatever they started
with. Splitting it this way is the whole point: RocketSim is single-core physics, so 4
workers means roughly 4x the environment steps/sec feeding the same one learner.
"""

import os
import sys
import time
import glob
import queue
import random
import argparse
import multiprocessing as mp

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from env import make_env, ACTION_SIZE, WEIGHTS_DIR
from rl_agent import RLAgent
from imitation import ImitationNet

IMITATION_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_imitation.pth")
POLICY_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_policy.pth")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_training_results.png")

POOL_DIR = os.path.join(WEIGHTS_DIR, "pool")  # self-play opponent snapshots live here
POOL_SIZE = 5           # oldest snapshot gets evicted once a 6th one is added
POOL_UPDATE_EVERY = 500  # episodes between adding the current policy to the pool

# Ryzen 7 8845HS has 8 cores -- 4 RocketSim workers leaves headroom for the learner
# process, matplotlib, and the OS instead of fighting them for the last core.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
WEIGHT_SYNC_EVERY = 100  # learner steps between pushing fresh actor weights to workers

N_EPISODES = 10_000
CHECKPOINT_EVERY = 500
REPORT_EVERY = 100
WINDOW = 100  # rolling window for the narrative stats
THROUGHPUT_EVERY = 5.0  # seconds between steps/sec readouts

PLOT_EVERY = 500   # episodes between regenerating rl_training_results.png mid-run
PLOT_WINDOW = 50    # rolling window for the plot specifically -- narrower than WINDOW
                    # so the graph reacts faster than the console narrative does


def _narrative(episode: int, touch_rate: float, goal_diff_avg: float) -> str:
    """Grounded in what we can actually measure -- touch rate and goal differential
    against whichever pool snapshot each episode happened to be played against --
    rather than claiming behaviors (aerials, dribbles) we have no detector for."""
    if episode <= 100:
        phase = "mostly random motion, any touch is a happy accident"
    elif touch_rate < 0.3:
        phase = "starting to close the distance to the ball more often"
    elif touch_rate < 0.6:
        phase = "getting a touch in most episodes now"
    elif goal_diff_avg <= 0.0:
        phase = "touching the ball reliably, not converting to goals yet"
    elif goal_diff_avg < 0.5:
        phase = "trading goals roughly evenly with its own recent history"
    else:
        phase = "outperforming its own past selves -- goal difference solidly positive"
    return (
        f"Episode {episode:5d} -- touch rate {touch_rate:4.0%} | "
        f"goal diff {goal_diff_avg:+.2f} | {phase}"
    )


def _pool_snapshot_paths() -> list:
    return sorted(glob.glob(os.path.join(POOL_DIR, "gen_*.pth")))


def _add_to_pool(agent: RLAgent, episode: int, pool_version: mp.Value):
    """Snapshots the current actor into the self-play pool, evicting the oldest entry
    once there are more than POOL_SIZE. Written to a temp file then moved into place
    (os.replace is atomic on both Windows and POSIX) so a worker mid-glob never picks
    up a half-written checkpoint.

    Bumping pool_version last (after the file is safely in place) is what tells
    workers a reload is worth doing -- see _worker_loop for why they don't just
    re-glob the pool directory every episode instead.
    """
    os.makedirs(POOL_DIR, exist_ok=True)
    path = os.path.join(POOL_DIR, f"gen_{episode:06d}.pth")
    tmp_path = path + ".tmp"
    torch.save({k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}, tmp_path)
    os.replace(tmp_path, path)

    snapshots = _pool_snapshot_paths()
    while len(snapshots) > POOL_SIZE:
        os.remove(snapshots.pop(0))

    pool_version.value += 1  # single writer (this process) -- no lock needed


def _opponent_action(net: ImitationNet, obs: np.ndarray) -> np.ndarray:
    """Pure exploitation, same as RLAgent.best_action -- a frozen sparring partner
    should play its snapshot as well as it can, not explore on top of it."""
    state_t = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        return net(state_t).cpu().numpy()[0]


def _load_random_opponent(net: ImitationNet):
    """Points `net` at a random snapshot from the self-play pool and returns the action
    function to call each step. Only called from _worker_loop when pool_version has
    actually changed -- not every episode -- so this is the one place that ever touches
    disk for the opponent. Empty pool (the very start of training, before anything has
    been added yet) falls back to pure random actions -- there's no "beginner bot" skill
    floor to fall back on anymore, so a blank pool just means the first episodes are
    against noise instead of a fixed opponent.
    """
    snapshots = _pool_snapshot_paths()
    if snapshots:
        path = random.choice(snapshots)
        try:
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            return lambda obs: _opponent_action(net, obs)
        except FileNotFoundError:
            pass  # pool eviction beat us to it -- fall through to random actions this episode

    return lambda obs: np.random.uniform(-1.0, 1.0, size=ACTION_SIZE).astype(np.float32)


def _worker_loop(worker_id: int, weights_queue: mp.Queue, experience_queue: mp.Queue,
                  stats_queue: mp.Queue, stop_event, init_weights: str, pool_version: mp.Value):
    """Runs in its own process: plays 1v1 self-play forever, shipping transitions out
    and picking up fresh actor weights whenever the learner has some.

    Stays on CPU end to end -- RocketSim is single-core physics regardless, and a CUDA
    context per worker would cost VRAM for a network this small with no upside.
    Epsilon decays locally per worker rather than being broadcast from the learner --
    exploration only needs to happen somewhere, it doesn't need to be perfectly
    synchronized across 4 processes to do its job.

    The opponent is a bare ImitationNet rather than a full RLAgent -- it only ever
    needs a forward pass against a frozen snapshot, so there's no reason to carry a
    critic, optimizers, and a replay buffer it will never touch. It's kept in memory
    across episodes and only reloaded from disk when pool_version changes (i.e. the
    learner actually added a new snapshot, every POOL_UPDATE_EVERY episodes) -- the
    pool doesn't change every episode, so re-globbing mind/weights/pool/ and hitting
    disk with torch.load every single episode was pure waste for no fresher data.
    """
    env = make_env(spawn_opponents=True)
    agent = RLAgent(imitation_weights=init_weights, device=torch.device("cpu"))
    opponent_net = ImitationNet().to(torch.device("cpu"))
    print(f"Worker {worker_id} ready", flush=True)

    obs, info = env.reset(return_info=True)
    blue0, orange0 = info["state"].blue_score, info["state"].orange_score
    touched = False
    opponent_action = _load_random_opponent(opponent_net)
    last_seen_pool_version = pool_version.value

    while not stop_event.is_set():
        action_agent = agent.act(np.asarray(obs[0]))
        action_bot = opponent_action(np.asarray(obs[1]))

        next_obs, rewards, done, info = env.step(np.stack([action_agent, action_bot]))
        experience_queue.put((np.asarray(obs[0]), action_agent, rewards[0], np.asarray(next_obs[0]), done))

        touched = touched or info["state"].players[0].ball_touched
        obs = next_obs

        if done:
            blue1, orange1 = info["state"].blue_score, info["state"].orange_score
            stats_queue.put((float(touched), (blue1 - blue0) - (orange1 - orange0)))
            agent.decay_epsilon()

            obs, info = env.reset(return_info=True)
            blue0, orange0 = info["state"].blue_score, info["state"].orange_score
            touched = False

            if pool_version.value != last_seen_pool_version:
                opponent_action = _load_random_opponent(opponent_net)
                last_seen_pool_version = pool_version.value

        _apply_latest_weights(agent, weights_queue)

    env.close()


def _apply_latest_weights(agent: RLAgent, weights_queue: mp.Queue):
    """Drains the queue instead of popping one -- if the learner pushed several updates
    while this worker was mid-episode, only the newest one is worth loading."""
    latest = None
    while True:
        try:
            latest = weights_queue.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        agent.actor.load_state_dict(latest)


def _broadcast_weights(agent: RLAgent, weight_queues: list):
    """Pushed as plain CPU tensors -- workers never touch CUDA, so there's nothing to
    convert on their end, just load_state_dict and keep playing."""
    state_dict = {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}
    for q in weight_queues:
        try:
            q.get_nowait()  # drop whatever update the worker hasn't picked up yet
        except queue.Empty:
            pass
        q.put(state_dict)


def _spawn_workers(init_weights: str) -> tuple:
    """Starts the worker pool and the queues/shared state it shares with the learner."""
    experience_queue = mp.Queue(maxsize=50_000)
    stats_queue = mp.Queue()
    weight_queues = [mp.Queue(maxsize=1) for _ in range(NUM_WORKERS)]
    stop_event = mp.Event()
    pool_version = mp.Value("i", 0)  # bumped by _add_to_pool -- the "shared flag" workers poll for a reload

    workers = [
        mp.Process(
            target=_worker_loop,
            args=(i, weight_queues[i], experience_queue, stats_queue, stop_event, init_weights, pool_version),
            daemon=True,
        )
        for i in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()

    return workers, experience_queue, stats_queue, weight_queues, stop_event, pool_version


def _shutdown_workers(workers: list, stop_event):
    stop_event.set()
    for w in workers:
        w.join(timeout=5.0)
        if w.is_alive():
            w.terminate()


def train():
    print("=" * 70)
    print("  Ashby x RLGym -- autonomous RL, self-play vs its own history")
    print(f"  {NUM_WORKERS} CPU workers -> 1 GPU learner, {N_EPISODES} episodes, checkpoint every {CHECKPOINT_EVERY}")
    print(f"  self-play pool: {POOL_SIZE} snapshots, refreshed every {POOL_UPDATE_EVERY} episodes -> {POOL_DIR}")
    print("=" * 70)

    init_weights = IMITATION_WEIGHTS if os.path.exists(IMITATION_WEIGHTS) else None
    agent = RLAgent(imitation_weights=init_weights)  # the one and only GPU copy

    workers, experience_queue, stats_queue, weight_queues, stop_event, pool_version = _spawn_workers(init_weights)

    touches_per_ep, goal_diffs_per_ep = [], []
    episode = 0
    steps_since_sync = 0
    throughput_steps = 0
    throughput_clock = time.perf_counter()

    try:
        while episode < N_EPISODES:
            try:
                state, action, reward, next_state, done = experience_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # all 4 workers mid-physics-step -- nothing to learn from yet

            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            steps_since_sync += 1
            throughput_steps += 1

            if steps_since_sync >= WEIGHT_SYNC_EVERY:
                _broadcast_weights(agent, weight_queues)
                steps_since_sync = 0

            while True:
                try:
                    touched, goal_diff = stats_queue.get_nowait()
                except queue.Empty:
                    break
                episode += 1
                touches_per_ep.append(touched)
                goal_diffs_per_ep.append(goal_diff)

                if episode == 1 or episode % REPORT_EVERY == 0:
                    touch_rate = float(np.mean(touches_per_ep[-WINDOW:]))
                    goal_diff_avg = float(np.mean(goal_diffs_per_ep[-WINDOW:]))
                    print(_narrative(episode, touch_rate, goal_diff_avg))

                if episode % CHECKPOINT_EVERY == 0:
                    agent.save_weights(POLICY_WEIGHTS)
                    print(f"  -> checkpoint saved ({episode}/{N_EPISODES}) -> {POLICY_WEIGHTS}")

                if episode % POOL_UPDATE_EVERY == 0:
                    _add_to_pool(agent, episode, pool_version)
                    print(f"  -> self-play pool updated ({len(_pool_snapshot_paths())}/{POOL_SIZE} snapshots)")

                if episode % PLOT_EVERY == 0:
                    _plot(touches_per_ep, goal_diffs_per_ep)
                    print(f"  -> plot updated ({episode}/{N_EPISODES}) -> {RESULTS_PATH}")

            elapsed = time.perf_counter() - throughput_clock
            if elapsed >= THROUGHPUT_EVERY:
                print(f"  [throughput] {throughput_steps / elapsed:.1f} steps/sec across {NUM_WORKERS} workers")
                throughput_steps = 0
                throughput_clock = time.perf_counter()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        _shutdown_workers(workers, stop_event)

    agent.save_weights(POLICY_WEIGHTS)
    print(f"\nFinal policy saved -> {POLICY_WEIGHTS}")

    _plot(touches_per_ep, goal_diffs_per_ep)
    print(f"Plot saved         -> {RESULTS_PATH}")


def _plot(touches: list, goal_diffs: list):
    """Two subplots, not three -- reward is what the learner optimizes, but touch rate
    and goal diff are what's actually legible at a glance about whether training is
    going anywhere, so those are the two numbers worth checking in on mid-run.
    """
    n = len(touches)
    w = min(PLOT_WINDOW, n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle("Ashby x RLGym -- autonomous RL, self-play vs its own history", fontsize=13)

    if w > 1:
        ax1.plot(range(w, n + 1), np.convolve(touches, np.ones(w) / w, mode="valid"), color="tomato")
    ax1.set_ylabel(f"Touch rate ({w}-ep avg)")
    ax1.set_ylim(0, 1)
    ax1.set_title("Ball touch rate")
    ax1.grid(alpha=0.3)

    if w > 1:
        ax2.plot(range(w, n + 1), np.convolve(goal_diffs, np.ones(w) / w, mode="valid"), color="seagreen")
    ax2.axhline(0, linestyle="--", color="gray", alpha=0.5)
    ax2.set_ylabel(f"Goal diff ({w}-ep avg)")
    ax2.set_xlabel("Episode")
    ax2.set_title("Goal differential vs self-play pool")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH, dpi=120)
    plt.close(fig)


def smoke_test():
    """--test: spins up the real NUM_WORKERS worker pool, lets the learner consume a
    handful of transitions and push one weight sync, then tears everything down. Proves
    the worker/queue/learner wiring works before committing to a real N_EPISODES run."""
    print(f"Spawning {NUM_WORKERS} workers (CPU) + 1 GPU learner...")
    init_weights = IMITATION_WEIGHTS if os.path.exists(IMITATION_WEIGHTS) else None
    agent = RLAgent(imitation_weights=init_weights, batch_size=8)

    workers, experience_queue, stats_queue, weight_queues, stop_event, pool_version = _spawn_workers(init_weights)

    try:
        collected = 0
        deadline = time.perf_counter() + 30.0  # rlgym_sim/RocketSim import takes a moment per worker
        while collected < 20:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    "Workers never produced experience -- check that rlgym_sim/RocketSim "
                    "import cleanly in a worker process (run without --test for full logs)."
                )
            try:
                state, action, reward, next_state, done = experience_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            collected += 1

        _broadcast_weights(agent, weight_queues)
        print(f"  learner: ok ({collected} transitions consumed, one weight sync pushed)")

        _add_to_pool(agent, episode=1, pool_version=pool_version)
        print(f"  self-play pool: ok (pool_version={pool_version.value}, workers reload on their next episode boundary)")
        os.remove(os.path.join(POOL_DIR, "gen_000001.pth"))  # a smoke test shouldn't leave a snapshot in a real pool
    finally:
        _shutdown_workers(workers, stop_event)

    print("rl_train.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLGym -- autonomous RL training")
    parser.add_argument("--test", action="store_true", help="smoke-test the worker pool + learner without a full run")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        train()
