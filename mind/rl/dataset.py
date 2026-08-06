"""Loads recorded sessions into a train/val split ready for behavioral cloning.

One thing this deliberately does NOT do: re-normalize states from dataset statistics.
AshbyObsBuilder (env.py) already emits a pre-scaled state -- positions and velocities
divided down to a small range, quaternion already unit-length, boost already in [0,1].
If this file computed its own mean/std from whatever sessions happen to be on disk, the
imitation net would learn to expect that particular distribution, and rl_agent.py (which
feeds it raw env.py observations with no such normalization) would hand it something
subtly different at fine-tuning time. Same obs pipeline everywhere, or the transferred
weights are transferring a lie.
"""

import os
import glob
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import sys
sys.path.insert(0, os.path.dirname(__file__))
from env import DATA_DIR

# Physics glitches (a car clipping a wall corner, a kickoff double-spawn) can produce a
# one-frame outlier that dwarfs everything else in an MSE loss. Clip instead of dropping
# the frame -- keeps the sequence intact for anyone who later wants temporal context.
CLIP_RANGE = 5.0


class ImitationDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def _load_sessions(data_dir: str) -> tuple:
    paths = sorted(glob.glob(os.path.join(data_dir, "session_*.pkl")))
    if not paths:
        raise FileNotFoundError(
            f"No session_*.pkl files found in {data_dir} -- run `make rl-record` (or "
            "record.py directly) to capture some gameplay first."
        )

    all_states, all_actions = [], []
    for path in paths:
        with open(path, "rb") as f:
            session = pickle.load(f)
        all_states.append(session["states"])
        all_actions.append(session["actions"])
        print(f"  loaded {os.path.basename(path)}: {len(session['states'])} frames")

    states = np.concatenate(all_states).astype(np.float32)
    actions = np.concatenate(all_actions).astype(np.float32)
    return states, actions


def load_datasets(data_dir: str = DATA_DIR, val_split: float = 0.2, seed: int = 0):
    states, actions = _load_sessions(data_dir)
    states = np.clip(states, -CLIP_RANGE, CLIP_RANGE)

    full = ImitationDataset(states, actions)
    n_val = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=generator)
    print(f"  total frames: {len(full)}  (train {n_train} / val {n_val})")
    return train_set, val_set


def make_dataloaders(data_dir: str = DATA_DIR, batch_size: int = 256, val_split: float = 0.2, seed: int = 0):
    train_set, val_set = load_datasets(data_dir, val_split, seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_replay_dataset(path: str, val_split: float = 0.2, seed: int = 0, max_frames: int | None = None):
    """Same split logic as load_datasets, but for parse_replays.py's single
    consolidated dataset_replays.pkl instead of many session_*.pkl files --
    replay states already go through AshbyObsBuilder's own scaling (see
    parse_replays.py), so the same CLIP_RANGE outlier guard applies as-is.

    max_frames randomly subsamples down to that many rows (uniformly, no
    replacement) when the dataset has more -- at 127.98M frames the full
    dataset alone is well over 10GB in memory, which starts fighting the OS
    for RAM and slows everything down, not just the DataLoader.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No dataset at {path} -- run `make rl-parse` (or parse_replays.py "
            "directly) to build it from downloaded replays first."
        )

    with open(path, "rb") as f:
        data = pickle.load(f)
    states = data["states"]
    actions = data["actions"]
    total_frames = len(states)
    n_replays = data.get("n_replays", "?")

    if max_frames is not None and total_frames > max_frames:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total_frames, size=max_frames, replace=False)
        states = states[idx]
        actions = actions[idx]
        print(f"  subsampled {max_frames} of {total_frames} frames (RAM guard)")

    del data  # drop the last reference to the pre-subsample arrays so they can be freed

    states = np.clip(states.astype(np.float32), -CLIP_RANGE, CLIP_RANGE)
    actions = actions.astype(np.float32)

    full = ImitationDataset(states, actions)
    n_val = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=generator)
    print(f"  total frames: {len(full)} from {n_replays} replay(s)  (train {n_train} / val {n_val})")
    return train_set, val_set


def make_replay_dataloaders(
    path: str,
    batch_size: int = 256,
    val_split: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_frames: int | None = None,
):
    # dataset_replays.pkl is big enough (127.98M frames as of last parse) that
    # CPU->GPU transfer, not compute, is the bottleneck on a 4060 -- pin_memory
    # speeds up the actual H2D copy. num_workers defaults to 0 on purpose:
    # ImitationDataset.__getitem__ does no real per-sample CPU work (it just
    # indexes an already-loaded in-memory tensor), so workers have nothing to
    # overlap with the GPU, and on Windows (no fork) each worker has to
    # reconstruct this multi-GB in-memory dataset via spawn -- measured ~150s of
    # pure overhead per epoch for zero benefit. Override if a future dataset
    # actually does per-sample work worth parallelizing.
    train_set, val_set = load_replay_dataset(path, val_split, seed, max_frames)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader
