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
import json
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
    """Same split logic as load_datasets, but for parse_replays.py's replay dataset
    -- replay states already go through AshbyObsBuilder's own scaling (see
    parse_replays.py), so the same CLIP_RANGE outlier guard applies as-is.

    `path` is a directory (parse_replays.py's DATASET_DIR): a manifest.json plus
    shard_NNNNN.pkl files, not one monolithic pickle. That split exists because a
    single-file dataset means loading it means holding the whole thing in memory
    twice, however large it's grown to -- once as whatever np.concatenate needs to
    read from, once as the array it produces. Shards let both this function and
    parse_replays.py's writer avoid that: only load what's actually needed, and
    never hold more than one shard's extra copy in memory at a time.

    max_frames randomly subsamples down to that many rows when the dataset has
    more -- at 127.98M+ frames the full dataset is well over 10GB in memory,
    which starts fighting the OS for RAM and slows down everything, not just the
    DataLoader. When set, only the shards actually needed to reach max_frames are
    ever opened -- the rest are never loaded at all.
    """
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No dataset manifest at {manifest_path} -- run `make rl-parse` (or "
            "parse_replays.py directly) to build it from downloaded replays first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    shards = manifest["shards"]
    total_frames = manifest["n_frames"]
    n_replays = manifest.get("n_replays", "?")
    if not shards:
        raise RuntimeError(f"Dataset manifest at {manifest_path} lists no shards.")

    rng = np.random.default_rng(seed)

    if max_frames is not None and total_frames > max_frames:
        # Pick whole shards, in random order, until their frames cover max_frames --
        # shards are ~500-replay batches assembled from imap_unordered completion
        # order (not correlated with any replay's own characteristics), so this is
        # a close approximation of true frame-level uniform sampling without ever
        # opening shards that aren't needed.
        order = rng.permutation(len(shards))
        selected, running = [], 0
        for shard_idx in order:
            selected.append(shards[shard_idx])
            running += shards[shard_idx]["n_frames"]
            if running >= max_frames:
                break

        state_parts, action_parts = [], []
        for shard in selected:
            with open(os.path.join(path, shard["file"]), "rb") as f:
                shard_data = pickle.load(f)
            state_parts.append(shard_data["states"])
            action_parts.append(shard_data["actions"])
            del shard_data

        states = np.concatenate(state_parts)
        actions = np.concatenate(action_parts)
        del state_parts, action_parts

        if len(states) > max_frames:
            idx = rng.choice(len(states), size=max_frames, replace=False)
            states = states[idx]
            actions = actions[idx]

        print(f"  loaded {len(selected)}/{len(shards)} shard(s), subsampled to {len(states)} of {total_frames} frames (RAM guard)")
    else:
        # Loading everything -- preallocate the destination once and fill it shard by
        # shard instead of np.concatenate-ing a list of per-shard arrays, so peak
        # memory is "one destination array + one shard" rather than "every shard
        # held at once, plus a second copy for the concatenated result".
        states = None
        actions = None
        offset = 0
        for shard in shards:
            with open(os.path.join(path, shard["file"]), "rb") as f:
                shard_data = pickle.load(f)
            shard_states = shard_data["states"]
            shard_actions = shard_data["actions"]
            if states is None:
                states = np.empty((total_frames, shard_states.shape[1]), dtype=np.float32)
                actions = np.empty((total_frames, shard_actions.shape[1]), dtype=np.float32)
            n = len(shard_states)
            states[offset:offset + n] = shard_states
            actions[offset:offset + n] = shard_actions
            offset += n
            del shard_data, shard_states, shard_actions

        print(f"  loaded all {len(shards)} shard(s), {total_frames} frames (no subsampling)")

    states = np.clip(states.astype(np.float32, copy=False), -CLIP_RANGE, CLIP_RANGE)
    actions = actions.astype(np.float32, copy=False)

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
    # The replay dataset is big enough (127.98M+ frames as of last parse) that
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
