from collections import deque
import random


class ReplayBuffer:
    """Stores past (state, action, reward, next_state, done) transitions.

    Random sampling breaks the temporal correlation between consecutive steps —
    without this, the Q-table update sees the same sequence over and over and
    tends to overfit to recent trajectories.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
