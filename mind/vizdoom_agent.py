import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from memory import ReplayBuffer


# One-hot button vectors mapping action index → [MOVE_LEFT, MOVE_RIGHT, ATTACK]
ACTIONS = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]


def _preprocess(frame: np.ndarray, out_h: int = 84, out_w: int = 84) -> np.ndarray:
    """(H,W), (H,W,1) or (H,W,3) uint8 → (1, out_h, out_w) uint8, nearest-neighbor resize.

    Defaults to 84×84 so all existing agents keep working unchanged.
    Pass out_h=120, out_w=160 for the GPU live-training mode.
    Storing uint8 instead of float32 keeps replay buffer ~4x smaller.
    We normalize to [0,1] in learn() when we actually need the floats.
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = (frame[..., 0] * 0.299 + frame[..., 1] * 0.587 + frame[..., 2] * 0.114).astype(np.uint8)
    elif frame.ndim == 3:
        gray = frame[..., 0]
    else:
        gray = frame
    h, w = gray.shape
    rows = np.linspace(0, h - 1, out_h).astype(np.int32)
    cols = np.linspace(0, w - 1, out_w).astype(np.int32)
    return gray[np.ix_(rows, cols)][np.newaxis]  # (1, out_h, out_w) uint8


class CNNQNetwork(nn.Module):
    """Maps a grayscale frame to Q-values for each action.

    Mnih 2015 conv architecture. The flatten size is computed dynamically via a
    dummy forward pass so it stays correct across input resolutions — no hardcoded
    magic numbers that silently break when the resolution changes.
    """

    def __init__(self, n_actions: int, input_shape: tuple = (1, 84, 84)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Dummy pass to compute flatten size — avoids the classic "3136 is wrong
        # for this resolution" silent failure when input dimensions change.
        with torch.no_grad():
            flatten_size = self.conv(torch.zeros(1, *input_shape)).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).view(x.size(0), -1))


class VizDoomDQNAgent:
    """DQN for pixel-input environments — same mechanics as DQNAgent but with CNN.

    The lower learning rate (0.0001 vs 0.001) is because CNN gradients over raw
    pixels can be large and noisy. Gradient clipping handles outlier updates.
    Everything else — target network, experience replay, epsilon-greedy — is
    identical to the dense DQN used for MazeWorld and ResourceHunter.
    """

    def __init__(
        self,
        n_actions: int = 3,
        input_shape: tuple = (1, 84, 84),
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 1000,
        batch_size: int = 32,
        memory_capacity: int = 100_000,
    ):
        self.n_actions   = n_actions
        self.input_shape = input_shape
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay     = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size  = batch_size
        self._steps      = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = CNNQNetwork(n_actions, input_shape).to(self.device)
        self.target_net = CNNQNetwork(n_actions, input_shape).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=memory_capacity)

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        _, out_h, out_w = self.input_shape
        return _preprocess(frame, out_h, out_w)

    def act(self, frame: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(self._prep(frame).astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax().item())

    def best_action(self, frame: np.ndarray) -> int:
        """Pure exploitation — no random moves."""
        state_t = torch.FloatTensor(self._prep(frame).astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax().item())

    def remember(self, frame, action: int, reward: float, next_frame, done: bool):
        self.memory.push(self._prep(frame), action, reward, self._prep(next_frame), done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Normalize uint8 → float32, then move the whole batch to device in one shot.
        # Transferring one big tensor is faster than transferring many small ones.
        states_t      = torch.FloatTensor(np.array(states,      dtype=np.float32) / 255.0).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states, dtype=np.float32) / 255.0).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        dones_t       = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q   = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load_weights(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.target_net.eval()
