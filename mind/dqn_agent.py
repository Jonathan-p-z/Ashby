import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from memory import ReplayBuffer


class QNetwork(nn.Module):
    """The function approximator that replaces the Q-table.

    Three fully connected layers — small enough to train fast on CPU,
    large enough to capture meaningful patterns in a 53-dimensional maze state.
    """

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Deep Q-Network with experience replay and a target network.

    The target network is the key stability trick: it's a frozen copy of the
    policy network used to compute TD targets. Without it, the network is
    chasing a moving target in both senses — the gradients oscillate and
    learning often diverges. We sync the target every `target_update_freq` steps.
    """

    def __init__(
        self,
        state_size: int,
        n_actions: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 200,
        batch_size: int = 64,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self._steps = 0

        # policy_net trains every step; target_net is a frozen snapshot
        # that only gets synced periodically.
        self.policy_net = QNetwork(state_size, n_actions)
        self.target_net = QNetwork(state_size, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=50_000)

    def act(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax().item())

    def best_action(self, state) -> int:
        """Pure exploitation — used during evaluation."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax().item())

    def remember(self, state, action: int, reward: float, next_state, done: bool):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # np.array first — converting a list of lists to tensor this way is
        # significantly faster than letting torch parse it element by element.
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones, dtype=np.float32))

        # Q(s, a) predicted by the policy network for the actions actually taken.
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Bellman target: r + γ · max_a' Q_target(s', a') · (1 − done)
        # Using the target network here is what stops the training signal from
        # chasing itself — we hold the target fixed for `target_update_freq` steps.
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @property
    def is_exploring(self) -> bool:
        return self.epsilon > 0.1
