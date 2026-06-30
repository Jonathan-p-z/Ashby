import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayBuffer


class TransferNet(nn.Module):
    """Three-layer DQN with named sublayers so individual layers can be frozen or loaded selectively.

    input_layer  (state_size → 128)  environment-specific encoder
    hidden_layer (128 → 64)          universal feature extractor — the actually transferable part
    output_layer (64 → action_size)  environment-specific policy head
    """

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.input_layer = nn.Linear(state_size, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)


class TransferAgent:
    """DQN agent built for cross-environment weight transfer.

    Identical training loop to DQNAgent; adds save/load/freeze for the
    sequential pretraining → fine-tuning workflow.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 200,
        batch_size: int = 64,
    ):
        self.state_size = state_size
        self.n_actions = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._steps = 0
        self._input_transferred = False

        self.policy_net = TransferNet(state_size, action_size)
        self.target_net = TransferNet(state_size, action_size)
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

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones, dtype=np.float32))

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

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

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'state_size': self.state_size,
            'action_size': self.n_actions,
            'state_dict': self.policy_net.state_dict(),
        }, path)

    def load_weights(self, path: str) -> dict:
        """Load pretrained weights, transferring what's dimensionally compatible.

        hidden_layer (128→64) always transfers — it's the universal navigation
        feature extractor that survives domain shifts.
        input_layer only transfers when state dimensionality matches exactly.
        output_layer is always kept fresh for fine-tuning.

        Returns info about which layers were transferred.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        saved_sd = checkpoint['state_dict']
        saved_state_size = checkpoint['state_size']

        self.policy_net.hidden_layer.weight.data.copy_(saved_sd['hidden_layer.weight'])
        self.policy_net.hidden_layer.bias.data.copy_(saved_sd['hidden_layer.bias'])

        self._input_transferred = (saved_state_size == self.state_size)
        if self._input_transferred:
            self.policy_net.input_layer.weight.data.copy_(saved_sd['input_layer.weight'])
            self.policy_net.input_layer.bias.data.copy_(saved_sd['input_layer.bias'])

        self.target_net.load_state_dict(self.policy_net.state_dict())

        transferred = ['input', 'hidden'] if self._input_transferred else ['hidden']
        return {'layers_transferred': transferred, 'saved_state_size': saved_state_size}

    def freeze_base_layers(self):
        """Lock the transferred base layers so only the policy head fine-tunes.

        Rebuilds the optimizer over trainable params only — frozen params must
        not appear in the optimizer or they'll receive spurious zero updates.
        """
        for param in self.policy_net.hidden_layer.parameters():
            param.requires_grad = False

        if self._input_transferred:
            for param in self.policy_net.input_layer.parameters():
                param.requires_grad = False

        trainable = [p for p in self.policy_net.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=self.learning_rate)

    def weight_snapshot(self) -> dict:
        """Capture current layer weight tensors for later delta computation."""
        return {
            'input':  self.policy_net.input_layer.weight.data.clone(),
            'hidden': self.policy_net.hidden_layer.weight.data.clone(),
            'output': self.policy_net.output_layer.weight.data.clone(),
        }

    def weight_delta(self, snapshot: dict) -> dict:
        """L2 norm of (current - snapshot) per layer — how much each layer moved."""
        current = {
            'input':  self.policy_net.input_layer.weight.data,
            'hidden': self.policy_net.hidden_layer.weight.data,
            'output': self.policy_net.output_layer.weight.data,
        }
        return {name: (current[name] - snapshot[name]).norm().item() for name in current}
