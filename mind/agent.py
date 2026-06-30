import numpy as np
from collections import defaultdict
from memory import ReplayBuffer


class QLearningAgent:
    """Tabular Q-Learning with epsilon-greedy exploration.

    The Q-table is a dict keyed by (state_tuple, action) — sparse by default,
    which is fine for small discrete state spaces like GridWorld. If the state
    space explodes we'll need to discretize or move to function approximation,
    but let's not optimize for that yet.
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,       # discount: how much the agent cares about the future
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # defaultdict means unseen (state, action) pairs start at 0 —
        # optimistic zero initialization keeps things simple.
        self.q_table: dict = defaultdict(float)

        self.memory = ReplayBuffer(capacity=10_000)

    def _state_key(self, state) -> tuple:
        # Round to 2 decimals to bucket nearby continuous positions together.
        # GridWorld already normalizes to a small finite set, so this is mostly
        # defensive for future environments that might add noise.
        return tuple(round(s, 2) for s in state)

    def q(self, state, action: int) -> float:
        return self.q_table[(self._state_key(state), action)]

    def best_action(self, state) -> int:
        return int(np.argmax([self.q(state, a) for a in range(self.n_actions)]))

    def act(self, state) -> int:
        """Epsilon-greedy: explore randomly or exploit the best known action."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self.best_action(state)

    def remember(self, state, action: int, reward: float, next_state, done: bool):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self, batch_size: int = 32):
        """Sample a batch and apply the Bellman update to each transition."""
        if len(self.memory) < batch_size:
            return  # not enough experience yet

        batch = self.memory.sample(batch_size)

        for state, action, reward, next_state, done in batch:
            key = (self._state_key(state), action)

            if done:
                # Terminal state: no future rewards to discount.
                target = reward
            else:
                best_next = max(self.q(next_state, a) for a in range(self.n_actions))
                target = reward + self.gamma * best_next

            # Standard Q-Learning update: nudge toward the target with lr.
            self.q_table[key] += self.lr * (target - self.q_table[key])

    def decay_epsilon(self):
        """Called once per episode to shift from exploring toward exploiting."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @property
    def is_exploring(self) -> bool:
        return self.epsilon > 0.1
