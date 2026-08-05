"""The autonomous-improvement half of the pipeline.

Imitation learning gets Ashby to "drives around and vaguely goes for the ball", but a
regression network trained on human demonstrations has no idea whether an action was
actually *good* -- it just knows what a human did in a similar-looking state. To improve
past that ceiling it needs something that can score its own outcomes and push it toward
better ones, which is what plain behavioral cloning structurally cannot do.

That's the one addition beyond the brief: a small critic network Q(s, a) alongside the
policy. The policy ("actor") is architecturally identical to imitation.py's net -- same
18->256->256->8 shape -- so rl_imitation.pth loads into it with no adapter, and its
weights are the only thing rl_train.py saves as rl_policy.pth. The critic is scratch
plumbing that only exists to make "replay buffer + target network + epsilon-greedy
around a continuous action" a well-defined training signal instead of a contradiction --
you can't apply a Bellman update directly to raw actions, only to a value estimate of
them. This is DDPG stripped to the minimum that still does that job.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import STATE_SIZE, ACTION_SIZE, DEVICE, POS_COEF, VEL_COEF, common_values, rl_math, AshbyReward
from imitation import ImitationNet as ActorNet
from memory import ReplayBuffer


def _estimate_immediate_reward(state: np.ndarray) -> float:
    """A cheap, state-only stand-in for AshbyReward, used only to give the critic a
    sane initial baseline (see RLAgent._bootstrap_critic). Real AshbyReward needs a
    live GameState -- goal deltas, ball-touch flags, previous-tick boost -- none of
    which exist for a bare state vector with no simulation behind it. This reuses
    just the terms that AshbyReward computes from a single snapshot (no history
    needed): how well positioned/aligned/moving the car and ball already are toward
    the opponent's goal. AshbyObsBuilder always orients state so "+Y" means "toward
    the opponent's goal" regardless of team (see its own docstring), so the target
    here is always the canonical ORANGE_GOAL_BACK point -- no team_num needed.
    """
    car_pos = state[0:3] / POS_COEF
    car_vel = state[3:6] / VEL_COEF
    ball_pos = state[10:13] / POS_COEF
    ball_vel = state[13:16] / VEL_COEF
    goal_pos = np.array(common_values.ORANGE_GOAL_BACK, dtype=np.float32)

    to_ball = ball_pos - car_pos
    dist_to_ball = float(np.linalg.norm(to_ball))
    ball_to_goal = goal_pos - ball_pos
    ball_to_goal_dist = float(np.linalg.norm(ball_to_goal))

    vel_to_ball = (
        float(np.dot(car_vel, to_ball / dist_to_ball)) / common_values.CAR_MAX_SPEED
        if dist_to_ball > 1e-3 else 0.0
    )
    align = float(rl_math.cosine_similarity(ball_to_goal, to_ball)) if dist_to_ball > 1e-3 else 0.0
    dist_reward = float(np.exp(-dist_to_ball / AshbyReward.OFFENSIVE_DIST_DECAY))
    offensive_potential = AshbyReward._krc(vel_to_ball, align, dist_reward)

    vel_ball_to_goal = (
        float(np.dot(ball_vel, ball_to_goal / ball_to_goal_dist)) / common_values.BALL_MAX_SPEED
        if ball_to_goal_dist > 1e-3 else 0.0
    )
    dist_weighted_align = align * float(np.exp(-dist_to_ball / AshbyReward.ALIGN_DIST_DECAY))

    return (
        AshbyReward.OFFENSIVE_POTENTIAL_WEIGHT * offensive_potential
        + AshbyReward.VELOCITY_BALL_TO_GOAL_WEIGHT * vel_ball_to_goal
        + AshbyReward.DIST_WEIGHTED_ALIGN_WEIGHT * dist_weighted_align
    )


class CriticNet(nn.Module):
    """Q(s, a) -> scalar. Judges an action instead of proposing one."""

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1)).squeeze(-1)


class RLAgent:
    """Actor-critic with a target pair and a replay buffer -- DDPG's core recipe,
    kept to the minimum needed to fine-tune an imitation-pretrained policy."""

    def __init__(
        self,
        imitation_weights: str = None,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 500,
        batch_size: int = 128,
        memory_capacity: int = 100_000,
        device: torch.device = None,
    ):
        # rl_train.py's worker processes force this to CPU -- RocketSim never touches
        # the GPU anyway, and giving every worker its own CUDA context for a network
        # this small would burn VRAM for zero benefit. The learner process is the one
        # place that actually wants DEVICE (cuda if available).
        self.device = device if device is not None else DEVICE

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self._steps = 0

        self.actor = ActorNet().to(self.device)
        self.actor_target = ActorNet().to(self.device)
        self.critic = CriticNet().to(self.device)
        self.critic_target = CriticNet().to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(capacity=memory_capacity)

        if imitation_weights and os.path.exists(imitation_weights):
            state_dict = torch.load(imitation_weights, map_location=self.device, weights_only=True)
            self.actor.load_state_dict(state_dict)
            print(f"Actor initialized from imitation weights: {imitation_weights}")
            self._bootstrap_critic()
        else:
            print("No imitation weights found -- actor starts from random init (pure RL from scratch).")

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.eval()
        self.critic_target.eval()

    def _bootstrap_critic(self, n_random_states: int = 1000, epochs: int = 200):
        """Fits the critic to Q(s, actor(s)) ~= a cheap immediate-reward estimate of
        s, instead of leaving it at its random init. A freshly-initialized critic
        scores an already-competent imitation-pretrained actor's actions with pure
        noise, and the very first actor_loss = -critic(s, actor(s)) gradient step
        would drag the actor toward whatever nonsense that random critic happens to
        prefer -- undoing the imitation pretraining before real RL ever gets a
        chance to build on it. Giving the critic a sane starting baseline first
        avoids that.

        Uses whatever's already in the replay buffer if there's anything there yet
        (there normally isn't -- this runs at construction, before any real env
        steps), otherwise samples n_random_states plausible-looking states to fit
        against.
        """
        if len(self.memory) > 0:
            states = np.array([entry[0] for entry in self.memory.buffer], dtype=np.float32)
        else:
            states = np.random.uniform(-1.0, 1.0, size=(n_random_states, STATE_SIZE)).astype(np.float32)
            # Keep the quaternion slice unit-length and boost/on_ground in [0, 1] so
            # these look like real AshbyObsBuilder output instead of pure noise.
            quat = states[:, 6:10]
            states[:, 6:10] = quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-6, None)
            states[:, 16:18] = np.clip(states[:, 16:18], 0.0, 1.0)

        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            actions_t = self.actor(states_t)
        targets_t = torch.FloatTensor([_estimate_immediate_reward(s) for s in states]).to(self.device)

        for _ in range(epochs):
            q = self.critic(states_t, actions_t)
            loss = nn.MSELoss()(q, targets_t)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

        print(
            f"Critic bootstrapped on {len(states)} state(s) from the actor's own actions "
            f"(fit loss={loss.item():.5f}, mean target reward={targets_t.mean().item():.4f})."
        )

    def act(self, state: np.ndarray) -> np.ndarray:
        """Epsilon-greedy for a continuous action space: a coin flip between a fresh
        random guess and the actor's current best guess, rather than argmax over a
        discrete set that doesn't exist here."""
        if np.random.random() < self.epsilon:
            return np.random.uniform(-1.0, 1.0, size=ACTION_SIZE).astype(np.float32)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.actor(state_t).cpu().numpy()[0]

    def best_action(self, state: np.ndarray) -> np.ndarray:
        """Pure exploitation -- used for evaluation and watch.py."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.actor(state_t).cpu().numpy()[0]

    def remember(self, state, action: np.ndarray, reward: float, next_state, done: bool):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)

        # Critic learns to predict returns, same Bellman target as any DQN --
        # just bootstrapped off the actor's next action instead of an argmax over Q.
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target = rewards_t + self.gamma * target_q * (1.0 - dones_t)

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor learns to produce whatever action the critic currently rates highest --
        # gradient flows from the critic's judgment straight back into the policy.
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        return critic_loss.item(), actor_loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load_weights(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)
        self.actor.eval()
        self.actor_target.eval()
