"""Behavioral cloning -- Ashby learns to copy whatever got recorded by record.py.

No reward function, no exploration, just regression: given a state, predict the action
a human produced in that state. This is deliberately the dumbest possible way to bootstrap
a policy, and that's the point -- it gets the agent from "outputs random noise" to "roughly
knows which way to drive" in minutes instead of the thousands of episodes raw RL would need
to stumble into the same behavior by accident. rl_agent.py picks up from here.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from env import STATE_SIZE, ACTION_SIZE, DEVICE, WEIGHTS_DIR
from dataset import make_dataloaders

WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "rl_imitation.pth")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_imitation_results.png")

N_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001
REPORT_EVERY = 10


class ImitationNet(nn.Module):
    """The exact architecture rl_agent.py's Q-network reuses, so imitation weights
    load into it directly with no adapter layer in between."""

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _run_epoch(model, loader, criterion, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    n_batches = 0
    for states, actions in loader:
        states, actions = states.to(DEVICE), actions.to(DEVICE)

        if training:
            optimizer.zero_grad()
        preds = model(states)
        loss = criterion(preds, actions)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train():
    print("=" * 65)
    print("  Ashby x RLGym -- behavioral cloning")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    print("\nLoading recorded sessions...")
    train_loader, val_loader = make_dataloaders(batch_size=BATCH_SIZE)

    model = ImitationNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    print(f"\nTraining for {N_EPOCHS} epochs...")

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch == 1 or epoch % REPORT_EVERY == 0:
            print(f"  epoch {epoch:4d}/{N_EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nWeights saved -> {WEIGHTS_PATH}")

    _plot(train_losses, val_losses)
    print(f"Plot saved    -> {RESULTS_PATH}")


def _plot(train_losses: list, val_losses: list):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_losses, label="train", color="steelblue")
    ax.plot(val_losses, label="val", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Ashby x RLGym -- imitation learning")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH, dpi=120)
    plt.close(fig)


def smoke_test():
    """--test: exercise the network and training step on synthetic data -- no recorded
    sessions required, no real training loop, just proof the pieces fit together."""
    print(f"Device: {DEVICE}")

    model = ImitationNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    fake_states = torch.randn(32, STATE_SIZE, device=DEVICE)
    fake_actions = torch.rand(32, ACTION_SIZE, device=DEVICE) * 2 - 1

    preds = model(fake_states)
    assert preds.shape == (32, ACTION_SIZE), f"expected (32, {ACTION_SIZE}), got {preds.shape}"

    loss = criterion(preds, fake_actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  forward/backward pass: ok (loss={loss.item():.4f})")
    print(f"  output shape: {tuple(preds.shape)}")
    print("imitation.py --test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ashby x RLGym -- behavioral cloning")
    parser.add_argument("--test", action="store_true", help="smoke-test the network on synthetic data")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        train()
