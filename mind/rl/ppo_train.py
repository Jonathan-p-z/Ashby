"""PPO training via rlgym-ppo -- replaces rl_train.py's DDPG loop.

DDPG's actor update (`actor_loss = -critic(s, actor(s)).mean()`) has no brakes: the
very first gradient step can drag the actor arbitrarily far toward whatever a
freshly-initialized (i.e. still mostly random) critic happens to prefer, which is
exactly why rl_agent.py needed a critic-bootstrapping step to avoid wrecking the
imitation-pretrained actor on step one. PPO doesn't have that problem by
construction -- its clipped surrogate objective caps how far a single update is
allowed to move the policy, so an imitation-pretrained starting point survives
fine-tuning instead of getting overwritten in the first few gradient steps
(catastrophic forgetting). rlgym-ppo also runs its own multi-process rollout
collection, so this replaces rl_train.py's hand-rolled worker/queue/self-play-pool
plumbing entirely -- one Learner, n_proc workers, done.
"""

import os
import sys

# rlgym_ppo's own status reporting calls locale.setlocale(LC_ALL, '') and then
# number-formats with grouping=True -- on a machine whose Windows locale is French,
# that inserts a narrow no-break space (U+202F) as the thousands separator, which
# the default cp1252 console codepage can't encode. Left alone, that crashes the
# very first iteration report of every run. Reconfiguring stdout/stderr to UTF-8
# up front (before anything prints) fixes it regardless of the system locale.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

from rlgym_ppo import Learner

sys.path.insert(0, os.path.dirname(__file__))
from env import make_env, WEIGHTS_DIR, STATE_SIZE, ACTION_SIZE

N_PROC = 4
TIMESTEP_LIMIT = 50_000_000
SAVE_EVERY_TS = 1_000_000  # a few checkpoints across the run, not just one at the very end

CHECKPOINTS_DIR = os.path.join(WEIGHTS_DIR, "ppo")
IMITATION_WEIGHTS = os.path.join(WEIGHTS_DIR, "rl_imitation.pth")

# Deliberately matched to ImitationNet's hidden trunk (env.py's imitation.py: 18->256->256->8)
# instead of rlgym-ppo's own default (256, 256, 256) -- see _load_imitation_into_policy for why.
POLICY_LAYER_SIZES = (256, 256)


def _create_env():
    """Top-level (picklable) env factory -- rlgym-ppo spawns this in each worker
    process, and a lambda/closure can't cross that process boundary on Windows.
    spawn_opponents=True so the same policy plays both cars: PPO's self-play needs
    no separate opponent pool the way the old DDPG setup did, since a shared policy
    training against itself IS the self-play.
    """
    return make_env(spawn_opponents=True)


def _load_imitation_into_policy(policy: torch.nn.Module, weights_path: str, device: str):
    """Splices imitation.py's ImitationNet weights into rlgym-ppo's ContinuousPolicy.

    The two hidden layers transfer cleanly because POLICY_LAYER_SIZES was chosen to
    match ImitationNet's trunk exactly: model.0 and model.2 are the same shape as
    ImitationNet's net.0 and net.2. The output layer can't transfer as cleanly --
    ContinuousPolicy ends in Linear(256, ACTION_SIZE*2) + Tanh (the first half is
    the action mean, the second half maps to the Gaussian's std -- see
    rlgym_ppo.util.torch_functions.MapContinuousToAction), while ImitationNet ends
    in a bare Linear(256, ACTION_SIZE) with no distribution at all. Only the mean
    half has anything to inherit, so that's the only slice copied; the std half
    keeps rlgym-ppo's own random init, since imitation learning never produced
    anything resembling an uncertainty estimate to seed it with.
    """
    imitation_state = torch.load(weights_path, map_location=device, weights_only=True)
    policy_state = policy.state_dict()

    policy_state["model.0.weight"] = imitation_state["net.0.weight"].to(device)
    policy_state["model.0.bias"] = imitation_state["net.0.bias"].to(device)
    policy_state["model.2.weight"] = imitation_state["net.2.weight"].to(device)
    policy_state["model.2.bias"] = imitation_state["net.2.bias"].to(device)

    action_size = imitation_state["net.4.weight"].shape[0]
    policy_state["model.4.weight"][:action_size] = imitation_state["net.4.weight"].to(device)
    policy_state["model.4.bias"][:action_size] = imitation_state["net.4.bias"].to(device)

    policy.load_state_dict(policy_state)


def train():
    print("=" * 65)
    print("  Ashby x rlgym-ppo -- PPO fine-tuning")
    print(f"  workers: {N_PROC}  |  timestep limit: {TIMESTEP_LIMIT}")
    print("=" * 65)

    learner = Learner(
        env_create_function=_create_env,
        n_proc=N_PROC,
        policy_layer_sizes=POLICY_LAYER_SIZES,
        timestep_limit=TIMESTEP_LIMIT,
        checkpoints_save_folder=CHECKPOINTS_DIR,
        add_unix_timestamp=False,  # one stable mind/weights/ppo/<timesteps>/ history, not a new folder per run
        save_every_ts=SAVE_EVERY_TS,
    )

    # cumulative_timesteps > 0 here means Learner already auto-resumed a prior PPO
    # checkpoint from CHECKPOINTS_DIR (its default "latest" behavior) -- in that case
    # the policy is already mid-fine-tuning and splicing imitation weights back in
    # would just throw that progress away. Only seed from imitation on a fresh start.
    if learner.agent.cumulative_timesteps == 0 and os.path.exists(IMITATION_WEIGHTS):
        _load_imitation_into_policy(learner.ppo_learner.policy, IMITATION_WEIGHTS, learner.device)
        print(f"Policy seeded from imitation weights: {IMITATION_WEIGHTS}")
    elif learner.agent.cumulative_timesteps > 0:
        print(f"Resumed PPO checkpoint at {learner.agent.cumulative_timesteps} timesteps -- skipping imitation seed.")
    else:
        print("No imitation weights found -- policy starts from PPO's own random init.")

    learner.learn()

    # learn() only saves on its own save_every_ts cadence or a keyboard 'c'/'q' --
    # reaching timestep_limit normally doesn't trigger one, so save explicitly here
    # to guarantee a checkpoint exists at the end of every run.
    learner.save(learner.agent.cumulative_timesteps)
    print(f"\nFinal checkpoint saved -> {CHECKPOINTS_DIR}")


def smoke_test():
    """--test: verifies the one genuinely fragile piece of this file -- the
    imitation-weight splice -- without paying for a multi-process Learner spin-up.
    Builds a real ContinuousPolicy at the exact dimensions rlgym-ppo would use,
    splices in rl_imitation.pth (if present) or a freshly-saved ImitationNet
    otherwise, and checks the policy's mean output matches ImitationNet's forward
    pass exactly -- proof the layer-index assumptions in
    _load_imitation_into_policy actually hold, not just that they run without
    raising.
    """
    from rlgym_ppo.ppo import ContinuousPolicy
    from imitation import ImitationNet

    device = "cpu"
    weights_path = IMITATION_WEIGHTS
    tmp_weights = None
    if not os.path.exists(weights_path):
        print(f"No {weights_path} yet -- using a freshly-initialized ImitationNet instead.")
        tmp_weights = os.path.join(WEIGHTS_DIR, "_ppo_train_smoke_test.pth")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save(ImitationNet().state_dict(), tmp_weights)
        weights_path = tmp_weights

    try:
        policy = ContinuousPolicy(
            STATE_SIZE, ACTION_SIZE * 2, POLICY_LAYER_SIZES, device,
        )
        _load_imitation_into_policy(policy, weights_path, device)
        print("  splice: ok (no shape mismatch)")

        imitation_net = ImitationNet()
        imitation_net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        imitation_net.eval()

        states = torch.randn(16, STATE_SIZE)
        with torch.no_grad():
            spliced_mean, _ = policy.get_output(states)
            # ContinuousPolicy's trunk ends in Tanh (ImitationNet's doesn't) -- apply
            # it here so this compares the same thing on both sides instead of a
            # squashed output against a raw one.
            expected = torch.tanh(imitation_net(states))

        max_diff = (spliced_mean - expected).abs().max().item()
        assert max_diff < 1e-4, f"spliced policy mean diverges from ImitationNet by {max_diff}"
        print(f"  mean output matches ImitationNet (post-tanh): max diff {max_diff:.2e}")
    finally:
        if tmp_weights and os.path.exists(tmp_weights):
            os.remove(tmp_weights)

    print("ppo_train.py --test PASSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ashby x rlgym-ppo -- PPO fine-tuning from imitation weights")
    parser.add_argument("--test", action="store_true", help="smoke-test the imitation-weight splice, no real training")
    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        train()
