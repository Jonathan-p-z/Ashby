# Architecture

Three layers: Rust environments, a PyO3 bridge, and Python agents. Each layer has one job and doesn't bleed into the others.

---

## The Environment trait (Rust)

Everything an agent needs to interact with a world, and nothing more:

```rust
pub trait Environment {
    fn reset(&mut self) -> State;
    fn step(&mut self, action: Action) -> (State, Reward, bool);
    fn action_space(&self) -> Vec<Action>;
    fn state_shape(&self) -> Vec<usize>;
}

pub type State  = Vec<f32>;   // flat observation vector
pub type Action = usize;      // index into action_space()
pub type Reward = f32;        // scalar feedback signal
```

This is a stripped-down OpenAI Gym interface. The three-tuple from `step` — `(new_state, reward, done)` — is the Bellman update's raw input. Everything the agent needs, nothing about rendering or human-readable output.

**Why Rust for environments?** Three reasons:

1. **Performance** — the training loop calls `step()` millions of times. Rust pays no runtime overhead. This matters especially for MazeWorld's procedural maze generation (recursive backtracking DFS each episode) and ResourceHunter's multi-entity collision detection.

2. **Correctness** — environment bugs corrupt training silently. Rust's type system and borrow checker catch a class of bugs (off-by-one in grid bounds, mutable aliasing during predator pursuit) at compile time.

3. **`StdRng` instead of `ThreadRng`** — PyO3 requires `pyclass` types to implement `Send`. `ThreadRng` uses an `Rc` internally, which is not `Send`. Every environment uses `StdRng::from_entropy()` instead. This is a concrete example of how Rust's type system exposes concurrency properties that would be silent assumptions in Python.

---

## The PyO3 bridge (`bridge/src/lib.rs`)

The bridge wraps each Rust struct in a Python-visible class:

```rust
#[pyclass]
struct PyMazeWorld { inner: MazeWorld }

#[pymethods]
impl PyMazeWorld {
    #[new]
    fn new() -> Self { PyMazeWorld { inner: MazeWorld::new() } }
    fn reset(&mut self) -> Vec<f32> { self.inner.reset() }
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { self.inner.step(action) }
    fn action_space(&self) -> Vec<usize> { self.inner.action_space() }
    fn state_shape(&self) -> Vec<usize> { self.inner.state_shape() }
}
```

`maturin develop` compiles this into a native Python extension installed into `.venv`. From Python:

```python
import ashby
env = ashby.PyMazeWorld()
state = env.reset()        # list[float], length 53
state, reward, done = env.step(0)  # UP
```

The `Vec<f32>` → `list[float]` conversion happens at the bridge boundary. Python receives a regular list; NumPy and PyTorch consume it normally. No shared memory, no IPC overhead — it's a direct function call into compiled code.

---

## The tabular agent (`agent.py` — QLearningAgent)

Standard Q-learning with a sparse dict as the Q-table:

```
Q-table: dict[(state_tuple, action)] -> float
Update:  Q(s,a) += lr * (r + γ * max_a' Q(s',a') - Q(s,a))
```

State keys are rounded to 2 decimal places: `tuple(round(s, 2) for s in state)`. This is defensive bucketing — GridWorld's normalized positions land on a small finite set of values, so exact comparison works. FrozenLake adds a tile type; PredatorGrid adds predator position. All within tabular range.

**Hyperparameters:**
- `learning_rate=0.1` — standard for tabular Q-learning; high enough for fast updates on the small state space
- `gamma=0.99` — high discount preserves long-horizon credit (needed for PredatorGrid where the predator catches you 10+ steps after a bad choice)
- `epsilon_start=1.0, epsilon_end=0.05` — full exploration to start, 5% residual for robustness
- `epsilon_decay`: 0.995 (GridWorld), 0.997 (FrozenLake, PredatorGrid) — decays over the training budget

**Why experience replay in a tabular agent?** The same ReplayBuffer as DQN. Tabular Q-learning doesn't technically need it — you can update online. But it removes temporal correlation from the batch, which stabilizes updates on FrozenLake's stochastic transitions. And it's the same interface as DQN, which simplifies the training loop.

---

## The DQN agent (`dqn_agent.py` — DQNAgent)

When the state space is infinite (MazeWorld: new maze every episode, ResourceHunter: random resource positions), tabular Q-learning fails. DQN replaces the Q-table with a neural network:

```
Q(s, a) ≈ QNetwork(s)[a]
```

### Network architecture

```
Linear(state_size, 128)  →  ReLU
Linear(128, 64)          →  ReLU
Linear(64, n_actions)
```

Three fully-connected layers. Small enough to train fast on CPU; large enough to capture meaningful patterns in a 53-dimensional maze state.

### Hyperparameters and their rationale

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | 0.001 | Adam default; lower would slow convergence on CPU |
| `gamma` | 0.99 | Long-horizon discount; needed for multi-step goals |
| `batch_size` | 64 | Standard; balances gradient quality vs. speed |
| `buffer_capacity` | 50,000 | Enough history to break temporal correlation |
| `target_update_freq` | 200 | Frequent enough to track progress, slow enough to not chase a moving target |
| `epsilon_decay` | 0.997 (MazeWorld), 0.999 (ResourceHunter) | ResourceHunter needs 3× longer exploration — 5 sequential targets vs. 1 |

### The target network

The most important stability trick: a frozen copy of the policy network (`target_net`) is used to compute TD targets.

Without it, the training signal is:
```
loss = (Q_policy(s,a) - [r + γ · max Q_policy(s',a')])²
```

`Q_policy` appears on both sides. Every gradient step changes both the prediction AND the target. The network chases itself. This is notoriously unstable — it's the reason early DQN implementations would diverge.

With a target network that only syncs every 200 steps:
```
loss = (Q_policy(s,a) - [r + γ · max Q_target(s',a')])²
```

The target is held fixed for 200 steps. The policy network can move toward it without the target moving away at the same rate. This is the core DQN stability insight from Mnih et al. (2015).

### Batch construction

```python
states_t = torch.FloatTensor(np.array(states))
```

The `np.array(states)` conversion before `FloatTensor` is not cosmetic — it's a performance fix. Letting PyTorch convert a list-of-lists element by element is significantly slower than converting via NumPy first (which uses a single `memcpy`-style operation). On 50,000-transition replay buffers, this matters.

---

## The TransferAgent (`transfer_agent.py`)

Same training loop as DQNAgent. Different architecture: named sublayers instead of a `Sequential`, plus three transfer-specific methods.

### Architecture

```
TransferNet:
  input_layer  = Linear(state_size, 128)   # environment-specific input encoder
  hidden_layer = Linear(128, 64)           # universal feature extractor
  output_layer = Linear(64, action_size)   # environment-specific policy head
```

### `save_weights(path)`

Saves the full `policy_net` state dict plus metadata:

```python
torch.save({
    'state_size': self.state_size,
    'action_size': self.n_actions,
    'state_dict': self.policy_net.state_dict(),
}, path)
```

The `state_size` metadata is critical — it's how `load_weights` knows whether the `input_layer` can be transferred.

### `load_weights(path)`

```python
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
saved_sd = checkpoint['state_dict']

# hidden_layer always transfers — it's the actual universal feature extractor
self.policy_net.hidden_layer.weight.data.copy_(saved_sd['hidden_layer.weight'])
self.policy_net.hidden_layer.bias.data.copy_(saved_sd['hidden_layer.bias'])

# input_layer only transfers when state dimensionality matches exactly
self._input_transferred = (checkpoint['state_size'] == self.state_size)
if self._input_transferred:
    self.policy_net.input_layer.weight.data.copy_(saved_sd['input_layer.weight'])
    self.policy_net.input_layer.bias.data.copy_(saved_sd['input_layer.bias'])
```

`output_layer` is always left at random initialization. Fine-tuning is supposed to retrain it — loading pretrained output weights would defeat the purpose of measuring how fast the policy head adapts.

### `freeze_base_layers()`

```python
for param in self.policy_net.hidden_layer.parameters():
    param.requires_grad = False

if self._input_transferred:
    for param in self.policy_net.input_layer.parameters():
        param.requires_grad = False

# Rebuild optimizer over trainable params only
trainable = [p for p in self.policy_net.parameters() if p.requires_grad]
self.optimizer = optim.Adam(trainable, lr=self.learning_rate)
```

**Why rebuild the optimizer?** PyTorch optimizers hold direct references to parameter tensors. If you set `requires_grad=False` after optimizer creation, the optimizer still holds those parameters and may apply spurious zero updates or corrupt its internal state (momentum, adaptive learning rates). Rebuilding with only the trainable subset is the correct approach.

**Why does backprop still work with frozen layers?** This is a subtle PyTorch detail. When `input_layer` and `hidden_layer` have `requires_grad=False`, PyTorch's autograd doesn't accumulate gradients for them. But `output_layer.weight` still has `requires_grad=True`. When computing `y = output_layer(x)` where `x` has `requires_grad=False`, PyTorch still computes `∂loss/∂output_layer.weight` because `output_layer.weight` is a leaf tensor that requires grad. The gradient flows correctly to the trainable parameters even though the frozen parameters don't receive gradients.

### `weight_delta(snapshot)`

```python
snapshot = agent.weight_snapshot()   # before training
# ... train ...
deltas = agent.weight_delta(snapshot)  # L2 norm of (current - snapshot)
```

Measures how much each layer's weights actually moved during training. Used in `pretrain.py` to show which layers are most active in each environment — typically `hidden_layer` changes the most, which confirms it's the one doing real feature learning.
