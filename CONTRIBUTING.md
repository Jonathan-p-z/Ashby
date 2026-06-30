# Contributing

This is a personal research project. Contributions are welcome if they're honest and minimal — no feature creep, no premature abstractions.

---

## Adding a new environment

Five steps, in order. Don't skip any of them.

### 1. Implement the Rust struct

Create `games/src/your_env.rs`. Implement the `Environment` trait:

```rust
use ashby_core::{Action, Environment, Reward, State};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct YourEnv {
    // your fields
    rng: StdRng,  // use StdRng, not ThreadRng — ThreadRng is not Send
}

impl Environment for YourEnv {
    fn reset(&mut self) -> State { /* ... */ }
    fn step(&mut self, action: Action) -> (State, Reward, bool) { /* ... */ }
    fn action_space(&self) -> Vec<Action> { vec![0, 1, 2, 3] }
    fn state_shape(&self) -> Vec<usize> { vec![YOUR_STATE_DIM] }
}
```

State values should be normalized to `[0, 1]` — the neural network trains better on bounded inputs, and it keeps the Q-table key rounding consistent for tabular agents.

**Reward shaping note:** sparse rewards (only +1 on success, -1 on failure) work for simple environments. For anything requiring sequential actions, add potential-based shaping: a small reward proportional to progress toward the goal, measured before and after the action. See `resource_hunter.rs` for the pattern.

### 2. Export from `games/src/lib.rs`

```rust
pub mod your_env;
pub use your_env::YourEnv;
```

### 3. Wrap in the PyO3 bridge

Add to `bridge/src/lib.rs`:

```rust
use ashby_games::YourEnv;

#[pyclass]
struct PyYourEnv { inner: YourEnv }

#[pymethods]
impl PyYourEnv {
    #[new]
    fn new() -> Self { PyYourEnv { inner: YourEnv::new() } }
    fn reset(&mut self) -> Vec<f32> { self.inner.reset() }
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { self.inner.step(action) }
    fn action_space(&self) -> Vec<usize> { self.inner.action_space() }
    fn state_shape(&self) -> Vec<usize> { self.inner.state_shape() }
}
```

Register it in the module:

```rust
fn ashby(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // existing classes...
    m.add_class::<PyYourEnv>()?;
    Ok(())
}
```

Run `make build` and verify the import works: `python -c "import ashby; e = ashby.PyYourEnv(); print(e.reset())"`.

### 4. Add to the training pipeline

In `mind/train.py`, add your environment to `run_training()`. Choose the agent type:
- Tabular (`QLearningAgent`) if the state space is small and discrete
- DQN (`DQNAgent`) if the state space is large, continuous, or infinite (new layout every episode)

Choose `epsilon_decay` based on task difficulty — harder tasks need more exploration time.

### 5. Document it

Add a section to `docs/environments.md` with:
- An ASCII grid showing the layout
- State dimensions and what each value represents
- Reward structure and why it's shaped this way
- What skill this environment teaches the agent

---

## Training an agent and reading the results

```bash
make train    # full pipeline, saves training_results.png
make eval     # evaluation table without the plot
```

**Reading the convergence episode:** the training script reports the first episode where the 100-episode rolling success rate crosses 75%. This is the primary metric for comparing agents. It's sensitive to variance — run a second seed if you see an unexpected result before drawing conclusions.

**Reading the loss curve:** the reward subplot in `training_results.png` shows the raw per-episode total reward (noisy) and a smoothed moving average. A healthy curve climbs from negative to positive, with the noisy raw values spreading around the average. A flat or downward curve after 500+ episodes means the agent is stuck — check reward shaping, epsilon schedule, and Q-table key bucketing.

**Reading the outcome rate subplot:** the success/failure rates over time. A good sign: failure rate drops as success rate rises. If both stay flat or move together, the reward signal is probably ambiguous.

---

## Code rules

These are not suggestions:

**Language:** code and comments in English only.

**Comments:** explain the WHY, never the WHAT. If the comment would be removed by a junior dev who "cleans up obvious comments", it was the wrong comment. If the next developer would be surprised without it, it's the right comment.

```python
# Round to 2 decimals to bucket nearby continuous positions together.
# GridWorld already normalizes to a small finite set, so this is mostly
# defensive for future environments that might add noise.
return tuple(round(s, 2) for s in state)
```

**No complexity for its own sake.** A helper function that's called once is just bureaucracy. Three similar lines is better than a premature abstraction. Only abstract when you're repeating the same logic in three different places and it's genuinely the same logic.

**Logs that tell a story.** The logs are not debug output — they're the agent's diary. They should describe what the agent is experiencing, not what the code is doing. "success 78% | ε=0.090 (exploiting)" tells you the agent is starting to exploit what it learned. "step() called 47291 times" tells you nothing.

**No backwards-compatibility hacks.** If you change an interface, change all callers. Don't add `_old_method` aliases or `# removed in v2` comments. Version history is what git is for.

**No error handling for impossible cases.** Trust the framework. If `env.reset()` returns a state, it's a valid state. Don't write `if state is None: raise ValueError` around code that the framework guarantees won't produce `None`.
