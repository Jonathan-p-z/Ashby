# Transfer Learning in Ashby

The idea: train one agent sequentially on all five environments, then test whether that accumulated knowledge helps it learn faster on any single environment compared to starting from scratch.

The short answer: **yes, dramatically, but only under the right conditions.**

---

## The hypothesis

Neural networks trained on related tasks tend to develop shared internal representations in their early layers. The classic result from computer vision: a network trained on ImageNet develops edge detectors and texture extractors in its early layers that transfer to other visual tasks. The late layers (task-specific classifiers) need retraining, but the early layers generalize.

The analogy for navigation: a network that has learned to navigate a grid should develop internal representations for concepts like "direction to goal", "obstacle ahead", "proximity to target" — concepts that are useful regardless of which specific grid world it's in.

---

## The architecture

```
input_layer  (state_size → 128)   environment-specific encoder
hidden_layer (128 → 64)           universal feature extractor
output_layer (64 → action_size)   environment-specific policy head
```

The hypothesis is that `hidden_layer` is the transferable part. It receives a 128-dimensional representation of the current state (from `input_layer`) and compresses it to 64 dimensions of abstract navigation features. If the hypothesis holds, those 64 features should be meaningful in any navigation environment, not just the one that trained them.

### Sequential pretraining

Ashby trains on the five environments one after another. Each transition carries knowledge forward:

1. **GridWorld (4D)** — fresh weights, 800 episodes. Learns directed movement.
2. **FrozenLake (5D)** — loads `hidden_layer` from GridWorld checkpoint. 1500 episodes. Adapts to probabilistic movement.
3. **PredatorGrid (6D)** — loads `hidden_layer` from FrozenLake checkpoint. 2000 episodes. Adds threat avoidance.
4. **MazeWorld (53D)** — loads `hidden_layer` from PredatorGrid checkpoint. 2500 episodes. Learns spatial generalization.
5. **ResourceHunter (18D)** — loads `hidden_layer` from MazeWorld checkpoint. 4000 episodes. Multi-objective planning.

When state sizes match between consecutive environments, both `input_layer` and `hidden_layer` carry over. When they differ (e.g., PredatorGrid 6D → MazeWorld 53D), only `hidden_layer` transfers — the `input_layer` is re-initialized.

After pretraining, one checkpoint is saved per environment: `weights/gridworld.pth`, `weights/frozen_lake.pth`, etc.

### Fine-tuning with frozen base layers

For each benchmark comparison, the transfer agent:
1. Loads the matching checkpoint (e.g., `maze_world.pth` for MazeWorld)
2. Calls `freeze_base_layers()` — locks gradients on `input_layer` and `hidden_layer` (if `input_layer` was transferred)
3. Trains normally, but only `output_layer` (64→4) updates

This means the optimizer has ~260 parameters to update (64×4 weights + 4 biases) instead of the full ~28,000. The frozen layers act as a fixed feature extractor; the policy head learns to read those features for the current task.

---

## Benchmark results, environment by environment

### MazeWorld — +710 episodes (8x speedup)

**Scratch:** 811 episodes to reach 75% success.
**Transfer:** 101 episodes to reach 75% success.

This is the clearest result in the entire project, and it works for a specific reason: MazeWorld generates a **new random maze every episode**. The pretrained agent never saw any particular maze twice during pretraining. The fine-tuning agent doesn't either. Both work on the same distribution.

The pretrained `input_layer` (53D→128) has learned to encode "wall vs. open" patterns from the 49-cell grid. The `hidden_layer` has learned what those patterns mean for navigation. When we freeze both and retrain only the output head (64→4), the network already has the right vocabulary. The output layer just needs to learn: "when feature X is high, action 2 (LEFT) is better." That's a fast linear regression, not a full learning problem.

The speedup is real because the pretrained features are **distribution-general**. MazeWorld's reward structure and mechanics are identical between pretraining and fine-tuning — only the specific instance changes, and the features handle that.

### FrozenLake — -223 episodes (transfer is slower)

**Scratch:** 681 episodes.
**Transfer:** 904 episodes.

This is a failure case, and it's instructive. FrozenLake generates its map **once at construction time** and keeps it fixed forever. The pretrain and benchmark use different `FrozenLake` instances — different random hole placements, different ice density patterns.

The `frozen_lake.pth` checkpoint was trained on map instance A. Its `input_layer` (5D→128) learned to respond to the specific tile values it saw during training — encoding features like "this position is dangerous because there's a hole two steps right."

When we load that checkpoint into the benchmark (map instance B), `input_layer` is frozen but the map is different. The features it produces are misleading: they're tuned to a geometry that doesn't exist in this new map. The output layer gets wrong signals and has to fight against them. Scratch is better because at least it starts neutral.

**The lesson:** Transfer hurts when the pretrained encoder is instance-specific. It helps when the encoder is distribution-general.

### PredatorGrid — -121 episodes (same cause)

**Scratch:** 578 episodes.
**Transfer:** 699 episodes.

Same mechanism as FrozenLake, different surface. PredatorGrid's layout is fixed (the grid is always 5×5 with the same walls), but the corner pairings for agent and predator start positions are randomized on each reset. The pretrained `input_layer` (6D→128) learned to respond to the geometry of corner-to-center navigation; the benchmark might not have exactly the same starting conditions.

The effect is smaller than FrozenLake (121 vs 223 episodes penalty) because PredatorGrid's state is simpler — 6D with direct position readings — and the mechanics are more consistent across instances.

### GridWorld — 0 gain (no effect)

**Scratch:** 101 episodes.
**Transfer:** 101 episodes.

GridWorld is too simple to benefit from transfer. 101 episodes is the minimum needed for tabular Q-learning to visit enough states — it's not bottlenecked by feature quality. Even if the pretrained features were perfect, you can't go below ~100 episodes because the epsilon schedule needs time to decay.

GridWorld also illustrates why "0 gain" isn't the same as "transfer fails". The pretrained weights (from `gridworld.pth`, state_size=4 matches, full transfer) are loaded correctly. There's just no room for improvement at this scale.

### ResourceHunter — 0 gain (insufficient pretraining)

**Scratch:** >2500 episodes (didn't converge).
**Transfer:** >2500 episodes (didn't converge).

The pretrain allocated 4000 episodes for ResourceHunter, but with `epsilon_decay=0.999` and a 7-step minimum exploration phase, the agent barely gets 1000 episodes of real exploitation before the pretrain ends. The saved `resource_hunter.pth` reflects a partially trained agent, not a competent one.

The original pipeline uses 5000 episodes. The benchmark caps at 2500. Neither is enough. This isn't a transfer learning failure — it's a resource allocation problem.

---

## Why the negative results matter

It would be easy to tune the experiment to only show the MazeWorld success. That would be dishonest.

The FrozenLake and PredatorGrid results reveal something real: **transfer learning is not free**. Frozen features are a commitment. If the pretrained representation is wrong for the new task, you've locked yourself into a bad starting point that scratch training doesn't have.

The conditions where transfer helps:
- The environment is **procedural** — every episode generates a new instance from the same distribution
- The state encodes **raw observations** that remain meaningful across instances (MazeWorld's grid flattening)
- The pretrained features are **high-level** enough to generalize (128→64 abstract navigation signals)

The conditions where transfer hurts:
- The environment has a **fixed random instance** per object lifetime (FrozenLake's map, PredatorGrid's corner assignments)
- The `input_layer` is frozen to an instance-specific encoding
- There's a mismatch between the distribution during pretraining and fine-tuning

---

## Directions for improvement

### Progressive freezing
Don't freeze everything at once. Start with only `hidden_layer` frozen (more flexibility), and if convergence stalls, unfreeze it. This is particularly relevant for FrozenLake — letting `input_layer` retrain while keeping `hidden_layer`'s abstract features would likely turn the negative result into a positive one.

### Curriculum learning
The pretraining order matters. Going GridWorld → FrozenLake → PredatorGrid → MazeWorld → ResourceHunter is a rough difficulty curriculum. A more principled curriculum (sorting by state space complexity or reward sparsity) might produce better-transferred features.

### Adaptive fine-tuning
Detect at load time whether the pretrained checkpoint's state distribution is similar to the target environment. If distributions diverge (measured by e.g. KL divergence on state embeddings), reduce the freeze depth. Only freeze `hidden_layer` when input encoders differ; freeze both when they match and the task distribution is similar.

### Same-instance pretraining
For fixed-map environments, pretrain and fine-tune on the **same instance**. Create the environment once, save the instance, use it for both pretrain and benchmark. This would likely turn FrozenLake and PredatorGrid from negative to positive. The downside is reduced generalizability — you're measuring recall, not transfer.

### Elastic weight consolidation (EWC)
Catastrophic forgetting is a known problem in sequential training. EWC regularizes later training to preserve weights that were important for earlier tasks. Implementing it would require tracking Fisher information after each environment — non-trivial but it would directly address the "pretrain overwrites earlier knowledge" problem.
