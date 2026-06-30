# Environments

Five environments, each adding a new mechanic that forces Ashby to develop a different skill. They're ordered by difficulty — not arbitrary, but a deliberate curriculum: each one introduces exactly one new challenge on top of what came before.

---

## GridWorld

**The baseline.** Learn to navigate, nothing else.

```
┌───┬───┬───┬───┬───┐
│ S │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │ T │
└───┴───┴───┴───┴───┘

S = agent start (random, not target)
T = target (4,4), fixed
```

**State (4D):** `[agent_row, agent_col, target_row, target_col]`, all normalized to `[0, 1]` by dividing by 4.

**Actions:** UP, DOWN, LEFT, RIGHT (wall-clamped — stepping into a border stays in place).

**Rewards:**
- `+1.0` on reaching target (terminal)
- `-0.5` on wall bump (tried to move, didn't go anywhere)
- `-0.01` per step (discourages wandering)

**Why 5×5?** Large enough to require a non-trivial policy (25 states), small enough for tabular Q-learning to converge fast (~100 episodes). The benchmark needs a baseline to prove the tabular agent works before we add complexity.

**Why target at (4,4)?** Fixed target creates a clear Q-value gradient. Every cell has a deterministic shortest path. Tabular Q-learning should memorize this perfectly once it's seen all states.

**What Ashby learns here:** Basic directed movement. The Q-table maps every (position, action) pair to an expected return. After ~100 episodes the agent navigates optimally from any start position.

**Result:** 100% success rate, convergence at episode 101, Tabular Q.

---

## FrozenLake

**Add uncertainty.** Same grid, same goal — but ice makes movement probabilistic.

```
┌───┬───┬───┬───┬───┐
│ S │ . │ . │ . │ . │
├───┼───┼───┼───┼───┤
│ . │ . │ H │ . │ . │
├───┼───┼───┼───┼───┤
│ . │ . │ . │ . │ H │
├───┼───┼───┼───┼───┤
│ . │ H │ . │ . │ . │
├───┼───┼───┼───┼───┤
│ . │ . │ . │ . │ T │
└───┴───┴───┴───┴───┘

S = start (0,0), fixed
T = target (4,4), fixed
. = ice tile (70% forward, 15% left, 15% right)
H = hole (random placement, 2-3 per map)
(map layout is random at construction, then fixed)
```

**State (5D):** `[agent_row, agent_col, target_row, target_col, tile_type]` where `tile_type` is 0.0 (normal), 1.0 (ice), or 2.0 (hole, seen only at termination).

**Actions:** Same 4 directions. On an ice tile, the intended action succeeds 70% of the time; 15% chance of rotating 90° left, 15% right.

**Rewards:**
- `+1.0` on reaching target
- `-1.0` on falling into hole (terminal)
- `-0.01` per step

**Why stochastic?** A deterministic optimal policy becomes suboptimal under slip. The agent must learn to avoid ice near holes even when the detour costs extra steps. This tests whether the Q-table can internalize uncertainty.

**Why fixed map per instance?** The map is generated once at `FrozenLake::new()` and stays fixed across all episodes. This is intentional: the agent must learn THIS specific layout — hole positions, ice density, safe paths. It's memorization + uncertainty handling, not generalization.

**Critical design decision:** the map is random at construction but fixed for the lifetime of the object. This has consequences for transfer learning — see [transfer_learning.md](transfer_learning.md).

**Why 2–3 holes?** Enough to create genuine danger, not so many that safe paths disappear. The hole placement avoids the start cell's immediate neighborhood (`|row_diff| <= 1 && |col_diff| <= 1`).

**What Ashby learns here:** Probabilistic planning. The optimal policy avoids routes that pass near holes through ice tiles, even if those routes are shorter in expected steps.

**Result:** 97% success rate, convergence ~episode 391, Tabular Q.

---

## PredatorGrid

**Add an adversary.** The world is no longer passive — something is actively trying to catch you.

```
┌───┬───┬───┬───┬───┐
│ A │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │ T │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │ P │
└───┴───┴───┴───┴───┘

A = agent (starts at one of 4 corners)
P = predator (starts at opposite corner)
T = target, always at center (2,2)
```

**State (6D):** `[agent_row, agent_col, target_row, target_col, pred_row, pred_col]`, all normalized to `[0, 1]`.

**Actions:** Same 4 directions.

**Predator behavior:** Greedy one-step pursuit. Closes the largest axis gap first; ties go to the row axis. The bias is arbitrary but consistent — the agent can learn to exploit it.

**Episode logic:**
1. Agent moves.
2. If agent reaches target → `+1.0`, done.
3. If agent is on predator's cell → `-1.0`, done.
4. Predator moves.
5. If predator is on agent's cell → `-1.0`, done.
6. Otherwise → `-0.01`, continue.

**Rewards:**
- `+1.0` on reaching target (agent priority — stepping onto target before predator is a win)
- `-1.0` on capture
- `-0.01` per step

**Why corners?** Four corner pairings (agent at one corner, predator at opposite) randomized each reset. Forces the policy to generalize across multiple starting configurations, not memorize one approach direction.

**Why center target?** Equidistant from all corners — the agent can't take a straight-line path without passing near the predator's starting zone.

**What Ashby learns here:** Simultaneous goal-seeking and threat avoidance. Two objectives in the same action. The Q-table must encode the value of every (agent, target, predator) triple.

**Result:** 96% success rate, convergence ~episode 484, Tabular Q.

---

## MazeWorld

**Break tabular.** A new maze every episode makes memorization impossible.

```
Episode N:                  Episode N+1:
┌─────────────────┐        ┌─────────────────┐
│ # # # # # # # # │        │ # # # # # # # # │
│ # A   #       # │        │ #   # # #     # │
│ #   # # # #   # │        │ #   #   # # # # │
│ #   #       # # │        │ #         #   # │
│ # # # # # #   # │        │ # # # # # # #   │
│ #           # # │        │ #   #   # # # # │
│ # # # # # T # # │        │ # # # #   # T # │
│ # # # # # # # # │        │ # # # # # # # # │
└─────────────────┘        └─────────────────┘

# = wall
  = open corridor
A = agent (random open cell, ≥5 Manhattan steps from target)
T = target (random open cell)
(Different layout every reset — tabular Q-learning fails here)
```

**State (53D):** `[agent_row, agent_col, target_row, target_col]` (4 values) + all 49 grid cells flattened (1.0 = wall, 0.0 = open). The agent sees the full map.

**Actions:** Same 4 directions.

**Maze generation:** Recursive backtracking (DFS). Maze cells are at odd row/col indices in the 7×7 grid; corridors are carved at midpoints. Every generated maze is a perfect maze — no loops, exactly one path between any two cells.

**Rewards:**
- `+1.0` on reaching target
- `-0.5` on wall bump (heavy penalty — the agent must learn the walls fast)
- `-0.01` per step

**Why 7×7?** Small enough to train on CPU without a GPU, large enough that the maze topology is non-trivial. The 5×5 grid was too constrained to generate interesting maze structure.

**Why recursive backtracking?** It produces perfect mazes with a single solution path. This forces the DQN to actually navigate rather than stumble into the goal through shortcuts. It's also cheap to generate — no external dependency.

**Why expose the full grid in the state?** The agent needs the map to plan. Without it, every cell looks identical and the network can't generalize across layouts. The 49-cell binary grid is the key input that lets DQN succeed where tabular fails.

**Why DQN?** The state space is effectively infinite (new layout every episode). Tabular Q-learning would need a new entry per unique maze — it can't generalize. DQN learns features (wall ahead, open corridor, direction to target) that apply to any layout.

**What Ashby learns here:** Spatial generalization. The network develops internal representations for "wall", "open corridor", "heading toward goal", "dead end" — concepts that are useful regardless of which specific maze it's in.

**Result:** 91% success rate, convergence ~episode 883, DQN.

---

## ResourceHunter

**Add multiple objectives and a time limit.**

```
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │ R │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │ R │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │ A │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │ R │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │ R │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │ R │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘

A = agent (random position each episode)
R = resource (5 placed randomly, no repeats)
Hunger counter: starts at 50, decrements each step,
resets to 50 on collection. Hits 0 = death.
```

**State (18D):** `[agent_row, agent_col, hunger_norm, res1_row, res1_col, res1_collected, ..., res5_row, res5_col, res5_collected]`. Hunger normalized to `[0, 1]` by dividing by 50. Resource positions normalized by 6.

**Actions:** Same 4 directions (open grid, no walls).

**Rewards (potential-based shaping):**
- `-0.01` per step (base cost)
- `+0.05 × (dist_before - dist_after)` — reward for moving toward the nearest uncollected resource (potential-based shaping, measured before eating so the target doesn't shift mid-step)
- `+0.5` on collecting a resource (hunger resets to 50)
- `+2.0` bonus on collecting all 5 (added to the last collection's reward)
- `-1.0` on starvation (terminal)

**Why potential-based shaping?** Without shaping, the sparse reward signal (+0.5 on collection, -1.0 on death) leaves the agent with no gradient signal during the long stretch between meals. Potential-based shaping is reward-equivalent for the optimal policy — it doesn't change what the agent *should* do, only how fast it gets feedback.

**Why epsilon_decay=0.999?** Collecting 5 sequential targets requires much more exploration time than finding 1. With 0.997 (like MazeWorld), epsilon hits minimum around episode 1000 — not enough exploration time for the 5-target task. 0.999 pushes that to episode ~3000.

**Why MAX_HUNGER=50?** A 7×7 grid means the agent is at most ~12 Manhattan steps from any resource. At 50 hunger, a competent agent can visit resources in sequence with time to spare. Too short and starvation is unavoidable; too long and there's no real pressure.

**What Ashby learns here:** Dynamic priority. The optimal target changes as resources are collected. The agent can't memorize a fixed route — positions re-randomize every episode. It must learn a policy: "go to the nearest uncollected resource, prioritize when hungry."

**Result:** Converges around episode 4000 with epsilon_decay=0.999. Needs the full 5000-episode budget from the original pipeline.
