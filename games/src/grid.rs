use ashby_core::{Action, Environment, Reward, State};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const GRID_SIZE: usize = 5;

// Action indices — named so the mapping is explicit when reading step()
const UP: usize = 0;
const DOWN: usize = 1;
const LEFT: usize = 2;
const RIGHT: usize = 3;

/// A 5x5 grid where the agent learns to reach a fixed target.
///
/// Simple enough to solve quickly, rich enough to validate that the Q-table
/// is actually learning a policy and not just memorizing noise.
pub struct GridWorld {
    agent: (usize, usize),
    target: (usize, usize),
    // StdRng instead of ThreadRng because PyO3 requires pyclass types to be Send,
    // and ThreadRng uses an Rc internally which prevents that.
    rng: StdRng,
}

impl GridWorld {
    pub fn new() -> Self {
        let mut rng = StdRng::from_entropy();
        // Target is fixed at the bottom-right — creates a clear gradient
        // in the Q-values once the agent starts converging.
        let target = (GRID_SIZE - 1, GRID_SIZE - 1);
        let agent = Self::random_position_excluding(&mut rng, target);

        GridWorld { agent, target, rng }
    }

    fn random_position_excluding(
        rng: &mut StdRng,
        excluded: (usize, usize),
    ) -> (usize, usize) {
        loop {
            let pos = (rng.gen_range(0..GRID_SIZE), rng.gen_range(0..GRID_SIZE));
            if pos != excluded {
                return pos;
            }
        }
    }

    fn observe(&self) -> State {
        // Normalize to [0, 1] — makes the values well-behaved if we ever
        // swap the Q-table for a neural net down the road.
        let scale = (GRID_SIZE - 1) as f32;
        vec![
            self.agent.0 as f32 / scale,
            self.agent.1 as f32 / scale,
            self.target.0 as f32 / scale,
            self.target.1 as f32 / scale,
        ]
    }

    fn apply_move(&self, action: Action) -> (usize, usize) {
        let (x, y) = self.agent;
        match action {
            UP => (x.saturating_sub(1), y),
            DOWN => ((x + 1).min(GRID_SIZE - 1), y),
            LEFT => (x, y.saturating_sub(1)),
            RIGHT => (x, (y + 1).min(GRID_SIZE - 1)),
            _ => (x, y),
        }
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> State {
        self.agent = Self::random_position_excluding(&mut self.rng, self.target);
        self.observe()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        let next = self.apply_move(action);

        // Penalize wall-bumping: the agent tried to move but didn't go anywhere.
        // Without this, it learns to spam wall moves near the target.
        let hit_wall = next == self.agent;
        self.agent = next;

        let reached_target = self.agent == self.target;

        let reward = if reached_target {
            1.0
        } else if hit_wall {
            -0.5
        } else {
            -0.01 // small step cost to discourage wandering
        };

        (self.observe(), reward, reached_target)
    }

    fn action_space(&self) -> Vec<Action> {
        vec![UP, DOWN, LEFT, RIGHT]
    }

    fn state_shape(&self) -> Vec<usize> {
        // [agent_x, agent_y, target_x, target_y]
        vec![4]
    }
}

impl Default for GridWorld {
    fn default() -> Self {
        Self::new()
    }
}
