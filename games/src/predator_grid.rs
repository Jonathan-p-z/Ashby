use ashby_core::{Action, Environment, Reward, State};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const SIZE: usize = 5;
const UP: usize = 0;
const DOWN: usize = 1;
const LEFT: usize = 2;
const RIGHT: usize = 3;

// Agent and predator start at opposite corners so the tension is immediate.
// Four possible pairings — randomized each reset to force the Q-table
// to learn evasion from multiple angles, not just one fixed approach.
const CORNERS: [(usize, usize); 4] = [(0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)];

// Target at the center — equidistant from all corners, maximally ambiguous
// about whether the agent should take a straight or evasive path.
const TARGET: (usize, usize) = (SIZE / 2, SIZE / 2);

/// A 5x5 grid where a greedy predator hunts the agent every single step.
///
/// The agent must reach the center while the predator closes in from the
/// opposite corner. Unlike GridWorld and FrozenLake, the threat moves —
/// so a static memorized path is not enough. The agent has to reason about
/// two things at once: where it wants to go, and where the danger is coming from.
pub struct PredatorGrid {
    agent: (usize, usize),
    predator: (usize, usize),
    target: (usize, usize),
    rng: StdRng,
}

impl PredatorGrid {
    pub fn new() -> Self {
        let mut rng = StdRng::from_entropy();
        let idx = rng.gen_range(0..CORNERS.len());
        let agent = CORNERS[idx];
        let predator = opposite_corner(agent);
        PredatorGrid { agent, predator, target: TARGET, rng }
    }

    fn random_start(&mut self) -> ((usize, usize), (usize, usize)) {
        let idx = self.rng.gen_range(0..CORNERS.len());
        let agent = CORNERS[idx];
        (agent, opposite_corner(agent))
    }

    fn apply_agent_move(&self, action: Action) -> (usize, usize) {
        let (r, c) = self.agent;
        match action {
            UP => (r.saturating_sub(1), c),
            DOWN => ((r + 1).min(SIZE - 1), c),
            LEFT => (r, c.saturating_sub(1)),
            RIGHT => (r, (c + 1).min(SIZE - 1)),
            _ => (r, c),
        }
    }

    // Greedy one-step move: close whichever axis has the larger gap first.
    // Ties go to the row axis — the bias is arbitrary but consistent,
    // which matters for the agent's ability to learn predictable evasion.
    fn predator_step(&self) -> (usize, usize) {
        let (pr, pc) = (self.predator.0 as isize, self.predator.1 as isize);
        let (ar, ac) = (self.agent.0 as isize, self.agent.1 as isize);
        let dr = ar - pr;
        let dc = ac - pc;

        if dr.abs() >= dc.abs() && dr != 0 {
            ((pr + dr.signum()) as usize, pc as usize)
        } else if dc != 0 {
            (pr as usize, (pc + dc.signum()) as usize)
        } else {
            // Predator is already on the agent — shouldn't be reached
            // mid-step since we check for capture before this call.
            self.predator
        }
    }

    fn observe(&self) -> State {
        let scale = (SIZE - 1) as f32;
        let (ar, ac) = self.agent;
        let (tr, tc) = self.target;
        let (pr, pc) = self.predator;
        vec![
            ar as f32 / scale,
            ac as f32 / scale,
            tr as f32 / scale,
            tc as f32 / scale,
            pr as f32 / scale,
            pc as f32 / scale,
        ]
    }
}

fn opposite_corner(pos: (usize, usize)) -> (usize, usize) {
    (SIZE - 1 - pos.0, SIZE - 1 - pos.1)
}

impl Environment for PredatorGrid {
    fn reset(&mut self) -> State {
        let (agent, predator) = self.random_start();
        self.agent = agent;
        self.predator = predator;
        self.observe()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        self.agent = self.apply_agent_move(action);

        // Agent priority: if it reaches the target this step, episode ends
        // before the predator gets to move.
        if self.agent == self.target {
            return (self.observe(), 1.0, true);
        }

        // Agent walked directly into the predator's current cell.
        if self.agent == self.predator {
            return (self.observe(), -1.0, true);
        }

        // Predator hunts.
        self.predator = self.predator_step();

        if self.predator == self.agent {
            return (self.observe(), -1.0, true);
        }

        (self.observe(), -0.01, false)
    }

    fn action_space(&self) -> Vec<Action> {
        vec![UP, DOWN, LEFT, RIGHT]
    }

    fn state_shape(&self) -> Vec<usize> {
        // [agent_row, agent_col, target_row, target_col, predator_row, predator_col]
        vec![6]
    }
}

impl Default for PredatorGrid {
    fn default() -> Self {
        Self::new()
    }
}
