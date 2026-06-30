use ashby_core::{Action, Environment, Reward, State};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const SIZE: usize = 5;

const UP: usize = 0;
const DOWN: usize = 1;
const LEFT: usize = 2;
const RIGHT: usize = 3;

// On an ice tile, 70% chance of going where intended, 15% each side.
const SLIP_FORWARD: f64 = 0.70;
const SLIP_SIDE: f64 = 0.15;

#[derive(Clone, Copy, PartialEq)]
enum Tile {
    Normal,
    Ice,
    Hole,
}

/// A 5x5 grid where ice makes movement probabilistic and holes punish mistakes.
///
/// The map is generated once and stays fixed across episodes — the agent must
/// learn THIS specific layout, holes included. That fixed-map constraint is what
/// forces the Q-table to build a genuine internal model rather than just
/// averaging over random configurations.
pub struct FrozenLake {
    grid: [[Tile; SIZE]; SIZE],
    target: (usize, usize),
    agent: (usize, usize),
    rng: StdRng,
}

impl FrozenLake {
    pub fn new() -> Self {
        let mut rng = StdRng::from_entropy();
        let start = (0, 0);
        let target = (SIZE - 1, SIZE - 1);

        let mut grid = [[Tile::Normal; SIZE]; SIZE];

        // Scatter ice tiles at ~40% density, leaving start and target clear.
        for r in 0..SIZE {
            for c in 0..SIZE {
                if (r, c) == start || (r, c) == target {
                    continue;
                }
                if rng.gen::<f64>() < 0.4 {
                    grid[r][c] = Tile::Ice;
                }
            }
        }

        // Drop 2–3 holes, keeping them away from start so the agent
        // isn't punished before it has a single step to reason about.
        let n_holes = 2 + rng.gen_range(0usize..=1);
        let mut placed = 0;
        let mut attempts = 0;
        while placed < n_holes && attempts < 200 {
            let r = rng.gen_range(0..SIZE);
            let c = rng.gen_range(0..SIZE);
            let near_start = r.abs_diff(start.0) <= 1 && c.abs_diff(start.1) <= 1;
            if (r, c) != target && !near_start && grid[r][c] != Tile::Hole {
                grid[r][c] = Tile::Hole;
                placed += 1;
            }
            attempts += 1;
        }

        FrozenLake { grid, target, agent: start, rng }
    }

    // When on ice, the intended action may rotate to a perpendicular direction.
    fn effective_action(&mut self, intended: Action) -> Action {
        let (r, c) = self.agent;
        if self.grid[r][c] != Tile::Ice {
            return intended;
        }
        let roll: f64 = self.rng.gen();
        if roll < SLIP_FORWARD {
            intended
        } else if roll < SLIP_FORWARD + SLIP_SIDE {
            rotate_left(intended)
        } else {
            rotate_right(intended)
        }
    }

    fn apply_move(&self, action: Action) -> (usize, usize) {
        let (r, c) = self.agent;
        match action {
            UP => (r.saturating_sub(1), c),
            DOWN => ((r + 1).min(SIZE - 1), c),
            LEFT => (r, c.saturating_sub(1)),
            RIGHT => (r, (c + 1).min(SIZE - 1)),
            _ => (r, c),
        }
    }

    fn observe(&self) -> State {
        let scale = (SIZE - 1) as f32;
        let (ar, ac) = self.agent;
        let (tr, tc) = self.target;
        let tile_code = match self.grid[ar][ac] {
            Tile::Normal => 0.0f32,
            Tile::Ice => 1.0,
            // Hole appears here only in the terminal transition —
            // the agent observes it once and the episode ends.
            Tile::Hole => 2.0,
        };
        vec![
            ar as f32 / scale,
            ac as f32 / scale,
            tr as f32 / scale,
            tc as f32 / scale,
            tile_code,
        ]
    }
}

fn rotate_left(action: Action) -> Action {
    match action {
        UP => LEFT,
        LEFT => DOWN,
        DOWN => RIGHT,
        RIGHT => UP,
        _ => action,
    }
}

fn rotate_right(action: Action) -> Action {
    match action {
        UP => RIGHT,
        RIGHT => DOWN,
        DOWN => LEFT,
        LEFT => UP,
        _ => action,
    }
}

impl Environment for FrozenLake {
    fn reset(&mut self) -> State {
        self.agent = (0, 0);
        self.observe()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        let actual = self.effective_action(action);
        self.agent = self.apply_move(actual);

        let (r, c) = self.agent;
        let (reward, done) = match self.grid[r][c] {
            Tile::Hole => (-1.0, true),
            _ if self.agent == self.target => (1.0, true),
            _ => (-0.01, false),
        };

        (self.observe(), reward, done)
    }

    fn action_space(&self) -> Vec<Action> {
        vec![UP, DOWN, LEFT, RIGHT]
    }

    fn state_shape(&self) -> Vec<usize> {
        // [agent_row, agent_col, target_row, target_col, tile_type]
        vec![5]
    }
}

impl Default for FrozenLake {
    fn default() -> Self {
        Self::new()
    }
}
