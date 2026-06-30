use ashby_core::{Action, Environment, Reward, State};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const SIZE: usize = 7;

const UP: usize = 0;
const DOWN: usize = 1;
const LEFT: usize = 2;
const RIGHT: usize = 3;

/// A 7×7 maze regenerated from scratch on every reset via recursive backtracking.
///
/// This is where tabular Q-learning breaks down: each episode produces a new
/// layout, so the state space is effectively infinite. The Q-table can't
/// memorize solutions — a function approximator that generalizes is the only
/// viable path forward.
pub struct MazeWorld {
    grid: [[bool; SIZE]; SIZE], // true = wall
    agent: (usize, usize),
    target: (usize, usize),
    rng: StdRng,
}

impl MazeWorld {
    pub fn new() -> Self {
        let rng = StdRng::from_entropy();
        let mut maze = MazeWorld {
            grid: [[true; SIZE]; SIZE],
            agent: (1, 1),
            target: (5, 5),
            rng,
        };
        maze.generate_maze();
        maze.pick_positions();
        maze
    }

    /// Recursive backtracking — the classical perfect-maze algorithm.
    ///
    /// Starts from cell (1,1) and carves passages by visiting unvisited
    /// neighbors two steps away, removing the wall between them.
    /// "Perfect" here means no loops and exactly one path between any two cells.
    fn generate_maze(&mut self) {
        self.grid = [[true; SIZE]; SIZE];
        self.carve(1, 1);
    }

    fn carve(&mut self, r: usize, c: usize) {
        self.grid[r][c] = false;

        // Shuffle the four cardinal directions before visiting — this is what
        // makes each generated maze unique rather than always the same spiral.
        let mut dirs: [(isize, isize); 4] = [(-2, 0), (2, 0), (0, -2), (0, 2)];
        for i in (1..4).rev() {
            let j = self.rng.gen_range(0..=i);
            dirs.swap(i, j);
        }
        // rng borrow fully released before any grid access below.

        for (dr, dc) in dirs {
            let nr = r as isize + dr;
            let nc = c as isize + dc;

            // Stay inside the interior (leave the outer border as walls).
            if nr <= 0 || nc <= 0 || nr >= SIZE as isize - 1 || nc >= SIZE as isize - 1 {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;

            if !self.grid[nr][nc] {
                continue; // already visited
            }

            // Open the corridor cell sitting halfway between the two maze cells.
            let wr = (r as isize + dr / 2) as usize;
            let wc = (c as isize + dc / 2) as usize;
            self.grid[wr][wc] = false;

            self.carve(nr, nc);
        }
    }

    fn pick_positions(&mut self) {
        // Collect open cells first so we can release the grid borrow before
        // calling self.rng, which needs a mutable borrow of self.
        let open: Vec<(usize, usize)> = {
            let mut cells = Vec::new();
            for r in 0..SIZE {
                for c in 0..SIZE {
                    if !self.grid[r][c] {
                        cells.push((r, c));
                    }
                }
            }
            cells
        };

        // Pick start and goal at least 5 Manhattan steps apart so the episode
        // is non-trivial. Fall back to first/last open cell if we can't find
        // a valid pair quickly (very rare for 7×7 mazes).
        let mut attempts = 0;
        loop {
            let ai = self.rng.gen_range(0..open.len());
            let ti = self.rng.gen_range(0..open.len());
            let a = open[ai];
            let t = open[ti];
            let dist = a.0.abs_diff(t.0) + a.1.abs_diff(t.1);
            if a != t && dist >= 5 {
                self.agent = a;
                self.target = t;
                return;
            }
            attempts += 1;
            if attempts > 1000 {
                self.agent = open[0];
                self.target = open[open.len() - 1];
                return;
            }
        }
    }

    fn observe(&self) -> State {
        let scale = (SIZE - 1) as f32;
        let (ar, ac) = self.agent;
        let (tr, tc) = self.target;

        let mut state = vec![
            ar as f32 / scale,
            ac as f32 / scale,
            tr as f32 / scale,
            tc as f32 / scale,
        ];

        // Flatten the full grid — the network needs to "see" all walls to plan.
        for r in 0..SIZE {
            for c in 0..SIZE {
                state.push(if self.grid[r][c] { 1.0 } else { 0.0 });
            }
        }
        state
    }
}

impl Environment for MazeWorld {
    fn reset(&mut self) -> State {
        // New layout every episode — this is what makes tabular Q-learning fail
        // and forces the use of a function approximator.
        self.generate_maze();
        self.pick_positions();
        self.observe()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        let (r, c) = self.agent;
        let (nr, nc) = match action {
            UP => (r.saturating_sub(1), c),
            DOWN => ((r + 1).min(SIZE - 1), c),
            LEFT => (r, c.saturating_sub(1)),
            RIGHT => (r, (c + 1).min(SIZE - 1)),
            _ => (r, c),
        };

        if self.grid[nr][nc] {
            // Wall bump — heavy penalty so the agent learns the map fast.
            return (self.observe(), -0.5, false);
        }

        self.agent = (nr, nc);

        if self.agent == self.target {
            return (self.observe(), 1.0, true);
        }

        (self.observe(), -0.01, false)
    }

    fn action_space(&self) -> Vec<Action> {
        vec![UP, DOWN, LEFT, RIGHT]
    }

    fn state_shape(&self) -> Vec<usize> {
        // 4 position values + 49 grid cells flattened = 53 dimensions
        vec![4 + SIZE * SIZE]
    }
}

impl Default for MazeWorld {
    fn default() -> Self {
        Self::new()
    }
}
