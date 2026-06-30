use ashby_core::{Action, Environment, Reward, State};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const SIZE: usize = 7;
const N_RESOURCES: usize = 5;
const MAX_HUNGER: u32 = 50;

const UP: usize = 0;
const DOWN: usize = 1;
const LEFT: usize = 2;
const RIGHT: usize = 3;

/// A 7×7 open grid where the agent must collect 5 randomly placed resources
/// before its hunger counter hits zero.
///
/// This environment introduces dynamic priorities: the optimal next target
/// changes as resources are collected, and the hunger pressure forces the agent
/// to trade off exploration radius against survival. A pure memorizer fails
/// because resource positions re-randomize every episode — the agent must learn
/// a generalizable "go eat the closest thing" policy under urgency.
pub struct ResourceHunter {
    agent: (usize, usize),
    resources: [(usize, usize, bool); N_RESOURCES], // (row, col, collected)
    hunger: u32,
    rng: StdRng,
}

impl ResourceHunter {
    pub fn new() -> Self {
        let rng = StdRng::from_entropy();
        let mut env = ResourceHunter {
            agent: (0, 0),
            resources: [(0, 0, false); N_RESOURCES],
            hunger: MAX_HUNGER,
            rng,
        };
        env.place_entities();
        env
    }

    fn place_entities(&mut self) {
        self.hunger = MAX_HUNGER;
        self.agent = (
            self.rng.gen_range(0..SIZE),
            self.rng.gen_range(0..SIZE),
        );

        // Collect positions already taken before the resource placement loop
        // to avoid the borrow conflict between self.rng and self.agent.
        let mut taken: Vec<(usize, usize)> = vec![self.agent];

        let mut placed = 0;
        while placed < N_RESOURCES {
            let r = self.rng.gen_range(0..SIZE);
            let c = self.rng.gen_range(0..SIZE);
            if !taken.contains(&(r, c)) {
                self.resources[placed] = (r, c, false);
                taken.push((r, c));
                placed += 1;
            }
        }
    }

    // Manhattan distance to the nearest uncollected resource.
    // Returns 0 if all collected (shouldn't happen mid-episode, but safe fallback).
    fn nearest_dist(&self) -> usize {
        let (ar, ac) = self.agent;
        self.resources
            .iter()
            .filter(|r| !r.2)
            .map(|r| ar.abs_diff(r.0) + ac.abs_diff(r.1))
            .min()
            .unwrap_or(0)
    }

    fn observe(&self) -> State {
        let scale = (SIZE - 1) as f32;
        let (ar, ac) = self.agent;
        let mut state = vec![
            ar as f32 / scale,
            ac as f32 / scale,
            self.hunger as f32 / MAX_HUNGER as f32,
        ];
        for &(rr, rc, collected) in &self.resources {
            state.push(rr as f32 / scale);
            state.push(rc as f32 / scale);
            state.push(if collected { 1.0 } else { 0.0 });
        }
        state
    }
}

impl Environment for ResourceHunter {
    fn reset(&mut self) -> State {
        self.place_entities();
        self.observe()
    }

    fn step(&mut self, action: Action) -> (State, Reward, bool) {
        // Snapshot distance BEFORE the move — used for potential-based shaping below.
        let dist_before = self.nearest_dist();

        let (r, c) = self.agent;
        self.agent = match action {
            UP => (r.saturating_sub(1), c),
            DOWN => ((r + 1).min(SIZE - 1), c),
            LEFT => (r, c.saturating_sub(1)),
            RIGHT => (r, (c + 1).min(SIZE - 1)),
            _ => (r, c),
        };

        self.hunger = self.hunger.saturating_sub(1);

        let mut reward = -0.01;

        // Potential-based shaping: reward proportional to improvement in distance
        // to the nearest uncollected resource, measured BEFORE eating so the
        // reference target doesn't change under our feet on collection steps.
        let dist_after_move = self.nearest_dist();
        reward += 0.05 * (dist_before as f32 - dist_after_move as f32);

        // Eating happens before starvation is checked — stepping onto a resource
        // on the last hunger tick saves the agent rather than killing it.
        let (nr, nc) = self.agent;
        for res in &mut self.resources {
            if !res.2 && (res.0, res.1) == (nr, nc) {
                res.2 = true;
                reward += 0.5;
                self.hunger = MAX_HUNGER;
                break;
            }
        }

        if self.hunger == 0 {
            return (self.observe(), -1.0, true);
        }

        if self.resources.iter().all(|res| res.2) {
            reward += 2.0;
            return (self.observe(), reward, true);
        }

        (self.observe(), reward, false)
    }

    fn action_space(&self) -> Vec<Action> {
        vec![UP, DOWN, LEFT, RIGHT]
    }

    fn state_shape(&self) -> Vec<usize> {
        // [agent_row, agent_col, hunger_norm] + 5×[res_row, res_col, collected] = 18
        vec![3 + N_RESOURCES * 3]
    }
}

impl Default for ResourceHunter {
    fn default() -> Self {
        Self::new()
    }
}
