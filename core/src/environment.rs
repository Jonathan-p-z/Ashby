use crate::types::{Action, Reward, State};

/// The contract every environment in Ashby must fulfill.
///
/// Inspired by the OpenAI Gym interface but stripped to its essence —
/// everything an agent needs to interact with a world, nothing more.
pub trait Environment {
    /// Wipe the world and return the agent's first observation.
    fn reset(&mut self) -> State;

    /// Apply an action and return what happened:
    /// the new observation, the reward signal, and whether the episode ended.
    fn step(&mut self, action: Action) -> (State, Reward, bool);

    /// All moves the agent can legally attempt in this environment.
    fn action_space(&self) -> Vec<Action>;

    /// Dimensions of the state vector — lets the agent allocate its Q-table
    /// without knowing what's inside the state.
    fn state_shape(&self) -> Vec<usize>;
}
