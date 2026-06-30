/// A snapshot of what the agent can observe at a given moment.
/// Represented as a flat vector of floats — keeps the bridge to Python trivial.
pub type State = Vec<f32>;

/// An integer index into the environment's action space.
/// Concrete environments map these indices to meaningful moves.
pub type Action = usize;

/// Scalar signal the environment sends back after each transition.
pub type Reward = f32;
