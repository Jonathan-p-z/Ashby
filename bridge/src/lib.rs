use ashby_core::Environment;
use ashby_games::{FrozenLake, GridWorld, MazeWorld, PredatorGrid, ResourceHunter};
use pyo3::prelude::*;

#[pyclass]
struct PyGridWorld {
    inner: GridWorld,
}
#[pymethods]
impl PyGridWorld {
    #[new]
    fn new() -> Self { PyGridWorld { inner: GridWorld::new() } }
    fn reset(&mut self) -> Vec<f32> { self.inner.reset() }
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { self.inner.step(action) }
    fn action_space(&self) -> Vec<usize> { self.inner.action_space() }
    fn state_shape(&self) -> Vec<usize> { self.inner.state_shape() }
}

#[pyclass]
struct PyFrozenLake {
    inner: FrozenLake,
}
#[pymethods]
impl PyFrozenLake {
    #[new]
    fn new() -> Self { PyFrozenLake { inner: FrozenLake::new() } }
    fn reset(&mut self) -> Vec<f32> { self.inner.reset() }
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { self.inner.step(action) }
    fn action_space(&self) -> Vec<usize> { self.inner.action_space() }
    fn state_shape(&self) -> Vec<usize> { self.inner.state_shape() }
}

#[pyclass]
struct PyPredatorGrid {
    inner: PredatorGrid,
}
#[pymethods]
impl PyPredatorGrid {
    #[new]
    fn new() -> Self { PyPredatorGrid { inner: PredatorGrid::new() } }
    fn reset(&mut self) -> Vec<f32> { self.inner.reset() }
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { self.inner.step(action) }
    fn action_space(&self) -> Vec<usize> { self.inner.action_space() }
    fn state_shape(&self) -> Vec<usize> { self.inner.state_shape() }
}

#[pyclass]
struct PyMazeWorld {
    inner: MazeWorld,
}
#[pymethods]
impl PyMazeWorld {
    #[new]
    fn new() -> Self { PyMazeWorld { inner: MazeWorld::new() } }
    fn reset(&mut self) -> Vec<f32> { self.inner.reset() }
    /// State is 53-dimensional: [agent_pos, target_pos, grid_flattened(49)].
    /// The maze layout changes on every reset — state includes the full map.
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { self.inner.step(action) }
    fn action_space(&self) -> Vec<usize> { self.inner.action_space() }
    fn state_shape(&self) -> Vec<usize> { self.inner.state_shape() }
}

#[pyclass]
struct PyResourceHunter {
    inner: ResourceHunter,
}
#[pymethods]
impl PyResourceHunter {
    #[new]
    fn new() -> Self { PyResourceHunter { inner: ResourceHunter::new() } }
    fn reset(&mut self) -> Vec<f32> { self.inner.reset() }
    /// State is 18-dimensional: [agent_pos(2), hunger_norm(1), 5×(res_pos(2), collected(1))].
    /// Resource positions re-randomize every reset — DQN must generalize the "eat nearest" policy.
    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) { self.inner.step(action) }
    fn action_space(&self) -> Vec<usize> { self.inner.action_space() }
    fn state_shape(&self) -> Vec<usize> { self.inner.state_shape() }
}

#[pymodule]
fn ashby(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGridWorld>()?;
    m.add_class::<PyFrozenLake>()?;
    m.add_class::<PyPredatorGrid>()?;
    m.add_class::<PyMazeWorld>()?;
    m.add_class::<PyResourceHunter>()?;
    Ok(())
}
