mod linear;
mod nonlinear;
mod traits;

pub use linear::{ContinuousLinearSystem, DiscreteLinearSystem};
pub use nonlinear::DiscretisedNonlinear;
pub use traits::{
    ContinuousDynamics, DifferentiableContinuousDynamics, DifferentiableDiscreteDynamics,
    DiscreteDynamics,
};
