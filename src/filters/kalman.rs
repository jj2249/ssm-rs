use nalgebra::{SMatrix, SVector};

use crate::{
    dynamics::{DifferentiableDiscreteDynamics, DiscreteDynamics},
    filters::{Filter, StateEstimate},
    types::Real,
};

pub struct KalmanFilter<'d, D, const X: usize, const U: usize, const Y: usize, const Z: usize>
where
    D: DiscreteDynamics<X, U, Y, Z> + DifferentiableDiscreteDynamics<X, U, Y, Z>,
{
    dynamics: &'d D,
    q: SMatrix<Real, Z, Z>,
    r: SMatrix<Real, Y, Y>,
}

impl<'d, D, const X: usize, const U: usize, const Y: usize, const Z: usize>
    KalmanFilter<'d, D, X, U, Y, Z>
where
    D: DiscreteDynamics<X, U, Y, Z> + DifferentiableDiscreteDynamics<X, U, Y, Z>,
{
    pub fn new(dynamics: &'d D, q: SMatrix<Real, Z, Z>, r: SMatrix<Real, Y, Y>) -> Self {
        Self { dynamics, q, r }
    }
}

impl<'d, D, const X: usize, const U: usize, const Y: usize, const Z: usize>
    Filter<'d, D, X, U, Y, Z> for KalmanFilter<'d, D, X, U, Y, Z>
where
    D: DiscreteDynamics<X, U, Y, Z> + DifferentiableDiscreteDynamics<X, U, Y, Z>,
{
    fn dynamics(&self) -> &'d D {
        self.dynamics
    }
    fn predict(&self, state: &StateEstimate<X>, u: &SVector<Real, U>) -> StateEstimate<X> {
        let dynamics = self.dynamics();
        // predict
        let m = dynamics.f(state.m(), u);
        let jacobian = dynamics.f_jacobian(state.m(), u);
        let p = jacobian * state.p() * jacobian.transpose()
            + dynamics.h_matrix() * self.q * dynamics.h_matrix().transpose();
        StateEstimate::new(m, p)
    }
    fn update(&self, state: &StateEstimate<X>, y: &SVector<Real, Y>) -> StateEstimate<X> {
        let jacobian = self.dynamics().g_jacobian(state.m());
        // Update
        let e = y - jacobian * state.m();
        let s = jacobian * state.p() * jacobian.transpose() + self.r;
        let k = s
            .cholesky()
            .expect("Failed Cholesky decomposition")
            .solve(&(jacobian * state.p().transpose()))
            .transpose();
        let i = SMatrix::<Real, X, X>::identity();
        StateEstimate::new(
            state.m() + k * e,
            (i - k * jacobian) * state.p() * (i - k * jacobian).transpose()
                + k * self.r * k.transpose(),
        )
    }
}
