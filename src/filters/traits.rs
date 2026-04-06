use nalgebra::{SMatrix, SVector};

use crate::{
    dynamics::{DiscreteDynamics, DiscreteLinearSystem},
    types::Real,
};

use super::estimate::StateEstimate;

pub trait Filter<
    D: DiscreteDynamics<X, U, Y, Z>,
    const X: usize,
    const U: usize,
    const Y: usize,
    const Z: usize,
>
{
    fn predict(&self, state: &StateEstimate<X>, u: &SVector<Real, U>) -> StateEstimate<X>;
    fn update(&self, state: &StateEstimate<X>, y: &SVector<Real, Y>) -> StateEstimate<X>;
    fn dynamics(&self) -> D;
}

pub struct KalmanFilter<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    dynamics: DiscreteLinearSystem<X, U, Y, Z>,
    q: SMatrix<Real, Z, Z>,
    r: SMatrix<Real, Y, Y>,
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize> KalmanFilter<X, U, Y, Z> {
    pub fn new(
        dynamics: DiscreteLinearSystem<X, U, Y, Z>,
        q: SMatrix<Real, Z, Z>,
        r: SMatrix<Real, Y, Y>,
    ) -> Self {
        Self { dynamics, q, r }
    }
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize>
    Filter<DiscreteLinearSystem<X, U, Y, Z>, X, U, Y, Z> for KalmanFilter<X, U, Y, Z>
{
    fn dynamics(&self) -> DiscreteLinearSystem<X, U, Y, Z> {
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
        let dynamics = self.dynamics();
        // Update
        let e = y - dynamics.c_matrix() * state.m();
        let s = dynamics.c_matrix() * state.p() * dynamics.c_matrix().transpose() + self.r;
        let k = s
            .cholesky()
            .expect("Failed Cholesky decomposition")
            .solve(&(dynamics.c_matrix() * state.p().transpose()))
            .transpose();
        let i = SMatrix::<Real, X, X>::identity();
        StateEstimate::new(
            state.m() + k * e,
            (i - k * dynamics.c_matrix()) * state.p() * (i - k * dynamics.c_matrix()).transpose()
                + k * self.r * k.transpose(),
        )
    }
}
