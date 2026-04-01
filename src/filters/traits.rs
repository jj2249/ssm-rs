use nalgebra::{SMatrix, SVector};

use crate::{dynamics::Dynamics, types::Real};

use super::estimate::StateEstimate;

pub trait Filter<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    fn predict<D: Dynamics<X, U, Y, Z>>(
        &self,
        dynamics: &D,
        state: &StateEstimate<X>,
        u: &SVector<Real, U>,
    ) -> StateEstimate<X>;
    fn update<D: Dynamics<X, U, Y, Z>>(
        &self,
        dynamics: &D,
        state: &StateEstimate<X>,
        y: &SVector<Real, Y>,
    ) -> StateEstimate<X>;
}

pub struct KalmanFilter<const Y: usize, const Z: usize> {
    q: SMatrix<Real, Z, Z>,
    r: SMatrix<Real, Y, Y>,
}

impl<const Y: usize, const Z: usize> KalmanFilter<Y, Z> {
    pub fn new(q: SMatrix<Real, Z, Z>, r: SMatrix<Real, Y, Y>) -> Self {
        Self { q, r }
    }
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize> Filter<X, U, Y, Z>
    for KalmanFilter<Y, Z>
{
    fn predict<D: Dynamics<X, U, Y, Z>>(
        &self,
        dynamics: &D,
        state: &StateEstimate<X>,
        u: &SVector<Real, U>,
    ) -> StateEstimate<X> {
        // predict
        let m = dynamics.f(state.m(), u);
        let jacobian = dynamics.f_jacobian(state.m(), u);
        let p = jacobian * state.p() * jacobian.transpose()
            + dynamics.h_matrix() * self.q * dynamics.h_matrix().transpose();
        StateEstimate::new(m, p)
    }
    fn update<D: Dynamics<X, U, Y, Z>>(
        &self,
        dynamics: &D,
        state: &StateEstimate<X>,
        y: &SVector<Real, Y>,
    ) -> StateEstimate<X> {
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
