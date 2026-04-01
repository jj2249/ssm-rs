use nalgebra::{SMatrix, SVector};

use crate::{dynamics::Dynamics, types::Real};

use super::estimate::StateEstimate;

pub trait Filter<D: Dynamics<X, U, Y, Z>, const X: usize, const U: usize, const Y: usize, const Z: usize> {
    fn predict(
        &self,
        state: &StateEstimate<X>,
        u: &SVector<Real, U>,
    ) -> StateEstimate<X>;
    fn update(
        &self,
        state: &StateEstimate<X>,
        y: &SVector<Real, Y>,
    ) -> StateEstimate<X>;
    fn dynamics(&self) -> D;
}

pub struct KalmanFilter<D, const X: usize, const U:usize, const Y: usize, const Z: usize>
where D :Dynamics<X, U, Y, Z> + Copy,
{
    dynamics: D,
    q: SMatrix<Real, Z, Z>,
    r: SMatrix<Real, Y, Y>,
}

impl<D, const X: usize, const U:usize, const Y: usize, const Z: usize> KalmanFilter<D, X, U, Y, Z>
where D :Dynamics<X, U, Y, Z> + Copy,
{
    pub fn new(dynamics: D, q: SMatrix<Real, Z, Z>, r: SMatrix<Real, Y, Y>) -> Self {
        Self { dynamics, q, r }
    }
}

impl<D,const X: usize, const U: usize, const Y: usize, const Z: usize> Filter<D, X, U, Y, Z>
    for KalmanFilter<D, X, U, Y, Z>
where D :Dynamics<X, U, Y, Z> + Copy,
{
    fn dynamics(&self) -> D {
        self.dynamics
    }
    fn predict(
        &self,
        state: &StateEstimate<X>,
        u: &SVector<Real, U>,
    ) -> StateEstimate<X> {
        let dynamics = self.dynamics();
        // predict
        let m = dynamics.f(state.m(), u);
        let jacobian = dynamics.f_jacobian(state.m(), u);
        let p = jacobian * state.p() * jacobian.transpose()
            + dynamics.h_matrix() * self.q * dynamics.h_matrix().transpose();
        StateEstimate::new(m, p)
    }
    fn update(
        &self,
        state: &StateEstimate<X>,
        y: &SVector<Real, Y>,
    ) -> StateEstimate<X> {
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
