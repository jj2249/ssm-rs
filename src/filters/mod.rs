use nalgebra::{SMatrix, SVector};

use crate::types::Real;

#[derive(Clone)]
pub struct StateEstimate<const X: usize> {
    m: SVector<Real, X>,
    p: SMatrix<Real, X, X>,
}

impl<const X: usize> StateEstimate<X> {
    pub fn new(m: SVector<Real, X>, p: SMatrix<Real, X, X>) -> Self {
        Self { m, p }
    }
    pub fn m(&self) -> &SVector<Real, X> {
        &self.m
    }
    pub fn p(&self) -> &SMatrix<Real, X, X> {
        &self.p
    }
}

pub enum Filter<const X: usize, const U: usize, const Z: usize, const Y: usize> {
    None,
    Kalman {
        q: SMatrix<Real, Z, Z>,
        r: SMatrix<Real, Y, Y>,
    },
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize> Filter<X, U, Z, Y> {
    pub(crate) fn predict(
        &self,
        a: &SMatrix<Real, X, X>,
        b: &SMatrix<Real, X, U>,
        h: &SMatrix<Real, X, Z>,
        u: &SVector<Real, U>,
        state: &StateEstimate<X>,
    ) -> StateEstimate<X> {
        match self {
            Filter::None => panic!("predict called on system with no filter"),
            Filter::Kalman { q, .. } => StateEstimate::new(
                a * state.m() + b * u,
                a * state.p() * a.transpose() + h * q * h.transpose(),
            ),
        }
    }

    pub(crate) fn update(
        &self,
        c: &SMatrix<Real, Y, X>,
        y: &SVector<Real, Y>,
        state: &StateEstimate<X>,
    ) -> StateEstimate<X> {
        match self {
            Filter::None => panic!("update called on system with no filter"),
            Filter::Kalman { r, .. } => {
                let e = y - c * state.m();
                let s = c * state.p() * c.transpose() + r;
                let k = s
                    .cholesky()
                    .unwrap()
                    .solve(&(c * state.p().transpose()))
                    .transpose();
                let i = SMatrix::<Real, X, X>::identity();
                StateEstimate::new(
                    state.m() + k * e,
                    (i - k * c) * state.p() * (i - k * c).transpose() + k * r * k.transpose(),
                )
            }
        }
    }
}
