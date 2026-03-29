mod kalman;

pub use kalman::StateEstimate;
use kalman::KalmanFilter;

use nalgebra::{SMatrix, SVector};

use crate::{models::DiscreteLinearSystem, types::Real};

pub enum Filter<const X: usize, const U: usize, const Z: usize, const Y: usize> {
    Kalman(KalmanFilter<X, U, Z, Y>),
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize> Filter<X, U, Z, Y> {
    pub fn kalman(q: SMatrix<Real, Z, Z>, r: SMatrix<Real, Y, Y>) -> Self {
        Self::Kalman(KalmanFilter::new(q, r))
    }

    pub fn predict(&self, model: &DiscreteLinearSystem<X, U, Z, Y>, state: &StateEstimate<X>, u: &SVector<Real, U>) -> StateEstimate<X> {
        match self {
            Self::Kalman(kf) => kf.predict(model, state, u),
        }
    }

    pub fn update(&self, model: &DiscreteLinearSystem<X, U, Z, Y>, state: &StateEstimate<X>, y: &SVector<Real, Y>) -> StateEstimate<X> {
        match self {
            Self::Kalman(kf) => kf.update(model, state, y),
        }
    }
}
