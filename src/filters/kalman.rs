use nalgebra::{SMatrix, SVector};

use crate::{models::DiscreteLinearSystem, types::Real};

pub struct KalmanState<const X: usize> {
    m: SVector<Real, X>,
    p: SMatrix<Real, X, X>,
}
impl<const X: usize> KalmanState<X> {
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

pub struct KalmanFilter<const X: usize, const U: usize, const Z: usize, const Y: usize> {
    q: SMatrix<Real, Z, Z>,
    r: SMatrix<Real, Y, Y>,
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize> KalmanFilter<X, U, Z, Y> {
    pub fn new(
        q: SMatrix<Real, Z, Z>,
        r: SMatrix<Real, Y, Y>,
    ) -> Self {
        Self {
            q,
            r,
        }
    }

    pub fn predict(&self, model:&DiscreteLinearSystem<X, U, Z, Y>, state: &KalmanState<X>, u: &SVector<Real, U>) -> KalmanState<X> {
        KalmanState {
            m: model.a() * state.m + model.b() * u,
            p: model.a() * state.p * model.a().transpose() + model.h() * self.q * model.h().transpose(),
        }
    }

    pub fn update(&self, model:&DiscreteLinearSystem<X, U, Z, Y>, state: &KalmanState<X>, y: &SVector<Real, Y>) -> KalmanState<X> {
        let e = y - model.c() * state.m;
        let s = model.c() * state.p * model.c().transpose() + self.r;
        let k = s
            .cholesky()
            .unwrap()
            .solve(&(model.c() * state.p.transpose()))
            .transpose();

        let i = SMatrix::<Real, X, X>::identity();

        KalmanState {
            m: state.m + k * e,
            p: (i - k * model.c()) * state.p * (i - k * model.c()).transpose()
                + k * self.r * k.transpose(),
        }
    }
}
