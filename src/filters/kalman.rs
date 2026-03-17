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
    a: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    c: SMatrix<Real, Y, X>,

    q: SMatrix<Real, Z, Z>,
    r: SMatrix<Real, Y, Y>,
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize> KalmanFilter<X, U, Z, Y> {
    pub fn new(
        model: &DiscreteLinearSystem<X, U, Z, Y>,
        q: SMatrix<Real, Z, Z>,
        r: SMatrix<Real, Y, Y>,
    ) -> Self {
        Self {
            a: *model.a(),
            b: *model.b(),
            h: *model.h(),
            c: *model.c(),
            q,
            r,
        }
    }

    pub fn predict(&self, state: &KalmanState<X>, u: &SVector<Real, U>) -> KalmanState<X> {
        KalmanState {
            m: self.a * state.m + self.b * u,
            p: self.a * state.p * self.a.transpose() + self.h * self.q * self.h.transpose(),
        }
    }

    pub fn update(&self, state: &KalmanState<X>, y: &SVector<Real, Y>) -> KalmanState<X> {
        let e = y - self.c * state.m;
        let s = self.c * state.p * self.c.transpose() + self.r;
        let k = s
            .cholesky()
            .unwrap()
            .solve(&(self.c * state.p.transpose()))
            .transpose();

        let i = SMatrix::<Real, X, X>::identity();

        KalmanState {
            m: state.m + k * e,
            p: (i - k * self.c) * state.p * (i - k * self.c).transpose()
                + k * self.r * k.transpose(),
        }
    }
}
