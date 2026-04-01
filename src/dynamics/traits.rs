use crate::types::Real;
use nalgebra::{SMatrix, SVector};

pub trait Dynamics<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X>;

    fn f_jacobian(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> &SMatrix<Real, X, X>;

    fn h_matrix(&self) -> &SMatrix<Real, X, Z>;

    fn c_matrix(&self) -> &SMatrix<Real, Y, X>;

    fn propagate(
        &self,
        x: &SVector<Real, X>,
        u: &SVector<Real, U>,
        z: &SVector<Real, Z>,
    ) -> SVector<Real, X> {
        self.f(x, u) + self.h_matrix() * z
    }
    fn observe(&self, x: &SVector<Real, X>, v: &SVector<Real, Y>) -> SVector<Real, Y> {
        self.c_matrix() * x + v
    }
}
