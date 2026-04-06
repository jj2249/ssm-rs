use crate::types::Real;
use nalgebra::{SMatrix, SVector};

pub trait DiscreteDynamics<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X>;

    fn f_jacobian(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SMatrix<Real, X, X>;

    fn h_matrix(&self) -> &SMatrix<Real, X, Z>;

    fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y>;

    fn step(
        &self,
        x: &SVector<Real, X>,
        u: &SVector<Real, U>,
        z: &SVector<Real, Z>,
    ) -> SVector<Real, X> {
        self.f(x, u) + self.h_matrix() * z
    }
    fn observe(&self, x: &SVector<Real, X>, v: &SVector<Real, Y>) -> SVector<Real, Y> {
        self.g(x) + v
    }
}

pub trait ContinuousDynamics<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X>;

    fn f_jacobian(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SMatrix<Real, X, X>;

    fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y>;

    fn h_matrix(&self) -> &SMatrix<Real, X, Z>;

    fn step_rk4(
        &self,
        x: &SVector<Real, X>,
        u: &SVector<Real, U>,
        z: &SVector<Real, Z>,
        dt: Real,
    ) -> SVector<Real, X> {
        let k1 = self.f(x, u);
        let k2 = self.f(&(x + 0.5 * dt * k1), u);
        let k3 = self.f(&(x + 0.5 * dt * k2), u);
        let k4 = self.f(&(x + dt * k3), u);
        x + (dt / 6.) * (k1 + 2. * (k2 + k3) + k4) + self.h_matrix() * z
    }
    fn observe(&self, x: &SVector<Real, X>, v: &SVector<Real, Y>) -> SVector<Real, Y> {
        self.g(x) + v
    }
}
