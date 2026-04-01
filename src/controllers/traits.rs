use nalgebra::SVector;

use crate::types::Real;

pub trait Controller<const X: usize, const U: usize> {
    fn control_law(&self, x: &SVector<Real, X>) -> SVector<Real, U>;
}
