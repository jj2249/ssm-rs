use nalgebra::SVector;

use super::traits::Controller;
use crate::types::Real;

pub struct Nontroller;

impl<const X: usize, const U: usize> Controller<X, U> for Nontroller {
    fn control_law(&self, _x: &SVector<Real, X>) -> SVector<Real, U> {
        SVector::zeros()
    }
}
