use nalgebra::SVector;

use crate::types::Real;

pub enum Controller<const U: usize> {
    Null
}

impl<const U:usize> Controller<U> {
    pub fn control_law(&self) -> SVector<Real, U> {
        match self {
            Self::Null => SVector::zeros()
        }
    }
}