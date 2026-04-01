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
