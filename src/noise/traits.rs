use nalgebra::SVector;
use rand::rngs::ThreadRng;
use rand_distr::Distribution;

use crate::types::Real;

#[derive(Debug, Clone, Copy)]
pub enum PdfError {
    UndefinedPdf,
    NotImplemented
}

pub trait NoiseProcess<const N: usize> {
    fn sample(&self, dt: Real, rng: &mut ThreadRng) -> SVector<Real, N>;
    fn log_pdf(&self, x: SVector<Real, N>, dt: Real) -> Result<Real, PdfError>;
}

pub trait RandomVariable<const N: usize>: Distribution<SVector<Real, N>> {
    fn log_pdf(&self, x: SVector<Real, N>) -> Result<Real, PdfError>;
}
