use nalgebra::SVector;
use rand::rngs::ThreadRng;

use crate::types::Real;

#[derive(Debug, Clone, Copy)]
pub enum PdfError {
    UndefinedPdf,
}

pub trait Noise<const N: usize> {
    fn sample(&self, rng: &mut ThreadRng) -> SVector<Real, N>;
    fn log_pdf(&self, x: SVector<Real, N>) -> Result<Real, PdfError>;
}
