use super::traits::{NoiseProcess, PdfError};
use crate::types::Real;
use nalgebra::SVector;
use rand::rngs::ThreadRng;

pub struct Noiseless;

impl<const N: usize> NoiseProcess<N> for Noiseless {
    fn sample(&self, _dt: Real, _rng: &mut ThreadRng) -> SVector<Real, N> {
        SVector::zeros()
    }

    fn log_pdf(&self, _x: SVector<Real, N>, _dt: Real) -> Result<Real, PdfError> {
        Err(PdfError::UndefinedPdf)
    }
}
