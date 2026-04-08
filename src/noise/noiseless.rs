use super::traits::Noise;
use crate::{noise::traits::PdfError, types::Real};
use nalgebra::SVector;
use rand::rngs::ThreadRng;
pub struct Noiseless;

impl<const N: usize> Noise<N> for Noiseless {
    fn sample(&self, _rng: &mut ThreadRng) -> SVector<Real, N> {
        SVector::zeros()
    }
    fn log_pdf(&self, _x: SVector<Real, N>) -> Result<Real, PdfError> {
        Err(PdfError::UndefinedPdf)
    }
}
