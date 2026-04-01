use super::traits::Noise;
use crate::types::Real;
use nalgebra::SVector;
use rand::rngs::ThreadRng;
pub struct Noiseless;

impl<const X: usize> Noise<X> for Noiseless {
    fn sample(&self, _rng: &mut ThreadRng) -> SVector<Real, X> {
        SVector::zeros()
    }
}
