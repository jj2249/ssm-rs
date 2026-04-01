use nalgebra::SVector;
use rand::rngs::ThreadRng;

use crate::types::Real;

pub trait Noise<const X: usize> {
    fn sample(&self, rng: &mut ThreadRng) -> SVector<Real, X>;
}
