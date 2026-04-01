use super::traits::Noise;
use crate::types::Real;
use nalgebra::SVector;

use rand::{RngExt, rngs::ThreadRng};
use rand_distr::StandardNormal;

pub struct WhiteNoise {
    sigma: Real,
}

impl WhiteNoise {
    pub fn new(sigma: Real) -> Self {
        Self { sigma }
    }
}

impl<const X: usize> Noise<X> for WhiteNoise {
    fn sample(&self, rng: &mut ThreadRng) -> SVector<Real, X> {
        let z = SVector::from_fn(|_, _| rng.sample(StandardNormal));
        z * self.sigma
    }
}
