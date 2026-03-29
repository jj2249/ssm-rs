use nalgebra::SVector;
use rand::{RngExt, rngs::ThreadRng};
use rand_distr::StandardNormal;

use crate::types::Real;

pub fn sr_state<const X: usize>(rng: &mut ThreadRng) -> SVector<Real, X> {
    SVector::from_fn(|_, _| rng.sample(StandardNormal))
}

pub fn r_state<const X: usize>(rng: &mut ThreadRng, mu: Real, sigma: Real) -> SVector<Real, X> {
    let z = sr_state(rng);
    z * sigma + SVector::repeat(mu)
}

#[derive(Clone, Copy)]
pub enum Noise<const D: usize> {
    Noiseless,
    Gaussian(Real, Real),
}

impl<const D: usize> Noise<D> {
    pub fn sample(&self, rng: &mut ThreadRng) -> SVector<Real, D> {
        match self {
            Self::Noiseless => SVector::zeros(),
            Self::Gaussian(mu, sigma) => r_state(rng, *mu, *sigma),
        }
    }
    pub fn discretise(&self, dt: Real) -> Self {
        match self {
            Self::Noiseless => Self::Noiseless,
            Self::Gaussian(mu, sigma) => Self::Gaussian(*mu, *sigma * dt.sqrt()),
        }
    }
}
