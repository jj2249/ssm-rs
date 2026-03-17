use nalgebra::SVector;
use rand_distr::{StandardNormal};
use rand::{RngExt, rngs::ThreadRng};

use crate::types::Real;

pub fn sr_state<const X: usize>(rng: &mut ThreadRng) -> SVector<Real, X> {
    SVector::from_fn(|_,_| rng.sample(StandardNormal))
}

pub fn r_state<const X: usize>(rng: &mut ThreadRng, mu: Real, sigma: Real) -> SVector<Real, X> {
    let z = sr_state(rng);
    z * sigma + SVector::repeat(mu)
}

