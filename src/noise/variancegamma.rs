use super::traits::{NoiseProcess, PdfError};
use crate::types::Real;
use nalgebra::{Cholesky, SMatrix, SVector};

use rand::{RngExt, rngs::ThreadRng};
use rand_distr::{Gamma, StandardNormal};

pub struct VarianceGammaNoise<const N: usize> {
    mean: SVector<Real, N>,
    cholesky: SMatrix<Real, N, N>,
    nu: Real,
}

impl<const N: usize> VarianceGammaNoise<N> {
    pub fn new(mean: SVector<Real, N>, covar: SMatrix<Real, N, N>, nu: Real) -> Self {
        let cholesky = Cholesky::new(covar).unwrap().l().into_owned();
        Self { mean, cholesky, nu }
    }
}

impl<const N: usize> NoiseProcess<N> for VarianceGammaNoise<N> {
    fn sample(&self, dt: Real, rng: &mut ThreadRng) -> SVector<Real, N> {
        let g = rng.sample(Gamma::new(dt / self.nu, self.nu).unwrap());
        let z = SVector::from_fn(|_, _| rng.sample(StandardNormal));
        self.mean * g + self.cholesky * z * g.sqrt()
    }

    fn log_pdf(&self, _x: SVector<Real, N>, _dt: Real) -> Result<Real, PdfError> {
        Err(PdfError::NotImplemented)
    }
}
