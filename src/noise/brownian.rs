use super::traits::{NoiseProcess, PdfError};
use crate::types::Real;
use nalgebra::{Cholesky, SMatrix, SVector};

use rand::{RngExt, rngs::ThreadRng};
use rand_distr::StandardNormal;

pub struct BrownianNoise<const N: usize> {
    mean: SVector<Real, N>,
    cholesky: SMatrix<Real, N, N>,
}

impl<const N: usize> BrownianNoise<N> {
    pub fn new(mean: SVector<Real, N>, covar: SMatrix<Real, N, N>) -> Self {
        let cholesky = Cholesky::new(covar).unwrap().l().into_owned();
        Self { mean, cholesky }
    }
}

impl<const N: usize> NoiseProcess<N> for BrownianNoise<N> {
    fn sample(&self, dt: Real, rng: &mut ThreadRng) -> SVector<Real, N> {
        let z = SVector::from_fn(|_, _| rng.sample(StandardNormal));
        self.mean * dt + self.cholesky * z * dt.sqrt()
    }

    fn log_pdf(&self, x: SVector<Real, N>, dt: Real) -> Result<Real, PdfError> {
        let diff = x - self.mean * dt;
        // Cholesky of (Σ·dt) is √dt · L, so (√dt · L)⁻¹ · diff = L⁻¹ · diff / √dt
        let y = self.cholesky.solve_lower_triangular(&diff).unwrap() / dt.sqrt();
        let log_det = 2.
            * self
                .cholesky
                .diagonal()
                .iter()
                .map(|s| s.ln())
                .sum::<Real>()
            + (N as Real) * dt.ln();
        Ok(-0.5 * ((N as Real) * (2.0 * std::f64::consts::PI).ln() + log_det + y.dot(&y)))
    }
}
