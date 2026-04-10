use super::traits::{PdfError, RandomVariable};
use crate::types::Real;
use nalgebra::{Cholesky, SMatrix, SVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub struct Gaussian<const N: usize> {
    mean: SVector<Real, N>,
    cholesky: SMatrix<Real, N, N>,
}

impl<const N: usize> Gaussian<N> {
    pub fn new(mean: SVector<Real, N>, covar: SMatrix<Real, N, N>) -> Self {
        let cholesky = Cholesky::new(covar).unwrap().l().into_owned();
        Self { mean, cholesky }
    }
}

impl<const N: usize> Distribution<SVector<Real, N>> for Gaussian<N> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SVector<Real, N> {
        let z: SVector<Real, N> = SVector::from_fn(|_, _| StandardNormal.sample(rng));
        self.mean + self.cholesky * z
    }
}

impl<const N: usize> RandomVariable<N> for Gaussian<N> {
    fn log_pdf(&self, x: SVector<Real, N>) -> Result<Real, PdfError> {
        let diff = x - self.mean;
        let y = self.cholesky.solve_lower_triangular(&diff).unwrap();
        let log_det = 2. * self.cholesky.diagonal().iter().map(|s| s.ln()).sum::<Real>();
        Ok(-0.5 * ((N as Real) * (2.0 * std::f64::consts::PI).ln() + log_det + y.dot(&y)))
    }
}
