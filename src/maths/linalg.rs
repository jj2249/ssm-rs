use nalgebra::{DefaultAllocator, Dim, OMatrix, UniformNorm, allocator::Allocator};
use std::cmp;

use crate::types::Real;

pub fn expm<N>(a: &OMatrix<Real, N, N>) -> OMatrix<Real, N, N>
where
    N: Dim,
    DefaultAllocator: Allocator<N, N>,
{
    let tol = 1e-12;
    let norm_a = a.apply_norm(&UniformNorm);
    let s_t = (norm_a / (5.4 as Real)).log2().ceil();
    let s = cmp::max(0, s_t as i32);
    let a_scaled: OMatrix<Real, N, N> = a / (2. as Real).powi(s);

    let (nrows, ncols) = a.shape_generic();
    let mut exp_scaled = OMatrix::<Real, N, N>::identity_generic(nrows, ncols);
    let mut term = OMatrix::<Real, N, N>::identity_generic(nrows, ncols);
    let mut n: usize = 1;
    while term.apply_norm(&UniformNorm) > tol {
        term *= &a_scaled;
        term /= n as Real;
        exp_scaled += &term;
        n += 1;
    }
    for _ in 0..s {
        let temp = exp_scaled.clone();
        exp_scaled *= temp;
    }
    exp_scaled
}
