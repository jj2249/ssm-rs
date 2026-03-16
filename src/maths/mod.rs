use nalgebra::{
    DefaultAllocator, Dim, Matrix, OMatrix, RealField, Storage, UniformNorm, allocator::Allocator,
    convert,
};
use simba::scalar::SubsetOf;
use std::cmp;

pub fn expm<T, N, S>(a: &Matrix<T, N, N, S>) -> OMatrix<T, N, N>
where
    T: RealField + SubsetOf<f64>,
    N: Dim,
    S: Storage<T, N, N>,
    DefaultAllocator: Allocator<N, N>,
{
    let tol = T::from_f64(1e-12).unwrap();
    let norm_a = a.apply_norm(&UniformNorm);
    let s_t = (norm_a / T::from_f64(5.4).unwrap()).log2().ceil();
    let s = cmp::max(0, convert::<T, f64>(s_t) as i32);
    let a_scaled: OMatrix<T, N, N> = a / T::from_f64(2f64.powi(s)).unwrap();

    let (nrows, ncols) = a.shape_generic();
    let mut exp_scaled = OMatrix::<T, N, N>::identity_generic(nrows, ncols);
    let mut term = OMatrix::<T, N, N>::identity_generic(nrows, ncols);
    let mut n: usize = 1;
    while term.apply_norm(&UniformNorm) > tol {
        term *= &a_scaled;
        term /= T::from_f64(n as f64).unwrap();
        exp_scaled += &term;
        n += 1;
    }
    for _ in 0..s {
        let temp = exp_scaled.clone();
        exp_scaled *= temp;
    }
    exp_scaled
}
