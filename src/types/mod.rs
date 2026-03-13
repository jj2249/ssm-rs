use nalgebra::{OMatrix, OVector};

pub type Real = f64;

pub type State<N> = OVector<Real, N>;
pub type Input<N> = OVector<Real, N>;
pub type Output<N> = OVector<Real, N>;
pub type Matrix<N,M> = OMatrix<Real, N, M>;
