use nalgebra::{DMatrix, SMatrix, SVector};

use crate::{
    dynamics::{DiscreteDynamics, traits::ContinuousDynamics},
    maths::expm,
    types::Real,
};

pub struct ContinuousLinearSystem<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    f: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    g: SMatrix<Real, Y, X>,
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize>
    ContinuousLinearSystem<X, U, Y, Z>
{
    pub fn new(
        f: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        g: SMatrix<Real, Y, X>,
    ) -> Self {
        Self { f, b, h, g }
    }
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize> ContinuousDynamics<X, U, Y, Z>
    for ContinuousLinearSystem<X, U, Y, Z>
{
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X> {
        self.f * x + self.b * u
    }
    fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y> {
        self.g * x
    }
    fn f_jacobian(&self, _x: &SVector<Real, X>, _u: &SVector<Real, U>) -> SMatrix<Real, X, X> {
        self.f
    }
    fn b_jacobian(&self, _x: &SVector<Real, X>, _u: &SVector<Real, U>) -> SMatrix<Real, X, U> {
        self.b
    }
    fn g_jacobian(&self, _x: &SVector<Real, X>) -> SMatrix<Real, Y, X> {
        self.g
    }
    fn h_matrix(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
}

#[derive(Clone, Copy)]
pub struct DiscreteLinearSystem<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    f: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    g: SMatrix<Real, Y, X>,
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize>
    DiscreteLinearSystem<X, U, Y, Z>
{
    pub fn new(
        f: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        g: SMatrix<Real, Y, X>,
    ) -> Self {
        Self { f, b, h, g }
    }

    pub fn from_expm<D>(model: &D, dt: Real) -> Self
    where D: ContinuousDynamics<X, U, Y, Z>
    {
        let mut m = DMatrix::<Real>::zeros(X + U, X + U);
        m.view_mut((0, 0), (X, X)).copy_from(&(model.f_jacobian(&SVector::zeros(), &SVector::zeros()) * dt));
        m.view_mut((0, X), (X, U)).copy_from(&(model.b_jacobian(&SVector::zeros(), &SVector::zeros()) * dt));

        let m_exp = expm(&m);

        let mut f = SMatrix::<Real, X, X>::zeros();
        let mut b = SMatrix::<Real, X, U>::zeros();
        f.copy_from(&m_exp.view((0, 0), (X, X)));
        b.copy_from(&m_exp.view((0, X), (X, U)));

        Self {
            f,
            b,
            h: *model.h_matrix(),
            g: model.g_jacobian(&SVector::zeros()),
        }
    }
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize> DiscreteDynamics<X, U, Y, Z>
    for DiscreteLinearSystem<X, U, Y, Z>
{
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X> {
        self.f * x + self.b * u
    }
    fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y> {
        self.g * x
    }
    fn f_jacobian(&self, _x: &SVector<Real, X>, _u: &SVector<Real, U>) -> SMatrix<Real, X, X> {
        self.f
    }
    fn b_jacobian(&self, _x: &SVector<Real, X>, _u: &SVector<Real, U>) -> SMatrix<Real, X, U> {
        self.b
    }
    fn g_jacobian(&self, _x: &SVector<Real, X>) -> SMatrix<Real, Y, X> {
        self.g
    }
    fn h_matrix(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
}
