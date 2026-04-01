use nalgebra::{DMatrix, SMatrix, SVector};

use crate::{dynamics::Dynamics, maths::expm, types::Real};

pub struct ContinuousLinearSystem<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    a: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    c: SMatrix<Real, Y, X>,
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize>
    ContinuousLinearSystem<X, U, Y, Z>
{
    pub fn new(
        a: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        c: SMatrix<Real, Y, X>,
    ) -> Self {
        Self { a, b, h, c }
    }
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize> Dynamics<X, U, Y, Z>
    for ContinuousLinearSystem<X, U, Y, Z>
{
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X> {
        self.a * x + self.b * u
    }
    fn f_jacobian(&self, _x: &SVector<Real, X>, _u: &SVector<Real, U>) -> &SMatrix<Real, X, X> {
        &self.a
    }
    fn h_matrix(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
    fn c_matrix(&self) -> &SMatrix<Real, Y, X> {
        &self.c
    }
}

pub struct DiscreteLinearSystem<const X: usize, const U: usize, const Y: usize, const Z: usize> {
    a: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    c: SMatrix<Real, Y, X>,
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize>
    DiscreteLinearSystem<X, U, Y, Z>
{
    pub fn new(
        a: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        c: SMatrix<Real, Y, X>,
    ) -> Self {
        Self { a, b, h, c }
    }

    pub fn from_expm(model: &ContinuousLinearSystem<X, U, Y, Z>, dt: Real) -> Self {
        let mut m = DMatrix::<Real>::zeros(X + U, X + U);
        m.view_mut((0, 0), (X, X)).copy_from(&(model.a * dt));
        m.view_mut((0, X), (X, U)).copy_from(&(model.b * dt));

        let m_exp = expm(&m);

        let mut a = SMatrix::<Real, X, X>::zeros();
        let mut b = SMatrix::<Real, X, U>::zeros();
        a.copy_from(&m_exp.view((0, 0), (X, X)));
        b.copy_from(&m_exp.view((0, X), (X, U)));

        Self {
            a,
            b,
            h: model.h,
            c: model.c,
        }
    }
}

impl<const X: usize, const U: usize, const Y: usize, const Z: usize> Dynamics<X, U, Y, Z>
    for DiscreteLinearSystem<X, U, Y, Z>
{
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X> {
        &self.a * x + &self.b * u
    }

    fn f_jacobian(&self, _x: &SVector<Real, X>, _u: &SVector<Real, U>) -> &SMatrix<Real, X, X> {
        &self.a
    }

    fn h_matrix(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
    fn c_matrix(&self) -> &SMatrix<Real, Y, X> {
        &self.c
    }
}
