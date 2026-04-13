use nalgebra::{DMatrix, SMatrix, SVector};

use crate::{
    dynamics::traits::{
        ContinuousDynamics, DifferentiableContinuousDynamics, DifferentiableDiscreteDynamics,
        DiscreteDynamics,
    },
    maths::expm,
    types::Real,
};

pub struct DiscretisedNonlinear<C, const X: usize, const U: usize, const Y: usize, const Z: usize>
{
    dynamics: C,
    dt: Real,
}

impl<C, const X: usize, const U: usize, const Y: usize, const Z: usize>
    DiscretisedNonlinear<C, X, U, Y, Z>
{
    pub fn new(dynamics: C, dt: Real) -> Self {
        Self { dynamics, dt }
    }
}

impl<C, const X: usize, const U: usize, const Y: usize, const Z: usize> DiscreteDynamics<X, U, Y, Z>
    for DiscretisedNonlinear<C, X, U, Y, Z>
where
    C: ContinuousDynamics<X, U, Y, Z>,
{
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X> {
        self.dynamics.propagate_rk4(x, u, self.dt)
    }

    fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y> {
        self.dynamics.g(x)
    }

    fn h_matrix(&self) -> &SMatrix<Real, X, Z> {
        self.dynamics.h_matrix()
    }
}

impl<C, const X: usize, const U: usize, const Y: usize, const Z: usize>
    DifferentiableDiscreteDynamics<X, U, Y, Z> for DiscretisedNonlinear<C, X, U, Y, Z>
where
    C: DifferentiableContinuousDynamics<X, U, Y, Z>,
{
    fn f_jacobian(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SMatrix<Real, X, X> {
        self.fb_jacobians(x, u).0
    }

    fn b_jacobian(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SMatrix<Real, X, U> {
        self.fb_jacobians(x, u).1
    }

    fn g_jacobian(&self, x: &SVector<Real, X>) -> SMatrix<Real, Y, X> {
        self.dynamics.g_jacobian(x)
    }

    fn fb_jacobians(
        &self,
        x: &SVector<Real, X>,
        u: &SVector<Real, U>,
    ) -> (SMatrix<Real, X, X>, SMatrix<Real, X, U>) {
        let a = self.dynamics.f_jacobian(x, u);
        let b = self.dynamics.b_jacobian(x, u);

        let mut m = DMatrix::<Real>::zeros(X + U, X + U);
        m.view_mut((0, 0), (X, X)).copy_from(&(a * self.dt));
        m.view_mut((0, X), (X, U)).copy_from(&(b * self.dt));

        let m_exp = expm(&m);

        let mut f_d = SMatrix::<Real, X, X>::zeros();
        let mut b_d = SMatrix::<Real, X, U>::zeros();
        f_d.copy_from(&m_exp.view((0, 0), (X, X)));
        b_d.copy_from(&m_exp.view((0, X), (X, U)));

        (f_d, b_d)
    }
}
