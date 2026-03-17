use crate::maths::expm;
use crate::types::Real;
use nalgebra::{DMatrix, SMatrix, SVector};

pub struct ContinuousLinearSystem<const X: usize, const U: usize, const Z: usize, const Y: usize> {
    a: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    c: SMatrix<Real, Y, X>,
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize>
    ContinuousLinearSystem<X, U, Z, Y>
{
    pub fn new(
        a: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        c: SMatrix<Real, Y, X>,
    ) -> Self {
        Self { a, b, h, c }
    }

    pub fn a(&self) -> &SMatrix<Real, X, X> {
        &self.a
    }
    pub fn b(&self) -> &SMatrix<Real, X, U> {
        &self.b
    }
    pub fn h(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
    pub fn c(&self) -> &SMatrix<Real, Y, X> {
        &self.c
    }

    pub fn f(
        &self,
        x: &SVector<Real, X>,
        u: &SVector<Real, U>,
        z: &SVector<Real, Z>,
    ) -> SVector<Real, X> {
        self.a * x + self.b * u + self.h * z
    }

    pub fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y> {
        self.c * x
    }

    pub fn is_open_loop_stable(&self) -> bool {
        let mut ad = DMatrix::<Real>::zeros(X, X);
        ad.copy_from(&self.a);
        ad.complex_eigenvalues().iter().all(|e| e.re < (0. as Real))
    }
}

pub struct DiscreteLinearSystem<const X: usize, const U: usize, const Z: usize, const Y: usize> {
    a: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    c: SMatrix<Real, Y, X>,
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize>
    DiscreteLinearSystem<X, U, Z, Y>
{
    pub fn new(
        a: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        c: SMatrix<Real, Y, X>,
    ) -> Self {
        Self { a, b, h, c }
    }

    pub fn from_euler(model: &ContinuousLinearSystem<X, U, Z, Y>, dt: Real) -> Self {
        let a = SMatrix::<Real, X, X>::identity() + model.a.scale(dt);
        let b = model.b.scale(dt);
        let h = model.h;
        let c = model.c;

        Self { a, b, h, c }
    }

    pub fn from_rk4(model: &ContinuousLinearSystem<X, U, Z, Y>, dt: Real) -> Self {
        let iden = SMatrix::<Real, X, X>::identity();
        let half_dt = dt / (2. as Real);

        // k1
        let ka = model.a;
        let kb = model.b;
        let mut a = ka;
        let mut b = kb;

        // k2
        let ka = model.a * (iden + ka * half_dt);
        let kb = model.b + model.a * kb * half_dt;
        a += ka * (2. as Real);
        b += kb * (2. as Real);

        // k3
        let ka = model.a * (iden + ka * half_dt);
        let kb = model.b + model.a * kb * half_dt;
        a += ka * (2. as Real);
        b += kb * (2. as Real);

        //k4
        let ka = model.a * (iden + ka * dt);
        let kb = model.b + model.a * kb * dt;
        a += ka;
        b += kb;

        Self {
            a: iden + a * (dt / (6. as Real)),
            b: b * (dt / (6. as Real)),
            h: model.h,
            c: model.c,
        }
    }

    pub fn from_exact(model: &ContinuousLinearSystem<X, U, Z, Y>, dt: Real) -> Self {
        let mut m = DMatrix::<Real>::zeros(X + U, X + U);
        m.view_mut((0, 0), (X, X)).copy_from(&(model.a * dt));
        m.view_mut((0, X), (X, U)).copy_from(&(model.b * dt));

        let m_exp = expm(&m);

        let mut a = SMatrix::<Real, X, X>::zeros();
        let mut b = SMatrix::<Real, X, U>::zeros();
        a.copy_from(&m_exp.view((0, 0), (X, X)));
        b.copy_from(&m_exp.view((0, 0), (X, U)));

        Self {
            a,
            b,
            h: model.h,
            c: model.c,
        }
    }

    pub fn a(&self) -> &SMatrix<Real, X, X> {
        &self.a
    }
    pub fn b(&self) -> &SMatrix<Real, X, U> {
        &self.b
    }
    pub fn h(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
    pub fn c(&self) -> &SMatrix<Real, Y, X> {
        &self.c
    }

    pub fn f(
        &self,
        x: &SVector<Real, X>,
        u: &SVector<Real, U>,
        z: &SVector<Real, Z>,
    ) -> SVector<Real, X> {
        self.a * x + self.b * u + self.h * z
    }

    pub fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y> {
        self.c * x
    }

    pub fn is_open_loop_stable(&self) -> bool {
        let mut ad = DMatrix::<Real>::zeros(X, X);
        ad.copy_from(&self.a);
        ad.complex_eigenvalues()
            .iter()
            .all(|&e| e.norm() < (1. as Real))
    }
}
