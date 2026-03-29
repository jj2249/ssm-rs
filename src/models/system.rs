use crate::controllers::Controller;
use crate::filters::{Filter, StateEstimate};
use crate::maths::expm;
use crate::maths::Noise;
use crate::types::Real;
use nalgebra::{DMatrix, SMatrix, SVector};
use rand::rngs::ThreadRng;

pub struct ContinuousLinearSystem<const X: usize, const U: usize, const Z: usize, const Y: usize> {
    a: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    c: SMatrix<Real, Y, X>,
    process_noise: Noise<Z>,
    observation_noise: Noise<Y>,
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize>
    ContinuousLinearSystem<X, U, Z, Y>
{
    pub fn new(
        a: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        c: SMatrix<Real, Y, X>,
        process_noise: Noise<Z>,
        observation_noise: Noise<Y>,
    ) -> Self {
        Self { a, b, h, c, process_noise, observation_noise}
    }

    pub fn is_open_loop_stable(&self) -> bool {
        let mut ad = DMatrix::<Real>::zeros(X, X);
        ad.copy_from(&self.a);
        ad.complex_eigenvalues().iter().all(|e| e.re < 0.)
    }
}

pub struct DiscreteLinearSystem<const X: usize, const U: usize, const Z: usize, const Y: usize> {
    a: SMatrix<Real, X, X>,
    b: SMatrix<Real, X, U>,
    h: SMatrix<Real, X, Z>,
    c: SMatrix<Real, Y, X>,
    process_noise: Noise<Z>,
    observation_noise: Noise<Y>,
    controller: Controller<U>,
    filter: Filter<X, U, Z, Y>,
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize>
    DiscreteLinearSystem<X, U, Z, Y>
{
    pub fn new(
        a: SMatrix<Real, X, X>,
        b: SMatrix<Real, X, U>,
        h: SMatrix<Real, X, Z>,
        c: SMatrix<Real, Y, X>,
        process_noise: Noise<Z>,
        observation_noise: Noise<Y>,
        controller: Controller<U>,
        filter: Filter<X, U, Z, Y>,
    ) -> Self {
        Self { a, b, h, c, process_noise, observation_noise, controller, filter }
    }

    pub fn from_euler(model: &ContinuousLinearSystem<X, U, Z, Y>, dt: Real, controller: Controller<U>, filter: Filter<X, U, Z, Y>) -> Self {
        Self {
            a: SMatrix::<Real, X, X>::identity() + model.a.scale(dt),
            b: model.b.scale(dt),
            h: model.h,
            c: model.c,
            process_noise: model.process_noise.discretise(dt),
            observation_noise: model.observation_noise,
            controller,
            filter,
        }
    }

    pub fn from_rk4(model: &ContinuousLinearSystem<X, U, Z, Y>, dt: Real, controller: Controller<U>, filter: Filter<X, U, Z, Y>) -> Self {
        let iden = SMatrix::<Real, X, X>::identity();
        let half_dt = dt / 2.;

        let ka = model.a;
        let kb = model.b;
        let mut a = ka;
        let mut b = kb;

        let ka = model.a * (iden + ka * half_dt);
        let kb = model.b + model.a * kb * half_dt;
        a += ka * 2.;
        b += kb * 2.;

        let ka = model.a * (iden + ka * half_dt);
        let kb = model.b + model.a * kb * half_dt;
        a += ka * 2.;
        b += kb * 2.;

        let ka = model.a * (iden + ka * dt);
        let kb = model.b + model.a * kb * dt;
        a += ka;
        b += kb;

        Self {
            a: iden + a * (dt / 6.),
            b: b * (dt / 6.),
            h: model.h,
            c: model.c,
            process_noise: model.process_noise.discretise(dt),
            observation_noise: model.observation_noise,
            controller,
            filter,
        }
    }

    pub fn from_expm(model: &ContinuousLinearSystem<X, U, Z, Y>, dt: Real, controller: Controller<U>, filter: Filter<X, U, Z, Y>) -> Self {
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
            process_noise: model.process_noise.discretise(dt),
            observation_noise: model.observation_noise,
            controller,
            filter,
        }
    }

    pub fn is_open_loop_stable(&self) -> bool {
        let mut ad = DMatrix::<Real>::zeros(X, X);
        ad.copy_from(&self.a);
        ad.complex_eigenvalues().iter().all(|&e| e.norm() < 1.)
    }
}

impl<const X: usize, const U: usize, const Z: usize, const Y: usize> DiscreteLinearSystem<X, U, Z, Y>
{
    pub fn a(&self) -> &SMatrix<Real, X, X> { &self.a }
    pub fn b(&self) -> &SMatrix<Real, X, U> { &self.b }
    pub fn h(&self) -> &SMatrix<Real, X, Z> { &self.h }
    pub fn c(&self) -> &SMatrix<Real, Y, X> { &self.c }

    fn f(&self, x: &SVector<Real, X>, rng: &mut ThreadRng) -> SVector<Real, X> {
        self.a * x + self.b * self.controller.control_law() + self.h * self.process_noise.sample(rng)
    }

    fn g(&self, x: &SVector<Real, X>, rng: &mut ThreadRng) -> SVector<Real, Y> {
        self.c * x + self.observation_noise.sample(rng)
    }

    /// Runs the system, generating observations and filtering them simultaneously.
    /// Yields `(true_state, observation, posterior_estimate)` per step.
    pub fn run<'a>(&'a self, x0: SVector<Real, X>, initial_estimate: StateEstimate<X>, rng: &'a mut ThreadRng) -> RunIter<'a, X, U, Z, Y> {
        RunIter { system: self, x: x0, estimate: initial_estimate, rng }
    }
}

pub struct RunIter<'a, const X: usize, const U: usize, const Z: usize, const Y: usize> {
    system: &'a DiscreteLinearSystem<X, U, Z, Y>,
    x: SVector<Real, X>,
    estimate: StateEstimate<X>,
    rng: &'a mut ThreadRng,
}

impl<'a, const X: usize, const U: usize, const Z: usize, const Y: usize> Iterator
    for RunIter<'a, X, U, Z, Y>
{
    type Item = (SVector<Real, X>, SVector<Real, Y>, StateEstimate<X>);

    fn next(&mut self) -> Option<Self::Item> {
        let y = self.system.g(&self.x, self.rng);
        let u = self.system.controller.control_law();
        let predicted = self.system.filter.predict(&self.system.a, &self.system.b, &self.system.h, &u, &self.estimate);
        self.estimate = self.system.filter.update(&self.system.c, &y, &predicted);
        let item = (self.x, y, self.estimate.clone());
        self.x = self.system.f(&self.x, self.rng);
        Some(item)
    }
}
