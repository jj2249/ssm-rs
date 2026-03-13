use nalgebra::{DefaultAllocator, Dim, allocator::Allocator};

use crate::types::{Input, Output, Real, State};

mod linear_system;
pub use linear_system::LinearSystem;

pub trait SystemModel<X:Dim, U:Dim, Y:Dim>
where
DefaultAllocator: Allocator<X, X> + Allocator<X, U> + Allocator<X> + Allocator<U> + Allocator<Y>
{
    fn f(&self, x: &State<X>, u: &Input<U>) -> State<X>;
    fn g(&self, x: &State<X>) -> Output<Y> ;

    fn f_rk4(&self, x: &State<X>, u: &Input<U>, dt: Real) -> State<X> {
        
        let k1 = self.f(x, u);

        let x1 = x + (dt * 0.5) * &k1;
        let k2 = self.f(&x1, u);

        let x2 = x + (dt * 0.5) * &k2;
        let k3 = self.f(&x2, u);

        let x3 = x + dt * &k3;
        let k4 = self.f(&x3, u);

        x + (dt / 6.) * ((&k1) + 2. * ((&k2) + (&k3)) + (&k4))
    }

    fn f_euler_fwd(&self, x: &State<X>, u: &Input<U>, dt: Real) -> State<X> {
        x + dt * &self.f(x, u)
    }
}