use nalgebra::{DefaultAllocator, Dim};
use nalgebra::allocator::Allocator;
use crate::model::SystemModel;
use crate::types::{Matrix, Output, State};

pub struct LinearSystem<X:Dim, U:Dim, Y:Dim>
where
DefaultAllocator: Allocator<X, X>+Allocator<X, U>+Allocator<Y, X>
{
    pub a: Matrix<X, X>,
    pub b: Matrix<X, U>,
    pub c: Matrix<Y, X>,
}

impl<X:Dim, U:Dim, Y:Dim> LinearSystem<X, U, Y>
where
DefaultAllocator: Allocator<X, X>+Allocator<X, U>+Allocator<Y, X>
{
    pub fn new(a: Matrix<X, X>, b: Matrix<X, U>, c: Matrix<Y, X>) -> Self {
        Self { a, b, c }
    }
}

impl<X: Dim, U: Dim, Y: Dim> SystemModel<X, U, Y>
    for LinearSystem<X, U, Y>
where
DefaultAllocator: Allocator<X, X>+Allocator<X, U>+Allocator<Y, X> + Allocator<X> + Allocator<U> + Allocator<Y>
{
    fn f(&self, x: &State<X>, u: &State<U>) -> State<X> {
        &self.a * x + &self.b * u
    }

    fn g(&self, x: &State<X>) -> Output<Y> {
        &self.c * x
    }
}

// use crate::model::SystemModel;
// use crate::types::{Matrix, Output, State};

// pub struct LinearSystem<const X: usize, const U: usize, const Y: usize> {
//     pub a: Matrix<X, X>,
//     pub b: Matrix<X, U>,
//     pub h: Matrix<Y, X>,
// }

// impl<const X: usize, const U: usize, const Y: usize> LinearSystem<X, U, Y> {
//     pub fn new(a: Matrix<X, X>, b: Matrix<X, U>, h: Matrix<Y, X>) -> Self {
//         Self { a, b, h }
//     }
// }

// impl<const X: usize, const U: usize, const Y: usize> SystemModel<X, U, Y>
//     for LinearSystem<X, U, Y>
// {
//     fn f(&self, x: &State<X>, u: &State<U>) -> State<X> {
//         self.a * x + self.b * u
//     }

//     fn g(&self, x: &State<X>) -> Output<Y> {
//         self.h * x
//     }
// }
