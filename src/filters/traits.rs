use nalgebra::SVector;

use crate::{dynamics::DiscreteDynamics, types::Real};

use super::estimate::StateEstimate;

pub trait Filter<
    'd,
    D: DiscreteDynamics<X, U, Y, Z>,
    const X: usize,
    const U: usize,
    const Y: usize,
    const Z: usize,
>
{
    fn predict(&self, state: &StateEstimate<X>, u: &SVector<Real, U>) -> StateEstimate<X>;
    fn update(&self, state: &StateEstimate<X>, y: &SVector<Real, Y>) -> StateEstimate<X>;
    fn dynamics(&self) -> &'d D;
}
