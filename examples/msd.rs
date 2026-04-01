use nalgebra::{SMatrix, vector};
use ssm_rs::dynamics::{ContinuousLinearSystem, DiscreteLinearSystem, Dynamics};
use ssm_rs::types::Real;
// use ssm_rs::noise::{Noise, Noiseless};

fn main() {
    let continuous_dynamics = ContinuousLinearSystem::new(
        SMatrix::<Real, 2, 2>::from_diagonal_element(-1.),
        SMatrix::<Real, 2, 1>::zeros(),
        SMatrix::<Real, 2, 1>::zeros(),
        SMatrix::<Real, 1, 2>::from_diagonal_element(1.),
    );
    let dt = 0.1;
    let dynamics = DiscreteLinearSystem::from_expm(&continuous_dynamics, dt);
    let mut x = vector![1., 0.];
    let u = vector![0.];
    let z = vector![0.];
    println!("{}", x);
    for _ in 0..100 {
        x = dynamics.propagate(&x, &u, &z);
        println!("{}", x);
    }
}
