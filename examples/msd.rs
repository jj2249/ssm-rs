#![allow(non_snake_case)]
use nalgebra::{U1, U2, matrix, vector};

use state_space::model::{LinearSystem, SystemModel};
use state_space::plots::plot_trajectory;
use state_space::types::{Real, State};

// Linear Mass-Spring-Damper system
fn mass_spring(m: Real, k: Real, c: Real) -> LinearSystem<U2, U1, U1> {
    let A = matrix![
        0., 1.;
        -k / m, -c / m;
    ];
    let B = matrix![
        0.;
        1. / m;
    ];
    let C = matrix![1., 0.];
    LinearSystem::new(A, B, C)
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt: Real = 0.01; // s
    let m: Real = 0.5; // kg
    let k: Real = 4.16; // kg s^-2
    let c: Real = 0.8; // kg s^-1

    let model = mass_spring(m, k, c);
    let mut x= vector![1., 0.];

    let u= vector![0.];

    let steps = (25.0 as Real / dt) as usize;

    let mut state_sequence: Vec<State<U2>> = Vec::new();
    // let mut state_sequence: Vec<State<U2>> = Vec::new();

    for _ in 0..steps {
        state_sequence.push(x);
        x = model.f_rk4(&x, &u, dt);
    }
    plot_trajectory(&state_sequence, "states.png")?;
    Ok(())
}
