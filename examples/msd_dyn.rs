#![allow(non_snake_case)]
use nalgebra::{Dyn, dvector, dmatrix};

use state_space::model::{LinearSystem, SystemModel};
use state_space::plots::plot_trajectory;
use state_space::types::{Real, State};

// Linear Mass-Spring-Damper system
fn mass_spring(m: Real, k: Real, c: Real) -> LinearSystem<Dyn, Dyn, Dyn> {
    let A = dmatrix![
        0., 1.;
        -k / m, -c / m;
    ];
    let B = dmatrix![
        0.;
        1. / m;
    ];
    let C = dmatrix![1., 0.];
    LinearSystem::new(A, B, C)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt: Real = 0.01; // s
    let m: Real = 0.5; // kg
    let k: Real = 4.16; // kg s^-2
    let c: Real = 0.8; // kg s^-1

    let model = mass_spring(m, k, c);
    let mut x= dvector![1., 0.];

    let u= dvector![0.];

    let steps = (25.0 as Real / dt) as usize;

    let mut state_sequence: Vec<State<Dyn>> = Vec::new();

    for _ in 0..steps {
        let x_next = model.f_rk4(&x, &u, dt);
        state_sequence.push(x);
        x = x_next;
    }
    plot_trajectory(&state_sequence, "states_dyn.png")?;
    Ok(())
}
