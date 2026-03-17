use nalgebra::{SVector, matrix, vector};
use state_space::filters::{KalmanFilter, KalmanState};
use state_space::maths::r_state;
use state_space::models::{ContinuousLinearSystem, DiscreteLinearSystem};
use state_space::plots::StatePlot;
use state_space::types::Real;

fn mass_spring_damper(m: Real, k: Real, c: Real) -> ContinuousLinearSystem<2, 1, 1, 1> {
    let a = matrix![
        0., 1.;
        -k/m, -c/m;
    ];
    let b = matrix![
        0.;
        1./m;
    ];
    let h = matrix![0.;1.];
    let c = matrix![1., 0.];
    ContinuousLinearSystem::new(a, b, h, c)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt = 0.01;
    let system = mass_spring_damper(0.5, 2.16, 0.8);
    println!(
        "Continuous system is stable: {}",
        system.is_open_loop_stable()
    );
    let dsystem = DiscreteLinearSystem::from_exact(&system, dt);
    println!(
        "Discretised system is stable: {}",
        dsystem.is_open_loop_stable()
    );

    let mut x = vector![1., 0.];
    let u = vector![0.];
    let mut y = dsystem.g(&x);

    let n_iters = (25. / dt) as usize;
    let mut trajectory: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);
    let mut means: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);
    let mut vars: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);

    let mut rng = rand::rng();

    let q = matrix![0.01];
    let r = matrix![1e-2];

    let kf = KalmanFilter::new(&dsystem, q, r);
    let m = vector![1., 0.];
    let p = matrix![1e-2, 0.; 0., 1e-2];
    let mut state = KalmanState::new(m, p);

    for _ in 0..n_iters {
        trajectory.push(x);
        let z = r_state(&mut rng, 0., 0.01);
        x = dsystem.f(&x, &u, &z);
        y = dsystem.g(&x);
        state = kf.predict(&state, &u);
        state = kf.update(&state, &y);
        means.push(*state.m());
        vars.push(state.p().diagonal());
    }
    StatePlot::new("kalman_output.png")
        .add_line("trajectory", &trajectory)
        .add_line("kalman mean", &means)
        .add_confidence_band("2σ bounds", &means, &vars, 2.0)
        .draw()?;
    Ok(())
}
