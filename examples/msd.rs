use nalgebra::{SMatrix, SVector, matrix, vector};
use smc_rs::filters::{KalmanFilter, KalmanState};
use smc_rs::maths::Gaussian;
use smc_rs::models::{ContinuousLinearSystem, DiscreteLinearSystem};
use smc_rs::plots::StatePlot;
use smc_rs::types::Real;

fn mass_spring_damper(m: Real, k: Real, c: Real) -> ContinuousLinearSystem<2, 1, 1, 1> {
    let a = matrix![
        0., 1.;
        -k/m, -c/m;
    ];
    let b = matrix![
        0.;
        1./m;
    ];
    let h = matrix![0.; 1.];
    let c = matrix![1., 0.];
    ContinuousLinearSystem::new(a, b, h, c, Gaussian(0f64, 0.01f64), Gaussian(0f64, 0.01f64))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt = 0.01;
    let system = mass_spring_damper(0.5, 2.16, 0.8);

    let dsystem = DiscreteLinearSystem::from_exact(&system, dt);

    let mut rng = rand::rng();

    let mut x = vector![0., 0.];
    let u = vector![0.];
    let mut y = dsystem.g(&x, &mut rng);

    let n_iters = (25. / dt) as usize;
    let mut observations: Vec<SVector<Real, 1>> = Vec::with_capacity(n_iters);
    let mut trajectory: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);
    let mut means: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);
    let mut std_devs: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);

    let kf = KalmanFilter::new(&dsystem, matrix![1e-4]*dt, matrix![1e-4]);
    let p = SMatrix::from_diagonal_element(0.001f64);
    let mut state = KalmanState::new(x.clone(), p);

    for _ in 0..n_iters {
        trajectory.push(x);
        observations.push(y);
        x = dsystem.f(&x, &u, &mut rng);
        y = dsystem.g(&x, &mut rng);
        state = kf.predict(&state, &u);
        state = kf.update(&state, &y);
        means.push(*state.m());
        std_devs.push(state.p().diagonal().map(|v| v.sqrt()));
    }
    StatePlot::new("kalman_output.png")
        .add_line("trajectory", &trajectory)
        .add_line("kalman mean", &means)
        .add_confidence_band("2σ bounds", &means, &std_devs, 2.0)
        .draw()?;
    Ok(())
}
