use nalgebra::{SMatrix, SVector, matrix, vector};
use smc_rs::filters::{KalmanFilter, KalmanState};
use smc_rs::maths::Gaussian;
use smc_rs::models::{ContinuousLinearSystem, DiscreteLinearSystem};
use smc_rs::plots::StatePlot;
use smc_rs::types::Real;

fn mass_spring_damper(m: Real, k: Real, c: Real, sp: Real, so:Real) -> ContinuousLinearSystem<2, 1, 1, 1> {
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
    ContinuousLinearSystem::new(a, b, h, c, Gaussian(0f64, sp), Gaussian(0f64, so))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let mut rng = rand::rng();
    
    let dt = 0.1;
    let sp = 1.;
    let so = 1e-1;

    let system = mass_spring_damper(0.5, 2.16, 0.8, sp, so);
    let dsystem = DiscreteLinearSystem::from_exact(&system, dt);

    let mut x = vector![1., 0.];
    let u = vector![0.];
    
    let mut y = dsystem.g(&x, &mut rng);

    let n_iters = (25. / dt) as usize;
    let mut observations: Vec<SVector<Real, 1>> = Vec::with_capacity(n_iters);
    let mut trajectory: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);
    let mut means: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);
    let mut vars: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);

    let kf = KalmanFilter::new(matrix![sp.powi(2)]*dt, matrix![so.powi(2)]);
    let p = SMatrix::from_diagonal_element(1.);
    let mut state = KalmanState::new(x.clone(), p);

    for _ in 0..n_iters {
        trajectory.push(x);
        observations.push(y);
        x = dsystem.f(&x, &u, &mut rng);
        y = dsystem.g(&x, &mut rng);
        state = kf.predict(&dsystem, &state, &u);
        state = kf.update(&dsystem, &state, &y);
        means.push(*state.m());
        vars.push(state.p().diagonal());
    }
    let obs_values: Vec<_> = observations.iter().map(|o| o[0]).collect();
    StatePlot::new("kalman_output.svg")
        .add_line("trajectory", &trajectory)
        .add_line("kalman mean", &means)
        .add_confidence_band("2σ bounds", &means, &vars, 2.0)
        .add_markers("observations", 0, &obs_values)
        .draw()?;
    Ok(())
}
