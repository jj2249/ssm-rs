use nalgebra::{SMatrix, matrix, vector};
use smc_rs::filters::{Filter, StateEstimate};
use smc_rs::maths::Noise;
use smc_rs::controllers::Controller;
use smc_rs::models::{ContinuousLinearSystem, DiscreteLinearSystem};
use smc_rs::plots::StatePlot;
use smc_rs::types::Real;

fn mass_spring_damper(m: Real, k: Real, c: Real, sp: Real, so: Real) -> ContinuousLinearSystem<2, 1, 1, 1> {
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
    ContinuousLinearSystem::new(a, b, h, c, Noise::Gaussian(0f64, sp), Noise::Gaussian(0f64, so))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let mut rng = rand::rng();

    let dt = 0.1;
    let sp = 1.;
    let so = 1e-1;

    let system = mass_spring_damper(0.5, 2.16, 0.8, sp, so);
    let dsystem = DiscreteLinearSystem::from_expm(
        &system,
        dt,
        Controller::Null,
        Filter::Kalman { q: matrix![sp.powi(2)] * dt, r: matrix![so.powi(2)] },
    );

    let x0 = vector![5., 0.];
    let initial_estimate = StateEstimate::new(vector![0., 0.], SMatrix::from_diagonal_element(1.));
    let n_iters = (25. / dt) as usize;

    let results: Vec<_> = dsystem.run(x0, initial_estimate, &mut rng).take(n_iters).collect();

    StatePlot::new("kalman_output.svg")
        .add_run(&results)
        .draw()?;
    Ok(())
}
