use nalgebra::{SMatrix, matrix, vector};
use ssm_rs::filters::{Filter, StateEstimate};
use ssm_rs::models::{ContinuousLinearSystem, DiscreteLinearSystem};
use ssm_rs::controllers::Controller;
use ssm_rs::plots::StatePlot;
use ssm_rs::maths::Noise;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();
    let theta = -0.5;
    let sp = 1.;
    let so = 1e-1;
    let dt = 0.1;
    
    let a = matrix![
        0., 1.;
        0., theta;
        ];
    
    let b = matrix![0.; 0.];
    let h = matrix![0.; 1.];
    let c = matrix![1., 0.];

    let system = ContinuousLinearSystem::new(
        a,
        b,
        h,
        c,
        Noise::Gaussian(0., sp),
        Noise::Gaussian(0., so));

    let dsystem = DiscreteLinearSystem::from_expm(
        &system,
        dt,
        Controller::Null,
        Filter::Kalman { q: matrix![sp.powi(2)] * dt, r: matrix![so.powi(2)] }
    );

    let x0 = vector![5., 0.];
    let initial_estimate = StateEstimate::new(vector![0., 0.], SMatrix::from_diagonal_element(1.));
    let n_iters = (25. / dt) as usize;

    let results: Vec<_> = dsystem.run(x0, initial_estimate, &mut rng).take(n_iters).collect();

    StatePlot::new("langevin.svg")
        .add_run(&results)
        .draw()?;

    Ok(())
}