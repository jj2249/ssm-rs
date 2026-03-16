use nalgebra::{SVector, matrix, vector};
use state_space::model::{ContinuousLinearSystem, DiscreteLinearSystem};
use state_space::plots::plot_trajectory;
use state_space::types::Real;

fn mass_spring_damper(m: Real, k: Real, c: Real) -> ContinuousLinearSystem<2, 1, 1> {
    let a = matrix![
        0., 1.;
        -k/m, -c/m;
    ];
    let b = matrix![
        0.;
        1./m;
    ];
    let c = matrix![1., 0.];
    ContinuousLinearSystem::new(a, b, c)
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

    let n_iters = (25. / dt) as usize;
    let mut trajectory: Vec<SVector<Real, 2>> = Vec::with_capacity(n_iters);

    for _ in 0..n_iters {
        trajectory.push(x);
        x = dsystem.f(&x, &u);
    }
    let _ = plot_trajectory(&trajectory, "states_exact.png");
    Ok(())
}
