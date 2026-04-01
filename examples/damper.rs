use nalgebra::{SMatrix, vector};
use ssm_rs::controllers::{Controller, Nontroller};
use ssm_rs::dynamics::{ContinuousLinearSystem, DiscreteLinearSystem, Dynamics};
use ssm_rs::noise::{GaussianNoise, Noise};
use ssm_rs::plots::StatePlot;
use ssm_rs::types::Real;

fn main() {
    let continuous_dynamics = ContinuousLinearSystem::new(
        SMatrix::<Real, 1, 1>::from_diagonal_element(-1.),
        SMatrix::<Real, 1, 1>::zeros(),
        SMatrix::<Real, 1, 1>::identity(),
        SMatrix::<Real, 1, 1>::identity(),
    );
    let dt = 0.1;
    let dynamics = DiscreteLinearSystem::from_expm(&continuous_dynamics, dt);

    let controller = Nontroller;
    let process_noise = GaussianNoise::new(0.1 * dt.sqrt());
    let observation_noise = GaussianNoise::new(0.1);
    let mut rng = rand::rng();

    let mut trajectory = Vec::new();
    let mut observations = Vec::new();

    let mut x = vector![5.];
    let mut y = dynamics.observe(&x, &observation_noise.sample(&mut rng));

    trajectory.push(x);
    observations.push(y);

    for _ in 0..100 {
        let u = controller.control_law(&x);
        x = dynamics.propagate(&x, &u, &process_noise.sample(&mut rng));
        y = dynamics.observe(&x, &observation_noise.sample(&mut rng));
        trajectory.push(x);
        observations.push(y);
    }
    StatePlot::<1, 1>::new("damper.svg")
        .add_line("trajectory", &trajectory)
        .add_markers("observations", &observations)
        .draw()
        .unwrap();
}
