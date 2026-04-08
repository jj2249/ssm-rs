use nalgebra::{SMatrix, matrix, vector};

use ssm_rs::controllers::{Controller, Nontroller};
use ssm_rs::dynamics::{
    ContinuousDynamics, ContinuousLinearSystem, DiscreteDynamics, DiscreteLinearSystem,
};
use ssm_rs::filters::{Filter, KalmanFilter, StateEstimate};
use ssm_rs::noise::{Noise, WhiteNoise};
use ssm_rs::plots::StatePlot;
use ssm_rs::types::Real;

fn main() {
    let m = 1.;
    let c = -1.;
    let continuous_dynamics = ContinuousLinearSystem::new(
        matrix![
        0., 1.;
        0., c/m;
        ],
        matrix![0.;0.],
        matrix![0.;1./m;],
        matrix![1., 0.],
    );
    let dt = 0.01;
    let dynamics = DiscreteLinearSystem::from_expm(&continuous_dynamics, dt);

    let controller = Nontroller;
    let sp = 1.;
    let so = 0.1;

    let process_noise = WhiteNoise::new(vector![0.], matrix![sp * sp * dt]);
    let observation_noise = WhiteNoise::new(vector![0.], matrix![so * so]);
    let mut rng = rand::rng();

    let mut x = vector![0., 0.];
    let mut trajectory = vec![x];
    let mut observations = vec![dynamics.observe(&x, &observation_noise.sample(&mut rng))];

    let filter = KalmanFilter::new(dynamics, matrix![sp * sp * dt], matrix![so * so]);
    let mut state = StateEstimate::new(x, SMatrix::<Real, 2, 2>::identity());

    let mut states = vec![state.clone()];

    let u = controller.control_law(&x);

    let t = 2.;
    let n = (t / dt) as usize;

    for _ in 0..n {
        x = continuous_dynamics.step_rk4(&x, &u, &process_noise.sample(&mut rng), dt);
        trajectory.push(x);

        state = filter.predict(&state, &u);

        let y = continuous_dynamics.observe(&x, &observation_noise.sample(&mut rng));
        observations.push(y);

        state = filter.update(&state, &y);
        states.push(state);
    }
    let means: Vec<_> = states.iter().map(|s| s.m().clone()).collect();
    let vars: Vec<_> = states.iter().map(|s| s.p().clone().diagonal()).collect();
    StatePlot::<2, 1>::new("langevin.svg")
        .add_markers("obs", &observations)
        .add_line("trajectory", &trajectory)
        .add_confidence_band("var", &means, &vars, 2.)
        .add_line("mean", &means)
        .draw()
        .unwrap();
}
