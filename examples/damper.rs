use nalgebra::{vector, matrix};

use ssm_rs::controllers::{Controller, Nontroller};
use ssm_rs::dynamics::{ContinuousLinearSystem, DiscreteLinearSystem, Dynamics};
use ssm_rs::noise::{WhiteNoise, Noise};
use ssm_rs::filters::{Filter, KalmanFilter, StateEstimate};

use ssm_rs::plots::StatePlot;

fn main() {
    let continuous_dynamics = ContinuousLinearSystem::new(
        matrix![-1.],
        matrix![0.],
        matrix![1.],
        matrix![1.],
    );
    let dt = 0.1;
    let dynamics = DiscreteLinearSystem::from_expm(&continuous_dynamics, dt);

    let controller = Nontroller;
    let sp = 0.5;
    let so = 1.;
    let process_noise = WhiteNoise::new(sp * dt.sqrt());
    let observation_noise = WhiteNoise::new(so);
    let mut rng = rand::rng();
    
    let mut x = vector![5.];
    let mut trajectory = vec![x];
    let mut observations = vec![dynamics.observe(&x, &observation_noise.sample(&mut rng))];

    let filter = KalmanFilter::new(dynamics, matrix![sp*sp*dt], matrix![so*so]);
    let mut state = StateEstimate::new(x, matrix![1.]);

    let mut states = vec![state.clone()];
    
    let u = controller.control_law(&x);

    let t = 10.;
    let n = (t / dt) as usize;

    for _ in 0..n {
        
        x = dynamics.propagate(&x, &u, &process_noise.sample(&mut rng));
        trajectory.push(x);
        
        state = filter.predict(&state, &u);
        
        let y = dynamics.observe(&x, &observation_noise.sample(&mut rng));
        observations.push(y);
        
        state = filter.update(&state, &y);
        states.push(state);
    }
    let means: Vec<_> = states.iter().map(|s| s.m().clone()).collect();
    let vars: Vec<_> = states.iter().map(|s| s.p().clone().diagonal()).collect();
    StatePlot::<1, 1>::new("damper.svg")
        .add_markers("obs", &observations)
        .add_line("trajectory", &trajectory)
        .add_confidence_band("var", &means, &vars, 2.)
        .add_line("mean", &means)
        .draw()
        .unwrap();
}
