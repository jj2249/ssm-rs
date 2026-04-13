use nalgebra::{SMatrix, SVector, matrix, vector};

use rand_distr::Distribution;
use ssm_rs::controllers::{Controller, Nontroller};
use ssm_rs::dynamics::{
    ContinuousDynamics, DifferentiableContinuousDynamics, DiscreteDynamics, DiscretisedNonlinear,
};
use ssm_rs::filters::{Filter, KalmanFilter, StateEstimate};
use ssm_rs::noise::{BrownianNoise, Gaussian, NoiseProcess};
use ssm_rs::plots::StatePlot;

use ssm_rs::types::Real;

const X: usize = 2;
const U: usize = 1;
const Y: usize = 1;
const Z: usize = 1;

struct Pendulum {
    g: Real,
    b: Real,
    l: Real,
    h: SMatrix<Real, X, Z>,
}

impl ContinuousDynamics<X, U, Y, Z> for Pendulum {
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X> {
        vector![
            x[1],
            -self.g * x[0].sin() / self.l - self.b * x[1] + u[0]
        ]
    }
    fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y> {
        matrix![1., 0.] * x
    }
    fn h_matrix(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
}

impl DifferentiableContinuousDynamics<X, U, Y, Z> for Pendulum {
    fn f_jacobian(&self, x: &SVector<Real, X>, _u: &SVector<Real, U>) -> SMatrix<Real, X, X> {
        matrix![0., 1.; -self.g * x[0].cos() / self.l, -self.b]
    }
    fn b_jacobian(&self, _x: &SVector<Real, X>, _u: &SVector<Real, U>) -> SMatrix<Real, X, U> {
        matrix![0.; 1.]
    }
    fn g_jacobian(&self, _x: &SVector<Real, X>) -> SMatrix<Real, Y, X> {
        matrix![1., 0.]
    }
}

fn main() {
    let dt: f64 = 0.1;
    let controller = Nontroller;
    let sp = 0.1;
    let so = 1.;

    let process_noise = BrownianNoise::new(vector![0.], matrix![sp * sp]);
    let observation_noise = Gaussian::new(vector![0.], matrix![so * so]);
    let mut rng = rand::rng();

    let mut x = vector![1., 0.];
    let mut trajectory = vec![x];

    let dynamics = DiscretisedNonlinear::new(
        Pendulum { g: 9.81, b: 0.6, l: 1., h: matrix![1.; 0.] },
        dt,
    );

    let mut observations = vec![dynamics.observe(&x, &observation_noise.sample(&mut rng))];
    let mut state = StateEstimate::new(x, SMatrix::<Real, 2, 2>::identity());
    let mut states = vec![state.clone()];

    let u = controller.control_law(&x);
    let filter = KalmanFilter::new(&dynamics, matrix![sp * sp * dt], matrix![so * so]);

    let t = 10.;
    let n = (t / dt) as usize;

    for _ in 0..n {
        x = dynamics.step(&x, &u, &process_noise.sample(dt, &mut rng));
        trajectory.push(x);
        state = filter.predict(&state, &u);

        let y = dynamics.observe(&x, &observation_noise.sample(&mut rng));
        observations.push(y);

        state = filter.update(&state, &y);
        states.push(state);
    }

    let means: Vec<_> = states.iter().map(|s| s.m().clone()).collect();
    let vars: Vec<_> = states.iter().map(|s| s.p().clone().diagonal()).collect();
    StatePlot::<2, 1>::new("pendulum.svg")
        .add_markers("obs", &observations)
        .add_line("trajectory", &trajectory)
        .add_confidence_band("var", &means, &vars, 2.)
        .add_line("mean", &means)
        .draw()
        .unwrap();
}
