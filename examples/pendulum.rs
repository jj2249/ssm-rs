use nalgebra::{SMatrix, SVector, matrix, vector};

use ssm_rs::controllers::{Controller, Nontroller};
use ssm_rs::dynamics::ContinuousDynamics;
use rand_distr::Distribution;
use ssm_rs::noise::{BrownianNoise, Gaussian, NoiseProcess};
use ssm_rs::plots::StatePlot;

use ssm_rs::types::Real;

struct Pendulum {
    g: Real,
    b: Real,
    l: Real,
    h: SMatrix<Real, X, Z>,
}
const X: usize = 2;
const U: usize = 1;
const Y: usize = 1;
const Z: usize = 1;

impl ContinuousDynamics<X, U, Y, Z> for Pendulum {
    fn f(&self, x: &SVector<Real, X>, u: &SVector<Real, U>) -> SVector<Real, X> {
        vector![
            x[1],
            -self.g * Real::sin(x[0]) / self.l - self.b * x[1] + u[0]
        ]
    }
    fn f_jacobian(&self, x: &SVector<Real, X>, _u: &SVector<Real, U>) -> SMatrix<Real, X, X> {
        matrix![0., 1.; -self.g * x[0].cos(), -self.b]
    }
    fn h_matrix(&self) -> &SMatrix<Real, X, Z> {
        &self.h
    }
    fn g(&self, x: &SVector<Real, X>) -> SVector<Real, Y> {
        matrix![1., 0.] * x
    }
}

fn main() {
    let dt: f64 = 0.01;
    let controller = Nontroller;
    let sp = 0.1;
    let so = 0.01;

    let process_noise = BrownianNoise::new(vector![0.], matrix![sp * sp * dt]);
    let observation_noise = Gaussian::new(vector![0.], matrix![so * so]);
    let mut rng = rand::rng();

    let mut x = vector![0., -1.];
    let mut trajectory = vec![x];

    let dynamics = Pendulum {
        g: 9.81,
        b: 0.6,
        l: 1.,
        h: matrix![1.; 0.],
    };

    let mut observations = vec![dynamics.observe(&x, &observation_noise.sample(&mut rng))];

    let u = controller.control_law(&x);

    let t = 10.;
    let n = (t / dt) as usize;

    for _ in 0..n {
        x = dynamics.step_rk4(&x, &u, &process_noise.sample(dt, &mut rng), dt);
        trajectory.push(x);

        let y = dynamics.observe(&x, &observation_noise.sample(&mut rng));
        observations.push(y);
    }
    StatePlot::<X, Y>::new("pendulum.svg")
        .add_markers("obs", &observations)
        .add_line("trajectory", &trajectory)
        .draw()
        .unwrap();
}
