use nalgebra::SVector;
use plotters::prelude::*;

use crate::types::Real;

enum SeriesKind<const N: usize, const M: usize> {
    Line {
        label: String,
        data: Vec<SVector<Real, N>>,
    },
    Band {
        data: Vec<SVector<Real, N>>,
        band: Vec<SVector<Real, N>>,
        k: Real,
    },
    Markers {
        data: Vec<SVector<Real, M>>,
    },
}
pub struct StatePlot<const N: usize, const M: usize> {
    filename: String,
    series: Vec<SeriesKind<N, M>>,
    time: Vec<Real>, // mandatory time axis
}

impl<const N: usize, const M: usize> StatePlot<N, M> {
    /// Create a new plot with a mandatory time axis
    pub fn new(filename: &str, time: &[Real]) -> Self {
        Self {
            filename: filename.to_string(),
            series: Vec::new(),
            time: time.to_vec(),
        }
    }

    pub fn add_line(mut self, label: &str, data: &[SVector<Real, N>]) -> Self {
        if data.len() != self.time.len() {
            panic!("Data length must match time axis length");
        }
        self.series.push(SeriesKind::Line {
            label: label.to_string(),
            data: data.to_vec(),
        });
        self
    }

    pub fn add_markers(mut self, data: &[SVector<Real, M>]) -> Self {
        if data.len() != self.time.len() {
            panic!("Data length must match time axis length");
        }
        self.series.push(SeriesKind::Markers {
            data: data.to_vec(),
        });
        self
    }

    pub fn add_confidence_band(
        mut self,
        means: &[SVector<Real, N>],
        vars: &[SVector<Real, N>],
        k: Real,
    ) -> Self {
        if means.len() != self.time.len() || vars.len() != self.time.len() {
            panic!("Mean and variance vectors must match time axis length");
        }
        self.series.push(SeriesKind::Band {
            data: means.to_vec(),
            band: vars.to_vec(),
            k,
        });
        self
    }

    pub fn draw(self) -> Result<(), Box<dyn std::error::Error>> {
        let y_min = self.series.iter().flat_map(|s| match s {
            SeriesKind::Line { data, .. } => data.iter().flat_map(|v| v.iter().cloned()).collect::<Vec<_>>(),
            SeriesKind::Markers { data, .. } => data.iter().flat_map(|v| v.iter().cloned()).collect(),
            SeriesKind::Band { data: means, band: vars, k, .. } => means.iter().zip(vars.iter()).flat_map(|(m, v)| {
                (0..N).map(move |i| m[i] - k * v[i].sqrt())
                    .chain((0..N).map(move |i| m[i] + k * v[i].sqrt()))
            }).collect(),
        }).fold(f64::INFINITY, |a, b| a.min(b));

        let y_max = self.series.iter().flat_map(|s| match s {
            SeriesKind::Line { data, .. } => data.iter().flat_map(|v| v.iter().cloned()).collect::<Vec<_>>(),
            SeriesKind::Markers { data, .. } => data.iter().flat_map(|v| v.iter().cloned()).collect(),
            SeriesKind::Band { data: means, band: vars, k, .. } => means.iter().zip(vars.iter()).flat_map(|(m, v)| {
                (0..N).map(move |i| m[i] - k * v[i].sqrt())
                    .chain((0..N).map(move |i| m[i] + k * v[i].sqrt()))
            }).collect(),
        }).fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let root = SVGBackend::new(&self.filename, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("State trajectory", ("sans-serif", 40))
            .build_cartesian_2d(*self.time.first().unwrap()..*self.time.last().unwrap(), y_min..y_max)?;

        chart.configure_mesh().x_desc("time").y_desc("value").draw()?;

        let mut color_idx = 0usize;

        for s in &self.series {
            match s {
                SeriesKind::Line { label, data } => {
                    for component in 0..N {
                        let ci = color_idx;
                        chart
                            .draw_series(LineSeries::new(
                                data.iter().enumerate().map(|(i, v)| (self.time[i], v[component])),
                                &Palette99::pick(ci),
                            ))?
                            .label(format!("{} x{}", label, component + 1))
                            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], Palette99::pick(ci)));
                        color_idx += 1;
                    }
                }
                SeriesKind::Band { data: means, band: variances, k } => {
                    for component in 0..N {
                        let ci = color_idx;
                        let k = *k;
                        chart.draw_series(means.iter().zip(variances.iter()).enumerate().map(|(i, (m, v))| {
                            let upper = m[component] + k * v[component].sqrt();
                            let lower = m[component] - k * v[component].sqrt();
                            Rectangle::new(
                                [(self.time[i], lower), (self.time[i], upper)],
                                Palette99::pick(ci).mix(0.3).filled(),
                            )
                        }))?;
                        color_idx += 1;
                    }
                }
                SeriesKind::Markers { data } => {
                    for component in 0..M {
                        let ci = color_idx;
                        chart.draw_series(data.iter().enumerate().map(|(i, v)| {
                            Cross::new((self.time[i], v[component]), 2, Palette99::pick(ci))
                        }))?;
                        color_idx += 1;
                    }
                }
            }
        }

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()?;

        root.present()?;
        Ok(())
    }
}