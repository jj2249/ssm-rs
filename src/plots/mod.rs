use nalgebra::SVector;
use plotters::prelude::*;

use crate::types::Real;

enum SeriesKind<const N: usize, const M: usize> {
    Line {
        label: String,
        data: Vec<SVector<Real, N>>,
    },
    Band {
        label: String,
        data: Vec<SVector<Real, N>>,
        band: Vec<SVector<Real, N>>,
        k: Real,
    },
    Markers {
        label: String,
        data: Vec<SVector<Real, M>>,
    },
}

pub struct StatePlot<const N: usize, const M: usize> {
    filename: String,
    series: Vec<SeriesKind<N, M>>,
}

impl<const N: usize, const M: usize> StatePlot<N, M> {
    pub fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            series: Vec::new(),
        }
    }

    pub fn add_line(mut self, label: &str, data: &[SVector<Real, N>]) -> Self {
        self.series.push(SeriesKind::Line {
            label: label.to_string(),
            data: data.to_vec(),
        });
        self
    }

    pub fn add_markers(mut self, label: &str, data: &[SVector<Real, M>]) -> Self {
        self.series.push(SeriesKind::Markers {
            label: label.to_string(),
            data: data.to_vec(),
        });
        self
    }

    /// Add a shaded confidence band: mean ± k * sqrt(variance) per component.
    pub fn add_confidence_band(
        mut self,
        label: &str,
        means: &[SVector<Real, N>],
        vars: &[SVector<Real, N>],
        k: Real,
    ) -> Self {
        self.series.push(SeriesKind::Band {
            label: label.to_string(),
            data: means.to_vec(),
            band: vars.to_vec(),
            k,
        });
        self
    }

    pub fn draw(self) -> Result<(), Box<dyn std::error::Error>> {
        let mut n_points = 0usize;
        let mut y_min = [f64::INFINITY; N];
        let mut y_max = [f64::NEG_INFINITY; N];

        for s in &self.series {
            match s {
                SeriesKind::Line { data, .. } => {
                    n_points = n_points.max(data.len());
                    for v in data {
                        for c in 0..N {
                            y_min[c] = y_min[c].min(v[c]);
                            y_max[c] = y_max[c].max(v[c]);
                        }
                    }
                }
                SeriesKind::Band {
                    data: means,
                    band: vars,
                    k,
                    ..
                } => {
                    n_points = n_points.max(means.len());
                    for (m, s) in means.iter().zip(vars.iter()) {
                        for c in 0..N {
                            y_min[c] = y_min[c].min(m[c] - k * s[c].sqrt());
                            y_max[c] = y_max[c].max(m[c] + k * s[c].sqrt());
                        }
                    }
                }
                SeriesKind::Markers { data, .. } => {
                    n_points = n_points.max(data.len());
                    for v in data {
                        for c in 0..M.min(N) {
                            y_min[c] = y_min[c].min(v[c]);
                            y_max[c] = y_max[c].max(v[c]);
                        }
                    }
                }
            }
        }

        let root = SVGBackend::new(&self.filename, (800, 300 * N as u32)).into_drawing_area();
        root.fill(&WHITE)?;

        let panels = root.split_evenly((N, 1));

        for (component, panel) in panels.iter().enumerate() {
            let lo = y_min[component];
            let hi = y_max[component];
            // guard against flat signals
            let (lo, hi) = if (hi - lo).abs() < 1e-12 {
                (lo - 1.0, hi + 1.0)
            } else {
                (lo, hi)
            };

            let mut chart = ChartBuilder::on(panel)
                .margin(10)
                .set_label_area_size(LabelAreaPosition::Left, 40)
                .set_label_area_size(LabelAreaPosition::Bottom, 40)
                .caption(format!("x{}", component + 1), ("sans-serif", 20))
                .build_cartesian_2d(0..n_points, lo..hi)?;

            chart.configure_mesh().x_desc("t").y_desc("value").draw()?;

            let mut color_idx = 0usize;
            for s in &self.series {
                match s {
                    SeriesKind::Line { label, data } => {
                        let ci = color_idx;
                        chart
                            .draw_series(LineSeries::new(
                                data.iter().enumerate().map(|(i, v)| (i, v[component])),
                                &Palette99::pick(ci),
                            ))?
                            .label(label.as_str())
                            .legend(move |(x, y)| {
                                PathElement::new([(x, y), (x + 20, y)], Palette99::pick(ci))
                            });
                        color_idx += 1;
                    }
                    SeriesKind::Band {
                        label,
                        data: means,
                        band: variances,
                        k,
                    } => {
                        let ci = color_idx;
                        let k = *k;
                        chart
                            .draw_series(means.iter().zip(variances.iter()).enumerate().map(
                                |(i, (m, v))| {
                                    let upper = m[component] + k * v[component].sqrt();
                                    let lower = m[component] - k * v[component].sqrt();
                                    Rectangle::new(
                                        [(i, lower), (i + 1, upper)],
                                        Palette99::pick(ci).mix(0.3).filled(),
                                    )
                                },
                            ))?
                            .label(label.as_str())
                            .legend(move |(x, y)| {
                                Rectangle::new(
                                    [(x, y - 5), (x + 20, y + 5)],
                                    Palette99::pick(ci).mix(0.5).filled(),
                                )
                            });
                        color_idx += 1;
                    }
                    SeriesKind::Markers { label, data } => {
                        if component < M {
                            let ci = color_idx;
                            chart
                                .draw_series(data.iter().enumerate().map(|(i, v)| {
                                    Cross::new((i, v[component]), 2, Palette99::pick(ci))
                                }))?
                                .label(label.as_str())
                                .legend(move |(x, y)| {
                                    Cross::new((x + 10, y), 5, Palette99::pick(ci))
                                });
                        }
                        color_idx += 1;
                    }
                }
            }

            chart
                .configure_series_labels()
                .border_style(BLACK)
                .background_style(WHITE.mix(0.8))
                .draw()?;
        }

        root.present()?;
        Ok(())
    }
}
