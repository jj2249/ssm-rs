use nalgebra::SVector;
use plotters::prelude::*;

use crate::types::Real;

enum SeriesKind<const N: usize> {
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
}

pub struct StatePlot<const N: usize> {
    filename: String,
    series: Vec<SeriesKind<N>>,
}

impl<const N: usize> StatePlot<N> {
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

    /// Add a shaded confidence band: mean ± k * sqrt(variance) per component.
    pub fn add_confidence_band(
        mut self,
        label: &str,
        means: &[SVector<Real, N>],
        std_devs: &[SVector<Real, N>],
        k: Real,
    ) -> Self {
        self.series.push(SeriesKind::Band {
            label: label.to_string(),
            data: means.to_vec(),
            band: std_devs.to_vec(),
            k,
        });
        self
    }

    pub fn draw(self) -> Result<(), Box<dyn std::error::Error>> {
        let mut n_points = 0usize;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for s in &self.series {
            match s {
                SeriesKind::Line { data, .. } => {
                    n_points = n_points.max(data.len());
                    for v in data {
                        for &val in v.iter() {
                            y_min = y_min.min(val);
                            y_max = y_max.max(val);
                        }
                    }
                }
                SeriesKind::Band {
                    data: means,
                    band: std_devs,
                    k,
                    ..
                } => {
                    n_points = n_points.max(means.len());
                    for (m, s) in means.iter().zip(std_devs.iter()) {
                        for c in 0..N {
                            y_min = y_min.min(m[c] - k * s[c]);
                            y_max = y_max.max(m[c] + k * s[c]);
                        }
                    }
                }
            }
        }

        let root = BitMapBackend::new(&self.filename, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("State trajectory", ("sans-serif", 40))
            .build_cartesian_2d(0..n_points, y_min..y_max)?;

        chart.configure_mesh().x_desc("t").y_desc("value").draw()?;

        let mut color_idx = 0usize;
        for s in &self.series {
            match s {
                SeriesKind::Line { label, data } => {
                    for component in 0..N {
                        let ci = color_idx;
                        chart
                            .draw_series(LineSeries::new(
                                data.iter().enumerate().map(|(i, v)| (i, v[component])),
                                &Palette99::pick(ci),
                            ))?
                            .label(format!("{} x{}", label, component + 1))
                            .legend(move |(x, y)| {
                                PathElement::new([(x, y), (x + 20, y)], Palette99::pick(ci))
                            });
                        color_idx += 1;
                    }
                }
                SeriesKind::Band {
                    label,
                    data: means,
                    band: variances,
                    k,
                } => {
                    for component in 0..N {
                        let ci = color_idx;
                        let k = *k;
                        chart
                            .draw_series(means.iter().zip(variances.iter()).enumerate().map(
                                |(i, (m, v))| {
                                    let upper = m[component] + k * v[component];
                                    let lower = m[component] - k * v[component];
                                    Rectangle::new(
                                        [(i, lower), (i + 1, upper)],
                                        Palette99::pick(ci).mix(0.3).filled(),
                                    )
                                },
                            ))?
                            .label(format!("{} x{}", label, component + 1))
                            .legend(move |(x, y)| {
                                Rectangle::new(
                                    [(x, y - 5), (x + 20, y + 5)],
                                    Palette99::pick(ci).mix(0.5).filled(),
                                )
                            });
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
