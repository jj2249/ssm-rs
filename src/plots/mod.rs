use nalgebra::SVector;
use plotters::prelude::*;

use crate::types::Real;

pub fn plot_trajectory<const N: usize>(
    trajectory: &[SVector<Real, N>],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create background
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Get min & max value across all states across all times
    let y_min = trajectory
        .iter()
        .flat_map(|v| v.iter())
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = trajectory
        .iter()
        .flat_map(|v| v.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Build the chart
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("State trajectory", ("sans-serif", 40))
        .build_cartesian_2d(0..trajectory.len(), y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc("t")
        .y_desc("trajectory")
        .draw()?;

    for component in 0..trajectory[0].nrows() {
        chart
            .draw_series(LineSeries::new(
                trajectory
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, v[component])),
                &Palette99::pick(component),
            ))?
            .label(format!("x{}", component + 1))
            .legend(move |(x, y)| {
                PathElement::new([(x, y), (x + 20, y)], Palette99::pick(component))
            });
    }
    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;
    root.present()?;
    Ok(())
}
