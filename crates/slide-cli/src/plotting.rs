use crate::output::Feature;
use anyhow::{ensure, Result};
use cairo::{Context, FontSlant, FontWeight, Format, ImageSurface, PdfSurface};
use slide_score::{ControlPerformance, ScoreMetric};
use std::{fs::File, path::Path};

fn text(ctx: &Context, x: f64, y: f64, label: &str) -> Result<()> {
    ctx.move_to(x, y);
    ctx.show_text(&label.replace(['\n', '\r', '\t', '\0'], " "))?;
    Ok(())
}

pub(crate) fn gene_plot(
    path: &Path,
    columns: &[(usize, Vec<Feature>)],
    marginals: &[usize],
    combined: bool,
) -> Result<()> {
    let columns: Vec<_> = columns
        .iter()
        .filter(|(lf, _)| combined || marginals.contains(lf))
        .collect();
    if columns.is_empty() || columns.iter().all(|(_, f)| f.is_empty()) {
        return Ok(());
    }
    let width = (columns.len() as f64 * 150.0 + 100.0).max(if combined { 600.0 } else { 500.0 });
    let height = 504.0;
    let surface = PdfSurface::new(width, height, path)?;
    let ctx = Context::new(&surface)?;
    ctx.set_source_rgb(1.0, 1.0, 1.0);
    ctx.paint()?;
    ctx.set_source_rgb(0.0, 0.0, 0.0);
    ctx.set_font_size(14.0);
    text(
        &ctx,
        50.0,
        25.0,
        if combined {
            "Significant Latent Factors - Marginals (bold/italic) and Interactions"
        } else {
            "SLIDE Marginal Variables"
        },
    )?;
    ctx.save()?;
    ctx.translate(18.0, 420.0);
    ctx.rotate(-std::f64::consts::FRAC_PI_2);
    ctx.set_font_size(11.0);
    text(
        &ctx,
        0.0,
        0.0,
        "Genes Associated with Significant Latent Factors",
    )?;
    ctx.restore()?;
    let n = columns.len();
    let step = (width - 80.0) / n as f64;
    let max_rows = columns.iter().map(|(_, f)| f.len()).max().unwrap_or(1);
    for (column, (lf, features)) in columns.iter().enumerate() {
        let x = 60.0 + step * column as f64;
        let emphasized = combined && marginals.contains(lf);
        ctx.select_font_face(
            "sans-serif",
            if emphasized {
                FontSlant::Italic
            } else {
                FontSlant::Normal
            },
            if emphasized {
                FontWeight::Bold
            } else {
                FontWeight::Normal
            },
        );
        for (row, feature) in features.iter().enumerate() {
            ctx.set_font_size(11.0);
            let label = feature.name.replace(['\n', '\r', '\t', '\0'], " ");
            let extent = ctx.text_extents(&label)?.width();
            if extent > step - 12.0 {
                ctx.set_font_size(11.0 * (step - 12.0) / extent);
            }
            match feature.color() {
                "red" => ctx.set_source_rgb(1.0, 0.0, 0.0),
                "blue" => ctx.set_source_rgb(0.0, 0.0, 1.0),
                _ => ctx.set_source_rgb(1.0, 1.0, 1.0),
            }
            text(&ctx, x, 60.0 + row as f64 * 360.0 / max_rows as f64, &label)?;
        }
        ctx.select_font_face("sans-serif", FontSlant::Normal, FontWeight::Normal);
        ctx.set_source_rgb(0.0, 0.0, 0.0);
        ctx.set_font_size(12.0);
        text(&ctx, x, 455.0, &format!("Z{lf}"))?;
    }
    text(&ctx, width / 2.0 - 70.0, 485.0, "Significant Latent Factor")?;
    ctx.show_page()?;
    surface.finish();
    surface.status()?;
    Ok(())
}

fn bandwidth(values: &[f64]) -> Result<f64> {
    ensure!(
        values.len() >= 2 && values.iter().all(|v| v.is_finite()),
        "density requires at least two finite control scores"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    let mean = values[0] + values.iter().map(|v| v - values[0]).sum::<f64>() / n;
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let quantile = |p: f64| {
        let i = p * (n - 1.0);
        let lo = i.floor() as usize;
        let hi = i.ceil() as usize;
        sorted[lo] + (sorted[hi] - sorted[lo]) * i.fract()
    };
    let mut scale = sd.min((quantile(0.75) - quantile(0.25)) / 1.34);
    if scale == 0.0 {
        scale = sd;
    }
    if scale == 0.0 {
        scale = values[0].abs();
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    Ok(0.9 * scale * n.powf(-0.2))
}

pub(crate) fn control_plot(path: &Path, scores: &ControlPerformance) -> Result<()> {
    ensure!(scores.real.is_finite(), "real control score must be finite");
    let mut curves = Vec::new();
    let mut lo = scores.real;
    let mut hi = scores.real;
    for values in [&scores.full_random, &scores.partial_random] {
        if values.is_empty() {
            continue;
        }
        let bw = bandwidth(values)?;
        for &v in values {
            lo = lo.min(v - 3.0 * bw);
            hi = hi.max(v + 3.0 * bw);
        }
        curves.push((values, bw));
    }
    ensure!(!curves.is_empty(), "no randomized control scores to plot");
    let pad = (hi - lo) * 0.04;
    lo -= pad;
    hi += pad;
    // shortcut: direct Gaussian KDE rather than R's FFT grid; use FFT if exact pixel parity becomes required.
    let densities: Vec<Vec<f64>> = curves
        .iter()
        .map(|(values, bw)| {
            (0..512)
                .map(|i| {
                    let x = lo + (hi - lo) * i as f64 / 511.0;
                    values
                        .iter()
                        .map(|v| (-0.5 * ((x - v) / bw).powi(2)).exp())
                        .sum::<f64>()
                        / (values.len() as f64 * bw * (2.0 * std::f64::consts::PI).sqrt())
                })
                .collect()
        })
        .collect();
    let ymax = densities.iter().flatten().copied().fold(0.0, f64::max) * 1.1;
    let surface = ImageSurface::create(Format::ARgb32, 1000, 700)?;
    let pdf = if path.extension().is_some_and(|v| v == "pdf") {
        Some(PdfSurface::new(1000.0, 700.0, path)?)
    } else {
        None
    };
    let ctx = if let Some(pdf) = &pdf {
        Context::new(pdf)?
    } else {
        Context::new(&surface)?
    };
    ctx.set_source_rgb(1.0, 1.0, 1.0);
    ctx.paint()?;
    for (j, density) in densities.iter().enumerate() {
        ctx.move_to(100.0, 590.0);
        for (i, d) in density.iter().enumerate() {
            ctx.line_to(100.0 + 820.0 * i as f64 / 511.0, 590.0 - 500.0 * d / ymax);
        }
        ctx.line_to(920.0, 590.0);
        ctx.close_path();
        if j == 0 {
            ctx.set_source_rgba(0.0, 0.0, 1.0, 0.7);
        } else {
            ctx.set_source_rgba(0.0, 1.0, 0.0, 0.7);
        }
        ctx.fill()?;
    }
    ctx.set_source_rgb(0.0, 0.0, 0.0);
    ctx.set_line_width(1.5);
    ctx.move_to(100.0, 90.0);
    ctx.line_to(100.0, 590.0);
    ctx.line_to(920.0, 590.0);
    ctx.stroke()?;
    ctx.set_font_size(18.0);
    for i in 0..=5 {
        text(
            &ctx,
            82.0 + 820.0 * i as f64 / 5.0,
            620.0,
            &format!("{:.2}", lo + (hi - lo) * i as f64 / 5.0),
        )?;
        text(
            &ctx,
            25.0,
            595.0 - 500.0 * i as f64 / 5.0,
            &format!("{:.1}", ymax * i as f64 / 5.0),
        )?;
    }
    text(
        &ctx,
        470.0,
        665.0,
        if scores.metric == ScoreMetric::Auc {
            "auc"
        } else {
            "corr"
        },
    )?;
    text(&ctx, 25.0, 60.0, "Density")?;
    ctx.set_source_rgb(0.0, 0.0, 1.0);
    text(&ctx, 540.0, 40.0, "FullRandom")?;
    if !scores.partial_random.is_empty() {
        ctx.set_source_rgb(0.0, 0.65, 0.0);
        text(&ctx, 700.0, 40.0, "PartialRandom")?;
    }
    ctx.set_source_rgb(1.0, 0.0, 0.0);
    text(&ctx, 110.0, 40.0, &format!("Real: {:.4}", scores.real))?;
    ctx.set_dash(&[12.0, 8.0], 0.0);
    ctx.set_line_width(3.0);
    let real_x = 100.0 + 820.0 * (scores.real - lo) / (hi - lo);
    ctx.move_to(real_x, 90.0);
    ctx.line_to(real_x, 590.0);
    ctx.stroke()?;
    if let Some(pdf) = pdf {
        ctx.show_page()?;
        pdf.finish();
        pdf.status()?;
    } else {
        surface.write_to_png(&mut File::create(path)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_density_and_native_artifacts() -> Result<()> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.tmpfiles/2026-10-17_native-output-fixture");
        std::fs::create_dir_all(&dir)?;
        assert!(bandwidth(&[0.0; 1000])? > 0.0);
        assert!((bandwidth(&[0.65; 1000])? - 0.9 * 0.65 * 1000_f64.powf(-0.2)).abs() < 1e-14);
        assert!(bandwidth(&[f64::NAN, 0.0]).is_err());
        control_plot(
            &dir.join("ControlPerformancePlot.png"),
            &ControlPerformance {
                real: 0.85,
                full_random: vec![0.5; 1000],
                partial_random: vec![0.65; 1000],
                metric: ScoreMetric::Auc,
            },
        )?;
        let png = ImageSurface::create_from_png(&mut File::open(
            dir.join("ControlPerformancePlot.png"),
        )?)?;
        assert_eq!(png.width(), 1000);
        control_plot(
            &dir.join("nonconstant.png"),
            &ControlPerformance {
                real: 0.85,
                full_random: (0..1000).map(|i| 0.45 + 0.1 * (i as f64).sin()).collect(),
                partial_random: (0..1000).map(|i| 0.65 + 0.1 * (i as f64).cos()).collect(),
                metric: ScoreMetric::Auc,
            },
        )?;
        control_plot(
            &dir.join("ControlPerformancePlot.pdf"),
            &ControlPerformance {
                real: 0.85,
                full_random: vec![0.5; 1000],
                partial_random: vec![],
                metric: ScoreMetric::Correlation,
            },
        )?;
        let features = vec![(
            0,
            vec![
                Feature {
                    name: "A, B & <gene> α".into(),
                    loading: 1.0,
                    auc: Some(0.8),
                    corr: Some(0.3),
                },
                Feature {
                    name: "Negative association".into(),
                    loading: 0.9,
                    auc: Some(0.2),
                    corr: Some(-0.3),
                },
            ],
        )];
        gene_plot(
            &dir.join("plotSigGenes_marginals.pdf"),
            &features,
            &[0],
            false,
        )?;
        assert!(std::fs::read(dir.join("plotSigGenes_marginals.pdf"))?.starts_with(b"%PDF-"));
        gene_plot(&dir.join("empty.pdf"), &[], &[], false)?;
        assert!(!dir.join("empty.pdf").exists());
        Ok(())
    }
}
