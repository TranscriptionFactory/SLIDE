use slide_core::{OptimizeSlideCheckpoint, OptimizeSlideConfig};

fn main() {
    let checkpoint = OptimizeSlideCheckpoint {
        x_ref: None,
        y_ref: None,
        delta_grid: vec![0.05, 0.10],
        lambda_grid: vec![0.1, 0.01],
        config: OptimizeSlideConfig::default(),
        evaluations: Vec::new(),
        final_cv: None,
    };
    println!("{}", serde_json::to_string_pretty(&checkpoint).unwrap());
}
