pub fn assert_tensors_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32, name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Tensor length mismatch for {}: actual={}, expected={}",
        name,
        actual.len(),
        expected.len()
    );

    let mut max_abs_error: f32 = 0.0;
    let mut sum_abs_error: f64 = 0.0;
    let mut failure_count = 0;

    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs_error = (a - b).abs();
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
        }
        sum_abs_error += abs_error as f64;

        let tolerance = atol + rtol * b.abs();
        if abs_error > tolerance {
            if failure_count < 10 {
                println!(
                    "FAIL [{}]: index {}, actual={}, expected={}, abs_error={}, tolerance={}",
                    name, i, a, b, abs_error, tolerance
                );
            }
            failure_count += 1;
        }
    }

    let mean_abs_error = (sum_abs_error / actual.len() as f64) as f32;

    println!(
        "STATS [{}]: Max Abs Error = {:.8}, Mean Abs Error = {:.8}",
        name, max_abs_error, mean_abs_error
    );

    if failure_count > 0 {
        panic!(
            "Tensor comparison failed for {} ({} elements exceeded tolerance). Max Abs Error = {:.8}",
            name, failure_count, max_abs_error
        );
    }
}
