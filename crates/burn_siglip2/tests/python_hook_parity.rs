#![cfg(feature = "flex")]

use std::path::PathBuf;

use burn_siglip2::load_model_from_bpk_path;

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(rel)
}

#[test]
fn tiny_fixture_bpk_is_rejected_before_hook_parity() {
    let device = burn::tensor::Device::flex();
    let bpk_path = fixture_path("fixtures/siglip2_tiny_model.bpk");
    let result = load_model_from_bpk_path(&device, &bpk_path);
    let message = result.expect_err("tiny fixture should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
}
