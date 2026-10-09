#![cfg(feature = "flex")]

use std::path::PathBuf;

use burn_siglip2::{LoadRequest, load_backend};

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(rel)
}

#[test]
fn tiny_fixture_bpk_is_rejected_for_inference_without_hooks() {
    let result = load_backend(LoadRequest::from_bpk(fixture_path(
        "fixtures/siglip2_tiny_model.bpk",
    )));
    let message = result.expect_err("tiny fixture should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
}

#[test]
fn tiny_fixture_bpk_is_rejected_for_inference_with_hooks() {
    let result = load_backend(LoadRequest::from_bpk(fixture_path(
        "fixtures/siglip2_tiny_model.bpk",
    )));
    let message = result.expect_err("tiny fixture should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
}
