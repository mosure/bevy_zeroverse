#![cfg(feature = "ndarray")]

use std::path::PathBuf;

use burn_siglip2::{LoadRequest, Siglip2Config, load_backend};

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(rel)
}

#[test]
fn load_backend_rejects_tiny_fixture_bpk() {
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
fn production_profile_check_rejects_tiny_test_config() {
    let cfg = Siglip2Config::tiny_for_tests();
    let err = cfg
        .validate_production_profile()
        .expect_err("tiny config should be rejected");
    assert!(
        err.contains("production SigLIP2 profile"),
        "unexpected error: {err}"
    );
}
