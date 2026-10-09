#![cfg(feature = "flex")]

use std::path::PathBuf;

use burn::tensor::{Int, Tensor, TensorData};
use burn_siglip2::{
    LoadRequest, PartLoadStats, Siglip2Config, Siglip2Model, Siglip2Runtime, load_backend_flex,
    load_backend_on_device,
};

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(rel)
}

#[test]
fn flex_loader_enforces_the_production_profile() {
    let result = load_backend_flex(LoadRequest::from_bpk(fixture_path(
        "fixtures/siglip2_tiny_model.bpk",
    )));
    let message = result.expect_err("tiny fixture should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
}

#[test]
fn generic_loader_accepts_an_explicit_flex_device() {
    let device = burn::tensor::Device::flex();
    let result = load_backend_on_device(
        LoadRequest::from_bpk(fixture_path("fixtures/siglip2_tiny_model.bpk")),
        device,
    );
    let message = result.expect_err("tiny fixture should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
}

#[test]
fn flex_executes_both_towers_and_calibrated_scoring() -> Result<(), String> {
    let config = Siglip2Config::tiny_for_tests();
    let device = burn::tensor::Device::flex();
    let model = Siglip2Model::zeros(config.clone(), &device)?;
    let runtime = Siglip2Runtime {
        model,
        device,
        load_stats: PartLoadStats::default(),
    };
    let image = Tensor::<4>::zeros(
        [1, config.channels, config.image_size, config.image_size],
        &runtime.device,
    );
    let input_ids = Tensor::<2, Int>::from_data(
        TensorData::new(
            vec![0i64; config.text_max_positions],
            [1, config.text_max_positions],
        ),
        &runtime.device,
    );

    let response = runtime.encode_image_and_text_tokens(image, input_ids, None, false)?;
    assert_eq!(
        response.evidence.backend_label,
        format!("{:?}", runtime.device)
    );
    assert!(response.evidence.device_label.contains("Flex"));
    assert!(!response.evidence.wgpu_executed);
    assert_eq!(
        response.image_embedding.shape().dims::<2>(),
        [1, config.projection_dim]
    );
    assert_eq!(
        response.text_embedding.shape().dims::<2>(),
        [1, config.projection_dim]
    );
    assert_eq!(response.logits_per_image.shape().dims::<2>(), [1, 1]);
    let probabilities = response
        .probabilities_per_image
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read Flex probabilities: {err:?}"))?;
    assert_eq!(probabilities, vec![0.5]);
    Ok(())
}
