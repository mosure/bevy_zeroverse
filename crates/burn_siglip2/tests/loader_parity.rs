#![cfg(feature = "flex")]

use burn_siglip2::{
    Siglip2Config, build_bpk_header, load_model_from_bpk_path, load_model_from_parts_manifest_path,
    load_model_from_parts_manifest_path_with_stream_reader, load_model_from_safetensors_path,
    write_bpk_parts, write_siglip2_bpk,
};
use safetensors::tensor::{Dtype, TensorView, serialize};
use tempfile::tempdir;

fn tiny_safetensors_payload() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let weight = vec![0u8; 8 * 48 * 4];
    let bias = vec![0u8; 8 * 4];
    let tensors = std::collections::BTreeMap::from([
        (
            "vision.patch_embed.weight".to_string(),
            TensorView::new(Dtype::F32, vec![8, 48], &weight)?,
        ),
        (
            "vision.patch_embed.bias".to_string(),
            TensorView::new(Dtype::F32, vec![8], &bias)?,
        ),
    ]);
    Ok(serialize(&tensors, None)?)
}

fn tiny_bpk_fixture() -> Result<(tempfile::TempDir, std::path::PathBuf), Box<dyn std::error::Error>>
{
    let dir = tempdir()?;
    let payload = tiny_safetensors_payload()?;
    let bpk_path = dir.path().join("siglip2_tiny_model.bpk");
    let header = build_bpk_header(Siglip2Config::tiny_for_tests(), &payload);
    write_siglip2_bpk(&bpk_path, &header, &payload)?;
    Ok((dir, bpk_path))
}

#[test]
fn safetensors_loader_rejects_tiny_test_profile() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let config = Siglip2Config::tiny_for_tests();
    let device = burn::tensor::Device::flex();
    let weights_path = dir.path().join("siglip2_tiny_weights.safetensors");
    std::fs::write(&weights_path, tiny_safetensors_payload()?)?;
    let result = load_model_from_safetensors_path(&config, &device, &weights_path);
    let message = result.expect_err("tiny test profile should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
    Ok(())
}

#[test]
fn bpk_parts_loader_rejects_tiny_test_profile() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, bpk_path) = tiny_bpk_fixture()?;
    let _report = write_bpk_parts(&bpk_path, 1, true)?;
    let device = burn::tensor::Device::flex();
    let manifest_path = bpk_path.with_file_name("siglip2_tiny_model.bpk.parts.json");
    let result = load_model_from_parts_manifest_path(&device, &manifest_path, true);
    let message = result.expect_err("tiny test profile should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
    Ok(())
}

#[test]
fn streamed_bpk_parts_loader_rejects_tiny_test_profile() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, bpk_path) = tiny_bpk_fixture()?;
    let _report = write_bpk_parts(&bpk_path, 1, true)?;
    let device = burn::tensor::Device::flex();
    let manifest_path = bpk_path.with_file_name("siglip2_tiny_model.bpk.parts.json");
    let result = load_model_from_parts_manifest_path_with_stream_reader::<_, _>(
        &device,
        &manifest_path,
        true,
        |part_path, _| {
            std::fs::File::open(part_path)
                .map_err(|err| format!("failed to open part '{}': {err}", part_path.display()))
        },
    );
    let message = result.expect_err("tiny test profile should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
    Ok(())
}

#[test]
fn bpk_loader_rejects_tiny_fixture_bpk() -> Result<(), Box<dyn std::error::Error>> {
    let (_dir, bpk_path) = tiny_bpk_fixture()?;
    let device = burn::tensor::Device::flex();
    let result = load_model_from_bpk_path(&device, &bpk_path);
    let message = result.expect_err("tiny fixture bpk should be rejected");
    assert!(
        message.contains("production SigLIP2 profile"),
        "unexpected error: {message}"
    );
    Ok(())
}
