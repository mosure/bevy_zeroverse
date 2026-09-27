#![cfg(all(feature = "wgpu", feature = "pipeline"))]

use std::{fs, path::Path, path::PathBuf};

use burn_siglip2::{LoadRequest, Siglip2Tokenizer, load_backend_wgpu};
use safetensors::SafeTensors;
use serde::Deserialize;

const REFERENCE_ENV: &str = "BURN_SIGLIP2_WGPU_REFERENCE";
const BUNDLE_ENV: &str = "BURN_SIGLIP2_WGPU_BUNDLE";
// Matches the F16 embedding-parity contract emitted by siglip2_reference.py.
const RAW_EMBEDDING_MAX_ABS: f32 = 5.0e-4;
const NORMALIZED_UNIT_NORM_MAX_ERROR: f32 = 5.0e-4;

#[derive(Deserialize)]
struct ReferenceManifest {
    inputs: ReferenceInputs,
    tensors: ReferenceTensors,
}

#[derive(Deserialize)]
struct ReferenceInputs {
    encoded_image: ReferenceFile,
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct ReferenceFile {
    file: String,
}

#[derive(Deserialize)]
struct ReferenceTensors {
    file: String,
}

#[test]
fn opt_in_real_wgpu_dual_tower_matches_reference_logits() -> Result<(), String> {
    let reference_path = std::env::var_os(REFERENCE_ENV).map(PathBuf::from);
    let bundle_path = std::env::var_os(BUNDLE_ENV).map(PathBuf::from);
    let (reference_path, bundle_path) = match (reference_path, bundle_path) {
        (None, None) => {
            eprintln!("skipping real WGPU parity; set {REFERENCE_ENV} and {BUNDLE_ENV}");
            return Ok(());
        }
        (Some(reference), Some(bundle)) => (reference, bundle),
        _ => {
            return Err(format!(
                "{REFERENCE_ENV} and {BUNDLE_ENV} must either both be set or both be absent"
            ));
        }
    };

    let reference: ReferenceManifest = serde_json::from_slice(
        &fs::read(&reference_path)
            .map_err(|err| format!("failed to read '{}': {err}", reference_path.display()))?,
    )
    .map_err(|err| format!("failed to parse '{}': {err}", reference_path.display()))?;
    let reference_dir = reference_path.parent().unwrap_or_else(|| Path::new("."));
    let encoded_image = fs::read(reference_dir.join(&reference.inputs.encoded_image.file))
        .map_err(|err| format!("failed to read reference image: {err}"))?;

    let runtime = load_backend_wgpu(LoadRequest::from_parts_manifest(&bundle_path, true))?;
    let bundle_name = bundle_path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".bpk.parts.json"))
        .ok_or_else(|| format!("invalid parts manifest name '{}'", bundle_path.display()))?;
    let tokenizer_path = bundle_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{bundle_name}.tokenizer.json"));
    let tokenizer = Siglip2Tokenizer::from_file(&tokenizer_path, &runtime.model.config)?;
    let response = runtime.encode_image_bytes_and_text_strings(
        &encoded_image,
        &tokenizer,
        &reference.inputs.texts,
        false,
    )?;
    let image_embedding_shape = response.image_embedding.shape().dims::<2>();
    let text_embedding_shape = response.text_embedding.shape().dims::<2>();
    let normalized_image_embedding_shape = response.normalized_image_embedding.shape().dims::<2>();
    let normalized_text_embedding_shape = response.normalized_text_embedding.shape().dims::<2>();
    let actual_image_embedding = response
        .image_embedding
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read WGPU image embedding: {err:?}"))?;
    let actual_text_embedding = response
        .text_embedding
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read WGPU text embedding: {err:?}"))?;
    let actual_normalized_image_embedding = response
        .normalized_image_embedding
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read normalized WGPU image embedding: {err:?}"))?;
    let actual_normalized_text_embedding = response
        .normalized_text_embedding
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read normalized WGPU text embedding: {err:?}"))?;
    let actual_logits = response
        .logits_per_image
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read WGPU logits: {err:?}"))?;
    let actual_probabilities = response
        .probabilities_per_image
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read WGPU probabilities: {err:?}"))?;
    let all_outputs_finite = [
        actual_image_embedding.as_slice(),
        actual_text_embedding.as_slice(),
        actual_normalized_image_embedding.as_slice(),
        actual_normalized_text_embedding.as_slice(),
        actual_logits.as_slice(),
        actual_probabilities.as_slice(),
    ]
    .into_iter()
    .flatten()
    .all(|value| value.is_finite());
    if !all_outputs_finite {
        let image = runtime.encode_image_bytes(&encoded_image, true)?;
        let text = runtime.encode_text_strings(&tokenizer, &reference.inputs.texts, true)?;
        let bad_image_hooks = image
            .hooks
            .iter()
            .filter(|(_, tensor)| tensor.data.iter().any(|value| !value.is_finite()))
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        let bad_text_hooks = text
            .hooks
            .iter()
            .filter(|(_, tensor)| tensor.data.iter().any(|value| !value.is_finite()))
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        return Err(format!(
            "WGPU produced non-finite outputs; image hooks={bad_image_hooks:?}, text hooks={bad_text_hooks:?}"
        ));
    }

    let tensor_path = reference_dir.join(&reference.tensors.file);
    let tensor_bytes = fs::read(&tensor_path)
        .map_err(|err| format!("failed to read '{}': {err}", tensor_path.display()))?;
    let tensors = SafeTensors::deserialize(&tensor_bytes)
        .map_err(|err| format!("failed to parse '{}': {err}", tensor_path.display()))?;
    let expected_image_embedding_view = tensors
        .tensor("output.image_embedding.raw")
        .map_err(|err| format!("missing reference image embedding: {err}"))?;
    let expected_text_embedding_view = tensors
        .tensor("output.text_embedding.raw")
        .map_err(|err| format!("missing reference text embedding: {err}"))?;
    ensure_shape(
        "image embedding",
        image_embedding_shape,
        expected_image_embedding_view.shape(),
    )?;
    ensure_shape(
        "text embedding",
        text_embedding_shape,
        expected_text_embedding_view.shape(),
    )?;
    let expected_image_embedding =
        burn_siglip2::hooks::decode_view_to_f32(&expected_image_embedding_view)?;
    let expected_text_embedding =
        burn_siglip2::hooks::decode_view_to_f32(&expected_text_embedding_view)?;
    let expected_logits = burn_siglip2::hooks::decode_view_to_f32(
        &tensors
            .tensor("output.logits_per_image")
            .map_err(|err| format!("missing reference logits: {err}"))?,
    )?;
    let expected_probabilities = burn_siglip2::hooks::decode_view_to_f32(
        &tensors
            .tensor("output.probabilities_per_image")
            .map_err(|err| format!("missing reference probabilities: {err}"))?,
    )?;
    let image_embedding_max_abs = max_abs(&actual_image_embedding, &expected_image_embedding)?;
    let text_embedding_max_abs = max_abs(&actual_text_embedding, &expected_text_embedding)?;
    let logits_max_abs = max_abs(&actual_logits, &expected_logits)?;
    let probabilities_max_abs = max_abs(&actual_probabilities, &expected_probabilities)?;
    let image_unit_norm_max_error = max_unit_norm_error(
        &actual_normalized_image_embedding,
        normalized_image_embedding_shape,
    )?;
    let text_unit_norm_max_error = max_unit_norm_error(
        &actual_normalized_text_embedding,
        normalized_text_embedding_shape,
    )?;
    if image_embedding_max_abs > RAW_EMBEDDING_MAX_ABS
        || text_embedding_max_abs > RAW_EMBEDDING_MAX_ABS
        || image_unit_norm_max_error > NORMALIZED_UNIT_NORM_MAX_ERROR
        || text_unit_norm_max_error > NORMALIZED_UNIT_NORM_MAX_ERROR
        || logits_max_abs > 5.0e-3
        || probabilities_max_abs > 1.0e-7
    {
        return Err(format!(
            "WGPU/reference mismatch: image embedding max_abs={image_embedding_max_abs:.6e}, text embedding max_abs={text_embedding_max_abs:.6e}, image unit-norm max_error={image_unit_norm_max_error:.6e}, text unit-norm max_error={text_unit_norm_max_error:.6e}, logits max_abs={logits_max_abs:.6e}, probabilities max_abs={probabilities_max_abs:.6e}"
        ));
    }
    eprintln!(
        "real WGPU parity passed: image embedding max_abs={image_embedding_max_abs:.6e}, text embedding max_abs={text_embedding_max_abs:.6e}, image unit-norm max_error={image_unit_norm_max_error:.6e}, text unit-norm max_error={text_unit_norm_max_error:.6e}, logits max_abs={logits_max_abs:.6e}, probabilities max_abs={probabilities_max_abs:.6e}"
    );
    Ok(())
}

fn ensure_shape(name: &str, actual: [usize; 2], expected: &[usize]) -> Result<(), String> {
    if expected != actual {
        return Err(format!(
            "{name} shape mismatch: WGPU={actual:?}, reference={expected:?}"
        ));
    }
    Ok(())
}

fn max_unit_norm_error(values: &[f32], shape: [usize; 2]) -> Result<f32, String> {
    let [rows, width] = shape;
    if rows == 0 || width == 0 || rows.checked_mul(width) != Some(values.len()) {
        return Err(format!(
            "invalid embedding shape {shape:?} for {} values",
            values.len()
        ));
    }
    Ok(values
        .chunks_exact(width)
        .map(|row| {
            let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            (norm - 1.0).abs()
        })
        .fold(0.0f32, f32::max))
}

fn max_abs(actual: &[f32], expected: &[f32]) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max))
}
