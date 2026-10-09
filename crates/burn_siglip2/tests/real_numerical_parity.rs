#![cfg(feature = "flex")]

use std::{fs, path::Path, path::PathBuf};

use burn::tensor::{Int, Tensor, TensorData};
use burn_siglip2::{LoadRequest, load_backend};
#[cfg(feature = "pipeline")]
use burn_siglip2::{Siglip2Backend, Siglip2ImageProcessor, Siglip2Tokenizer};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const REFERENCE_ENV: &str = "BURN_SIGLIP2_NUMERICAL_REFERENCE";
const BUNDLE_ENV: &str = "BURN_SIGLIP2_NUMERICAL_BUNDLE";

#[derive(Debug, Deserialize)]
struct ReferenceManifest {
    schema: String,
    schema_version: u32,
    model: ReferenceModel,
    #[cfg(feature = "pipeline")]
    inputs: Option<ReferenceInputs>,
    tensors: ReferenceTensorFile,
    tolerances: ReferenceTolerances,
}

#[cfg(feature = "pipeline")]
#[derive(Debug, Deserialize)]
struct ReferenceInputs {
    encoded_image: ReferenceInputFile,
    texts: Vec<String>,
}

#[cfg(feature = "pipeline")]
#[derive(Debug, Deserialize)]
struct ReferenceInputFile {
    file: String,
    file_sha256: String,
}

#[derive(Debug, Deserialize)]
struct ReferenceModel {
    variant: String,
    weight_precision: String,
    resolved_config: ReferenceConfig,
}

#[derive(Debug, Deserialize)]
struct ReferenceConfig {
    image_size: usize,
    patch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    projection_size: usize,
    text_max_positions: usize,
    text_vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct ReferenceTensorFile {
    file: String,
    file_sha256: String,
}

#[derive(Debug, Deserialize)]
struct ReferenceTolerances {
    parity: ParityTolerances,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct ParityTolerances {
    embedding_max_abs: f32,
    embedding_rmse: f32,
    embedding_min_cosine: f32,
    logit_max_abs: f32,
    logit_rmse: f32,
    probability_max_abs: f32,
    probability_rmse: f32,
}

#[derive(Debug, Clone)]
struct OwnedTensor<T> {
    shape: Vec<usize>,
    values: Vec<T>,
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    max_abs: f32,
    rmse: f32,
}

#[test]
fn comparison_math_and_schema_smoke_are_deterministic() -> Result<(), String> {
    let metrics = metrics(&[1.0, 2.0, 3.0], &[1.5, 1.0, 3.0])?;
    assert!((metrics.max_abs - 1.0).abs() <= f32::EPSILON);
    assert!((metrics.rmse - (1.25f32 / 3.0).sqrt()).abs() <= 1.0e-7);

    let cosine = minimum_row_cosine(&[3.0, 4.0, 1.0, 0.0], &[3.0, 4.0, 1.0, 0.0], 2, 2)?;
    assert!((cosine - 1.0).abs() <= 1.0e-7);

    let manifest: ReferenceManifest = serde_json::from_str(
        r#"{
          "schema":"burn_siglip2.hf_reference",
          "schema_version":1,
          "model":{"variant":"base-patch16-224","weight_precision":"f16","resolved_config":{
            "image_size":224,"patch_size":16,"hidden_size":768,"intermediate_size":3072,
            "num_hidden_layers":12,"num_attention_heads":12,"projection_size":768,
            "text_max_positions":64,"text_vocab_size":256000}},
          "tensors":{"file":"reference.safetensors","file_sha256":"00"},
          "tolerances":{"parity":{"embedding_max_abs":0.0005,"embedding_rmse":0.0001,
            "embedding_min_cosine":0.99999,"logit_max_abs":0.05,"logit_rmse":0.03,
            "probability_max_abs":0.005,"probability_rmse":0.003}}
        }"#,
    )
    .map_err(|err| format!("failed to parse smoke manifest: {err}"))?;
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.model.variant, "base-patch16-224");
    assert_eq!(manifest.model.weight_precision, "f16");
    Ok(())
}

#[test]
fn opt_in_real_hf_reference_matches_imported_bundle() -> Result<(), String> {
    let reference_path = std::env::var_os(REFERENCE_ENV).map(PathBuf::from);
    let bundle_path = std::env::var_os(BUNDLE_ENV).map(PathBuf::from);
    let (reference_path, bundle_path) = match (reference_path, bundle_path) {
        (None, None) => {
            eprintln!(
                "skipping real SigLIP2 numerical parity; set {REFERENCE_ENV} and {BUNDLE_ENV}"
            );
            return Ok(());
        }
        (Some(reference), Some(bundle)) => (reference, bundle),
        _ => {
            return Err(format!(
                "{REFERENCE_ENV} and {BUNDLE_ENV} must either both be set or both be absent"
            ));
        }
    };

    let manifest = read_manifest(&reference_path)?;
    if manifest.schema != "burn_siglip2.hf_reference" || manifest.schema_version != 1 {
        return Err(format!(
            "unsupported reference schema {} version {}",
            manifest.schema, manifest.schema_version
        ));
    }
    if !matches!(manifest.model.weight_precision.as_str(), "f16" | "f32") {
        return Err(format!(
            "unsupported reference weight precision '{}'",
            manifest.model.weight_precision
        ));
    }

    let tensor_path = reference_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&manifest.tensors.file);
    let tensor_bytes = fs::read(&tensor_path).map_err(|err| {
        format!(
            "failed to read reference tensors '{}': {err}",
            tensor_path.display()
        )
    })?;
    let actual_sha256 = hex::encode(Sha256::digest(&tensor_bytes));
    if actual_sha256 != manifest.tensors.file_sha256 {
        return Err(format!(
            "reference tensor checksum mismatch: manifest {}, actual {}",
            manifest.tensors.file_sha256, actual_sha256
        ));
    }
    let reference_tensors = SafeTensors::deserialize(&tensor_bytes)
        .map_err(|err| format!("failed to parse reference safetensors: {err}"))?;

    let request = load_request(&bundle_path)?;
    let runtime = load_backend(request)?;
    validate_runtime_config(&manifest, &runtime.model.config)?;

    let pixels = read_f32(&reference_tensors, "input.pixel_values")?;
    let input_ids = read_i64(&reference_tensors, "input.input_ids")?;
    let pixel_shape: [usize; 4] = pixels
        .shape
        .clone()
        .try_into()
        .map_err(|_| format!("input.pixel_values must be rank 4, got {:?}", pixels.shape))?;
    let input_id_shape: [usize; 2] = input_ids
        .shape
        .clone()
        .try_into()
        .map_err(|_| format!("input.input_ids must be rank 2, got {:?}", input_ids.shape))?;
    let pixel_tensor =
        Tensor::<4>::from_data(TensorData::new(pixels.values, pixel_shape), &runtime.device);
    let input_id_tensor = Tensor::<2, Int>::from_data(
        TensorData::new(input_ids.values, input_id_shape),
        &runtime.device,
    );

    // Fixed-resolution Hugging Face SigLIP2 does not forward a text attention mask.
    let response =
        runtime.encode_image_and_text_tokens(pixel_tensor, input_id_tensor, None, false)?;
    let actual = [
        (
            "output.image_embedding.raw",
            burn_f32(response.image_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "output.image_embedding.normalized",
            burn_f32(response.normalized_image_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "output.text_embedding.raw",
            burn_f32(response.text_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "output.text_embedding.normalized",
            burn_f32(response.normalized_text_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "output.logits_per_image",
            burn_f32(response.logits_per_image)?,
            TensorKind::Logit,
        ),
        (
            "output.logits_per_text",
            burn_f32(response.logits_per_text)?,
            TensorKind::Logit,
        ),
        (
            "output.probabilities_per_image",
            burn_f32(response.probabilities_per_image)?,
            TensorKind::Probability,
        ),
        (
            "output.probabilities_per_text",
            burn_f32(response.probabilities_per_text)?,
            TensorKind::Probability,
        ),
    ];

    for (name, actual, kind) in actual {
        let reference = read_f32(&reference_tensors, name)?;
        assert_tensor_parity(name, &actual, &reference, kind, manifest.tolerances.parity)?;
    }

    let reference_logits = read_f32(&reference_tensors, "output.logits_per_image")?;
    let reference_probabilities = read_f32(&reference_tensors, "output.probabilities_per_image")?;
    for (index, (logit, probability)) in reference_logits
        .values
        .iter()
        .zip(&reference_probabilities.values)
        .enumerate()
    {
        let expected = sigmoid(*logit);
        if (expected - probability).abs() > 2.0e-6 {
            return Err(format!(
                "reference probability {index} is not sigmoid(logit): {probability} != {expected}"
            ));
        }
    }

    #[cfg(feature = "pipeline")]
    assert_end_to_end_pipeline_parity(
        &reference_path,
        &bundle_path,
        &manifest,
        &reference_tensors,
        &runtime,
    )?;

    #[cfg(not(feature = "pipeline"))]
    eprintln!(
        "exact tensor parity passed; enable --features pipeline to additionally exercise encoded-image preprocessing and string tokenization"
    );

    eprintln!(
        "real SigLIP2 parity passed: variant={}, storage={}, bundle={}",
        manifest.model.variant,
        manifest.model.weight_precision,
        bundle_path.display()
    );
    Ok(())
}

#[cfg(feature = "pipeline")]
fn assert_end_to_end_pipeline_parity(
    reference_path: &Path,
    bundle_path: &Path,
    manifest: &ReferenceManifest,
    reference_tensors: &SafeTensors<'_>,
    runtime: &Siglip2Backend,
) -> Result<(), String> {
    let Some(inputs) = manifest.inputs.as_ref() else {
        eprintln!(
            "reference manifest has no encoded input sidecar; regenerate it to run end-to-end pipeline parity"
        );
        return Ok(());
    };
    if inputs.texts.is_empty() {
        return Err("reference manifest inputs.texts must not be empty".to_string());
    }

    let reference_dir = reference_path.parent().unwrap_or_else(|| Path::new("."));
    let image_path = reference_dir.join(&inputs.encoded_image.file);
    let image_bytes = fs::read(&image_path).map_err(|err| {
        format!(
            "failed to read reference input image '{}': {err}",
            image_path.display()
        )
    })?;
    let image_sha256 = hex::encode(Sha256::digest(&image_bytes));
    if image_sha256 != inputs.encoded_image.file_sha256 {
        return Err(format!(
            "reference input image checksum mismatch: manifest {}, actual {}",
            inputs.encoded_image.file_sha256, image_sha256
        ));
    }

    let rust_pixels = Siglip2ImageProcessor::new(&runtime.model.config)?
        .preprocess_bytes(&image_bytes, &runtime.device)?;
    let rust_pixels = burn_f32(rust_pixels)?;
    let reference_pixels = read_f32(reference_tensors, "input.pixel_values")?;
    let pixel_metrics = metrics(&rust_pixels.values, &reference_pixels.values)?;
    let worst_pixel = largest_delta_index(&rust_pixels.values, &reference_pixels.values)?;
    eprintln!(
        "pipeline.input.pixel_values: max_abs={:.6e}, rmse={:.6e}, worst_index={}, Rust={:.8}, HF={:.8}",
        pixel_metrics.max_abs,
        pixel_metrics.rmse,
        worst_pixel,
        rust_pixels.values[worst_pixel],
        reference_pixels.values[worst_pixel]
    );
    eprintln!(
        "pipeline.input.pixel_values first mismatches: {}",
        first_delta_samples(&rust_pixels.values, &reference_pixels.values, 8)?
    );

    let tokenizer_path = adjacent_bundle_asset(bundle_path, ".tokenizer.json")?;
    let tokenizer = Siglip2Tokenizer::from_file(&tokenizer_path, &runtime.model.config)?;
    let tokenized = tokenizer.encode_batch(&inputs.texts)?;
    let reference_ids = read_i64(reference_tensors, "input.input_ids")?;
    if tokenized.shape.as_slice() != reference_ids.shape.as_slice() {
        return Err(format!(
            "end-to-end tokenizer shape mismatch: Rust {:?}, HF {:?}",
            tokenized.shape, reference_ids.shape
        ));
    }
    if tokenized.input_ids != reference_ids.values {
        let mismatch = tokenized
            .input_ids
            .iter()
            .zip(&reference_ids.values)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(0);
        return Err(format!(
            "end-to-end tokenizer IDs differ from HF at flat index {mismatch}: Rust {}, HF {}",
            tokenized.input_ids[mismatch], reference_ids.values[mismatch]
        ));
    }

    let response = runtime.encode_image_bytes_and_text_strings(
        &image_bytes,
        &tokenizer,
        &inputs.texts,
        false,
    )?;
    let actual = [
        (
            "pipeline.output.image_embedding.raw",
            "output.image_embedding.raw",
            burn_f32(response.image_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "pipeline.output.image_embedding.normalized",
            "output.image_embedding.normalized",
            burn_f32(response.normalized_image_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "pipeline.output.text_embedding.raw",
            "output.text_embedding.raw",
            burn_f32(response.text_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "pipeline.output.text_embedding.normalized",
            "output.text_embedding.normalized",
            burn_f32(response.normalized_text_embedding)?,
            TensorKind::Embedding,
        ),
        (
            "pipeline.output.logits_per_image",
            "output.logits_per_image",
            burn_f32(response.logits_per_image)?,
            TensorKind::Logit,
        ),
        (
            "pipeline.output.logits_per_text",
            "output.logits_per_text",
            burn_f32(response.logits_per_text)?,
            TensorKind::Logit,
        ),
        (
            "pipeline.output.probabilities_per_image",
            "output.probabilities_per_image",
            burn_f32(response.probabilities_per_image)?,
            TensorKind::Probability,
        ),
        (
            "pipeline.output.probabilities_per_text",
            "output.probabilities_per_text",
            burn_f32(response.probabilities_per_text)?,
            TensorKind::Probability,
        ),
    ];
    for (display_name, reference_name, actual, kind) in actual {
        let reference = read_f32(reference_tensors, reference_name)?;
        assert_tensor_parity(
            display_name,
            &actual,
            &reference,
            kind,
            manifest.tolerances.parity,
        )?;
    }

    eprintln!(
        "end-to-end encoded-image/string pipeline parity passed with tokenizer {}",
        tokenizer_path.display()
    );
    Ok(())
}

#[cfg(feature = "pipeline")]
fn adjacent_bundle_asset(bundle_path: &Path, suffix: &str) -> Result<PathBuf, String> {
    let file_name = bundle_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("bundle path is not valid UTF-8: {}", bundle_path.display()))?;
    let stem = file_name
        .strip_suffix(".bpk.parts.json")
        .or_else(|| file_name.strip_suffix(".bpk"))
        .ok_or_else(|| {
            format!(
                "cannot derive pipeline sidecar name from bundle {}",
                bundle_path.display()
            )
        })?;
    Ok(bundle_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}{suffix}")))
}

#[derive(Debug, Clone, Copy)]
enum TensorKind {
    Embedding,
    Logit,
    Probability,
}

fn read_manifest(path: &Path) -> Result<ReferenceManifest, String> {
    let bytes = fs::read(path).map_err(|err| {
        format!(
            "failed to read reference manifest '{}': {err}",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|err| {
        format!(
            "failed to parse reference manifest '{}': {err}",
            path.display()
        )
    })
}

fn load_request(path: &Path) -> Result<LoadRequest, String> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("bundle path is not valid UTF-8: {}", path.display()))?;
    if name.ends_with(".parts.json") {
        Ok(LoadRequest::from_parts_manifest(path, true))
    } else if name.ends_with(".bpk") {
        Ok(LoadRequest::from_bpk(path))
    } else {
        Err(format!(
            "{BUNDLE_ENV} must name a .bpk or .parts.json artifact, got {}",
            path.display()
        ))
    }
}

fn validate_runtime_config(
    manifest: &ReferenceManifest,
    config: &burn_siglip2::Siglip2Config,
) -> Result<(), String> {
    let expected = &manifest.model.resolved_config;
    let pairs = [
        ("image_size", config.image_size, expected.image_size),
        ("patch_size", config.patch_size, expected.patch_size),
        ("hidden_dim", config.hidden_dim, expected.hidden_size),
        (
            "intermediate_dim",
            config.intermediate_dim,
            expected.intermediate_size,
        ),
        ("num_layers", config.num_layers, expected.num_hidden_layers),
        ("num_heads", config.num_heads, expected.num_attention_heads),
        (
            "projection_dim",
            config.projection_dim,
            expected.projection_size,
        ),
        (
            "text_max_positions",
            config.text_max_positions,
            expected.text_max_positions,
        ),
        (
            "text_vocab_size",
            config.text_vocab_size,
            expected.text_vocab_size,
        ),
    ];
    for (name, actual, reference) in pairs {
        if actual != reference {
            return Err(format!(
                "bundle/reference config mismatch for {name}: bundle {actual}, reference {reference}"
            ));
        }
    }
    Ok(())
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<OwnedTensor<f32>, String> {
    let view = tensors
        .tensor(name)
        .map_err(|err| format!("missing reference tensor '{name}': {err}"))?;
    if view.dtype() != Dtype::F32 {
        return Err(format!(
            "reference tensor '{name}' must be F32, got {:?}",
            view.dtype()
        ));
    }
    let (chunks, remainder) = view.data().as_chunks::<4>();
    let values = chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if !remainder.is_empty() {
        return Err(format!("reference tensor '{name}' has truncated F32 data"));
    }
    Ok(OwnedTensor {
        shape: view.shape().to_vec(),
        values,
    })
}

fn read_i64(tensors: &SafeTensors<'_>, name: &str) -> Result<OwnedTensor<i64>, String> {
    let view = tensors
        .tensor(name)
        .map_err(|err| format!("missing reference tensor '{name}': {err}"))?;
    if view.dtype() != Dtype::I64 {
        return Err(format!(
            "reference tensor '{name}' must be I64, got {:?}",
            view.dtype()
        ));
    }
    let (chunks, remainder) = view.data().as_chunks::<8>();
    let values = chunks
        .iter()
        .map(|bytes| i64::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if !remainder.is_empty() {
        return Err(format!("reference tensor '{name}' has truncated I64 data"));
    }
    Ok(OwnedTensor {
        shape: view.shape().to_vec(),
        values,
    })
}

fn burn_f32<const D: usize>(tensor: Tensor<D>) -> Result<OwnedTensor<f32>, String> {
    let shape = tensor.shape().dims::<D>().to_vec();
    let values = tensor
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read Burn output tensor: {err:?}"))?;
    Ok(OwnedTensor { shape, values })
}

fn assert_tensor_parity(
    name: &str,
    actual: &OwnedTensor<f32>,
    reference: &OwnedTensor<f32>,
    kind: TensorKind,
    tolerances: ParityTolerances,
) -> Result<(), String> {
    if actual.shape != reference.shape {
        return Err(format!(
            "{name} shape mismatch: Burn {:?}, reference {:?}",
            actual.shape, reference.shape
        ));
    }
    let metrics = metrics(&actual.values, &reference.values)?;
    let (max_abs, max_rmse) = match kind {
        TensorKind::Embedding => (tolerances.embedding_max_abs, tolerances.embedding_rmse),
        TensorKind::Logit => (tolerances.logit_max_abs, tolerances.logit_rmse),
        TensorKind::Probability => (tolerances.probability_max_abs, tolerances.probability_rmse),
    };
    if metrics.max_abs > max_abs || metrics.rmse > max_rmse {
        return Err(format!(
            "{name} parity failed: max_abs={:.6e} (limit {:.6e}), rmse={:.6e} (limit {:.6e})",
            metrics.max_abs, max_abs, metrics.rmse, max_rmse
        ));
    }
    if matches!(kind, TensorKind::Embedding) {
        if actual.shape.len() != 2 {
            return Err(format!("embedding tensor '{name}' must be rank 2"));
        }
        let cosine = minimum_row_cosine(
            &actual.values,
            &reference.values,
            actual.shape[0],
            actual.shape[1],
        )?;
        if cosine < tolerances.embedding_min_cosine {
            return Err(format!(
                "{name} cosine parity failed: min={cosine:.8}, limit={:.8}",
                tolerances.embedding_min_cosine
            ));
        }
    }
    eprintln!(
        "{name}: max_abs={:.6e}, rmse={:.6e}",
        metrics.max_abs, metrics.rmse
    );
    Ok(())
}

fn metrics(actual: &[f32], reference: &[f32]) -> Result<Metrics, String> {
    if actual.len() != reference.len() {
        return Err(format!(
            "tensor length mismatch: {} != {}",
            actual.len(),
            reference.len()
        ));
    }
    if actual.is_empty() {
        return Err("cannot compare empty tensors".to_string());
    }
    let mut max_abs = 0.0f64;
    let mut sum_squared = 0.0f64;
    for (actual, reference) in actual.iter().zip(reference) {
        if !actual.is_finite() || !reference.is_finite() {
            return Err("parity comparison encountered a non-finite value".to_string());
        }
        let delta = f64::from(*actual) - f64::from(*reference);
        max_abs = max_abs.max(delta.abs());
        sum_squared += delta * delta;
    }
    Ok(Metrics {
        max_abs: max_abs as f32,
        rmse: (sum_squared / actual.len() as f64).sqrt() as f32,
    })
}

#[cfg(feature = "pipeline")]
fn largest_delta_index(actual: &[f32], reference: &[f32]) -> Result<usize, String> {
    if actual.len() != reference.len() || actual.is_empty() {
        return Err("cannot locate largest delta in differently sized or empty inputs".to_string());
    }
    let mut largest_index = 0;
    let mut largest_delta = 0.0f32;
    for (index, (actual, reference)) in actual.iter().zip(reference).enumerate() {
        let delta = (*actual - *reference).abs();
        if delta > largest_delta {
            largest_index = index;
            largest_delta = delta;
        }
    }
    Ok(largest_index)
}

#[cfg(feature = "pipeline")]
fn first_delta_samples(actual: &[f32], reference: &[f32], limit: usize) -> Result<String, String> {
    if actual.len() != reference.len() {
        return Err("cannot sample deltas from differently sized inputs".to_string());
    }
    let samples = actual
        .iter()
        .zip(reference)
        .enumerate()
        .filter(|(_, (actual, reference))| (*actual - *reference).abs() > 1.0e-7)
        .take(limit)
        .map(|(index, (actual, reference))| format!("{index}: {actual:.8} != {reference:.8}"))
        .collect::<Vec<_>>();
    if samples.is_empty() {
        Ok("none".to_string())
    } else {
        Ok(samples.join(", "))
    }
}

fn minimum_row_cosine(
    actual: &[f32],
    reference: &[f32],
    rows: usize,
    columns: usize,
) -> Result<f32, String> {
    if rows.checked_mul(columns) != Some(actual.len()) || actual.len() != reference.len() {
        return Err("invalid row shape for cosine comparison".to_string());
    }
    let mut minimum = 1.0f64;
    for row in 0..rows {
        let range = row * columns..(row + 1) * columns;
        let mut dot = 0.0f64;
        let mut norm_actual = 0.0f64;
        let mut norm_reference = 0.0f64;
        for (actual, reference) in actual[range.clone()].iter().zip(&reference[range]) {
            let actual = f64::from(*actual);
            let reference = f64::from(*reference);
            dot += actual * reference;
            norm_actual += actual * actual;
            norm_reference += reference * reference;
        }
        if norm_actual == 0.0 || norm_reference == 0.0 {
            return Err(format!("cannot compute cosine for zero-norm row {row}"));
        }
        minimum = minimum.min(dot / (norm_actual.sqrt() * norm_reference.sqrt()));
    }
    Ok(minimum as f32)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
