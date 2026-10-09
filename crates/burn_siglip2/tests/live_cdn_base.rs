#![cfg(all(feature = "bootstrap", feature = "flex", feature = "pipeline"))]

use std::{io::Cursor, io::Read, path::PathBuf};

use burn_siglip2::{
    LoadRequest, Siglip2BootstrapConfig, Siglip2ModelVariant, Siglip2Tokenizer, load_backend,
    resolve_or_bootstrap_siglip2_weights_with_config_and_progress,
};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const CACHE_ENV: &str = "BURN_SIGLIP2_LIVE_CDN_CACHE";
const EXPECTED_LOGITS: [f32; 3] = [-12.288_71, -13.581_768, -13.254_331];
const EXPECTED_PROBABILITIES: [f32; 3] = [4.603_405_6e-6, 1.263_317_6e-6, 1.752_736_2e-6];

/// This opt-in regression downloads roughly 790 MB and proves the public Base URL, every shard
/// digest, image preprocessing, tokenizer, both towers, and calibrated similarity outputs.
#[test]
#[ignore = "downloads and runs the public Base F16 CDN bundle"]
fn public_default_base_bundle_matches_reference() -> Result<(), String> {
    let explicit_cache = std::env::var_os(CACHE_ENV).map(PathBuf::from);
    let temporary_cache = if explicit_cache.is_none() {
        Some(tempfile::tempdir().map_err(|err| format!("failed to create test cache: {err}"))?)
    } else {
        None
    };
    let cache_root = explicit_cache.unwrap_or_else(|| {
        temporary_cache
            .as_ref()
            .expect("temporary cache exists")
            .path()
            .to_path_buf()
    });
    let config = Siglip2BootstrapConfig {
        cache_root: Some(cache_root),
        ..Siglip2BootstrapConfig::default()
    };

    let artifacts =
        resolve_or_bootstrap_siglip2_weights_with_config_and_progress(&config, |message| {
            eprintln!("[live-cdn-base] {message}")
        })
        .map_err(|err| err.to_string())?;
    if !artifacts.using_parts {
        return Err("public CDN bootstrap unexpectedly used a monolithic BPK".to_string());
    }

    let runtime = load_backend(LoadRequest::from_parts_manifest(
        &artifacts.parts_manifest_path,
        true,
    ))?;
    if runtime.model.config.supported_variant() != Some(Siglip2ModelVariant::BasePatch16_224) {
        return Err("public default did not load the exact Base patch16-224 profile".to_string());
    }
    if runtime.load_stats.part_count != 14 || runtime.load_stats.sha256_verified != 14 {
        return Err(format!(
            "unexpected verified shard evidence: parts={}, sha256_verified={}",
            runtime.load_stats.part_count, runtime.load_stats.sha256_verified
        ));
    }

    let tokenizer =
        Siglip2Tokenizer::from_file(&artifacts.tokenizer_json_path, &runtime.model.config)?;
    let encoded_image = deterministic_reference_png()?;
    let response = runtime.encode_image_bytes_and_text_strings(
        &encoded_image,
        &tokenizer,
        &[
            "this is a photo of a cat.",
            "this is a photo of a dog.",
            "this is a photo of an airplane.",
        ],
        false,
    )?;

    let image_embedding = response
        .normalized_image_embedding
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read image embedding: {err:?}"))?;
    let text_embeddings = response
        .normalized_text_embedding
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read text embeddings: {err:?}"))?;
    let logits = response
        .logits_per_image
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read logits: {err:?}"))?;
    let probabilities = response
        .probabilities_per_image
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read probabilities: {err:?}"))?;

    assert_unit_rows(&image_embedding, 1, 768, "image")?;
    assert_unit_rows(&text_embeddings, 3, 768, "text")?;
    assert_close(&logits, &EXPECTED_LOGITS, 5.0e-5, "logits")?;
    assert_close(
        &probabilities,
        &EXPECTED_PROBABILITIES,
        1.0e-8,
        "probabilities",
    )?;
    Ok(())
}

#[derive(Deserialize)]
struct CdnIndex {
    schema_version: u32,
    manifest_kind: String,
    bundles: Vec<CdnIndexEntry>,
}

#[derive(Deserialize)]
struct CdnIndexEntry {
    model_variant: String,
    manifest: String,
    sha256: String,
    upstream_model_id: String,
}

#[derive(Deserialize)]
struct CdnBundle {
    schema_version: u32,
    manifest_kind: String,
    model_variant: String,
    upstream_model_id: String,
    parts_manifest: String,
    files: Vec<CdnFile>,
}

#[derive(Deserialize)]
struct CdnFile {
    path: String,
    role: String,
    sha256: String,
}

/// Lightweight live coverage for the public index and every model-size bundle/inner manifest.
#[test]
#[ignore = "contacts the public SigLIP2 CDN"]
fn public_cdn_inventory_is_complete_and_hash_linked() -> Result<(), String> {
    let root = burn_siglip2::SIGLIP2_DEFAULT_CDN_ROOT_URL;
    let index_bytes = fetch_small_json(&format!("{root}/index.json"))?;
    let index: CdnIndex = serde_json::from_slice(&index_bytes)
        .map_err(|err| format!("failed to parse public CDN index: {err}"))?;
    if index.schema_version != 1 || index.manifest_kind != "siglip2_cdn_index" {
        return Err("public CDN index has an unsupported schema".to_string());
    }

    for variant in Siglip2ModelVariant::ALL {
        let entry = index
            .bundles
            .iter()
            .find(|entry| entry.model_variant == variant.model_size())
            .ok_or_else(|| format!("CDN index is missing {}", variant.model_size()))?;
        if entry.upstream_model_id != variant.hf_model_id() {
            return Err(format!(
                "CDN index model ID mismatch for {}",
                variant.model_size()
            ));
        }
        let bundle_bytes = fetch_small_json(&format!("{root}/{}", entry.manifest))?;
        assert_sha256(&bundle_bytes, &entry.sha256, "bundle manifest")?;
        let bundle: CdnBundle = serde_json::from_slice(&bundle_bytes)
            .map_err(|err| format!("failed to parse {} bundle: {err}", variant.model_size()))?;
        if bundle.schema_version != 1
            || bundle.manifest_kind != "siglip2_cdn_bundle"
            || bundle.model_variant != variant.model_size()
            || bundle.upstream_model_id != variant.hf_model_id()
        {
            return Err(format!(
                "public bundle identity mismatch for {}",
                variant.model_size()
            ));
        }
        let parts_entry = bundle
            .files
            .iter()
            .find(|file| file.role == "parts_manifest" && file.path == bundle.parts_manifest)
            .ok_or_else(|| {
                format!(
                    "public bundle is missing the parts manifest for {}",
                    variant.model_size()
                )
            })?;
        let parts_bytes = fetch_small_json(&format!(
            "{root}/{}/{}",
            variant.model_size(),
            parts_entry.path
        ))?;
        assert_sha256(&parts_bytes, &parts_entry.sha256, "parts manifest")?;
        let parts: burn_siglip2::Siglip2BpkPartsManifest = serde_json::from_slice(&parts_bytes)
            .map_err(|err| format!("failed to parse verified parts manifest: {err}"))?;
        let expected_parts = match variant {
            Siglip2ModelVariant::BasePatch16_224 => 14,
            Siglip2ModelVariant::LargePatch16_256 => 35,
            Siglip2ModelVariant::So400mPatch14_224 => 42,
        };
        if parts.parts.len() != expected_parts
            || parts.config != burn_siglip2::Siglip2Config::for_variant(variant)
        {
            return Err(format!(
                "public parts manifest mismatch for {}",
                variant.model_size()
            ));
        }
    }
    Ok(())
}

fn fetch_small_json(url: &str) -> Result<Vec<u8>, String> {
    const MAX_BYTES: u64 = 1024 * 1024;
    let response = ureq::get(url)
        .set("Origin", "https://aberration.technology")
        .call()
        .map_err(|err| format!("failed to fetch '{url}': {err}"))?;
    if response.header("Access-Control-Allow-Origin") != Some("*") {
        return Err(format!("public CDN response lacks wildcard CORS: {url}"));
    }
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed reading '{url}': {err}"))?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_BYTES {
        return Err(format!("public CDN JSON has an invalid length: {url}"));
    }
    Ok(bytes)
}

fn assert_sha256(bytes: &[u8], expected: &str, label: &str) -> Result<(), String> {
    let actual = hex::encode(Sha256::digest(bytes));
    if !actual.eq_ignore_ascii_case(expected.trim()) {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn deterministic_reference_png() -> Result<Vec<u8>, String> {
    let image = RgbImage::from_fn(83, 61, |x, y| {
        Rgb([
            ((x * 3 + y * 5 + 17) % 256) as u8,
            ((x * 7 + y * 11 + 29) % 256) as u8,
            ((x * 13 + y * 17 + 43) % 256) as u8,
        ])
    });
    let mut output = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut output, ImageFormat::Png)
        .map_err(|err| format!("failed to encode deterministic PNG: {err}"))?;
    Ok(output.into_inner())
}

fn assert_unit_rows(values: &[f32], rows: usize, columns: usize, name: &str) -> Result<(), String> {
    if values.len() != rows * columns {
        return Err(format!(
            "{name} embedding length {}, expected {}",
            values.len(),
            rows * columns
        ));
    }
    for (row, values) in values.chunks_exact(columns).enumerate() {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || (norm - 1.0).abs() > 1.0e-4 {
            return Err(format!("{name} row {row} has invalid L2 norm {norm}"));
        }
    }
    Ok(())
}

fn assert_close(
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
    name: &str,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{name} length {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if !max_abs.is_finite() || max_abs > tolerance {
        return Err(format!(
            "{name} max absolute error {max_abs} exceeds {tolerance}; actual={actual:?} expected={expected:?}"
        ));
    }
    Ok(())
}
