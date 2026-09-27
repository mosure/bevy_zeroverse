//! Browser WebGPU entry points for incremental CDN model loading and multimodal inference.

use std::{cell::RefCell, path::Path};

use burn::tensor::Tensor;
use burn_wgpu::{self as wgpu, WebGpu};
use js_sys::{Array, Function, JsString, Promise, Reflect, Uint8Array};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::JsFuture;

use crate::{
    Siglip2BpkPartsManifest, Siglip2ModelVariant, Siglip2PartsLoader, Siglip2Runtime,
    Siglip2Tokenizer, WebLoadLimits, validate_bpk_part_entry, validate_bpk_parts_layout,
    validate_manifest_for_web,
    wasm_schema::{
        EmbeddingResponse as WasmEmbeddingResponse, IMAGE_EMBEDDING_METHOD,
        RESPONSE_SCHEMA_VERSION as WASM_RESPONSE_SCHEMA_VERSION, SCORE_METHOD,
        ScoreResponse as WasmScoreResponse, TEXT_EMBEDDING_METHOD,
        serialize_response as serialize_response_json,
    },
};
use crate::{
    preprocess::SIGLIP2_MAX_ENCODED_IMAGE_BYTES,
    tokenizer::{
        SIGLIP2_MAX_TEXT_BATCH_BYTES, SIGLIP2_MAX_TEXT_BATCH_SIZE, SIGLIP2_MAX_TEXT_BYTES,
    },
};

type WasmBackend = WebGpu<f32, i32>;
type WasmDevice = wgpu::WgpuDevice;

const MAX_REMOTE_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
const MAX_REMOTE_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_REMOTE_TOKENIZER_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_REMOTE_URL_BYTES: u64 = 16 * 1024;
const MAX_MODEL_SIZE_BYTES: u64 = 64;
const MAX_REMOTE_BUNDLE_PAYLOAD_BYTES: u64 =
    4 * 1024 * 1024 * 1024 + MAX_REMOTE_TOKENIZER_BYTES + 3 * MAX_REMOTE_TOKENIZER_CONFIG_BYTES;

struct FetchedBytes {
    bytes: Vec<u8>,
    sha256: String,
    final_url: String,
}

thread_local! {
    static WEBGPU_RUNTIME_DEVICE: RefCell<Option<WasmDevice>> = const { RefCell::new(None) };
}

/// A browser-owned SigLIP2 runtime backed by WebGPU.
///
/// Use [`WasmSiglip2::create_default`], [`WasmSiglip2::create_from_model_size`], or
/// [`WasmSiglip2::create_from_bundle_manifest_url`] to fetch and apply one verified shard at a
/// time. This avoids retaining the complete model package in JavaScript or Wasm memory.
#[wasm_bindgen]
pub struct WasmSiglip2 {
    runtime: Siglip2Runtime<WasmBackend>,
    tokenizer: Option<Siglip2Tokenizer>,
}

#[derive(Deserialize)]
struct CdnBundleManifest {
    schema_version: u32,
    manifest_kind: String,
    model_family: String,
    model_variant: String,
    upstream_model_id: String,
    upstream_revision: String,
    storage_dtype: String,
    parts_manifest: String,
    payload_bytes: u64,
    files: Vec<CdnBundleFile>,
}

#[derive(Deserialize)]
struct CdnBundleFile {
    path: String,
    role: String,
    bytes: u64,
    sha256: String,
}

#[wasm_bindgen]
impl WasmSiglip2 {
    /// Return the canonical public CDN bundle URL used by [`WasmSiglip2::create_default`].
    #[wasm_bindgen(js_name = defaultBundleManifestUrl)]
    pub fn default_bundle_manifest_url() -> String {
        Siglip2ModelVariant::BasePatch16_224.cdn_bundle_manifest_url()
    }

    /// Download and initialize the Base patch16-224 bundle from the public CDN.
    #[wasm_bindgen(js_name = createDefault)]
    pub async fn create_default() -> Result<WasmSiglip2, JsValue> {
        let url = Self::default_bundle_manifest_url();
        create_from_bundle_manifest_url_expected(url, Some(Siglip2ModelVariant::BasePatch16_224))
            .await
    }

    /// Download a public CDN bundle by short (`base`, `large`, `so400m`) or canonical size name.
    #[wasm_bindgen(js_name = createFromModelSize)]
    pub async fn create_from_model_size(model_size: JsString) -> Result<WasmSiglip2, JsValue> {
        let model_size =
            copy_bounded_js_string_value(&model_size, MAX_MODEL_SIZE_BYTES, "model_size")?;
        let variant = Siglip2ModelVariant::from_model_size(&model_size).ok_or_else(|| {
            js_error(format!(
                "unsupported SigLIP2 model size '{model_size}'; expected base, large, so400m, base-patch16-224, large-patch16-256, or so400m-patch14-224"
            ))
        })?;
        let url = variant.cdn_bundle_manifest_url();
        create_from_bundle_manifest_url_expected(url, Some(variant)).await
    }

    /// Open a complete `bundle.manifest.json` produced by `bundle_siglip2_assets.sh`.
    ///
    /// The model parts are still fetched sequentially. Tokenizer files are additionally checked
    /// against the outer bundle inventory before they are parsed.
    #[wasm_bindgen(js_name = createFromBundleManifestUrl)]
    pub async fn create_from_bundle_manifest_url(
        bundle_manifest_url: JsString,
    ) -> Result<WasmSiglip2, JsValue> {
        let bundle_manifest_url = copy_bounded_js_string_value(
            &bundle_manifest_url,
            MAX_REMOTE_URL_BYTES,
            "bundle_manifest_url",
        )?;
        create_from_bundle_manifest_url_expected(bundle_manifest_url, None).await
    }

    /// Fetch a CDN parts manifest and its shards sequentially, verifying byte lengths and SHA-256.
    ///
    /// This compatibility entry point is deliberately image-only. Supplying the legacy tokenizer
    /// URL arguments is rejected because they carry no trusted checksums; use the hash-bearing or
    /// complete bundle entry point for text inference.
    #[wasm_bindgen(js_name = createFromManifestUrl)]
    pub async fn create_from_manifest_url(
        manifest_url: JsString,
        tokenizer_url: Option<JsString>,
        tokenizer_config_url: Option<JsString>,
    ) -> Result<WasmSiglip2, JsValue> {
        if tokenizer_url.is_some() || tokenizer_config_url.is_some() {
            return Err(js_error(
                "unverified tokenizer URLs are disabled for createFromManifestUrl; use createFromManifestUrlWithTokenizerHashes or createFromBundleManifestUrl",
            ));
        }
        let manifest_url =
            copy_bounded_js_string_value(&manifest_url, MAX_REMOTE_URL_BYTES, "manifest_url")?;
        // Fail before transferring model metadata or shards when no usable WebGPU device exists.
        webgpu_device().await?;
        let manifest_fetch =
            fetch_url_bytes_limited(&manifest_url, None, MAX_REMOTE_MANIFEST_BYTES).await?;
        let manifest: Siglip2BpkPartsManifest = serde_json::from_slice(&manifest_fetch.bytes)
            .map_err(|err| js_error(format!("failed to parse manifest '{manifest_url}': {err}")))?;
        let runtime = load_remote_parts(manifest, &manifest_fetch.final_url).await?;
        Ok(Self {
            runtime,
            tokenizer: None,
        })
    }

    /// Fetch a verified model plus tokenizer files whose SHA-256 values are supplied by the caller.
    #[wasm_bindgen(js_name = createFromManifestUrlWithTokenizerHashes)]
    pub async fn create_from_manifest_url_with_tokenizer_hashes(
        manifest_url: JsString,
        tokenizer_url: JsString,
        tokenizer_sha256: JsString,
        tokenizer_config_url: JsString,
        tokenizer_config_sha256: JsString,
    ) -> Result<WasmSiglip2, JsValue> {
        let manifest_url =
            copy_bounded_js_string_value(&manifest_url, MAX_REMOTE_URL_BYTES, "manifest_url")?;
        let tokenizer_url =
            copy_bounded_js_string_value(&tokenizer_url, MAX_REMOTE_URL_BYTES, "tokenizer_url")?;
        let tokenizer_sha256 =
            copy_bounded_js_string_value(&tokenizer_sha256, 64, "tokenizer_sha256")?;
        let tokenizer_config_url = copy_bounded_js_string_value(
            &tokenizer_config_url,
            MAX_REMOTE_URL_BYTES,
            "tokenizer_config_url",
        )?;
        let tokenizer_config_sha256 =
            copy_bounded_js_string_value(&tokenizer_config_sha256, 64, "tokenizer_config_sha256")?;
        validate_sha256(&tokenizer_sha256, "tokenizer_sha256")?;
        validate_sha256(&tokenizer_config_sha256, "tokenizer_config_sha256")?;

        // Fail before transferring model/tokenizer assets when no usable WebGPU device exists.
        webgpu_device().await?;
        let manifest_fetch =
            fetch_url_bytes_limited(&manifest_url, None, MAX_REMOTE_MANIFEST_BYTES).await?;
        let manifest: Siglip2BpkPartsManifest = serde_json::from_slice(&manifest_fetch.bytes)
            .map_err(|err| js_error(format!("failed to parse manifest '{manifest_url}': {err}")))?;
        let runtime = load_remote_parts(manifest, &manifest_fetch.final_url).await?;
        let tokenizer = load_verified_tokenizer_urls(
            &tokenizer_url,
            &tokenizer_sha256,
            &tokenizer_config_url,
            &tokenizer_config_sha256,
            &runtime.model.config,
        )
        .await?;
        Ok(Self {
            runtime,
            tokenizer: Some(tokenizer),
        })
    }

    /// Construct from a manifest string and already-fetched `Uint8Array` shards.
    ///
    /// This is useful for service-worker or IndexedDB caches. For lowest peak Wasm memory, prefer
    /// `createFromManifestUrl`, which fetches and releases each shard in turn.
    #[wasm_bindgen(js_name = createFromParts)]
    pub async fn create_from_parts(
        manifest_json: JsString,
        #[wasm_bindgen(unchecked_param_type = "Uint8Array[]")] parts: Array,
        tokenizer_json: Option<Uint8Array>,
        tokenizer_config_json: Option<Uint8Array>,
    ) -> Result<WasmSiglip2, JsValue> {
        let manifest_json = copy_bounded_js_string_value(
            &manifest_json,
            MAX_REMOTE_MANIFEST_BYTES,
            "parts manifest JSON",
        )?;
        if tokenizer_json.is_some() != tokenizer_config_json.is_some() {
            return Err(js_error(
                "tokenizer_json and tokenizer_config_json must either both be supplied or both be omitted",
            ));
        }
        validate_optional_uint8_array(
            tokenizer_json.as_ref(),
            MAX_REMOTE_TOKENIZER_BYTES,
            "tokenizer_json",
        )?;
        validate_optional_uint8_array(
            tokenizer_config_json.as_ref(),
            MAX_REMOTE_TOKENIZER_CONFIG_BYTES,
            "tokenizer_config_json",
        )?;
        if !Array::is_array(parts.as_ref()) {
            return Err(js_error("parts must be supplied as a JavaScript Array"));
        }
        let manifest: Siglip2BpkPartsManifest = serde_json::from_str(&manifest_json)
            .map_err(|err| js_error(format!("failed to parse manifest: {err}")))?;
        validate_remote_manifest(&manifest)?;
        if parts.length() as usize != manifest.parts.len() {
            return Err(js_error(format!(
                "manifest declares {} parts but JavaScript supplied {}",
                manifest.parts.len(),
                parts.length()
            )));
        }
        validate_parts_array(&parts, &manifest)?;

        let device = webgpu_device().await?;
        let mut loader = Siglip2PartsLoader::<WasmBackend>::new(manifest, &device, true)
            .map_err(|err| js_error(format!("failed to initialize sharded model: {err}")))?;
        for index in 0..parts.length() as usize {
            let array = parts.get(index as u32).unchecked_into::<Uint8Array>();
            let bytes = copy_bounded_uint8_array(
                &array,
                loader.manifest().parts[index].bytes,
                &format!("part {index}"),
            )?;
            loader
                .apply_part(index, &bytes, &device)
                .map_err(|err| js_error(format!("failed to apply part {index}: {err}")))?;
        }
        let (model, load_stats) = loader
            .finish()
            .map_err(|err| js_error(format!("failed to finish sharded model load: {err}")))?;
        let tokenizer_json = tokenizer_json
            .as_ref()
            .map(|array| {
                copy_bounded_uint8_array(array, MAX_REMOTE_TOKENIZER_BYTES, "tokenizer_json")
            })
            .transpose()?;
        let tokenizer_config_json = tokenizer_config_json
            .as_ref()
            .map(|array| {
                copy_bounded_uint8_array(
                    array,
                    MAX_REMOTE_TOKENIZER_CONFIG_BYTES,
                    "tokenizer_config_json",
                )
            })
            .transpose()?;
        let tokenizer = tokenizer_json
            .as_deref()
            .map(|bytes| {
                Siglip2Tokenizer::from_bytes(bytes, tokenizer_config_json.as_deref(), &model.config)
            })
            .transpose()
            .map_err(js_error)?;
        Ok(Self {
            runtime: Siglip2Runtime {
                model,
                device,
                load_stats,
            },
            tokenizer,
        })
    }

    /// Decode and preprocess one encoded image and return its raw and normalized embedding.
    #[wasm_bindgen(js_name = encodeImageBytesJson)]
    pub async fn encode_image_bytes_json(
        &self,
        encoded_image: Uint8Array,
    ) -> Result<String, JsValue> {
        let encoded_image = copy_encoded_image(&encoded_image, "encodeImageBytesJson")?;
        let response = self
            .runtime
            .encode_image_bytes(&encoded_image, false)
            .map_err(js_error)?;
        embedding_response_json(IMAGE_EMBEDDING_METHOD, response.embedding).await
    }

    /// Tokenize one or more texts and return their raw and normalized embeddings.
    ///
    /// The fixed-resolution SigLIP2 inference default intentionally omits the attention mask.
    #[wasm_bindgen(js_name = encodeTextsJson)]
    pub async fn encode_texts_json(
        &self,
        #[wasm_bindgen(unchecked_param_type = "string[]")] texts: Array,
    ) -> Result<String, JsValue> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| js_error("this runtime was created without a tokenizer"))?;
        let texts = copy_bounded_texts(&texts, "encodeTextsJson")?;
        let response = self
            .runtime
            .encode_text_strings(tokenizer, &texts, false)
            .map_err(js_error)?;
        embedding_response_json(TEXT_EMBEDDING_METHOD, response.embedding).await
    }

    /// Decode and preprocess one encoded image, tokenize the supplied texts, and return SigLIP2
    /// embeddings, logits, and sigmoid probabilities as JSON.
    #[wasm_bindgen(js_name = scoreImageTextsJson)]
    pub async fn score_image_texts_json(
        &self,
        encoded_image: Uint8Array,
        #[wasm_bindgen(unchecked_param_type = "string[]")] texts: Array,
    ) -> Result<String, JsValue> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| js_error("this runtime was created without a tokenizer"))?;
        let encoded_image = copy_encoded_image(&encoded_image, "scoreImageTextsJson")?;
        let texts = copy_bounded_texts(&texts, "scoreImageTextsJson")?;
        let response = self
            .runtime
            .encode_image_bytes_and_text_strings(&encoded_image, tokenizer, &texts, false)
            .map_err(js_error)?;
        let image_embedding_shape = response.image_embedding.shape().dims::<2>();
        let text_embedding_shape = response.text_embedding.shape().dims::<2>();
        let logits_shape = response.logits_per_image.shape().dims::<2>();
        let raw_image_embedding =
            tensor_values(response.image_embedding, "raw_image_embedding").await?;
        let normalized_image_embedding = tensor_values(
            response.normalized_image_embedding,
            "normalized_image_embedding",
        )
        .await?;
        let raw_text_embedding =
            tensor_values(response.text_embedding, "raw_text_embedding").await?;
        let normalized_text_embedding = tensor_values(
            response.normalized_text_embedding,
            "normalized_text_embedding",
        )
        .await?;
        let logits_per_image = tensor_values(response.logits_per_image, "logits_per_image").await?;
        let probabilities_per_image =
            tensor_values(response.probabilities_per_image, "probabilities_per_image").await?;
        serialize_response(&WasmScoreResponse {
            schema_version: WASM_RESPONSE_SCHEMA_VERSION,
            method: SCORE_METHOD,
            image_embedding_shape,
            text_embedding_shape,
            logits_shape,
            raw_image_embedding,
            normalized_image_embedding,
            raw_text_embedding,
            normalized_text_embedding,
            logits_per_image,
            probabilities_per_image,
        })
    }

    #[wasm_bindgen(getter, js_name = loadedPartCount)]
    pub fn loaded_part_count(&self) -> usize {
        self.runtime.load_stats.part_count
    }

    #[wasm_bindgen(getter, js_name = loadedBytes)]
    pub fn loaded_bytes(&self) -> u64 {
        self.runtime.load_stats.loaded_bytes
    }

    /// Canonical SHA-256 identity of the complete tensor set accepted by the loader.
    ///
    /// Unlike the source BPK payload checksum, this value is stable across monolithic and
    /// canonically sharded representations of the same stored weights. Browser embedding caches
    /// should include it in their cache key.
    #[wasm_bindgen(getter, js_name = loadedWeightSha256)]
    pub fn loaded_weight_sha256(&self) -> String {
        self.runtime.load_stats.loaded_weight_sha256.clone()
    }

    /// SHA-256 identity of the exact tokenizer and tokenizer-config bytes parsed by this runtime.
    ///
    /// Image-only runtimes created with `createFromManifestUrl` return `undefined`.
    #[wasm_bindgen(getter, js_name = tokenizerSha256)]
    pub fn tokenizer_sha256(&self) -> Option<String> {
        self.tokenizer
            .as_ref()
            .map(|tokenizer| tokenizer.artifact_sha256().to_string())
    }

    /// Immutable upstream model identifier carried by the validated production artifact.
    #[wasm_bindgen(getter, js_name = upstreamModelId)]
    pub fn upstream_model_id(&self) -> Option<String> {
        self.runtime
            .load_stats
            .artifact
            .as_ref()
            .map(|artifact| artifact.upstream_model_id.clone())
    }

    /// Full upstream git revision carried by the validated production artifact.
    #[wasm_bindgen(getter, js_name = upstreamRevision)]
    pub fn upstream_revision(&self) -> Option<String> {
        self.runtime
            .load_stats
            .artifact
            .as_ref()
            .map(|artifact| artifact.upstream_revision.clone())
    }

    /// Canonical CDN/model-size identifier for the loaded production checkpoint.
    #[wasm_bindgen(getter, js_name = modelSize)]
    pub fn model_size(&self) -> String {
        self.runtime
            .model
            .config
            .supported_variant()
            .map(|variant| variant.model_size().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Square image resolution expected by the loaded checkpoint.
    #[wasm_bindgen(getter, js_name = imageSize)]
    pub fn image_size(&self) -> usize {
        self.runtime.model.config.image_size
    }

    /// Width of image and text embeddings returned by this checkpoint.
    #[wasm_bindgen(getter, js_name = embeddingSize)]
    pub fn embedding_size(&self) -> usize {
        self.runtime.model.config.projection_dim
    }
}

async fn create_from_bundle_manifest_url_expected(
    bundle_manifest_url: String,
    expected_variant: Option<Siglip2ModelVariant>,
) -> Result<WasmSiglip2, JsValue> {
    // Device initialization is intentionally first: a browser without a working adapter should
    // not download tens of megabytes of metadata/tokenizer data, let alone model shards.
    webgpu_device().await?;
    let fetched =
        fetch_url_bytes_limited(&bundle_manifest_url, None, MAX_REMOTE_MANIFEST_BYTES).await?;
    let bundle: CdnBundleManifest = serde_json::from_slice(&fetched.bytes).map_err(|err| {
        js_error(format!(
            "failed to parse CDN bundle manifest '{bundle_manifest_url}': {err}"
        ))
    })?;
    validate_cdn_bundle(&bundle)?;
    if let Some(expected) = expected_variant {
        validate_expected_bundle_variant(&bundle, expected)?;
    }

    let bundle_url = fetched.final_url;
    let parts_entry = bundle
        .files
        .iter()
        .find(|file| file.path == bundle.parts_manifest && file.role == "parts_manifest")
        .ok_or_else(|| js_error("CDN bundle has no parts manifest inventory entry"))?;
    let parts_fetch = fetch_verified_bundle_file(&bundle_url, parts_entry).await?;
    let parts_manifest: Siglip2BpkPartsManifest = serde_json::from_slice(&parts_fetch.bytes)
        .map_err(|err| js_error(format!("failed to parse verified parts manifest: {err}")))?;
    validate_bundle_parts_inventory(&bundle, &parts_manifest)?;
    if let Some(expected) = expected_variant {
        validate_expected_parts_variant(&parts_manifest, expected)?;
    }

    // Validate the comparatively small text assets before transferring shards or allocating the
    // complete WebGPU model. This makes a missing/corrupt tokenizer fail early.
    let tokenizer_entry = bundle
        .files
        .iter()
        .find(|file| file.role == "tokenizer")
        .ok_or_else(|| js_error("CDN bundle has no tokenizer file"))?;
    let tokenizer_config_entry = bundle.files.iter().find(|file| {
        file.role == "tokenizer_sidecar" && file.path.ends_with(".tokenizer_config.json")
    });
    let tokenizer_json = fetch_verified_bundle_file(&bundle_url, tokenizer_entry).await?;
    let tokenizer_config_json = match tokenizer_config_entry {
        Some(entry) => Some(fetch_verified_bundle_file(&bundle_url, entry).await?),
        None => None,
    };
    let tokenizer = Siglip2Tokenizer::from_bytes(
        &tokenizer_json.bytes,
        tokenizer_config_json
            .as_ref()
            .map(|fetched| fetched.bytes.as_slice()),
        &parts_manifest.config,
    )
    .map_err(js_error)?;
    let runtime = load_remote_parts(parts_manifest, &bundle_url).await?;
    Ok(WasmSiglip2 {
        runtime,
        tokenizer: Some(tokenizer),
    })
}

async fn embedding_response_json(
    method: &'static str,
    raw_embedding: Tensor<WasmBackend, 2>,
) -> Result<String, JsValue> {
    let shape = raw_embedding.shape().dims::<2>();
    let normalized_embedding =
        crate::Siglip2Model::<WasmBackend>::normalize_embeddings(raw_embedding.clone());
    let raw_embedding = tensor_values(raw_embedding, "raw_embedding").await?;
    let normalized_embedding = tensor_values(normalized_embedding, "normalized_embedding").await?;
    serialize_response(&WasmEmbeddingResponse {
        schema_version: WASM_RESPONSE_SCHEMA_VERSION,
        method,
        shape,
        raw_embedding,
        normalized_embedding,
    })
}

async fn tensor_values(
    tensor: Tensor<WasmBackend, 2>,
    field: &'static str,
) -> Result<Vec<f32>, JsValue> {
    let values = tensor
        .into_data_async()
        .await
        .map_err(|err| js_error(format!("failed to read WebGPU output: {err:?}")))?
        .to_vec::<f32>()
        .map_err(|err| js_error(format!("failed to decode WebGPU output: {err:?}")))?;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(js_error(format!(
            "WebGPU output '{field}' contains non-finite value {value} at index {index}"
        )));
    }
    Ok(values)
}

fn serialize_response(response: &impl serde::Serialize) -> Result<String, JsValue> {
    serialize_response_json(response)
        .map_err(|err| js_error(format!("failed to serialize inference response: {err}")))
}

fn validate_remote_manifest(manifest: &Siglip2BpkPartsManifest) -> Result<(), JsValue> {
    validate_manifest_for_web(manifest, WebLoadLimits::default()).map_err(js_error)?;
    // Use a synthetic safe file name because the remote URL itself is not a filesystem path.
    let logical_path = Path::new("remote.bpk.parts.json");
    validate_bpk_parts_layout(logical_path, manifest).map_err(js_error)?;
    for entry in &manifest.parts {
        validate_bpk_part_entry(logical_path, entry).map_err(js_error)?;
        if entry.bytes == 0 {
            return Err(js_error(format!(
                "remote part '{}' must declare a non-zero byte length",
                entry.path
            )));
        }
        if entry.sha256.trim().is_empty() {
            return Err(js_error(format!(
                "remote part '{}' must declare a SHA-256 checksum",
                entry.path
            )));
        }
    }
    Ok(())
}

fn validate_cdn_bundle(bundle: &CdnBundleManifest) -> Result<(), JsValue> {
    if bundle.schema_version != 1
        || bundle.manifest_kind != "siglip2_cdn_bundle"
        || bundle.model_family != "siglip2"
    {
        return Err(js_error("unsupported SigLIP2 CDN bundle manifest"));
    }
    if bundle.storage_dtype != "f16" {
        return Err(js_error(format!(
            "browser CDN bundle must use f16 storage, got '{}'",
            bundle.storage_dtype
        )));
    }
    validate_bundle_file_name(&bundle.parts_manifest)?;
    if bundle.files.is_empty() {
        return Err(js_error("CDN bundle inventory is empty"));
    }
    if bundle.payload_bytes > MAX_REMOTE_BUNDLE_PAYLOAD_BYTES {
        return Err(js_error(format!(
            "CDN bundle payload {} exceeds browser limit {MAX_REMOTE_BUNDLE_PAYLOAD_BYTES}",
            bundle.payload_bytes
        )));
    }
    let mut unique = std::collections::BTreeSet::new();
    let mut total = 0u64;
    let mut parts_manifest_count = 0usize;
    let mut model_part_count = 0usize;
    let mut tokenizer_count = 0usize;
    let mut tokenizer_config_count = 0usize;
    for file in &bundle.files {
        validate_bundle_file_name(&file.path)?;
        if !unique.insert(file.path.as_str()) {
            return Err(js_error(format!(
                "duplicate CDN bundle inventory path '{}'",
                file.path
            )));
        }
        if file.bytes == 0 {
            return Err(js_error(format!(
                "CDN bundle file '{}' has zero declared bytes",
                file.path
            )));
        }
        let limit = bundle_file_limit(file)?;
        if file.bytes > limit {
            return Err(js_error(format!(
                "CDN bundle file '{}' role '{}' declares {} bytes, exceeding browser limit {limit}",
                file.path, file.role, file.bytes
            )));
        }
        let checksum = file.sha256.trim();
        if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(js_error(format!(
                "CDN bundle file '{}' has an invalid SHA-256",
                file.path
            )));
        }
        total = total
            .checked_add(file.bytes)
            .ok_or_else(|| js_error("CDN bundle payload byte count overflow"))?;
        match file.role.as_str() {
            "model_part" => model_part_count += 1,
            "parts_manifest" => {
                parts_manifest_count += 1;
                if file.path != bundle.parts_manifest {
                    return Err(js_error(format!(
                        "CDN bundle parts manifest role points at '{}', expected '{}'",
                        file.path, bundle.parts_manifest
                    )));
                }
            }
            "tokenizer" => tokenizer_count += 1,
            "tokenizer_sidecar" if file.path.ends_with(".tokenizer_config.json") => {
                tokenizer_config_count += 1;
            }
            "tokenizer_sidecar" | "image_preprocessor" => {}
            _ => unreachable!("bundle_file_limit rejects unsupported roles"),
        }
    }
    if parts_manifest_count != 1 {
        return Err(js_error(format!(
            "CDN bundle must contain exactly one parts_manifest file, got {parts_manifest_count}"
        )));
    }
    if model_part_count == 0 {
        return Err(js_error("CDN bundle contains no model_part files"));
    }
    if tokenizer_count != 1 {
        return Err(js_error(format!(
            "CDN bundle must contain exactly one tokenizer file, got {tokenizer_count}"
        )));
    }
    if tokenizer_config_count != 1 {
        return Err(js_error(format!(
            "CDN bundle must contain exactly one tokenizer config sidecar, got {tokenizer_config_count}"
        )));
    }
    if total != bundle.payload_bytes {
        return Err(js_error(format!(
            "CDN bundle payload byte mismatch: manifest={}, inventory={total}",
            bundle.payload_bytes
        )));
    }
    Ok(())
}

fn validate_expected_bundle_variant(
    bundle: &CdnBundleManifest,
    expected: Siglip2ModelVariant,
) -> Result<(), JsValue> {
    if bundle.model_variant != expected.model_size()
        || bundle.upstream_model_id != expected.hf_model_id()
    {
        return Err(js_error(format!(
            "requested SigLIP2 model size '{}' but CDN bundle identifies '{}' / '{}'",
            expected.model_size(),
            bundle.model_variant,
            bundle.upstream_model_id
        )));
    }
    Ok(())
}

fn validate_expected_parts_variant(
    manifest: &Siglip2BpkPartsManifest,
    expected: Siglip2ModelVariant,
) -> Result<(), JsValue> {
    let artifact = manifest
        .artifact
        .as_ref()
        .ok_or_else(|| js_error("verified inner parts manifest is missing artifact provenance"))?;
    if manifest.config != crate::Siglip2Config::for_variant(expected)
        || artifact.model_variant != expected.model_size()
        || artifact.upstream_model_id != expected.hf_model_id()
    {
        return Err(js_error(format!(
            "requested SigLIP2 model size '{}' but verified parts identify '{}' / '{}' with a different config",
            expected.model_size(),
            artifact.model_variant,
            artifact.upstream_model_id
        )));
    }
    Ok(())
}

fn validate_bundle_parts_inventory(
    bundle: &CdnBundleManifest,
    manifest: &Siglip2BpkPartsManifest,
) -> Result<(), JsValue> {
    validate_remote_manifest(manifest)?;
    let artifact = manifest
        .artifact
        .as_ref()
        .ok_or_else(|| js_error("verified inner parts manifest is missing artifact provenance"))?;
    if bundle.model_variant != artifact.model_variant
        || bundle.upstream_model_id != artifact.upstream_model_id
        || bundle.upstream_revision != artifact.upstream_revision
        || bundle.storage_dtype != artifact.storage_dtype
    {
        return Err(js_error(format!(
            "CDN bundle provenance does not exactly match inner parts manifest artifact (outer={}/{}/{}/{}, inner={}/{}/{}/{})",
            bundle.model_variant,
            bundle.upstream_model_id,
            bundle.upstream_revision,
            bundle.storage_dtype,
            artifact.model_variant,
            artifact.upstream_model_id,
            artifact.upstream_revision,
            artifact.storage_dtype
        )));
    }
    let outer_model_parts = bundle
        .files
        .iter()
        .filter(|file| file.role == "model_part")
        .count();
    if outer_model_parts != manifest.parts.len() {
        return Err(js_error(format!(
            "CDN bundle declares {outer_model_parts} model parts but inner manifest declares {}",
            manifest.parts.len()
        )));
    }
    for part in &manifest.parts {
        let outer = bundle
            .files
            .iter()
            .find(|file| file.path == part.path && file.role == "model_part")
            .ok_or_else(|| {
                js_error(format!(
                    "parts manifest entry '{}' is absent from the CDN bundle inventory",
                    part.path
                ))
            })?;
        if outer.bytes != part.bytes || !outer.sha256.eq_ignore_ascii_case(part.sha256.trim()) {
            return Err(js_error(format!(
                "CDN bundle and parts manifest disagree for '{}'",
                part.path
            )));
        }
    }
    Ok(())
}

fn bundle_file_limit(file: &CdnBundleFile) -> Result<u64, JsValue> {
    match file.role.as_str() {
        "model_part" => Ok(WebLoadLimits::default().max_part_bytes),
        "parts_manifest" => Ok(MAX_REMOTE_MANIFEST_BYTES),
        "tokenizer" => Ok(MAX_REMOTE_TOKENIZER_BYTES),
        "tokenizer_sidecar" if file.path.ends_with(".tokenizer_config.json") => {
            Ok(MAX_REMOTE_TOKENIZER_CONFIG_BYTES)
        }
        "tokenizer_sidecar" => Ok(MAX_REMOTE_TOKENIZER_BYTES),
        "image_preprocessor" => Ok(MAX_REMOTE_TOKENIZER_CONFIG_BYTES),
        role => Err(js_error(format!(
            "CDN bundle file '{}' has unsupported role '{role}'",
            file.path
        ))),
    }
}

fn validate_bundle_file_name(value: &str) -> Result<(), JsValue> {
    if value.is_empty()
        || value.contains('/')
        || value.contains('\\')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || value == "."
        || value == ".."
    {
        return Err(js_error(format!("unsafe CDN bundle file name '{value}'")));
    }
    Ok(())
}

async fn fetch_verified_bundle_file(
    bundle_manifest_url: &str,
    entry: &CdnBundleFile,
) -> Result<FetchedBytes, JsValue> {
    let url = resolve_url(bundle_manifest_url, &entry.path)?;
    let fetched =
        fetch_url_bytes_limited(&url, Some(entry.bytes), bundle_file_limit(entry)?).await?;
    if fetched.bytes.len() as u64 != entry.bytes {
        return Err(js_error(format!(
            "CDN file '{}' byte mismatch: expected {}, got {}",
            entry.path,
            entry.bytes,
            fetched.bytes.len()
        )));
    }
    if !fetched.sha256.eq_ignore_ascii_case(entry.sha256.trim()) {
        return Err(js_error(format!(
            "CDN file '{}' checksum mismatch: expected {}, got {}",
            entry.path, entry.sha256, fetched.sha256
        )));
    }
    Ok(fetched)
}

async fn load_remote_parts(
    manifest: Siglip2BpkPartsManifest,
    manifest_url: &str,
) -> Result<Siglip2Runtime<WasmBackend>, JsValue> {
    validate_remote_manifest(&manifest)?;
    let device = webgpu_device().await?;
    let mut loader = Siglip2PartsLoader::<WasmBackend>::new(manifest, &device, true)
        .map_err(|err| js_error(format!("failed to initialize sharded model: {err}")))?;
    for index in 0..loader.manifest().parts.len() {
        let part_name = loader.manifest().parts[index].path.clone();
        let part_bytes = loader.manifest().parts[index].bytes;
        let expected_sha256 = loader.manifest().parts[index].sha256.clone();
        let part_url = resolve_url(manifest_url, &part_name)?;
        let fetched = fetch_url_bytes_limited(&part_url, Some(part_bytes), part_bytes).await?;
        if !fetched.sha256.eq_ignore_ascii_case(expected_sha256.trim()) {
            return Err(js_error(format!(
                "model part '{part_name}' checksum mismatch: expected {expected_sha256}, got {}",
                fetched.sha256
            )));
        }
        loader
            .apply_part(index, &fetched.bytes, &device)
            .map_err(|err| js_error(format!("failed to apply '{part_url}': {err}")))?;
        // `fetched` is dropped here before the next network request.
    }
    let (model, load_stats) = loader
        .finish()
        .map_err(|err| js_error(format!("failed to finish sharded model load: {err}")))?;
    Ok(Siglip2Runtime {
        model,
        device,
        load_stats,
    })
}

async fn load_verified_tokenizer_urls(
    tokenizer_url: &str,
    tokenizer_sha256: &str,
    tokenizer_config_url: &str,
    tokenizer_config_sha256: &str,
    config: &crate::Siglip2Config,
) -> Result<Siglip2Tokenizer, JsValue> {
    validate_sha256(tokenizer_sha256, "tokenizer_sha256")?;
    validate_sha256(tokenizer_config_sha256, "tokenizer_config_sha256")?;
    let tokenizer_fetch =
        fetch_url_bytes_limited(tokenizer_url, None, MAX_REMOTE_TOKENIZER_BYTES).await?;
    verify_fetched_sha256(&tokenizer_fetch, tokenizer_sha256, "tokenizer")?;
    let tokenizer_config_fetch = fetch_url_bytes_limited(
        tokenizer_config_url,
        None,
        MAX_REMOTE_TOKENIZER_CONFIG_BYTES,
    )
    .await?;
    verify_fetched_sha256(
        &tokenizer_config_fetch,
        tokenizer_config_sha256,
        "tokenizer config",
    )?;
    Siglip2Tokenizer::from_bytes(
        &tokenizer_fetch.bytes,
        Some(&tokenizer_config_fetch.bytes),
        config,
    )
    .map_err(js_error)
}

fn validate_sha256(value: &str, field: &str) -> Result<(), JsValue> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(js_error(format!(
            "{field} must be a 64-character hexadecimal SHA-256"
        )));
    }
    Ok(())
}

fn verify_fetched_sha256(
    fetched: &FetchedBytes,
    expected: &str,
    field: &str,
) -> Result<(), JsValue> {
    validate_sha256(expected, &format!("{field}_sha256"))?;
    if !fetched.sha256.eq_ignore_ascii_case(expected.trim()) {
        return Err(js_error(format!(
            "{field} checksum mismatch: expected {expected}, got {}",
            fetched.sha256
        )));
    }
    Ok(())
}

fn copy_bounded_js_string(value: &JsValue, max_bytes: u64, field: &str) -> Result<String, JsValue> {
    if !value.is_string() {
        return Err(js_error(format!("{field} must be a JavaScript string")));
    }
    let code_units = u64::from(JsString::from(value.clone()).length());
    // UTF-8 is never shorter than a JavaScript string's UTF-16 code-unit count. This rejects a
    // definitely oversized input before wasm-bindgen copies it into a Rust String.
    if code_units > max_bytes {
        return Err(js_error(format!(
            "{field} has {code_units} UTF-16 code units, exceeding pre-copy limit {max_bytes}"
        )));
    }
    let value = value
        .as_string()
        .ok_or_else(|| js_error(format!("{field} must be a JavaScript string")))?;
    if value.len() as u64 > max_bytes {
        return Err(js_error(format!(
            "{field} has {} UTF-8 bytes, exceeding browser limit {max_bytes}",
            value.len()
        )));
    }
    Ok(value)
}

fn copy_bounded_js_string_value(
    value: &JsString,
    max_bytes: u64,
    field: &str,
) -> Result<String, JsValue> {
    let value: &JsValue = value.as_ref();
    copy_bounded_js_string(value, max_bytes, field)
}

fn validate_optional_uint8_array(
    value: Option<&Uint8Array>,
    max_bytes: u64,
    field: &str,
) -> Result<(), JsValue> {
    let Some(value) = value else {
        return Ok(());
    };
    validate_uint8_array(value, max_bytes, field)
}

fn validate_uint8_array(value: &Uint8Array, max_bytes: u64, field: &str) -> Result<(), JsValue> {
    let js_value: &JsValue = value.as_ref();
    if !js_value.is_instance_of::<Uint8Array>() {
        return Err(js_error(format!(
            "{field} must be supplied as a Uint8Array"
        )));
    }
    let bytes = u64::from(value.length());
    if bytes == 0 {
        return Err(js_error(format!("{field} must not be empty")));
    }
    if bytes > max_bytes {
        return Err(js_error(format!(
            "{field} has {bytes} bytes, exceeding browser limit {max_bytes}"
        )));
    }
    Ok(())
}

fn copy_bounded_uint8_array(
    value: &Uint8Array,
    max_bytes: u64,
    field: &str,
) -> Result<Vec<u8>, JsValue> {
    validate_uint8_array(value, max_bytes, field)?;
    let len = value.length() as usize;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|err| {
        js_error(format!(
            "failed to reserve {len} bytes for {field} in Wasm memory: {err}"
        ))
    })?;
    bytes.resize(len, 0);
    value.copy_to(&mut bytes);
    Ok(bytes)
}

fn copy_encoded_image(value: &Uint8Array, method: &str) -> Result<Vec<u8>, JsValue> {
    copy_bounded_uint8_array(
        value,
        SIGLIP2_MAX_ENCODED_IMAGE_BYTES as u64,
        &format!("{method} encoded image"),
    )
}

fn copy_bounded_texts(texts: &Array, method: &str) -> Result<Vec<String>, JsValue> {
    if !Array::is_array(texts.as_ref()) {
        return Err(js_error(format!(
            "{method} texts must be supplied as a JavaScript Array"
        )));
    }
    let count = texts.length() as usize;
    if count == 0 {
        return Err(js_error(format!("{method} requires at least one text")));
    }
    if count > SIGLIP2_MAX_TEXT_BATCH_SIZE {
        return Err(js_error(format!(
            "{method} received {count} texts, exceeding browser limit {SIGLIP2_MAX_TEXT_BATCH_SIZE}"
        )));
    }

    let mut output = Vec::with_capacity(count);
    let mut utf16_total = 0u64;
    let mut utf8_total = 0usize;
    for index in 0..count {
        let value = texts.get(index as u32);
        if !value.is_string() {
            return Err(js_error(format!(
                "{method} text at index {index} must be a JavaScript string"
            )));
        }
        let code_units = u64::from(JsString::from(value.clone()).length());
        if code_units > SIGLIP2_MAX_TEXT_BYTES as u64 {
            return Err(js_error(format!(
                "{method} text at index {index} has {code_units} UTF-16 code units, exceeding pre-copy limit {SIGLIP2_MAX_TEXT_BYTES}"
            )));
        }
        utf16_total = utf16_total
            .checked_add(code_units)
            .ok_or_else(|| js_error(format!("{method} aggregate text length overflowed u64")))?;
        if utf16_total > SIGLIP2_MAX_TEXT_BATCH_BYTES as u64 {
            return Err(js_error(format!(
                "{method} texts exceed pre-copy aggregate limit {SIGLIP2_MAX_TEXT_BATCH_BYTES} UTF-16 code units"
            )));
        }

        let text = value.as_string().ok_or_else(|| {
            js_error(format!(
                "{method} text at index {index} must be a JavaScript string"
            ))
        })?;
        let bytes = text.len();
        if bytes > SIGLIP2_MAX_TEXT_BYTES {
            return Err(js_error(format!(
                "{method} text at index {index} has {bytes} UTF-8 bytes, exceeding browser limit {SIGLIP2_MAX_TEXT_BYTES}"
            )));
        }
        utf8_total = utf8_total
            .checked_add(bytes)
            .ok_or_else(|| js_error(format!("{method} aggregate UTF-8 length overflowed usize")))?;
        if utf8_total > SIGLIP2_MAX_TEXT_BATCH_BYTES {
            return Err(js_error(format!(
                "{method} texts have {utf8_total} UTF-8 bytes, exceeding browser limit {SIGLIP2_MAX_TEXT_BATCH_BYTES}"
            )));
        }
        output.push(text);
    }
    Ok(output)
}

fn validate_parts_array(parts: &Array, manifest: &Siglip2BpkPartsManifest) -> Result<(), JsValue> {
    for (index, entry) in manifest.parts.iter().enumerate() {
        let value = parts.get(index as u32);
        if !value.is_instance_of::<Uint8Array>() {
            return Err(js_error(format!(
                "part {index} must be supplied as a Uint8Array"
            )));
        }
        let array = value.unchecked_into::<Uint8Array>();
        let actual = u64::from(array.length());
        if actual != entry.bytes {
            return Err(js_error(format!(
                "part {index} byte mismatch before copying into Wasm: expected {}, got {actual}",
                entry.bytes
            )));
        }
    }
    Ok(())
}

async fn webgpu_device() -> Result<WasmDevice, JsValue> {
    let window = web_sys::window().ok_or_else(|| js_error("browser window is unavailable"))?;
    let navigator = Reflect::get(window.as_ref(), &JsValue::from_str("navigator"))
        .map_err(|err| js_error(format!("failed to inspect navigator: {err:?}")))?;
    let gpu = Reflect::get(&navigator, &JsValue::from_str("gpu"))
        .map_err(|err| js_error(format!("failed to inspect navigator.gpu: {err:?}")))?;
    if gpu.is_null() || gpu.is_undefined() {
        return Err(js_error(
            "WebGPU is unavailable; this browser or execution context has no navigator.gpu",
        ));
    }

    if let Some(device) = WEBGPU_RUNTIME_DEVICE.with(|slot| slot.borrow().clone()) {
        return Ok(device);
    }
    // CubeCL's adapter selection currently panics when the browser returns no adapter. Probe the
    // same WebGPU paths through JavaScript first so callers receive a normal rejected Promise
    // carrying an `Error` instead of an opaque Wasm abort. Prefer the high-performance adapter
    // CubeCL's default device requests, then retry with its integrated-GPU/low-power path.
    let request_adapter = Reflect::get(&gpu, &JsValue::from_str("requestAdapter"))
        .map_err(|err| {
            js_error(format!(
                "failed to inspect navigator.gpu.requestAdapter: {err:?}"
            ))
        })?
        .dyn_into::<Function>()
        .map_err(|_| js_error("navigator.gpu.requestAdapter is unavailable"))?;

    let high_performance_error = probe_webgpu_adapter(&gpu, &request_adapter, "high-performance")
        .await
        .err();
    let device = if let Some(high_performance_error) = high_performance_error {
        if let Err(low_power_error) =
            probe_webgpu_adapter(&gpu, &request_adapter, "low-power").await
        {
            return Err(js_error(format!(
                "WebGPU is present but no usable adapter is available (high-performance: {high_performance_error}; low-power: {low_power_error})"
            )));
        }
        WasmDevice::IntegratedGpu(0)
    } else {
        WasmDevice::default()
    };

    wgpu::init_setup_async::<wgpu::graphics::WebGpu>(&device, wgpu::RuntimeOptions::default())
        .await;
    WEBGPU_RUNTIME_DEVICE.with(|slot| *slot.borrow_mut() = Some(device.clone()));
    Ok(device)
}

async fn probe_webgpu_adapter(
    gpu: &JsValue,
    request_adapter: &Function,
    power_preference: &'static str,
) -> Result<(), String> {
    let options = js_sys::Object::new();
    let option_set = Reflect::set(
        &options,
        &JsValue::from_str("powerPreference"),
        &JsValue::from_str(power_preference),
    )
    .map_err(|err| format!("failed to set adapter options: {err:?}"))?;
    if !option_set {
        return Err("browser rejected the adapter power-preference option".to_string());
    }
    let adapter_promise = request_adapter
        .call1(gpu, &options)
        .map_err(|err| format!("adapter request threw: {err:?}"))?
        .dyn_into::<Promise>()
        .map_err(|_| "navigator.gpu.requestAdapter returned a non-Promise value".to_string())?;
    let adapter = JsFuture::from(adapter_promise)
        .await
        .map_err(|err| format!("adapter request failed: {err:?}"))?;
    if adapter.is_null() || adapter.is_undefined() {
        return Err("browser returned no adapter".to_string());
    }
    Ok(())
}

async fn fetch_url_bytes_limited(
    url: &str,
    expected_bytes: Option<u64>,
    max_bytes: u64,
) -> Result<FetchedBytes, JsValue> {
    if max_bytes == 0 {
        return Err(js_error(format!(
            "refusing to fetch '{url}' with a zero-byte limit"
        )));
    }
    let window = web_sys::window().ok_or_else(|| js_error("window is unavailable"))?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| js_error(format!("fetch failed for '{url}': {err:?}")))?;
    let response: web_sys::Response = response_value
        .dyn_into()
        .map_err(|_| js_error(format!("fetch returned a non-response value for '{url}'")))?;
    if !response.ok() {
        return Err(js_error(format!(
            "HTTP {} while fetching '{url}'",
            response.status()
        )));
    }
    let final_url = match response.url() {
        value if value.is_empty() => url.to_string(),
        value => value,
    };
    if let Some(value) = response.headers().get("content-length").map_err(|err| {
        js_error(format!(
            "failed to read Content-Length for '{url}': {err:?}"
        ))
    })? {
        let declared = value.parse::<u64>().map_err(|_| {
            js_error(format!(
                "invalid Content-Length '{value}' while fetching '{url}'"
            ))
        })?;
        // Content-Length describes the encoded HTTP representation, while Fetch exposes a decoded
        // stream. Content-Encoding is not necessarily CORS-exposed, so a declared length cannot be
        // compared with the decoded manifest size. It is only an early upper-bound hint for
        // resources without an exact decoded length; the stream limit below is authoritative.
        if expected_bytes.is_none() && declared > max_bytes {
            return Err(js_error(format!(
                "Content-Length {declared} for '{url}' exceeds browser limit {max_bytes}"
            )));
        }
    }

    let body = response
        .body()
        .ok_or_else(|| js_error(format!("response for '{url}' has no readable body")))?;
    let reader = web_sys::ReadableStreamDefaultReader::new(&body).map_err(|err| {
        js_error(format!(
            "failed to open response stream for '{url}': {err:?}"
        ))
    })?;
    let done_key = JsValue::from_str("done");
    let value_key = JsValue::from_str("value");
    let mut bytes = Vec::new();
    let mut digest = Sha256::new();
    let mut total = 0u64;

    loop {
        let read = JsFuture::from(reader.read())
            .await
            .map_err(|err| js_error(format!("response stream failed for '{url}': {err:?}")))?;
        let done = Reflect::get(&read, &done_key)
            .map_err(|err| js_error(format!("invalid stream result for '{url}': {err:?}")))?
            .as_bool()
            .ok_or_else(|| js_error(format!("stream result for '{url}' has no boolean done")))?;
        if done {
            break;
        }
        let value = Reflect::get(&read, &value_key)
            .map_err(|err| js_error(format!("invalid stream chunk for '{url}': {err:?}")))?;
        if !value.is_instance_of::<Uint8Array>() {
            let _ =
                reader.cancel_with_reason(&js_error("response stream yielded a non-byte chunk"));
            return Err(js_error(format!(
                "response stream for '{url}' yielded a non-Uint8Array chunk"
            )));
        }
        // This is a type cast only. `Uint8Array::new(&value)` would duplicate the JS chunk.
        let chunk = value.unchecked_into::<Uint8Array>();
        let chunk_len = u64::from(chunk.length());
        total = total
            .checked_add(chunk_len)
            .ok_or_else(|| js_error(format!("response body size overflow for '{url}'")))?;
        if total > max_bytes {
            let _ = reader.cancel_with_reason(&js_error("response exceeded browser byte limit"));
            return Err(js_error(format!(
                "response body for '{url}' exceeded browser limit {max_bytes} bytes"
            )));
        }
        let new_len = usize::try_from(total).map_err(|_| {
            js_error(format!(
                "response body for '{url}' exceeds Wasm addressable memory"
            ))
        })?;
        let start = bytes.len();
        bytes.try_reserve(new_len - start).map_err(|err| {
            js_error(format!(
                "failed to reserve response buffer for '{url}' ({new_len} bytes): {err}"
            ))
        })?;
        bytes.resize(new_len, 0);
        chunk.copy_to(&mut bytes[start..]);
        digest.update(&bytes[start..]);
    }
    reader.release_lock();

    if let Some(expected) = expected_bytes
        && total != expected
    {
        return Err(js_error(format!(
            "response body length mismatch for '{url}': expected {expected}, got {total}"
        )));
    }
    Ok(FetchedBytes {
        bytes,
        sha256: hex::encode(digest.finalize()),
        final_url,
    })
}

fn resolve_url(base_url: &str, child: &str) -> Result<String, JsValue> {
    validate_bundle_file_name(child)?;
    web_sys::Url::new_with_base(child, base_url)
        .map(|url| url.href())
        .map_err(|err| {
            js_error(format!(
                "failed to resolve bundle file '{child}' against '{base_url}': {err:?}"
            ))
        })
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    js_sys::Error::new(message.as_ref()).into()
}
