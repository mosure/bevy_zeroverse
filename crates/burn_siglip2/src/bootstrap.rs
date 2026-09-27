use std::path::PathBuf;
use std::sync::Arc;

use crate::{SIGLIP2_DEFAULT_CDN_ROOT_URL, Siglip2ModelVariant};

pub type BootstrapProgressCallback = Arc<dyn Fn(String) + Send + Sync + 'static>;

#[derive(Debug, Clone)]
pub struct Siglip2BootstrapConfig {
    pub cache_root: Option<PathBuf>,
    pub model_base_url: String,
    pub remote_root: String,
    pub model_stem: String,
    pub bpk_url: Option<String>,
    pub parts_manifest_url: Option<String>,
    pub tokenizer_json_url: Option<String>,
    pub tokenizer_config_url: Option<String>,
    pub tokenizer_model_url: Option<String>,
    pub preprocessor_config_url: Option<String>,
    pub prefer_bpk_parts: bool,
    pub download_tokenizer_assets: bool,
}

impl Default for Siglip2BootstrapConfig {
    fn default() -> Self {
        Self::for_variant(Siglip2ModelVariant::BasePatch16_224)
    }
}

impl Siglip2BootstrapConfig {
    /// Create a native bootstrap configuration for one supported public CDN model size.
    pub fn for_variant(variant: Siglip2ModelVariant) -> Self {
        Self {
            cache_root: None,
            model_base_url: SIGLIP2_DEFAULT_CDN_ROOT_URL.to_string(),
            remote_root: variant.model_size().to_string(),
            model_stem: variant.model_stem().to_string(),
            bpk_url: None,
            parts_manifest_url: None,
            tokenizer_json_url: None,
            tokenizer_config_url: None,
            tokenizer_model_url: None,
            preprocessor_config_url: None,
            prefer_bpk_parts: true,
            download_tokenizer_assets: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Siglip2Artifacts {
    pub cache_root: PathBuf,
    pub bpk_path: PathBuf,
    pub parts_manifest_path: PathBuf,
    pub tokenizer_json_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
    pub tokenizer_model_path: PathBuf,
    pub preprocessor_config_path: PathBuf,
    pub using_parts: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelBootstrapError {
    #[error("automatic model bootstrap is not supported on wasm32 targets")]
    UnsupportedTarget,
    #[error("failed to resolve user home directory for model cache")]
    MissingHomeDir,
    #[error("failed to create cache directory `{path}`: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to download model `{url}`: {message}")]
    Download { url: String, message: String },
    #[error("failed to write model file `{path}`: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("download returned invalid content for `{url}`: {message}")]
    InvalidContent { url: String, message: String },
}

pub fn resolve_or_bootstrap_siglip2_weights() -> Result<Siglip2Artifacts, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let cfg = native::apply_env_overrides(Siglip2BootstrapConfig::default());
        native::resolve_or_bootstrap_siglip2_weights_native(&cfg, None)
    }
}

pub fn resolve_or_bootstrap_siglip2_weights_with_config(
    cfg: &Siglip2BootstrapConfig,
) -> Result<Siglip2Artifacts, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = cfg;
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native::resolve_or_bootstrap_siglip2_weights_native(cfg, None)
    }
}

pub fn resolve_or_bootstrap_siglip2_weights_with_config_and_progress<F>(
    cfg: &Siglip2BootstrapConfig,
    progress: F,
) -> Result<Siglip2Artifacts, ModelBootstrapError>
where
    F: Fn(String) + Send + Sync + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (cfg, progress);
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native::resolve_or_bootstrap_siglip2_weights_native(cfg, Some(Arc::new(progress)))
    }
}

pub fn default_cache_root() -> Result<PathBuf, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let cfg = native::apply_env_overrides(Siglip2BootstrapConfig::default());
        native::default_cache_root_native(&cfg)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{
        fs,
        io::{BufReader, Read, Write},
        path::{Path, PathBuf},
        thread::sleep,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use ureq::AgentBuilder;

    use super::{
        BootstrapProgressCallback, ModelBootstrapError, Siglip2Artifacts, Siglip2BootstrapConfig,
    };
    use crate::{
        bpk::parse_siglip2_bpk_bytes,
        parts::{
            MAX_PRODUCTION_AGGREGATE_PART_BYTES, MAX_PRODUCTION_MANIFEST_BYTES,
            MAX_PRODUCTION_PART_BYTES, read_bpk_parts_manifest, resolve_part_entry_path,
            validate_production_bpk_parts_manifest,
        },
    };

    const CACHE_ROOT_DIR: &str = ".burn_siglip2";
    const CACHE_MODELS_SUBDIR: &str = "models";
    const DOWNLOAD_ATTEMPTS: u32 = 4;
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
    const READ_TIMEOUT: Duration = Duration::from_secs(60);
    const BACKOFF_MILLIS: u64 = 400;
    const MAX_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;
    const MAX_SMALL_SIDECAR_BYTES: u64 = 1024 * 1024;
    const MAX_CACHED_BUNDLE_SOURCE_BYTES: u64 = 32 * 1024;
    const MAX_BUNDLE_PAYLOAD_BYTES: u64 =
        MAX_PRODUCTION_AGGREGATE_PART_BYTES + 2 * MAX_TOKENIZER_BYTES + 2 * MAX_SMALL_SIDECAR_BYTES;

    struct DownloadedBytes {
        bytes: Vec<u8>,
        final_url: String,
    }

    #[derive(Debug, Clone, serde::Deserialize)]
    struct CdnBundleFile {
        path: String,
        role: String,
        bytes: u64,
        sha256: String,
    }

    #[derive(Debug, serde::Deserialize)]
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

    struct VerifiedBundle {
        source_url: String,
        manifest: CdnBundleManifest,
    }

    #[derive(serde::Deserialize)]
    struct CachedBundleSource {
        schema_version: u32,
        manifest_sha256: String,
        final_url: String,
    }

    pub fn apply_env_overrides(mut cfg: Siglip2BootstrapConfig) -> Siglip2BootstrapConfig {
        if let Ok(value) = std::env::var("BURN_SIGLIP2_MODEL_BASE_URL") {
            cfg.model_base_url = value;
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_REMOTE_ROOT") {
            cfg.remote_root = value;
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_MODEL_STEM") {
            cfg.model_stem = value;
        }
        if let Some(explicit) = std::env::var_os("BURN_SIGLIP2_CACHE_DIR") {
            cfg.cache_root = Some(
                PathBuf::from(explicit)
                    .join(CACHE_MODELS_SUBDIR)
                    .join(&cfg.model_stem),
            );
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_BPK_URL") {
            cfg.bpk_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_PARTS_MANIFEST_URL") {
            cfg.parts_manifest_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_TOKENIZER_JSON_URL") {
            cfg.tokenizer_json_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_TOKENIZER_CONFIG_URL") {
            cfg.tokenizer_config_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_TOKENIZER_MODEL_URL") {
            cfg.tokenizer_model_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_PREPROCESSOR_CONFIG_URL") {
            cfg.preprocessor_config_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_PREFER_PARTS") {
            cfg.prefer_bpk_parts = parse_bool(&value).unwrap_or(cfg.prefer_bpk_parts);
        }
        if let Ok(value) = std::env::var("BURN_SIGLIP2_DOWNLOAD_TOKENIZER_ASSETS") {
            cfg.download_tokenizer_assets =
                parse_bool(&value).unwrap_or(cfg.download_tokenizer_assets);
        }
        cfg
    }

    pub fn resolve_or_bootstrap_siglip2_weights_native(
        cfg: &Siglip2BootstrapConfig,
        progress: Option<BootstrapProgressCallback>,
    ) -> Result<Siglip2Artifacts, ModelBootstrapError> {
        let cache_root = default_cache_root_native(cfg)?;
        fs::create_dir_all(&cache_root).map_err(|source| ModelBootstrapError::CreateDir {
            path: cache_root.clone(),
            source,
        })?;
        let artifacts = local_artifacts(&cache_root, &cfg.model_stem);

        if cfg.prefer_bpk_parts && cached_parts_are_complete(&artifacts.parts_manifest_path) {
            let manifest =
                read_bpk_parts_manifest(&artifacts.parts_manifest_path).map_err(|message| {
                    ModelBootstrapError::InvalidContent {
                        url: artifacts.parts_manifest_path.display().to_string(),
                        message,
                    }
                })?;
            validate_requested_variant(cfg, &manifest.config, manifest.artifact.as_ref()).map_err(
                |message| ModelBootstrapError::InvalidContent {
                    url: artifacts.parts_manifest_path.display().to_string(),
                    message,
                },
            )?;
            maybe_download_sidecars(cfg, &artifacts, true, progress.as_ref())?;
            return Ok(Siglip2Artifacts {
                using_parts: true,
                ..artifacts
            });
        }

        if cfg.prefer_bpk_parts
            && let Some(manifest_url) = manifest_url(cfg)
        {
            match ensure_parts_present(cfg, &artifacts, &manifest_url, progress.as_ref()) {
                Ok(()) => {
                    maybe_download_sidecars(cfg, &artifacts, true, progress.as_ref())?;
                    return Ok(Siglip2Artifacts {
                        using_parts: true,
                        ..artifacts
                    });
                }
                Err(err) => {
                    emit_progress(
                        progress.as_ref(),
                        format!("parts bootstrap unavailable: {err}"),
                    );
                    if cfg.bpk_url.is_none() {
                        return Err(ModelBootstrapError::Download {
                            url: manifest_url,
                            message: format!(
                                "parts bootstrap failed and no explicit monolithic BPK fallback was configured: {err}"
                            ),
                        });
                    }
                    emit_progress(
                        progress.as_ref(),
                        "falling back to the explicitly configured monolithic BPK".to_string(),
                    );
                }
            }
        }

        ensure_bpk_present(cfg, &artifacts, progress.as_ref())?;
        maybe_download_sidecars(cfg, &artifacts, false, progress.as_ref())?;
        Ok(Siglip2Artifacts {
            using_parts: false,
            ..artifacts
        })
    }

    pub fn default_cache_root_native(
        cfg: &Siglip2BootstrapConfig,
    ) -> Result<PathBuf, ModelBootstrapError> {
        if let Some(explicit) = &cfg.cache_root {
            return Ok(explicit.clone());
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or(ModelBootstrapError::MissingHomeDir)?;
        Ok(home
            .join(CACHE_ROOT_DIR)
            .join(CACHE_MODELS_SUBDIR)
            .join(&cfg.model_stem))
    }

    fn ensure_parts_present(
        cfg: &Siglip2BootstrapConfig,
        artifacts: &Siglip2Artifacts,
        manifest_url: &str,
        progress_cb: Option<&BootstrapProgressCallback>,
    ) -> Result<(), String> {
        emit_progress(
            progress_cb,
            format!("downloading SigLIP2 parts manifest from {manifest_url}"),
        );
        let manifest_download =
            download_bytes_with_retries(manifest_url, MAX_PRODUCTION_MANIFEST_BYTES)
                .map_err(|err| format!("manifest download failed: {err}"))?;
        let manifest = serde_json::from_slice::<crate::parts::Siglip2BpkPartsManifest>(
            &manifest_download.bytes,
        )
        .map_err(|err| format!("invalid parts manifest JSON: {err}"))?;
        validate_production_bpk_parts_manifest(&artifacts.parts_manifest_path, &manifest)?;
        validate_requested_variant(cfg, &manifest.config, manifest.artifact.as_ref())?;
        write_bytes_atomically(&artifacts.parts_manifest_path, &manifest_download.bytes)
            .map_err(|err| format!("failed to cache manifest: {err}"))?;

        for entry in &manifest.parts {
            let local_part_path =
                resolve_part_entry_path(&artifacts.parts_manifest_path, &entry.path)
                    .map_err(|err| format!("invalid part path '{}': {err}", entry.path))?;
            if local_part_path.exists() && verify_cached_part(&local_part_path, entry).is_ok() {
                continue;
            }
            let part_url = join_url(&manifest_download.final_url, &entry.path).map_err(|err| {
                format!(
                    "failed to resolve part '{}' against final manifest URL '{}': {err}",
                    entry.path, manifest_download.final_url
                )
            })?;
            emit_progress(
                progress_cb,
                format!("downloading SigLIP2 part {}", entry.path),
            );
            let download = download_bytes_with_retries(&part_url, entry.bytes)
                .map_err(|err| format!("part download failed for '{}': {err}", entry.path))?;
            let bytes = download.bytes;
            if bytes.len() as u64 != entry.bytes {
                return Err(format!(
                    "downloaded part '{}' byte mismatch: expected {}, got {}",
                    entry.path,
                    entry.bytes,
                    bytes.len()
                ));
            }
            let actual = sha256_hex(&bytes);
            if !actual.eq_ignore_ascii_case(entry.sha256.trim()) {
                return Err(format!(
                    "downloaded part '{}' checksum mismatch: expected {}, got {}",
                    entry.path, entry.sha256, actual
                ));
            }
            write_bytes_atomically(&local_part_path, &bytes)
                .map_err(|err| format!("failed to cache part '{}': {err}", entry.path))?;
            verify_cached_part(&local_part_path, entry)?;
        }

        Ok(())
    }

    fn ensure_bpk_present(
        cfg: &Siglip2BootstrapConfig,
        artifacts: &Siglip2Artifacts,
        progress_cb: Option<&BootstrapProgressCallback>,
    ) -> Result<(), ModelBootstrapError> {
        if artifacts.bpk_path.exists() {
            let bytes =
                fs::read(&artifacts.bpk_path).map_err(|source| ModelBootstrapError::Write {
                    path: artifacts.bpk_path.clone(),
                    source,
                })?;
            let bpk =
                parse_siglip2_bpk_bytes(&bytes, Some(&artifacts.bpk_path)).map_err(|message| {
                    ModelBootstrapError::InvalidContent {
                        url: artifacts.bpk_path.display().to_string(),
                        message,
                    }
                })?;
            validate_requested_variant(cfg, &bpk.header.config, bpk.header.artifact.as_ref())
                .map_err(|message| ModelBootstrapError::InvalidContent {
                    url: artifacts.bpk_path.display().to_string(),
                    message,
                })?;
            return Ok(());
        }

        let url = bpk_url(cfg);
        emit_progress(
            progress_cb,
            format!("downloading SigLIP2 monolithic BPK from {url}"),
        );
        let download = download_bytes_with_retries(&url, MAX_PRODUCTION_AGGREGATE_PART_BYTES)
            .map_err(|message| ModelBootstrapError::Download {
                url: url.clone(),
                message,
            })?;
        let bytes = download.bytes;
        let bpk = parse_siglip2_bpk_bytes(&bytes, None).map_err(|message| {
            ModelBootstrapError::InvalidContent {
                url: url.clone(),
                message,
            }
        })?;
        validate_requested_variant(cfg, &bpk.header.config, bpk.header.artifact.as_ref()).map_err(
            |message| ModelBootstrapError::InvalidContent {
                url: url.clone(),
                message,
            },
        )?;
        write_bytes_atomically(&artifacts.bpk_path, &bytes).map_err(|source| {
            ModelBootstrapError::Write {
                path: artifacts.bpk_path.clone(),
                source,
            }
        })?;
        Ok(())
    }

    fn maybe_download_sidecars(
        cfg: &Siglip2BootstrapConfig,
        artifacts: &Siglip2Artifacts,
        using_parts: bool,
        progress_cb: Option<&BootstrapProgressCallback>,
    ) -> Result<(), ModelBootstrapError> {
        if !cfg.download_tokenizer_assets {
            return Ok(());
        }
        let bundle = resolve_verified_bundle(cfg, artifacts, using_parts, progress_cb)?;
        let requests = sidecar_requests(cfg, artifacts, bundle.as_ref())?;
        for request in requests {
            let path = request.path;
            let url = request.url;
            let Some(url) = url else {
                continue;
            };
            if path.exists() {
                match validate_cached_sidecar(path, request.max_bytes, request.expected.as_ref()) {
                    Ok(()) => continue,
                    Err(message) if request.expected.is_some() => emit_progress(
                        progress_cb,
                        format!(
                            "cached sidecar {} failed bundle verification and will be refreshed: {message}",
                            path.display()
                        ),
                    ),
                    Err(message) => {
                        return Err(ModelBootstrapError::InvalidContent {
                            url: path.display().to_string(),
                            message,
                        });
                    }
                }
            }
            emit_progress(
                progress_cb,
                format!(
                    "downloading sidecar {}",
                    path.file_name().and_then(|v| v.to_str()).unwrap_or("asset")
                ),
            );
            match download_bytes_with_retries(&url, request.max_bytes) {
                Ok(download) => {
                    validate_downloaded_sidecar(
                        &download.bytes,
                        request.max_bytes,
                        request.expected.as_ref(),
                    )
                    .map_err(|message| {
                        ModelBootstrapError::InvalidContent {
                            url: url.clone(),
                            message,
                        }
                    })?;
                    write_bytes_atomically(path, &download.bytes).map_err(|source| {
                        ModelBootstrapError::Write {
                            path: path.clone(),
                            source,
                        }
                    })?;
                    validate_cached_sidecar(path, request.max_bytes, request.expected.as_ref())
                        .map_err(|message| ModelBootstrapError::InvalidContent {
                            url: path.display().to_string(),
                            message,
                        })?;
                }
                Err(message) => {
                    if !message.contains("404") || request.required {
                        return Err(ModelBootstrapError::Download {
                            url,
                            message: if request.required && message.contains("404") {
                                format!("required model sidecar is unavailable: {message}")
                            } else {
                                message
                            },
                        });
                    }
                }
            }
        }
        Ok(())
    }

    struct SidecarRequest<'a> {
        path: &'a PathBuf,
        url: Option<String>,
        max_bytes: u64,
        required: bool,
        expected: Option<CdnBundleFile>,
    }

    fn sidecar_requests<'a>(
        cfg: &Siglip2BootstrapConfig,
        artifacts: &'a Siglip2Artifacts,
        bundle: Option<&VerifiedBundle>,
    ) -> Result<Vec<SidecarRequest<'a>>, ModelBootstrapError> {
        let bundle_entry = |suffix: &str, role: &str| {
            bundle.and_then(|bundle| {
                let expected_path = format!("{}.{}", cfg.model_stem, suffix);
                bundle
                    .manifest
                    .files
                    .iter()
                    .find(|file| file.path == expected_path && file.role == role)
                    .cloned()
            })
        };
        let resolve = |explicit: Option<String>, expected: Option<&CdnBundleFile>, suffix: &str| {
            if let Some(explicit) = explicit {
                return Ok(Some(explicit));
            }
            if let (Some(bundle), Some(expected)) = (bundle, expected) {
                return join_url(&bundle.source_url, &expected.path)
                    .map(Some)
                    .map_err(|message| ModelBootstrapError::InvalidContent {
                        url: bundle.source_url.clone(),
                        message,
                    });
            }
            Ok(sidecar_url(cfg, None, suffix))
        };

        let tokenizer = bundle_entry("tokenizer.json", "tokenizer");
        let tokenizer_config = bundle_entry("tokenizer_config.json", "tokenizer_sidecar");
        let tokenizer_model = bundle_entry("tokenizer.model", "tokenizer_sidecar");
        let preprocessor = bundle_entry("preprocessor_config.json", "image_preprocessor");

        Ok(vec![
            SidecarRequest {
                path: &artifacts.tokenizer_json_path,
                url: resolve(
                    cfg.tokenizer_json_url.clone(),
                    tokenizer.as_ref(),
                    "tokenizer.json",
                )?,
                max_bytes: MAX_TOKENIZER_BYTES,
                required: true,
                expected: tokenizer,
            },
            SidecarRequest {
                path: &artifacts.tokenizer_config_path,
                url: resolve(
                    cfg.tokenizer_config_url.clone(),
                    tokenizer_config.as_ref(),
                    "tokenizer_config.json",
                )?,
                max_bytes: MAX_SMALL_SIDECAR_BYTES,
                required: true,
                expected: tokenizer_config,
            },
            SidecarRequest {
                path: &artifacts.tokenizer_model_path,
                url: if bundle.is_none()
                    || tokenizer_model.is_some()
                    || cfg.tokenizer_model_url.is_some()
                {
                    resolve(
                        cfg.tokenizer_model_url.clone(),
                        tokenizer_model.as_ref(),
                        "tokenizer.model",
                    )?
                } else {
                    None
                },
                max_bytes: MAX_TOKENIZER_BYTES,
                required: false,
                expected: tokenizer_model,
            },
            SidecarRequest {
                path: &artifacts.preprocessor_config_path,
                url: if bundle.is_none()
                    || preprocessor.is_some()
                    || cfg.preprocessor_config_url.is_some()
                {
                    resolve(
                        cfg.preprocessor_config_url.clone(),
                        preprocessor.as_ref(),
                        "preprocessor_config.json",
                    )?
                } else {
                    None
                },
                max_bytes: MAX_SMALL_SIDECAR_BYTES,
                required: false,
                expected: preprocessor,
            },
        ])
    }

    fn resolve_verified_bundle(
        cfg: &Siglip2BootstrapConfig,
        artifacts: &Siglip2Artifacts,
        using_parts: bool,
        progress_cb: Option<&BootstrapProgressCallback>,
    ) -> Result<Option<VerifiedBundle>, ModelBootstrapError> {
        let explicit_model_url = if using_parts {
            cfg.parts_manifest_url.is_some()
        } else {
            cfg.bpk_url.is_some()
        };
        if explicit_model_url
            && cfg.tokenizer_json_url.is_some()
            && cfg.tokenizer_config_url.is_some()
        {
            emit_progress(
                progress_cb,
                "complete explicit model/tokenizer URL configuration bypasses the derived CDN bundle inventory; caller is responsible for authenticating those URLs"
                    .to_string(),
            );
            return Ok(None);
        }
        let url = bundle_manifest_url(cfg);
        let cache_path = artifacts
            .cache_root
            .join(format!("{}.bundle.manifest.json", cfg.model_stem));
        let source_cache_path = artifacts
            .cache_root
            .join(format!("{}.bundle.source.json", cfg.model_stem));
        if cache_path.exists() {
            match read_file_bounded(&cache_path, MAX_PRODUCTION_MANIFEST_BYTES).and_then(|bytes| {
                let source_url = read_cached_bundle_source(&source_cache_path, &bytes)?;
                parse_and_validate_bundle(cfg, artifacts, using_parts, &bytes, &source_url).map(
                    |manifest| VerifiedBundle {
                        source_url,
                        manifest,
                    },
                )
            }) {
                Ok(bundle) => return Ok(Some(bundle)),
                Err(message) => emit_progress(
                    progress_cb,
                    format!(
                        "cached CDN bundle manifest failed validation and will be refreshed: {message}"
                    ),
                ),
            }
        }

        emit_progress(
            progress_cb,
            format!("downloading SigLIP2 CDN bundle manifest from {url}"),
        );
        let download = match download_bytes_with_retries(&url, MAX_PRODUCTION_MANIFEST_BYTES) {
            Ok(download) => download,
            Err(message)
                if cfg.tokenizer_json_url.is_some() && cfg.tokenizer_config_url.is_some() =>
            {
                emit_progress(
                    progress_cb,
                    format!(
                        "CDN bundle inventory unavailable; using explicitly configured tokenizer URLs without bundle hashes: {message}"
                    ),
                );
                return Ok(None);
            }
            Err(message) => return Err(ModelBootstrapError::Download { url, message }),
        };
        let source_url = canonical_bundle_source_url(&download.final_url).map_err(|message| {
            ModelBootstrapError::InvalidContent {
                url: download.final_url.clone(),
                message,
            }
        })?;
        let manifest =
            parse_and_validate_bundle(cfg, artifacts, using_parts, &download.bytes, &source_url)
                .map_err(|message| ModelBootstrapError::InvalidContent {
                    url: source_url.clone(),
                    message,
                })?;
        write_bytes_atomically(&cache_path, &download.bytes).map_err(|source| {
            ModelBootstrapError::Write {
                path: cache_path.clone(),
                source,
            }
        })?;
        write_cached_bundle_source(&source_cache_path, &download.bytes, &source_url).map_err(
            |source| ModelBootstrapError::Write {
                path: source_cache_path,
                source,
            },
        )?;
        Ok(Some(VerifiedBundle {
            source_url,
            manifest,
        }))
    }

    fn canonical_bundle_source_url(value: &str) -> Result<String, String> {
        let mut url =
            url::Url::parse(value).map_err(|err| format!("invalid final CDN bundle URL: {err}"))?;
        if !matches!(url.scheme(), "http" | "https") || url.cannot_be_a_base() {
            return Err(format!(
                "final CDN bundle URL must be a hierarchical HTTP(S) URL, got '{value}'"
            ));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err("final CDN bundle URL must not contain credentials".to_string());
        }
        // Relative bundle assets never inherit a manifest query or fragment. Avoid persisting
        // transient signed query strings or other URL secrets in the cache metadata.
        url.set_query(None);
        url.set_fragment(None);
        let canonical = url.to_string();
        if canonical.len() as u64 > MAX_CACHED_BUNDLE_SOURCE_BYTES {
            return Err(format!(
                "final CDN bundle URL exceeds {} bytes",
                MAX_CACHED_BUNDLE_SOURCE_BYTES
            ));
        }
        Ok(canonical)
    }

    fn read_cached_bundle_source(path: &Path, manifest_bytes: &[u8]) -> Result<String, String> {
        let bytes = read_file_bounded(path, MAX_CACHED_BUNDLE_SOURCE_BYTES)?;
        let cached = serde_json::from_slice::<CachedBundleSource>(&bytes)
            .map_err(|err| format!("failed to parse cached bundle source metadata: {err}"))?;
        if cached.schema_version != 1 {
            return Err(format!(
                "unsupported cached bundle source schema {}",
                cached.schema_version
            ));
        }
        validate_sha256(&cached.manifest_sha256, "cached bundle manifest")?;
        let actual = sha256_hex(manifest_bytes);
        if !actual.eq_ignore_ascii_case(cached.manifest_sha256.trim()) {
            return Err(format!(
                "cached bundle source metadata checksum mismatch: expected {}, got {actual}",
                cached.manifest_sha256
            ));
        }
        canonical_bundle_source_url(&cached.final_url)
    }

    fn write_cached_bundle_source(
        path: &Path,
        manifest_bytes: &[u8],
        final_url: &str,
    ) -> Result<(), std::io::Error> {
        let bytes = serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "manifest_sha256": sha256_hex(manifest_bytes),
            "final_url": final_url,
        }))
        .map_err(std::io::Error::other)?;
        if bytes.len() as u64 > MAX_CACHED_BUNDLE_SOURCE_BYTES {
            return Err(std::io::Error::other(
                "cached bundle source metadata exceeds its byte limit",
            ));
        }
        write_bytes_atomically(path, &bytes)
    }

    fn parse_and_validate_bundle(
        cfg: &Siglip2BootstrapConfig,
        artifacts: &Siglip2Artifacts,
        using_parts: bool,
        bytes: &[u8],
        source: &str,
    ) -> Result<CdnBundleManifest, String> {
        let bundle = serde_json::from_slice::<CdnBundleManifest>(bytes)
            .map_err(|err| format!("failed to parse CDN bundle manifest '{source}': {err}"))?;
        validate_cdn_bundle(cfg, &bundle)?;
        if using_parts {
            validate_bundle_against_cached_parts(&bundle, &artifacts.parts_manifest_path)?;
        }
        Ok(bundle)
    }

    fn validate_cdn_bundle(
        cfg: &Siglip2BootstrapConfig,
        bundle: &CdnBundleManifest,
    ) -> Result<(), String> {
        if bundle.schema_version != 1
            || bundle.manifest_kind != "siglip2_cdn_bundle"
            || bundle.model_family != "siglip2"
        {
            return Err("unsupported SigLIP2 CDN bundle manifest".to_string());
        }
        if bundle.storage_dtype != "f16" {
            return Err(format!(
                "CDN bundle must use f16 storage, got '{}'",
                bundle.storage_dtype
            ));
        }
        if bundle.upstream_revision.len() != 40
            || !bundle
                .upstream_revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("CDN bundle upstream_revision must be a 40-character git commit".into());
        }
        if let Some(expected) = requested_variant(cfg)
            && (bundle.model_variant != expected.model_size()
                || bundle.upstream_model_id != expected.hf_model_id())
        {
            return Err(format!(
                "requested SigLIP2 model size '{}' but CDN bundle identifies '{}' / '{}'",
                expected.model_size(),
                bundle.model_variant,
                bundle.upstream_model_id
            ));
        }

        let expected_parts_manifest = format!("{}.bpk.parts.json", cfg.model_stem);
        if bundle.parts_manifest != expected_parts_manifest {
            return Err(format!(
                "CDN bundle parts_manifest mismatch: expected '{expected_parts_manifest}', got '{}'",
                bundle.parts_manifest
            ));
        }
        validate_bundle_file_name(&bundle.parts_manifest)?;
        if bundle.files.is_empty() {
            return Err("CDN bundle inventory is empty".to_string());
        }
        if bundle.payload_bytes > MAX_BUNDLE_PAYLOAD_BYTES {
            return Err(format!(
                "CDN bundle payload {} exceeds native bootstrap limit {MAX_BUNDLE_PAYLOAD_BYTES}",
                bundle.payload_bytes
            ));
        }

        let expected_tokenizer = format!("{}.tokenizer.json", cfg.model_stem);
        let expected_tokenizer_config = format!("{}.tokenizer_config.json", cfg.model_stem);
        let mut unique = std::collections::BTreeSet::new();
        let mut total = 0u64;
        let mut parts_manifests = 0usize;
        let mut model_parts = 0usize;
        let mut tokenizers = 0usize;
        let mut tokenizer_configs = 0usize;
        for file in &bundle.files {
            validate_bundle_file_name(&file.path)?;
            if !unique.insert(file.path.as_str()) {
                return Err(format!(
                    "duplicate CDN bundle inventory path '{}'",
                    file.path
                ));
            }
            if file.bytes == 0 {
                return Err(format!(
                    "CDN bundle file '{}' has zero declared bytes",
                    file.path
                ));
            }
            let limit = bundle_file_limit(file)?;
            if file.bytes > limit {
                return Err(format!(
                    "CDN bundle file '{}' declares {} bytes, exceeding limit {limit}",
                    file.path, file.bytes
                ));
            }
            validate_sha256(&file.sha256, &format!("CDN bundle file '{}'", file.path))?;
            total = total
                .checked_add(file.bytes)
                .ok_or_else(|| "CDN bundle payload byte count overflow".to_string())?;
            match file.role.as_str() {
                "model_part" => model_parts += 1,
                "parts_manifest" => {
                    parts_manifests += 1;
                    if file.path != bundle.parts_manifest {
                        return Err(format!(
                            "CDN bundle parts manifest role points at '{}', expected '{}'",
                            file.path, bundle.parts_manifest
                        ));
                    }
                }
                "tokenizer" => {
                    tokenizers += 1;
                    if file.path != expected_tokenizer {
                        return Err(format!(
                            "CDN bundle tokenizer path mismatch: expected '{expected_tokenizer}', got '{}'",
                            file.path
                        ));
                    }
                }
                "tokenizer_sidecar" if file.path == expected_tokenizer_config => {
                    tokenizer_configs += 1;
                }
                "tokenizer_sidecar" | "image_preprocessor" => {}
                role => {
                    return Err(format!(
                        "CDN bundle file '{}' has unsupported role '{role}'",
                        file.path
                    ));
                }
            }
        }
        if parts_manifests != 1 || model_parts == 0 {
            return Err(format!(
                "CDN bundle must contain one parts manifest and at least one model part; got {parts_manifests} and {model_parts}"
            ));
        }
        if tokenizers != 1 {
            return Err(format!(
                "CDN bundle must contain exactly one canonical tokenizer file, got {tokenizers}"
            ));
        }
        if tokenizer_configs != 1 {
            return Err(format!(
                "CDN bundle must contain exactly one canonical tokenizer config, got {tokenizer_configs}"
            ));
        }
        if total != bundle.payload_bytes {
            return Err(format!(
                "CDN bundle payload byte mismatch: manifest={}, inventory={total}",
                bundle.payload_bytes
            ));
        }
        Ok(())
    }

    fn validate_bundle_against_cached_parts(
        bundle: &CdnBundleManifest,
        manifest_path: &Path,
    ) -> Result<(), String> {
        let manifest = read_bpk_parts_manifest(manifest_path)?;
        let artifact = manifest.artifact.as_ref().ok_or_else(|| {
            "cached parts manifest is missing immutable artifact provenance".to_string()
        })?;
        if bundle.model_variant != artifact.model_variant
            || bundle.upstream_model_id != artifact.upstream_model_id
            || bundle.upstream_revision != artifact.upstream_revision
            || bundle.storage_dtype != artifact.storage_dtype
        {
            return Err("CDN bundle provenance does not match cached parts manifest".to_string());
        }
        let outer_manifest = bundle
            .files
            .iter()
            .find(|file| file.path == bundle.parts_manifest && file.role == "parts_manifest")
            .ok_or_else(|| "CDN bundle has no parts manifest inventory entry".to_string())?;
        verify_cached_bundle_file(manifest_path, MAX_PRODUCTION_MANIFEST_BYTES, outer_manifest)?;
        if bundle
            .files
            .iter()
            .filter(|file| file.role == "model_part")
            .count()
            != manifest.parts.len()
        {
            return Err(
                "CDN bundle and cached parts manifest declare different part counts".into(),
            );
        }
        for part in &manifest.parts {
            let outer = bundle
                .files
                .iter()
                .find(|file| file.path == part.path && file.role == "model_part")
                .ok_or_else(|| {
                    format!(
                        "cached parts manifest entry '{}' is absent from CDN bundle inventory",
                        part.path
                    )
                })?;
            if outer.bytes != part.bytes || !outer.sha256.eq_ignore_ascii_case(part.sha256.trim()) {
                return Err(format!(
                    "CDN bundle and cached parts manifest disagree for '{}'",
                    part.path
                ));
            }
        }
        Ok(())
    }

    fn bundle_file_limit(file: &CdnBundleFile) -> Result<u64, String> {
        match file.role.as_str() {
            "model_part" => Ok(MAX_PRODUCTION_PART_BYTES),
            "parts_manifest" => Ok(MAX_PRODUCTION_MANIFEST_BYTES),
            "tokenizer" => Ok(MAX_TOKENIZER_BYTES),
            "tokenizer_sidecar" if file.path.ends_with(".tokenizer_config.json") => {
                Ok(MAX_SMALL_SIDECAR_BYTES)
            }
            "tokenizer_sidecar" => Ok(MAX_TOKENIZER_BYTES),
            "image_preprocessor" => Ok(MAX_SMALL_SIDECAR_BYTES),
            role => Err(format!(
                "CDN bundle file '{}' has unsupported role '{role}'",
                file.path
            )),
        }
    }

    fn validate_bundle_file_name(value: &str) -> Result<(), String> {
        if value.is_empty()
            || value.contains('/')
            || value.contains('\\')
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            || value == "."
            || value == ".."
        {
            return Err(format!("unsafe CDN bundle file name '{value}'"));
        }
        Ok(())
    }

    fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
        let value = value.trim();
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "{field} must declare a 64-character hexadecimal SHA-256"
            ));
        }
        Ok(())
    }

    fn local_artifacts(cache_root: &Path, model_stem: &str) -> Siglip2Artifacts {
        Siglip2Artifacts {
            cache_root: cache_root.to_path_buf(),
            bpk_path: cache_root.join(format!("{model_stem}.bpk")),
            parts_manifest_path: cache_root.join(format!("{model_stem}.bpk.parts.json")),
            tokenizer_json_path: cache_root.join(format!("{model_stem}.tokenizer.json")),
            tokenizer_config_path: cache_root.join(format!("{model_stem}.tokenizer_config.json")),
            tokenizer_model_path: cache_root.join(format!("{model_stem}.tokenizer.model")),
            preprocessor_config_path: cache_root
                .join(format!("{model_stem}.preprocessor_config.json")),
            using_parts: false,
        }
    }

    fn requested_variant(cfg: &Siglip2BootstrapConfig) -> Option<crate::Siglip2ModelVariant> {
        cfg.model_stem
            .strip_prefix("siglip2-")
            .and_then(crate::Siglip2ModelVariant::from_model_size)
    }

    fn validate_requested_variant(
        cfg: &Siglip2BootstrapConfig,
        config: &crate::Siglip2Config,
        artifact: Option<&crate::Siglip2ArtifactMetadata>,
    ) -> Result<(), String> {
        let Some(expected) = requested_variant(cfg) else {
            return Ok(());
        };
        let artifact = artifact.ok_or_else(|| {
            format!(
                "requested SigLIP2 model size '{}' but artifact provenance is missing",
                expected.model_size()
            )
        })?;
        let expected_config = crate::Siglip2Config::for_variant(expected);
        if config != &expected_config
            || artifact.model_variant != expected.model_size()
            || artifact.upstream_model_id != expected.hf_model_id()
        {
            return Err(format!(
                "requested SigLIP2 model size '{}' but artifact identifies '{}' / '{}' with a different config",
                expected.model_size(),
                artifact.model_variant,
                artifact.upstream_model_id
            ));
        }
        Ok(())
    }

    fn validate_cached_sidecar(
        path: &Path,
        max_bytes: u64,
        expected: Option<&CdnBundleFile>,
    ) -> Result<(), String> {
        let bytes = fs::metadata(path)
            .map_err(|err| {
                format!(
                    "failed to inspect cached sidecar '{}': {err}",
                    path.display()
                )
            })?
            .len();
        if bytes == 0 || bytes > max_bytes {
            return Err(format!(
                "cached sidecar '{}' has invalid length {bytes}; expected 1..={max_bytes} bytes",
                path.display()
            ));
        }
        if let Some(expected) = expected {
            verify_cached_bundle_file(path, max_bytes, expected)?;
        }
        Ok(())
    }

    fn validate_downloaded_sidecar(
        bytes: &[u8],
        max_bytes: u64,
        expected: Option<&CdnBundleFile>,
    ) -> Result<(), String> {
        if bytes.is_empty() || bytes.len() as u64 > max_bytes {
            return Err(format!(
                "model sidecar has invalid length {}; expected 1..={max_bytes} bytes",
                bytes.len()
            ));
        }
        if let Some(expected) = expected {
            if bytes.len() as u64 != expected.bytes {
                return Err(format!(
                    "CDN sidecar '{}' byte mismatch: expected {}, got {}",
                    expected.path,
                    expected.bytes,
                    bytes.len()
                ));
            }
            validate_sha256(
                &expected.sha256,
                &format!("CDN sidecar '{}'", expected.path),
            )?;
            let actual = sha256_hex(bytes);
            if !actual.eq_ignore_ascii_case(expected.sha256.trim()) {
                return Err(format!(
                    "CDN sidecar '{}' checksum mismatch: expected {}, got {actual}",
                    expected.path, expected.sha256
                ));
            }
        }
        Ok(())
    }

    fn verify_cached_bundle_file(
        path: &Path,
        max_bytes: u64,
        expected: &CdnBundleFile,
    ) -> Result<(), String> {
        validate_sha256(
            &expected.sha256,
            &format!("CDN bundle file '{}'", expected.path),
        )?;
        if expected.bytes == 0 || expected.bytes > max_bytes {
            return Err(format!(
                "CDN bundle file '{}' declares invalid byte length {}",
                expected.path, expected.bytes
            ));
        }
        let actual_bytes = fs::metadata(path)
            .map_err(|err| format!("failed to stat '{}': {err}", path.display()))?
            .len();
        if actual_bytes != expected.bytes {
            return Err(format!(
                "cached CDN file '{}' byte mismatch: expected {}, got {actual_bytes}",
                path.display(),
                expected.bytes
            ));
        }
        let actual_sha256 = sha256_file(path)?;
        if !actual_sha256.eq_ignore_ascii_case(expected.sha256.trim()) {
            return Err(format!(
                "cached CDN file '{}' checksum mismatch: expected {}, got {actual_sha256}",
                path.display(),
                expected.sha256
            ));
        }
        Ok(())
    }

    fn read_file_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
        let bytes = fs::metadata(path)
            .map_err(|err| format!("failed to inspect '{}': {err}", path.display()))?
            .len();
        if bytes == 0 || bytes > max_bytes {
            return Err(format!(
                "file '{}' has invalid length {bytes}; expected 1..={max_bytes} bytes",
                path.display()
            ));
        }
        let file = fs::File::open(path)
            .map_err(|err| format!("failed to open '{}': {err}", path.display()))?;
        read_response_body_bounded(file, max_bytes)
    }

    fn manifest_url(cfg: &Siglip2BootstrapConfig) -> Option<String> {
        cfg.parts_manifest_url.clone().or_else(|| {
            Some(format!(
                "{}/{}/{}.bpk.parts.json",
                cfg.model_base_url.trim_end_matches('/'),
                cfg.remote_root.trim_matches('/'),
                cfg.model_stem
            ))
        })
    }

    fn bundle_manifest_url(cfg: &Siglip2BootstrapConfig) -> String {
        let base = cfg.model_base_url.trim_end_matches('/');
        let remote_root = cfg.remote_root.trim_matches('/');
        if remote_root.is_empty() {
            format!("{base}/bundle.manifest.json")
        } else {
            format!("{base}/{remote_root}/bundle.manifest.json")
        }
    }

    fn bpk_url(cfg: &Siglip2BootstrapConfig) -> String {
        cfg.bpk_url.clone().unwrap_or_else(|| {
            format!(
                "{}/{}/{}.bpk",
                cfg.model_base_url.trim_end_matches('/'),
                cfg.remote_root.trim_matches('/'),
                cfg.model_stem
            )
        })
    }

    fn sidecar_url(
        cfg: &Siglip2BootstrapConfig,
        explicit: Option<String>,
        suffix: &str,
    ) -> Option<String> {
        explicit.or_else(|| {
            Some(format!(
                "{}/{}/{}.{}",
                cfg.model_base_url.trim_end_matches('/'),
                cfg.remote_root.trim_matches('/'),
                cfg.model_stem,
                suffix
            ))
        })
    }

    fn join_url(base_url: &str, file_name: &str) -> Result<String, String> {
        let base = url::Url::parse(base_url)
            .map_err(|err| format!("invalid final manifest URL: {err}"))?;
        base.join(file_name)
            .map(|url| url.to_string())
            .map_err(|err| format!("invalid relative part URL: {err}"))
    }

    fn verify_cached_part(
        path: &Path,
        entry: &crate::parts::Siglip2BpkPartEntry,
    ) -> Result<(), String> {
        let bytes = fs::metadata(path)
            .map_err(|err| format!("failed to stat '{}': {err}", path.display()))?
            .len();
        if bytes != entry.bytes {
            return Err(format!(
                "cached part '{}' byte mismatch: expected {}, got {}",
                path.display(),
                entry.bytes,
                bytes
            ));
        }
        let checksum = entry.sha256.trim();
        if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "cached part '{}' has no valid required SHA-256",
                path.display()
            ));
        }
        let actual = sha256_file(path)?;
        if !actual.eq_ignore_ascii_case(checksum) {
            return Err(format!(
                "cached part '{}' checksum mismatch: expected {}, got {}",
                path.display(),
                entry.sha256,
                actual
            ));
        }
        Ok(())
    }

    fn cached_parts_are_complete(manifest_path: &Path) -> bool {
        let Ok(metadata) = fs::metadata(manifest_path) else {
            return false;
        };
        if metadata.len() == 0 || metadata.len() > MAX_PRODUCTION_MANIFEST_BYTES {
            return false;
        }
        let Ok(manifest) = read_bpk_parts_manifest(manifest_path) else {
            return false;
        };
        if validate_production_bpk_parts_manifest(manifest_path, &manifest).is_err() {
            return false;
        }
        manifest.parts.iter().all(|entry| {
            resolve_part_entry_path(manifest_path, &entry.path)
                .and_then(|path| verify_cached_part(&path, entry))
                .is_ok()
        })
    }

    fn emit_progress(progress: Option<&BootstrapProgressCallback>, message: String) {
        if let Some(progress) = progress {
            progress(message);
        }
    }

    fn download_bytes_with_retries(url: &str, max_bytes: u64) -> Result<DownloadedBytes, String> {
        let agent = AgentBuilder::new()
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout_read(READ_TIMEOUT)
            .timeout_write(READ_TIMEOUT)
            .build();

        let mut last_error = None;
        for attempt in 1..=DOWNLOAD_ATTEMPTS {
            match agent.get(url).call() {
                Ok(response) => {
                    let final_url = response.get_url().to_string();
                    if let Some(content_length) = response
                        .header("Content-Length")
                        .and_then(|value| value.parse::<u64>().ok())
                        && content_length > max_bytes
                    {
                        return Err(format!(
                            "response body declares {content_length} bytes, exceeds limit {max_bytes}"
                        ));
                    }
                    match read_response_body_bounded(response.into_reader(), max_bytes) {
                        Ok(bytes) => return Ok(DownloadedBytes { bytes, final_url }),
                        Err(message) => {
                            last_error = Some(message);
                            if attempt < DOWNLOAD_ATTEMPTS {
                                sleep(Duration::from_millis(BACKOFF_MILLIS * attempt as u64));
                            }
                        }
                    }
                }
                Err(err) => {
                    let message = format!("{err}");
                    if message.contains("404") {
                        return Err(message);
                    }
                    last_error = Some(message);
                    if attempt < DOWNLOAD_ATTEMPTS {
                        sleep(Duration::from_millis(BACKOFF_MILLIS * attempt as u64));
                    }
                }
            }
        }
        Err(last_error.unwrap_or_else(|| "unknown download failure".to_string()))
    }

    fn read_response_body_bounded<R: Read>(reader: R, max_bytes: u64) -> Result<Vec<u8>, String> {
        let read_limit = max_bytes
            .checked_add(1)
            .ok_or_else(|| "response body byte limit overflow".to_string())?;
        let mut limited = reader.take(read_limit);
        let mut out = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        loop {
            let read = limited
                .read(&mut chunk)
                .map_err(|err| format!("failed reading response body: {err}"))?;
            if read == 0 {
                break;
            }
            let new_len = (out.len() as u64)
                .checked_add(read as u64)
                .ok_or_else(|| "response body byte count overflow".to_string())?;
            if new_len > max_bytes {
                return Err(format!(
                    "response body exceeds bounded limit {max_bytes} bytes"
                ));
            }
            out.try_reserve(read)
                .map_err(|err| format!("failed to reserve response buffer: {err}"))?;
            out.extend_from_slice(&chunk[..read]);
        }
        Ok(out)
    }

    fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temp_path = path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        {
            let mut file = fs::File::create(&temp_path)?;
            file.write_all(bytes)?;
            file.flush()?;
        }
        fs::rename(&temp_path, path)?;
        Ok(())
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};

        let mut digest = Sha256::new();
        digest.update(bytes);
        hex::encode(digest.finalize())
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        use sha2::{Digest, Sha256};

        let file = fs::File::open(path)
            .map_err(|err| format!("failed to open '{}' for checksum: {err}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let read = reader.read(&mut buffer).map_err(|err| {
                format!("failed to read '{}' for checksum: {err}", path.display())
            })?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        Ok(hex::encode(digest.finalize()))
    }

    fn parse_bool(value: &str) -> Option<bool> {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use std::{
            fs,
            io::{Read, Write},
            net::{SocketAddr, TcpListener, TcpStream},
            path::Path,
            thread,
            time::Duration,
        };

        use crate::{
            bpk::{Siglip2ArtifactMetadata, build_bpk_header_with_metadata, write_siglip2_bpk},
            config::Siglip2Config,
            parts::{Siglip2BpkPartsManifest, write_bpk_parts},
        };

        use super::{
            Siglip2BootstrapConfig, bpk_url, cached_parts_are_complete, manifest_url,
            read_response_body_bounded, resolve_or_bootstrap_siglip2_weights_native, sha256_hex,
            sidecar_url,
        };

        #[test]
        fn public_cdn_defaults_include_the_model_size_directory() {
            let cfg = Siglip2BootstrapConfig::default();
            let root = "https://aberration.technology/model/siglip2/base-patch16-224";
            assert_eq!(
                manifest_url(&cfg).as_deref(),
                Some(
                    "https://aberration.technology/model/siglip2/base-patch16-224/siglip2-base-patch16-224.bpk.parts.json"
                )
            );
            assert_eq!(
                sidecar_url(&cfg, None, "tokenizer.json").as_deref(),
                Some(
                    "https://aberration.technology/model/siglip2/base-patch16-224/siglip2-base-patch16-224.tokenizer.json"
                )
            );
            assert_eq!(
                bpk_url(&cfg),
                format!("{root}/siglip2-base-patch16-224.bpk")
            );
        }

        #[test]
        fn public_cdn_variant_constructor_keeps_directory_and_stem_in_sync() {
            for variant in crate::Siglip2ModelVariant::ALL {
                let cfg = Siglip2BootstrapConfig::for_variant(variant);
                assert_eq!(cfg.remote_root, variant.model_size());
                assert_eq!(cfg.model_stem, variant.model_stem());
                assert_eq!(
                    manifest_url(&cfg),
                    Some(format!(
                        "{}/{}/{}.bpk.parts.json",
                        crate::SIGLIP2_DEFAULT_CDN_ROOT_URL,
                        variant.model_size(),
                        variant.model_stem()
                    ))
                );
            }
        }

        fn write_production_fixture_bpk(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
            let payload = safetensors::tensor::serialize(
                &std::collections::BTreeMap::from([(
                    "logit_scale".to_string(),
                    safetensors::tensor::TensorView::new(
                        safetensors::tensor::Dtype::F16,
                        vec![1],
                        &[0u8; 2],
                    )?,
                )]),
                None,
            )?;
            let artifact = Siglip2ArtifactMetadata {
                model_variant: "base-patch16-224".to_string(),
                upstream_model_id: "google/siglip2-base-patch16-224".to_string(),
                upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
                storage_dtype: "f16".to_string(),
            };
            let header =
                build_bpk_header_with_metadata(Siglip2Config::default(), &payload, Some(artifact));
            write_siglip2_bpk(path, &header, &payload)?;
            Ok(())
        }

        fn write_bundle_fixture(root: &Path, stem: &str) -> Result<(), Box<dyn std::error::Error>> {
            let parts_manifest_name = format!("{stem}.bpk.parts.json");
            let parts_manifest_bytes = fs::read(root.join(&parts_manifest_name))?;
            let parts_manifest =
                serde_json::from_slice::<Siglip2BpkPartsManifest>(&parts_manifest_bytes)?;
            let artifact = parts_manifest
                .artifact
                .as_ref()
                .ok_or("fixture parts manifest has no artifact provenance")?;
            let mut files = Vec::new();
            let mut payload_bytes = 0u64;
            let mut add_file = |path: String,
                                role: &str,
                                bytes: Vec<u8>|
             -> Result<(), Box<dyn std::error::Error>> {
                payload_bytes = payload_bytes
                    .checked_add(bytes.len() as u64)
                    .ok_or("fixture payload byte overflow")?;
                files.push(serde_json::json!({
                    "path": path,
                    "role": role,
                    "bytes": bytes.len(),
                    "sha256": sha256_hex(&bytes),
                }));
                Ok(())
            };
            add_file(
                parts_manifest_name.clone(),
                "parts_manifest",
                parts_manifest_bytes,
            )?;
            for part in &parts_manifest.parts {
                add_file(
                    part.path.clone(),
                    "model_part",
                    fs::read(root.join(&part.path))?,
                )?;
            }
            for (suffix, role) in [
                ("tokenizer.json", "tokenizer"),
                ("tokenizer_config.json", "tokenizer_sidecar"),
                ("tokenizer.model", "tokenizer_sidecar"),
                ("preprocessor_config.json", "image_preprocessor"),
            ] {
                let path = format!("{stem}.{suffix}");
                if let Ok(bytes) = fs::read(root.join(&path)) {
                    add_file(path, role, bytes)?;
                }
            }
            let bundle = serde_json::json!({
                "schema_version": 1,
                "manifest_kind": "siglip2_cdn_bundle",
                "model_family": "siglip2",
                "model_variant": artifact.model_variant,
                "upstream_model_id": artifact.upstream_model_id,
                "upstream_revision": artifact.upstream_revision,
                "storage_dtype": artifact.storage_dtype,
                "parts_manifest": parts_manifest_name,
                "payload_bytes": payload_bytes,
                "files": files,
            });
            fs::write(
                root.join("bundle.manifest.json"),
                serde_json::to_vec_pretty(&bundle)?,
            )?;
            Ok(())
        }

        fn write_complete_bundle_fixture(
            root: &Path,
            stem: &str,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let bpk_path = root.join(format!("{stem}.bpk"));
            write_production_fixture_bpk(&bpk_path)?;
            write_bpk_parts(&bpk_path, 1, true)?;
            fs::write(root.join(format!("{stem}.tokenizer.json")), "{}")?;
            fs::write(root.join(format!("{stem}.tokenizer_config.json")), "{}")?;
            write_bundle_fixture(root, stem)
        }

        #[test]
        fn bootstrap_prefers_parts_and_downloads_sidecars() -> Result<(), Box<dyn std::error::Error>>
        {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            let bpk_path = source_dir.path().join(format!("{stem}.bpk"));
            write_production_fixture_bpk(&bpk_path)?;
            write_bpk_parts(&bpk_path, 1, true)?;
            fs::write(
                source_dir.path().join(format!("{stem}.tokenizer.json")),
                "{}",
            )?;
            fs::write(
                source_dir
                    .path()
                    .join(format!("{stem}.tokenizer_config.json")),
                "{}",
            )?;
            write_bundle_fixture(source_dir.path(), stem)?;

            let server = StaticFileServer::spawn(source_dir.path())?;
            let cfg = Siglip2BootstrapConfig {
                cache_root: Some(cache_dir.path().to_path_buf()),
                model_base_url: format!("http://{}", server.addr),
                remote_root: String::new(),
                model_stem: stem.to_string(),
                prefer_bpk_parts: true,
                download_tokenizer_assets: true,
                ..Siglip2BootstrapConfig::default()
            };
            let artifacts = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            assert!(artifacts.using_parts);
            assert!(artifacts.parts_manifest_path.exists());
            assert!(artifacts.tokenizer_json_path.exists());
            Ok(())
        }

        #[test]
        fn bootstrap_rejects_a_tampered_bundle_sidecar_download()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            write_complete_bundle_fixture(source_dir.path(), stem)?;
            // Keep the byte length unchanged so this specifically proves checksum enforcement.
            fs::write(
                source_dir.path().join(format!("{stem}.tokenizer.json")),
                "[]",
            )?;

            let server = StaticFileServer::spawn(source_dir.path())?;
            let cfg = Siglip2BootstrapConfig {
                cache_root: Some(cache_dir.path().to_path_buf()),
                model_base_url: format!("http://{}", server.addr),
                remote_root: String::new(),
                model_stem: stem.to_string(),
                prefer_bpk_parts: true,
                download_tokenizer_assets: true,
                ..Siglip2BootstrapConfig::default()
            };
            let error = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)
                .expect_err("tampered tokenizer download must be rejected");
            let message = error.to_string();
            assert!(message.contains("tokenizer.json"), "{message}");
            assert!(message.contains("checksum mismatch"), "{message}");
            Ok(())
        }

        #[test]
        fn bootstrap_refreshes_a_same_length_tampered_cached_sidecar()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            write_complete_bundle_fixture(source_dir.path(), stem)?;
            let server = StaticFileServer::spawn(source_dir.path())?;
            let cfg = Siglip2BootstrapConfig {
                cache_root: Some(cache_dir.path().to_path_buf()),
                model_base_url: format!("http://{}", server.addr),
                remote_root: String::new(),
                model_stem: stem.to_string(),
                prefer_bpk_parts: true,
                download_tokenizer_assets: true,
                ..Siglip2BootstrapConfig::default()
            };
            let artifacts = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            fs::write(&artifacts.tokenizer_json_path, "[]")?;

            let repaired = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            assert_eq!(fs::read(&repaired.tokenizer_json_path)?, b"{}");
            Ok(())
        }

        #[test]
        fn cached_bundle_reload_preserves_the_redirected_asset_base()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            let release_dir = source_dir.path().join("cdn/releases/v2");
            fs::create_dir_all(&release_dir)?;
            write_complete_bundle_fixture(&release_dir, stem)?;

            let server = StaticFileServer::spawn_with_redirect(
                source_dir.path(),
                "/models/current/bundle.manifest.json".to_string(),
                "/cdn/releases/v2/bundle.manifest.json?release=current".to_string(),
            )?;
            let root = format!("http://{}", server.addr);
            let cfg = Siglip2BootstrapConfig {
                cache_root: Some(cache_dir.path().to_path_buf()),
                model_base_url: format!("{root}/models/current"),
                remote_root: String::new(),
                model_stem: stem.to_string(),
                parts_manifest_url: Some(format!("{root}/cdn/releases/v2/{stem}.bpk.parts.json")),
                prefer_bpk_parts: true,
                download_tokenizer_assets: true,
                ..Siglip2BootstrapConfig::default()
            };

            let artifacts = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            assert_eq!(fs::read(&artifacts.tokenizer_json_path)?, b"{}");
            fs::write(&artifacts.tokenizer_json_path, "[]")?;

            // This reload uses the cached outer manifest. Repair only succeeds if its persisted
            // final URL still resolves assets under /cdn/releases/v2 rather than /models/current.
            let repaired = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            assert_eq!(fs::read(repaired.tokenizer_json_path)?, b"{}");
            Ok(())
        }

        #[test]
        fn bundle_validation_rejects_missing_and_mismatched_tokenizer_inventory()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            write_complete_bundle_fixture(source_dir.path(), stem)?;
            let bytes = fs::read(source_dir.path().join("bundle.manifest.json"))?;
            let document = serde_json::from_slice::<serde_json::Value>(&bytes)?;
            let cfg = Siglip2BootstrapConfig::default();

            let mut missing = document.clone();
            missing["files"]
                .as_array_mut()
                .expect("fixture files array")
                .retain(|entry| {
                    !entry["path"]
                        .as_str()
                        .is_some_and(|path| path.ends_with(".tokenizer_config.json"))
                });
            missing["payload_bytes"] = serde_json::json!(
                missing["files"]
                    .as_array()
                    .expect("fixture files array")
                    .iter()
                    .map(|entry| entry["bytes"].as_u64().expect("fixture byte length"))
                    .sum::<u64>()
            );
            let missing = serde_json::from_value::<super::CdnBundleManifest>(missing)?;
            let error = super::validate_cdn_bundle(&cfg, &missing)
                .expect_err("missing tokenizer config inventory must fail");
            assert!(error.contains("canonical tokenizer config"), "{error}");

            let mut mismatched = document;
            let tokenizer = mismatched["files"]
                .as_array_mut()
                .expect("fixture files array")
                .iter_mut()
                .find(|entry| entry["role"] == "tokenizer")
                .expect("fixture tokenizer entry");
            tokenizer["path"] = serde_json::json!("other.tokenizer.json");
            let mismatched = serde_json::from_value::<super::CdnBundleManifest>(mismatched)?;
            let error = super::validate_cdn_bundle(&cfg, &mismatched)
                .expect_err("mismatched tokenizer inventory path must fail");
            assert!(error.contains("tokenizer path mismatch"), "{error}");
            Ok(())
        }

        #[test]
        fn complete_explicit_urls_bypass_a_mismatched_derived_bundle()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            write_complete_bundle_fixture(source_dir.path(), stem)?;
            fs::write(
                source_dir.path().join("bundle.manifest.json"),
                br#"{"schema_version":999,"manifest_kind":"wrong"}"#,
            )?;
            let server = StaticFileServer::spawn(source_dir.path())?;
            let root = format!("http://{}", server.addr);
            let cfg = Siglip2BootstrapConfig {
                cache_root: Some(cache_dir.path().to_path_buf()),
                model_base_url: root.clone(),
                remote_root: String::new(),
                model_stem: stem.to_string(),
                parts_manifest_url: Some(format!("{root}/{stem}.bpk.parts.json")),
                tokenizer_json_url: Some(format!("{root}/{stem}.tokenizer.json")),
                tokenizer_config_url: Some(format!("{root}/{stem}.tokenizer_config.json")),
                prefer_bpk_parts: true,
                download_tokenizer_assets: true,
                ..Siglip2BootstrapConfig::default()
            };

            let artifacts = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            assert!(artifacts.using_parts);
            assert_eq!(fs::read(artifacts.tokenizer_json_path)?, b"{}");
            Ok(())
        }

        #[test]
        fn typed_bootstrap_rejects_a_valid_wrong_model_variant()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let base_stem = "siglip2-base-patch16-224";
            let bpk_path = source_dir.path().join(format!("{base_stem}.bpk"));
            write_production_fixture_bpk(&bpk_path)?;
            write_bpk_parts(&bpk_path, 1, true)?;

            let server = StaticFileServer::spawn(source_dir.path())?;
            let mut cfg =
                Siglip2BootstrapConfig::for_variant(crate::Siglip2ModelVariant::LargePatch16_256);
            cfg.cache_root = Some(cache_dir.path().to_path_buf());
            cfg.parts_manifest_url = Some(format!(
                "http://{}/{}.bpk.parts.json",
                server.addr, base_stem
            ));
            cfg.download_tokenizer_assets = false;

            let error = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)
                .expect_err("typed Large bootstrap must reject a valid Base artifact");
            let message = error.to_string();
            assert!(message.contains("large-patch16-256"), "{message}");
            assert!(message.contains("base-patch16-224"), "{message}");
            Ok(())
        }

        #[test]
        fn bootstrap_uses_existing_cached_parts_without_monolith()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            let source_bpk_path = source_dir.path().join(format!("{stem}.bpk"));
            write_production_fixture_bpk(&source_bpk_path)?;
            let parts = write_bpk_parts(&source_bpk_path, 1, true)?.expect("parts report");

            let cached_manifest_path = cache_dir.path().join(format!("{stem}.bpk.parts.json"));
            fs::copy(&parts.manifest_path, &cached_manifest_path)?;
            for source_part in &parts.part_paths {
                let file_name = source_part.file_name().expect("part file name");
                fs::copy(source_part, cache_dir.path().join(file_name))?;
            }

            let cfg = Siglip2BootstrapConfig {
                cache_root: Some(cache_dir.path().to_path_buf()),
                model_base_url: "http://127.0.0.1:9".to_string(),
                remote_root: String::new(),
                model_stem: stem.to_string(),
                prefer_bpk_parts: true,
                download_tokenizer_assets: false,
                ..Siglip2BootstrapConfig::default()
            };
            let artifacts = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            assert!(artifacts.using_parts);
            assert!(artifacts.parts_manifest_path.exists());
            assert!(!artifacts.bpk_path.exists());
            Ok(())
        }

        #[test]
        fn bootstrap_resolves_parts_against_redirected_manifest_url()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let cache_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            let release_dir = source_dir.path().join("cdn/releases/v2");
            fs::create_dir_all(&release_dir)?;
            let bpk_path = release_dir.join(format!("{stem}.bpk"));
            write_production_fixture_bpk(&bpk_path)?;
            write_bpk_parts(&bpk_path, 1, true)?;

            let manifest_name = format!("{stem}.bpk.parts.json");
            let redirect_from = format!("/models/current/{manifest_name}");
            let redirect_to = format!("/cdn/releases/v2/{manifest_name}?release=current");
            let server = StaticFileServer::spawn_with_redirect(
                source_dir.path(),
                redirect_from,
                redirect_to,
            )?;
            let cfg = Siglip2BootstrapConfig {
                cache_root: Some(cache_dir.path().to_path_buf()),
                model_base_url: format!("http://{}", server.addr),
                remote_root: String::new(),
                model_stem: stem.to_string(),
                parts_manifest_url: Some(format!(
                    "http://{}/models/current/{manifest_name}?ignored=1",
                    server.addr
                )),
                prefer_bpk_parts: true,
                download_tokenizer_assets: false,
                ..Siglip2BootstrapConfig::default()
            };

            let artifacts = resolve_or_bootstrap_siglip2_weights_native(&cfg, None)?;
            assert!(artifacts.using_parts);
            assert!(cached_parts_are_complete(&artifacts.parts_manifest_path));
            assert!(!artifacts.bpk_path.exists());
            Ok(())
        }

        #[test]
        fn cached_parts_require_checksum_match_even_when_file_length_is_unchanged()
        -> Result<(), Box<dyn std::error::Error>> {
            let source_dir = tempfile::tempdir()?;
            let stem = "siglip2-base-patch16-224";
            let bpk_path = source_dir.path().join(format!("{stem}.bpk"));
            write_production_fixture_bpk(&bpk_path)?;
            let parts = write_bpk_parts(&bpk_path, 1, true)?.expect("parts report");
            assert!(cached_parts_are_complete(&parts.manifest_path));

            let part_path = &parts.part_paths[0];
            let mut bytes = fs::read(part_path)?;
            *bytes.last_mut().expect("non-empty BPK part") ^= 0xff;
            fs::write(part_path, bytes)?;
            assert!(!cached_parts_are_complete(&parts.manifest_path));
            Ok(())
        }

        #[test]
        fn response_body_reader_is_bounded() {
            let error = read_response_body_bounded(std::io::Cursor::new(vec![0u8; 17]), 16)
                .expect_err("body larger than its bound must fail");
            assert!(error.contains("exceeds bounded limit"));
        }

        struct StaticFileServer {
            addr: SocketAddr,
            shutdown: Option<std::sync::mpsc::Sender<()>>,
            join: Option<thread::JoinHandle<()>>,
        }

        impl StaticFileServer {
            fn spawn(root: &Path) -> Result<Self, Box<dyn std::error::Error>> {
                Self::spawn_inner(root, None)
            }

            fn spawn_with_redirect(
                root: &Path,
                from: String,
                to: String,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                Self::spawn_inner(root, Some((from, to)))
            }

            fn spawn_inner(
                root: &Path,
                redirect: Option<(String, String)>,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let listener = TcpListener::bind("127.0.0.1:0")?;
                let addr = listener.local_addr()?;
                let root = root.to_path_buf();
                let (tx, rx) = std::sync::mpsc::channel::<()>();
                let join = thread::spawn(move || {
                    listener.set_nonblocking(true).ok();
                    loop {
                        if rx.try_recv().is_ok() {
                            break;
                        }
                        match listener.accept() {
                            Ok((mut stream, _)) => {
                                let mut buffer = [0u8; 4096];
                                let Ok(read) = stream.read(&mut buffer) else {
                                    continue;
                                };
                                let request = String::from_utf8_lossy(&buffer[..read]);
                                let path = request
                                    .lines()
                                    .next()
                                    .and_then(|line| line.split_whitespace().nth(1))
                                    .unwrap_or("/");
                                let path_without_query = path.split('?').next().unwrap_or(path);
                                if let Some((from, to)) = redirect.as_ref()
                                    && path_without_query == from
                                {
                                    let _ = write!(
                                        stream,
                                        "HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                                    );
                                    continue;
                                }
                                let rel = path_without_query.trim_start_matches('/');
                                let file_path = root.join(rel);
                                if let Ok(bytes) = fs::read(&file_path) {
                                    let _ = write!(
                                        stream,
                                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                        bytes.len()
                                    );
                                    let _ = stream.write_all(&bytes);
                                } else {
                                    let _ = stream.write_all(
                                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                                    );
                                }
                            }
                            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                                thread::sleep(Duration::from_millis(10));
                            }
                            Err(_) => break,
                        }
                    }
                });
                Ok(Self {
                    addr,
                    shutdown: Some(tx),
                    join: Some(join),
                })
            }
        }

        impl Drop for StaticFileServer {
            fn drop(&mut self) {
                if let Some(shutdown) = self.shutdown.take() {
                    let _ = shutdown.send(());
                }
                let _ = TcpStream::connect(self.addr);
                if let Some(join) = self.join.take() {
                    let _ = join.join();
                }
            }
        }
    }
}
