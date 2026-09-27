use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::Path,
};

use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    bpk::{Siglip2ArtifactMetadata, Siglip2BpkHeader, parse_siglip2_bpk_view, read_siglip2_bpk},
    config::Siglip2Config,
    hooks::decode_view_to_f32,
    model::{
        Siglip2Model, Siglip2ModelBuilder, TEXT_TOKEN_EMBED_CHUNK_PREFIX,
        TEXT_TOKEN_EMBED_CHUNK_ROWS, validate_loaded_weight_keys,
    },
    parts::{
        MAX_PRODUCTION_MANIFEST_BYTES, Siglip2BpkPartEntry, Siglip2BpkPartsManifest,
        read_bpk_parts_manifest, resolve_part_entry_path, validate_production_bpk_parts_manifest,
    },
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartLoadStats {
    pub part_count: usize,
    pub loaded_bytes: u64,
    pub sha256_verified: usize,
    pub tensors_loaded: usize,
    /// Immutable upstream identity carried by production BPK artifacts.
    #[serde(default)]
    pub artifact: Option<Siglip2ArtifactMetadata>,
    /// Source-declared monolithic payload checksum, retained for artifact provenance.
    ///
    /// For a parts manifest this value is not cryptographically bound to the loaded shards and
    /// must not be used as the identity of the loaded model. Use [`Self::loaded_weight_sha256`]
    /// for cache or embedding-schema isolation.
    #[serde(default)]
    pub weight_payload_sha256: String,
    /// Canonical SHA-256 identity of the tensor values that were successfully loaded.
    ///
    /// This is derived from each parsed tensor's name, storage dtype, shape, and raw storage
    /// bytes, then finalized only after the complete expected tensor set has been accepted. It is
    /// independent of shard order and normalizes the full text embedding to the canonical CDN
    /// chunks, so native BPK, raw safetensors, streamed shards, and out-of-order browser loads of
    /// the same stored weights produce the same identity.
    #[serde(default)]
    pub loaded_weight_sha256: String,
}

fn stats_for_manifest(manifest: &Siglip2BpkPartsManifest) -> PartLoadStats {
    PartLoadStats {
        artifact: manifest.artifact.clone(),
        weight_payload_sha256: manifest.weight_payload_sha256.clone(),
        ..PartLoadStats::default()
    }
}

/// Incremental, order-independent commitment to the tensors accepted by the loader.
///
/// Leaves are keyed by canonical tensor name in a `BTreeMap`, so finalization is independent of
/// fetch/application order. The text token table is represented as the same fixed row chunks in
/// both monolithic and sharded inputs; this makes the commitment independent of the transport
/// representation as well.
#[derive(Debug, Default)]
struct LoadedWeightIdentity {
    tensor_digests: BTreeMap<String, [u8; 32]>,
}

impl LoadedWeightIdentity {
    fn record_view(
        &mut self,
        name: &str,
        view: &safetensors::tensor::TensorView<'_>,
        config: &Siglip2Config,
    ) -> Result<(), String> {
        const FULL_TEXT_EMBEDDING: &str = "text.token_embed.weight";

        if name != FULL_TEXT_EMBEDDING {
            return self.record_tensor(name, view.dtype(), view.shape(), view.data());
        }

        let expected_shape = [config.text_vocab_size, config.hidden_dim];
        if view.shape() != expected_shape {
            return Err(format!(
                "cannot fingerprint tensor '{name}': expected shape {expected_shape:?}, got {:?}",
                view.shape()
            ));
        }
        let element_bytes = safetensors_dtype_bytes(view.dtype()).ok_or_else(|| {
            format!(
                "cannot fingerprint tensor '{name}': unsupported dtype {:?}",
                view.dtype()
            )
        })?;
        let row_bytes = config
            .hidden_dim
            .checked_mul(element_bytes)
            .ok_or_else(|| format!("tensor '{name}' row byte count overflow"))?;
        let chunk_bytes = TEXT_TOKEN_EMBED_CHUNK_ROWS
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("tensor '{name}' chunk byte count overflow"))?;
        for (index, bytes) in view.data().chunks(chunk_bytes).enumerate() {
            if bytes.len() % row_bytes != 0 {
                return Err(format!(
                    "cannot fingerprint tensor '{name}' chunk {index}: incomplete row"
                ));
            }
            let chunk_name = format!("{TEXT_TOKEN_EMBED_CHUNK_PREFIX}{index:05}");
            let chunk_shape = [bytes.len() / row_bytes, config.hidden_dim];
            self.record_tensor(&chunk_name, view.dtype(), &chunk_shape, bytes)?;
        }
        Ok(())
    }

    fn record_tensor(
        &mut self,
        name: &str,
        dtype: safetensors::tensor::Dtype,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<(), String> {
        let mut digest = Sha256::new();
        digest.update(b"burn_siglip2.loaded_tensor.v1\0");
        update_len_prefixed(&mut digest, name.as_bytes(), "tensor name")?;
        update_len_prefixed(
            &mut digest,
            safetensors_dtype_tag(dtype)?.as_bytes(),
            "tensor dtype",
        )?;
        digest.update(usize_to_u64(shape.len(), "tensor rank")?.to_be_bytes());
        for &dimension in shape {
            digest.update(usize_to_u64(dimension, "tensor dimension")?.to_be_bytes());
        }
        update_len_prefixed(&mut digest, bytes, "tensor storage")?;
        let leaf: [u8; 32] = digest.finalize().into();
        if self.tensor_digests.insert(name.to_string(), leaf).is_some() {
            return Err(format!(
                "duplicate canonical tensor encountered while fingerprinting loaded weights: '{name}'"
            ));
        }
        Ok(())
    }

    fn finish(self) -> Result<String, String> {
        if self.tensor_digests.is_empty() {
            return Err("cannot fingerprint an empty loaded tensor set".to_string());
        }
        let mut digest = Sha256::new();
        digest.update(b"burn_siglip2.loaded_weight_set.v1\0");
        digest
            .update(usize_to_u64(self.tensor_digests.len(), "loaded tensor count")?.to_be_bytes());
        for (name, tensor_digest) in self.tensor_digests {
            update_len_prefixed(&mut digest, name.as_bytes(), "tensor name")?;
            digest.update(tensor_digest);
        }
        Ok(hex::encode(digest.finalize()))
    }
}

fn update_len_prefixed(digest: &mut Sha256, bytes: &[u8], field: &str) -> Result<(), String> {
    digest.update(usize_to_u64(bytes.len(), field)?.to_be_bytes());
    digest.update(bytes);
    Ok(())
}

fn usize_to_u64(value: usize, field: &str) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("{field} length exceeds 64-bit identity encoding"))
}

fn safetensors_dtype_tag(dtype: safetensors::tensor::Dtype) -> Result<&'static str, String> {
    use safetensors::tensor::Dtype;

    match dtype {
        Dtype::BOOL => Ok("bool"),
        Dtype::U8 => Ok("u8"),
        Dtype::I8 => Ok("i8"),
        Dtype::I16 => Ok("i16"),
        Dtype::U16 => Ok("u16"),
        Dtype::F16 => Ok("f16"),
        Dtype::BF16 => Ok("bf16"),
        Dtype::I32 => Ok("i32"),
        Dtype::U32 => Ok("u32"),
        Dtype::F32 => Ok("f32"),
        Dtype::I64 => Ok("i64"),
        Dtype::U64 => Ok("u64"),
        Dtype::F64 => Ok("f64"),
        _ => Err(format!(
            "unsupported safetensors dtype for loaded-weight identity: {dtype:?}"
        )),
    }
}

fn safetensors_dtype_bytes(dtype: safetensors::tensor::Dtype) -> Option<usize> {
    use safetensors::tensor::Dtype;

    match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => Some(1),
        Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => Some(2),
        Dtype::I32 | Dtype::U32 | Dtype::F32 => Some(4),
        Dtype::I64 | Dtype::U64 | Dtype::F64 => Some(8),
        _ => None,
    }
}

/// Incremental sharded-model loader used by both native filesystems and browser fetches.
///
/// Exactly one BPK shard needs to be resident on the host at a time. Each shard is verified,
/// applied to the backend, and can then be dropped by the caller before the next fetch.
pub struct Siglip2PartsLoader<B: Backend> {
    manifest: Siglip2BpkPartsManifest,
    builder: Siglip2ModelBuilder<B>,
    loaded_weights: BTreeSet<String>,
    applied_parts: BTreeSet<usize>,
    verify_checksums: bool,
    stats: PartLoadStats,
    loaded_weight_identity: LoadedWeightIdentity,
}

impl<B: Backend> Siglip2PartsLoader<B> {
    pub fn new(
        manifest: Siglip2BpkPartsManifest,
        _device: &B::Device,
        verify_checksums: bool,
    ) -> Result<Self, String> {
        validate_production_bpk_parts_manifest(Path::new("remote.bpk.parts.json"), &manifest)?;
        let builder = Siglip2ModelBuilder::new(manifest.config.clone())?;
        let stats = stats_for_manifest(&manifest);
        Ok(Self {
            manifest,
            builder,
            loaded_weights: BTreeSet::new(),
            applied_parts: BTreeSet::new(),
            verify_checksums,
            stats,
            loaded_weight_identity: LoadedWeightIdentity::default(),
        })
    }

    pub fn manifest(&self) -> &Siglip2BpkPartsManifest {
        &self.manifest
    }

    pub fn stats(&self) -> &PartLoadStats {
        &self.stats
    }

    pub fn apply_part(
        &mut self,
        part_index: usize,
        bytes: &[u8],
        device: &B::Device,
    ) -> Result<(), String> {
        let entry = self.manifest.parts.get(part_index).ok_or_else(|| {
            format!(
                "part index {part_index} is out of range for {} manifest entries",
                self.manifest.parts.len()
            )
        })?;
        if self.applied_parts.contains(&part_index) {
            return Err(format!(
                "part index {part_index} ('{}') was applied more than once",
                entry.path
            ));
        }
        if bytes.len() as u64 != entry.bytes {
            return Err(format!(
                "part '{}' byte mismatch: expected {}, got {}",
                entry.path,
                entry.bytes,
                bytes.len()
            ));
        }
        if self.verify_checksums {
            if entry.sha256.trim().is_empty() {
                return Err(format!(
                    "part '{}' has no SHA-256 checksum but verification is required",
                    entry.path
                ));
            }
            let actual = sha256_hex(bytes);
            if !actual.eq_ignore_ascii_case(entry.sha256.trim()) {
                return Err(format!(
                    "part '{}' checksum mismatch: expected {}, got {}",
                    entry.path, entry.sha256, actual
                ));
            }
            self.stats.sha256_verified = self.stats.sha256_verified.saturating_add(1);
        }

        let package = parse_siglip2_bpk_view(bytes, None)?;
        validate_shard_header(&self.manifest, entry, &package.header)?;
        apply_safetensor_part_bytes(
            &mut self.builder,
            device,
            package.weight_blob,
            &mut self.loaded_weights,
            &mut self.stats,
            &mut self.loaded_weight_identity,
            Some(safetensors::tensor::Dtype::F16),
        )?;
        self.stats.part_count = self.stats.part_count.saturating_add(1);
        self.stats.loaded_bytes = self.stats.loaded_bytes.saturating_add(bytes.len() as u64);
        self.applied_parts.insert(part_index);
        Ok(())
    }

    pub fn finish(mut self) -> Result<(Siglip2Model<B>, PartLoadStats), String> {
        if self.applied_parts.len() != self.manifest.parts.len() {
            let missing = (0..self.manifest.parts.len())
                .filter(|index| !self.applied_parts.contains(index))
                .take(16)
                .map(|index| self.manifest.parts[index].path.clone())
                .collect::<Vec<_>>();
            return Err(format!(
                "not all manifest parts were applied ({} of {}); missing: {}",
                self.applied_parts.len(),
                self.manifest.parts.len(),
                missing.join(", ")
            ));
        }
        verify_all_expected_weights_loaded::<B>(&self.manifest.config, &self.loaded_weights)?;
        self.stats.loaded_weight_sha256 = self.loaded_weight_identity.finish()?;
        let model = self.builder.finish()?;
        Ok((model, self.stats))
    }
}

/// Load a manifest and ordered shard byte buffers without requiring filesystem access.
pub fn load_model_from_parts_bytes<B: Backend>(
    device: &B::Device,
    manifest: Siglip2BpkPartsManifest,
    parts: impl IntoIterator<Item = Vec<u8>>,
    verify_checksums: bool,
) -> Result<(Siglip2Model<B>, PartLoadStats), String> {
    let mut loader = Siglip2PartsLoader::new(manifest, device, verify_checksums)?;
    for (index, bytes) in parts.into_iter().enumerate() {
        loader.apply_part(index, bytes.as_slice(), device)?;
    }
    loader.finish()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebLoadLimits {
    pub max_manifest_parts: usize,
    pub max_total_bytes: u64,
    pub max_part_bytes: u64,
}

pub const MAX_WEB_PART_BYTES: u64 = 64 * 1024 * 1024;

impl Default for WebLoadLimits {
    fn default() -> Self {
        Self {
            max_manifest_parts: 256,
            max_total_bytes: 4 * 1024 * 1024 * 1024,
            max_part_bytes: MAX_WEB_PART_BYTES,
        }
    }
}

pub fn read_parts_manifest(path: &Path) -> Result<Siglip2BpkPartsManifest, String> {
    read_bpk_parts_manifest(path)
}

pub fn validate_manifest_for_web(
    manifest: &Siglip2BpkPartsManifest,
    limits: WebLoadLimits,
) -> Result<(), String> {
    validate_production_bpk_parts_manifest(Path::new("remote.bpk.parts.json"), manifest)?;
    if manifest.parts.len() > limits.max_manifest_parts {
        return Err(format!(
            "parts manifest has {} parts, exceeds web limit {}",
            manifest.parts.len(),
            limits.max_manifest_parts
        ));
    }
    if manifest.total_bytes > 0 && manifest.total_bytes > limits.max_total_bytes {
        return Err(format!(
            "parts manifest total_bytes {} exceeds web limit {}",
            manifest.total_bytes, limits.max_total_bytes
        ));
    }
    let mut part_bytes_total = 0u64;
    for part in &manifest.parts {
        if part.bytes > limits.max_part_bytes {
            return Err(format!(
                "part '{}' size {} exceeds web part limit {}",
                part.path, part.bytes, limits.max_part_bytes
            ));
        }
        part_bytes_total = part_bytes_total.checked_add(part.bytes).ok_or_else(|| {
            "parts manifest shard byte total overflows a 64-bit counter".to_string()
        })?;
    }
    if part_bytes_total > limits.max_total_bytes {
        return Err(format!(
            "parts manifest shard byte total {} exceeds web limit {}",
            part_bytes_total, limits.max_total_bytes
        ));
    }
    Ok(())
}

pub fn load_model_from_safetensors_path<B: Backend>(
    config: &Siglip2Config,
    device: &B::Device,
    weights_path: &Path,
) -> Result<(Siglip2Model<B>, PartLoadStats), String> {
    config.validate_production_profile()?;
    let bytes = fs::read(weights_path).map_err(|err| {
        format!(
            "failed to read safetensors weights '{}': {err}",
            weights_path.display()
        )
    })?;
    load_model_from_safetensor_bytes(config, device, bytes.as_slice(), None)
}

pub fn load_model_from_bpk_path<B: Backend>(
    device: &B::Device,
    bpk_path: &Path,
) -> Result<(Siglip2Model<B>, PartLoadStats), String> {
    let package = read_siglip2_bpk(bpk_path)?;
    if package.header.weight_sha256.trim().is_empty() {
        return Err(format!(
            "BPK '{}' has no payload SHA-256 checksum; production loading requires verification",
            bpk_path.display()
        ));
    }
    package.header.config.validate_production_profile()?;
    let artifact = package.header.artifact.as_ref().ok_or_else(|| {
        format!(
            "BPK '{}' is missing immutable artifact provenance",
            bpk_path.display()
        )
    })?;
    artifact.validate_for_config(&package.header.config)?;
    let expected_storage_dtype = artifact_storage_dtype(artifact)?;
    let (model, mut stats) = load_model_from_safetensor_bytes(
        &package.header.config,
        device,
        package.weight_blob.as_slice(),
        Some(expected_storage_dtype),
    )?;
    stats.artifact = package.header.artifact;
    stats.weight_payload_sha256 = package.header.weight_sha256;
    Ok((model, stats))
}

pub fn load_model_from_parts_manifest_path<B: Backend>(
    device: &B::Device,
    manifest_path: &Path,
    verify_checksums: bool,
) -> Result<(Siglip2Model<B>, PartLoadStats), String> {
    load_model_from_parts_manifest_path_with_stream_reader(
        device,
        manifest_path,
        verify_checksums,
        |part_path, _entry| {
            fs::File::open(part_path)
                .map_err(|err| format!("failed to open part '{}': {err}", part_path.display()))
        },
    )
}

pub fn load_model_from_parts_manifest_path_with_reader<B: Backend, F>(
    device: &B::Device,
    manifest_path: &Path,
    verify_checksums: bool,
    mut read_part_bytes: F,
) -> Result<(Siglip2Model<B>, PartLoadStats), String>
where
    F: FnMut(&Path, &Siglip2BpkPartEntry) -> Result<Vec<u8>, String>,
{
    let manifest = read_and_validate_parts_manifest(manifest_path)?;

    let mut builder = Siglip2ModelBuilder::new(manifest.config.clone())?;
    let mut loaded = BTreeSet::new();
    let mut stats = stats_for_manifest(&manifest);
    let mut loaded_weight_identity = LoadedWeightIdentity::default();

    for part in &manifest.parts {
        let part_path = resolve_part_entry_path(manifest_path, &part.path)?;
        let bytes = read_part_bytes(&part_path, part)?;
        stats.part_count = stats.part_count.saturating_add(1);
        stats.loaded_bytes = stats.loaded_bytes.saturating_add(bytes.len() as u64);

        if bytes.len() as u64 != part.bytes {
            return Err(format!(
                "part '{}' byte mismatch: expected {}, got {}",
                part_path.display(),
                part.bytes,
                bytes.len()
            ));
        }
        if verify_checksums {
            let actual = sha256_hex(bytes.as_slice());
            if !actual.eq_ignore_ascii_case(part.sha256.trim()) {
                return Err(format!(
                    "part '{}' checksum mismatch: expected {}, got {}",
                    part_path.display(),
                    part.sha256,
                    actual
                ));
            }
            stats.sha256_verified = stats.sha256_verified.saturating_add(1);
        }

        let package = parse_siglip2_bpk_view(bytes.as_slice(), Some(&part_path))?;
        validate_shard_header(&manifest, part, &package.header)?;
        apply_safetensor_part_bytes(
            &mut builder,
            device,
            package.weight_blob,
            &mut loaded,
            &mut stats,
            &mut loaded_weight_identity,
            Some(safetensors::tensor::Dtype::F16),
        )?;
    }

    verify_all_expected_weights_loaded::<B>(&manifest.config, &loaded)?;
    stats.loaded_weight_sha256 = loaded_weight_identity.finish()?;
    let model = builder.finish()?;
    Ok((model, stats))
}

pub fn load_model_from_parts_manifest_path_with_stream_reader<B: Backend, F, R>(
    device: &B::Device,
    manifest_path: &Path,
    verify_checksums: bool,
    mut open_part_reader: F,
) -> Result<(Siglip2Model<B>, PartLoadStats), String>
where
    F: FnMut(&Path, &Siglip2BpkPartEntry) -> Result<R, String>,
    R: Read,
{
    let manifest = read_and_validate_parts_manifest(manifest_path)?;

    let mut builder = Siglip2ModelBuilder::new(manifest.config.clone())?;
    let mut loaded = BTreeSet::new();
    let mut stats = stats_for_manifest(&manifest);
    let mut loaded_weight_identity = LoadedWeightIdentity::default();
    let mut buffer = Vec::new();

    for part in &manifest.parts {
        let part_path = resolve_part_entry_path(manifest_path, &part.path)?;
        let part_reader = open_part_reader(&part_path, part)?;
        read_exact_part_bounded(part_reader, &mut buffer, part.bytes, &part_path)?;

        stats.part_count = stats.part_count.saturating_add(1);
        stats.loaded_bytes = stats.loaded_bytes.saturating_add(buffer.len() as u64);

        if verify_checksums {
            let actual = sha256_hex(buffer.as_slice());
            if !actual.eq_ignore_ascii_case(part.sha256.trim()) {
                return Err(format!(
                    "part '{}' checksum mismatch: expected {}, got {}",
                    part_path.display(),
                    part.sha256,
                    actual
                ));
            }
            stats.sha256_verified = stats.sha256_verified.saturating_add(1);
        }

        let package = parse_siglip2_bpk_view(buffer.as_slice(), Some(&part_path))?;
        validate_shard_header(&manifest, part, &package.header)?;
        apply_safetensor_part_bytes(
            &mut builder,
            device,
            package.weight_blob,
            &mut loaded,
            &mut stats,
            &mut loaded_weight_identity,
            Some(safetensors::tensor::Dtype::F16),
        )?;
    }

    verify_all_expected_weights_loaded::<B>(&manifest.config, &loaded)?;
    stats.loaded_weight_sha256 = loaded_weight_identity.finish()?;
    let model = builder.finish()?;
    Ok((model, stats))
}

fn read_and_validate_parts_manifest(
    manifest_path: &Path,
) -> Result<Siglip2BpkPartsManifest, String> {
    let manifest_bytes = fs::metadata(manifest_path)
        .map_err(|err| {
            format!(
                "failed to stat BPK parts manifest '{}': {err}",
                manifest_path.display()
            )
        })?
        .len();
    if manifest_bytes == 0 || manifest_bytes > MAX_PRODUCTION_MANIFEST_BYTES {
        return Err(format!(
            "BPK parts manifest '{}' size {} must be in 1..={}",
            manifest_path.display(),
            manifest_bytes,
            MAX_PRODUCTION_MANIFEST_BYTES
        ));
    }
    let manifest = read_parts_manifest(manifest_path)?;
    validate_production_bpk_parts_manifest(manifest_path, &manifest).map_err(|message| {
        format!(
            "invalid parts manifest '{}': {message}",
            manifest_path.display()
        )
    })?;
    Ok(manifest)
}

fn validate_shard_header(
    manifest: &Siglip2BpkPartsManifest,
    part: &Siglip2BpkPartEntry,
    header: &Siglip2BpkHeader,
) -> Result<(), String> {
    if header.config != manifest.config {
        return Err(format!(
            "part '{}' config mismatch with manifest",
            part.path
        ));
    }
    let expected_artifact = manifest
        .artifact
        .as_ref()
        .expect("production manifest validation requires artifact provenance");
    if header.artifact.as_ref() != Some(expected_artifact) {
        return Err(format!(
            "part '{}' artifact provenance does not exactly match the parts manifest",
            part.path
        ));
    }
    let payload_checksum = header.weight_sha256.trim();
    if payload_checksum.len() != 64
        || !payload_checksum
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(format!(
            "part '{}' BPK header must declare a valid payload SHA-256",
            part.path
        ));
    }
    Ok(())
}

fn read_exact_part_bounded<R: Read>(
    reader: R,
    buffer: &mut Vec<u8>,
    expected_bytes: u64,
    part_path: &Path,
) -> Result<(), String> {
    let expected = usize::try_from(expected_bytes).map_err(|_| {
        format!(
            "part '{}' size {} exceeds platform addressable memory",
            part_path.display(),
            expected_bytes
        )
    })?;
    buffer.clear();
    buffer.try_reserve_exact(expected).map_err(|err| {
        format!(
            "failed to reserve {} bytes for part '{}': {err}",
            expected_bytes,
            part_path.display()
        )
    })?;

    let read_limit = expected_bytes
        .checked_add(1)
        .ok_or_else(|| format!("part '{}' byte limit overflow", part_path.display()))?;
    let mut limited = reader.take(read_limit);
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let read = limited.read(&mut chunk).map_err(|err| {
            format!(
                "failed to stream-read part '{}': {err}",
                part_path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        let new_len = buffer
            .len()
            .checked_add(read)
            .ok_or_else(|| format!("part '{}' byte count overflow", part_path.display()))?;
        if new_len > expected {
            return Err(format!(
                "part '{}' exceeds declared byte length {}",
                part_path.display(),
                expected_bytes
            ));
        }
        buffer.try_reserve(read).map_err(|err| {
            format!(
                "failed to grow buffer for part '{}': {err}",
                part_path.display()
            )
        })?;
        buffer.extend_from_slice(&chunk[..read]);
    }
    if buffer.len() != expected {
        return Err(format!(
            "part '{}' byte mismatch: expected {}, got {}",
            part_path.display(),
            expected_bytes,
            buffer.len()
        ));
    }
    Ok(())
}

fn artifact_storage_dtype(
    artifact: &Siglip2ArtifactMetadata,
) -> Result<safetensors::tensor::Dtype, String> {
    match artifact.storage_dtype.as_str() {
        "f16" => Ok(safetensors::tensor::Dtype::F16),
        "f32" => Ok(safetensors::tensor::Dtype::F32),
        other => Err(format!(
            "unsupported artifact storage dtype '{other}' after provenance validation"
        )),
    }
}

fn validate_safetensor_storage_dtype(
    bytes: &[u8],
    expected: safetensors::tensor::Dtype,
) -> Result<(), String> {
    let tensors = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|err| format!("failed to parse safetensor payload for dtype validation: {err}"))?;
    for name in tensors.names() {
        let view = tensors
            .tensor(name)
            .map_err(|err| format!("missing tensor '{name}' during dtype validation: {err}"))?;
        if view.dtype() != expected {
            return Err(format!(
                "tensor '{name}' uses {:?} storage but artifact provenance declares {:?}",
                view.dtype(),
                expected
            ));
        }
    }
    Ok(())
}

fn reject_non_finite_weight_values(name: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "tensor '{name}' contains non-finite decoded weight at element {index}: {value}"
        ));
    }
    Ok(())
}

fn load_model_from_safetensor_bytes<B: Backend>(
    config: &Siglip2Config,
    device: &B::Device,
    bytes: &[u8],
    expected_storage_dtype: Option<safetensors::tensor::Dtype>,
) -> Result<(Siglip2Model<B>, PartLoadStats), String> {
    if let Some(expected_dtype) = expected_storage_dtype {
        validate_safetensor_storage_dtype(bytes, expected_dtype)?;
    }
    let mut builder = Siglip2ModelBuilder::new(config.clone())?;
    let mut loaded = BTreeSet::new();
    let mut loaded_weight_identity = LoadedWeightIdentity::default();
    let mut stats = PartLoadStats {
        part_count: 1,
        loaded_bytes: bytes.len() as u64,
        weight_payload_sha256: sha256_hex(bytes),
        ..PartLoadStats::default()
    };
    apply_safetensor_part_bytes(
        &mut builder,
        device,
        bytes,
        &mut loaded,
        &mut stats,
        &mut loaded_weight_identity,
        expected_storage_dtype,
    )?;
    verify_all_expected_weights_loaded::<B>(config, &loaded)?;
    stats.loaded_weight_sha256 = loaded_weight_identity.finish()?;
    let model = builder.finish()?;
    Ok((model, stats))
}

fn apply_safetensor_part_bytes<B: Backend>(
    builder: &mut Siglip2ModelBuilder<B>,
    device: &B::Device,
    bytes: &[u8],
    loaded: &mut BTreeSet<String>,
    stats: &mut PartLoadStats,
    loaded_weight_identity: &mut LoadedWeightIdentity,
    expected_storage_dtype: Option<safetensors::tensor::Dtype>,
) -> Result<(), String> {
    let safetensors = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|err| format!("failed to parse safetensor part bytes: {err}"))?;
    if let Some(expected_dtype) = expected_storage_dtype {
        for name in safetensors.names() {
            let view = safetensors
                .tensor(name)
                .map_err(|err| format!("missing tensor '{name}' in part: {err}"))?;
            if view.dtype() != expected_dtype {
                return Err(format!(
                    "tensor '{name}' uses {:?} storage, expected {:?}",
                    view.dtype(),
                    expected_dtype
                ));
            }
        }
    }
    for name in safetensors.names() {
        if loaded.contains(name) {
            return Err(format!("duplicate weight key loaded from parts: '{name}'"));
        }
        validate_text_embedding_representation_before_apply(name, loaded)?;
        let view = safetensors
            .tensor(name)
            .map_err(|err| format!("missing tensor '{name}' in part: {err}"))?;
        let shape = view.shape().to_vec();
        let data = decode_view_to_f32(&view)?;
        reject_non_finite_weight_values(name, &data)?;
        builder.apply_weight(name, shape.as_slice(), data, device)?;
        loaded_weight_identity.record_view(name, &view, builder.config())?;
        loaded.insert(name.to_string());
        stats.tensors_loaded = stats.tensors_loaded.saturating_add(1);
    }
    Ok(())
}

fn verify_all_expected_weights_loaded<B: Backend>(
    config: &Siglip2Config,
    loaded: &BTreeSet<String>,
) -> Result<(), String> {
    let _ = std::marker::PhantomData::<B>;
    validate_loaded_weight_keys(config, loaded)
}

fn validate_text_embedding_representation_before_apply(
    key: &str,
    loaded: &BTreeSet<String>,
) -> Result<(), String> {
    const FULL_KEY: &str = "text.token_embed.weight";

    if key == FULL_KEY
        && loaded
            .iter()
            .any(|loaded_key| loaded_key.starts_with(TEXT_TOKEN_EMBED_CHUNK_PREFIX))
    {
        return Err(
            "cannot apply a full text token embedding after chunk tensors were loaded".to_string(),
        );
    }
    if key.starts_with(TEXT_TOKEN_EMBED_CHUNK_PREFIX) && loaded.contains(FULL_KEY) {
        return Err(
            "cannot apply text token embedding chunks after the full tensor was loaded".to_string(),
        );
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex::encode(digest.finalize())
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use burn::backend::NdArray;
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use tempfile::tempdir;

    use super::{
        LoadedWeightIdentity, WebLoadLimits, load_model_from_bpk_path,
        load_model_from_parts_manifest_path_with_reader,
        load_model_from_parts_manifest_path_with_stream_reader, reject_non_finite_weight_values,
        stats_for_manifest, validate_manifest_for_web, verify_all_expected_weights_loaded,
    };
    use crate::{
        bpk::{
            Siglip2ArtifactMetadata, build_bpk_header, build_bpk_header_with_metadata,
            write_siglip2_bpk,
        },
        config::Siglip2Config,
        parts::{MAX_PRODUCTION_PART_BYTES, Siglip2BpkPartEntry, Siglip2BpkPartsManifest},
    };

    fn production_artifact() -> Siglip2ArtifactMetadata {
        Siglip2ArtifactMetadata {
            model_variant: "base-patch16-224".to_string(),
            upstream_model_id: "google/siglip2-base-patch16-224".to_string(),
            upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            storage_dtype: "f16".to_string(),
        }
    }

    fn production_manifest() -> Siglip2BpkPartsManifest {
        Siglip2BpkPartsManifest {
            version: 2,
            source_file: "model.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: 16,
            max_part_bytes: 16,
            model_family: "siglip2".to_string(),
            config: Siglip2Config::default(),
            manifest_kind: "siglip2_bpk_parts".to_string(),
            artifact: Some(production_artifact()),
            storage_dtypes: vec!["f16".to_string()],
            tensor_count: 1,
            weight_payload_sha256: "11".repeat(32),
            source_file_sha256: "22".repeat(32),
            parts: vec![Siglip2BpkPartEntry {
                path: "model.bpk.part-00000.bpk".to_string(),
                bytes: 16,
                sha256: "33".repeat(32),
                tensors: 1,
            }],
        }
    }

    fn legacy_test_manifest(part_path: &str) -> Siglip2BpkPartsManifest {
        Siglip2BpkPartsManifest {
            version: 2,
            source_file: "model.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: 16,
            max_part_bytes: 16,
            model_family: "siglip2".to_string(),
            config: Siglip2Config::tiny_for_tests(),
            manifest_kind: String::new(),
            artifact: None,
            storage_dtypes: Vec::new(),
            tensor_count: 0,
            weight_payload_sha256: String::new(),
            source_file_sha256: String::new(),
            parts: vec![Siglip2BpkPartEntry {
                path: part_path.to_string(),
                bytes: 16,
                sha256: String::new(),
                tensors: 1,
            }],
        }
    }

    fn tensor_payload(name: &str, values: &[f32]) -> Result<Vec<u8>, String> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, vec![values.len()], &bytes)
            .map_err(|err| format!("failed to create test tensor: {err}"))?;
        serialize(&BTreeMap::from([(name.to_string(), view)]), None)
            .map_err(|err| format!("failed to serialize test tensor: {err}"))
    }

    fn loaded_identity_for_payloads(
        config: &Siglip2Config,
        payloads: &[Vec<u8>],
    ) -> Result<String, String> {
        let mut identity = LoadedWeightIdentity::default();
        for payload in payloads {
            let tensors = safetensors::SafeTensors::deserialize(payload)
                .map_err(|err| format!("failed to parse test tensor: {err}"))?;
            for name in tensors.names() {
                let view = tensors
                    .tensor(name)
                    .map_err(|err| format!("failed to read test tensor: {err}"))?;
                identity.record_view(name, &view, config)?;
            }
        }
        identity.finish()
    }

    #[test]
    fn validates_web_limits() {
        let manifest = production_manifest();
        assert!(validate_manifest_for_web(&manifest, WebLoadLimits::default()).is_ok());
    }

    #[test]
    fn manifest_payload_claim_does_not_control_loaded_weight_identity() -> Result<(), String> {
        let payload = tensor_payload("logit_scale", &[1.25])?;
        let identity = loaded_identity_for_payloads(
            &Siglip2Config::tiny_for_tests(),
            std::slice::from_ref(&payload),
        )?;

        let first = production_manifest();
        let mut second = first.clone();
        second.weight_payload_sha256 = "ff".repeat(32);
        let mut first_stats = stats_for_manifest(&first);
        let mut second_stats = stats_for_manifest(&second);
        first_stats.loaded_weight_sha256 = identity.clone();
        second_stats.loaded_weight_sha256 = loaded_identity_for_payloads(
            &Siglip2Config::tiny_for_tests(),
            std::slice::from_ref(&payload),
        )?;

        assert_ne!(
            first_stats.weight_payload_sha256,
            second_stats.weight_payload_sha256
        );
        assert_eq!(
            first_stats.loaded_weight_sha256,
            second_stats.loaded_weight_sha256
        );
        Ok(())
    }

    #[test]
    fn actual_tensor_storage_changes_loaded_weight_identity() -> Result<(), String> {
        let first = tensor_payload("logit_scale", &[1.25])?;
        let second = tensor_payload("logit_scale", &[1.5])?;
        let config = Siglip2Config::tiny_for_tests();
        let first_identity = loaded_identity_for_payloads(&config, &[first])?;
        let second_identity = loaded_identity_for_payloads(&config, &[second])?;
        assert_ne!(first_identity, second_identity);
        Ok(())
    }

    #[test]
    fn loaded_weight_identity_is_independent_of_part_application_order() -> Result<(), String> {
        let first = tensor_payload("logit_scale", &[1.25])?;
        let second = tensor_payload("logit_bias", &[-0.5])?;
        let config = Siglip2Config::tiny_for_tests();
        let forward = loaded_identity_for_payloads(&config, &[first.clone(), second.clone()])?;
        let reverse = loaded_identity_for_payloads(&config, &[second, first])?;
        assert_eq!(forward, reverse);
        Ok(())
    }

    #[test]
    fn full_and_canonical_chunked_text_embeddings_have_one_identity() -> Result<(), String> {
        let config = Siglip2Config::tiny_for_tests();
        let values = (0..config.text_vocab_size * config.hidden_dim)
            .map(|index| index as f32 / 32.0)
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let full_view = TensorView::new(
            Dtype::F32,
            vec![config.text_vocab_size, config.hidden_dim],
            &bytes,
        )
        .map_err(|err| format!("failed to create full embedding: {err}"))?;
        let chunk_view = TensorView::new(
            Dtype::F32,
            vec![config.text_vocab_size, config.hidden_dim],
            &bytes,
        )
        .map_err(|err| format!("failed to create chunk embedding: {err}"))?;

        let mut full = LoadedWeightIdentity::default();
        full.record_view("text.token_embed.weight", &full_view, &config)?;
        let mut chunked = LoadedWeightIdentity::default();
        chunked.record_view("text.token_embed.weight.chunk.00000", &chunk_view, &config)?;
        assert_eq!(full.finish()?, chunked.finish()?);
        Ok(())
    }

    #[test]
    fn read_parts_manifest_delegates_to_parts_module() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let manifest_path = dir.path().join("model.bpk.parts.json");
        let manifest = legacy_test_manifest("model.bpk.part-00000.bpk");
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        let parsed = super::read_parts_manifest(&manifest_path)?;
        assert_eq!(parsed.source_file, "model.bpk");
        Ok(())
    }

    #[test]
    fn read_parts_manifest_rejects_non_bpk_entries() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let manifest_path = dir.path().join("model.bpk.parts.json");
        let manifest = legacy_test_manifest("model.part-00000.safetensors");
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        let err = super::read_parts_manifest(&manifest_path).expect_err("non-BPK part should fail");
        assert!(err.contains(".bpk"), "unexpected error: {err}");
        Ok(())
    }

    #[test]
    fn byte_reader_rejects_blank_required_checksum_before_reading_part()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let manifest_path = dir.path().join("model.bpk.parts.json");
        let mut manifest = production_manifest();
        manifest.parts[0].sha256 = "  ".to_string();
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let error = load_model_from_parts_manifest_path_with_reader::<NdArray, _>(
            &device,
            &manifest_path,
            false,
            |_, _| panic!("part reader must not be called for an unverifiable manifest"),
        )
        .expect_err("blank required checksum must fail");
        assert!(
            error.contains("64-character SHA-256"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn stream_reader_rejects_blank_required_checksum_before_opening_part()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let manifest_path = dir.path().join("model.bpk.parts.json");
        let mut manifest = production_manifest();
        manifest.parts[0].sha256 = "  ".to_string();
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let error = load_model_from_parts_manifest_path_with_stream_reader::<
            NdArray,
            _,
            std::io::Cursor<Vec<u8>>,
        >(&device, &manifest_path, true, |_, _| {
            panic!("part reader must not be opened for an unverifiable manifest")
        })
        .expect_err("blank required checksum must fail");
        assert!(
            error.contains("64-character SHA-256"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn native_reader_rejects_oversized_manifest_part_before_backend_allocation()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let manifest_path = dir.path().join("model.bpk.parts.json");
        let mut manifest = production_manifest();
        manifest.parts[0].bytes = MAX_PRODUCTION_PART_BYTES + 1;
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let error = load_model_from_parts_manifest_path_with_reader::<NdArray, _>(
            &device,
            &manifest_path,
            true,
            |_, _| panic!("oversized manifest must fail before invoking the reader"),
        )
        .expect_err("oversized production part must fail");
        assert!(
            error.contains("must be in 1..="),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn bounded_stream_reader_rejects_more_than_declared_bytes() {
        let mut buffer = Vec::new();
        let error = super::read_exact_part_bounded(
            std::io::Cursor::new(vec![0u8; 17]),
            &mut buffer,
            16,
            std::path::Path::new("part.bpk"),
        )
        .expect_err("reader must stop after one byte beyond its declaration");
        assert!(error.contains("exceeds declared byte length"));
        assert!(buffer.len() <= 16);
    }

    #[test]
    fn shard_header_must_match_outer_artifact_provenance() {
        let manifest = production_manifest();
        let header = build_bpk_header(Siglip2Config::default(), &[1, 2, 3, 4]);
        let error = super::validate_shard_header(&manifest, &manifest.parts[0], &header)
            .expect_err("missing per-shard provenance must fail");
        assert!(error.contains("artifact provenance"));
    }

    #[test]
    fn production_bpk_loader_rejects_blank_payload_checksum()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let bpk_path = dir.path().join("model.bpk");
        let payload = [1_u8, 2, 3, 4];
        let mut header = build_bpk_header(Siglip2Config::default(), payload.as_slice());
        header.weight_sha256 = "  ".to_string();
        write_siglip2_bpk(&bpk_path, &header, payload.as_slice())?;

        // The compatibility parser/writer accepts legacy blank checksums, but the
        // production model loader must reject one before allocating model tensors.
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let error = load_model_from_bpk_path::<NdArray>(&device, &bpk_path)
            .expect_err("production BPK loading must require a payload checksum");
        assert!(
            error.contains("no payload SHA-256 checksum"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn production_bpk_loader_requires_artifact_provenance_before_model_allocation()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let bpk_path = dir.path().join("model.bpk");
        let payload = tensor_payload("logit_scale", &[1.0])?;
        let header = build_bpk_header(Siglip2Config::default(), payload.as_slice());
        write_siglip2_bpk(&bpk_path, &header, payload.as_slice())?;

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let error = load_model_from_bpk_path::<NdArray>(&device, &bpk_path)
            .expect_err("production BPK without provenance must fail");
        assert!(
            error.contains("missing immutable artifact provenance"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn production_bpk_loader_rejects_provenance_dtype_mismatch_before_model_allocation()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let bpk_path = dir.path().join("model.bpk");
        let payload = tensor_payload("logit_scale", &[1.0])?;
        let header = build_bpk_header_with_metadata(
            Siglip2Config::default(),
            payload.as_slice(),
            Some(production_artifact()),
        );
        write_siglip2_bpk(&bpk_path, &header, payload.as_slice())?;

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let error = load_model_from_bpk_path::<NdArray>(&device, &bpk_path)
            .expect_err("F32 payload with F16 provenance must fail");
        assert!(
            error.contains("artifact provenance declares F16"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn decoded_weights_must_be_finite() {
        let error = reject_non_finite_weight_values("logit_scale", &[0.0, f32::NAN])
            .expect_err("NaN weight must fail");
        assert!(error.contains("non-finite decoded weight"));
        assert!(reject_non_finite_weight_values("logit_scale", &[0.0, 1.0]).is_ok());
    }

    #[test]
    fn accepts_exactly_one_complete_token_embedding_representation() {
        const FULL_KEY: &str = "text.token_embed.weight";

        let config = Siglip2Config::tiny_for_tests();
        let full = crate::Siglip2Model::<NdArray>::expected_weight_key_set(&config);
        assert!(verify_all_expected_weights_loaded::<NdArray>(&config, &full).is_ok());

        let chunks =
            crate::Siglip2Model::<NdArray>::expected_text_token_embedding_chunk_specs(&config)
                .into_iter()
                .map(|spec| spec.key)
                .collect::<BTreeSet<_>>();
        let mut chunked = full.clone();
        chunked.remove(FULL_KEY);
        chunked.extend(chunks.iter().cloned());
        assert!(verify_all_expected_weights_loaded::<NdArray>(&config, &chunked).is_ok());

        let mut mixed = chunked.clone();
        mixed.insert(FULL_KEY.to_string());
        let mixed_error = verify_all_expected_weights_loaded::<NdArray>(&config, &mixed)
            .expect_err("mixed full/chunk embedding must fail");
        assert!(mixed_error.contains("never both"));

        let mut incomplete = chunked;
        incomplete.remove(chunks.first().expect("tiny config has an embedding chunk"));
        let missing_error = verify_all_expected_weights_loaded::<NdArray>(&config, &incomplete)
            .expect_err("incomplete embedding chunks must fail");
        assert!(missing_error.contains("missing required weight tensors"));
    }
}
