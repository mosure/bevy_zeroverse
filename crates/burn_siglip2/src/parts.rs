use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
    time::UNIX_EPOCH,
};

use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView, serialize},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    bpk::{
        Siglip2ArtifactMetadata, build_bpk_header_with_metadata, read_siglip2_bpk,
        write_siglip2_bpk,
    },
    config::Siglip2Config,
    model::{TEXT_TOKEN_EMBED_CHUNK_PREFIX, TEXT_TOKEN_EMBED_CHUNK_ROWS},
};

pub const PARTS_MANIFEST_VERSION: u32 = 2;
pub const MAX_PRODUCTION_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
pub const MAX_PRODUCTION_MANIFEST_PARTS: usize = 256;
pub const MAX_PRODUCTION_PART_BYTES: u64 = 1024 * 1024 * 1024;
pub const MAX_PRODUCTION_AGGREGATE_PART_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const ONE_MIB: u64 = 1024 * 1024;
const BPK_PARTS_MANIFEST_SUFFIX: &str = ".bpk.parts.json";
const BPK_PART_FILE_SUFFIX: &str = ".bpk";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Siglip2BpkPartsManifest {
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub source_modified_unix_ms: u64,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub max_part_bytes: u64,
    #[serde(default = "default_model_family")]
    pub model_family: String,
    pub config: Siglip2Config,
    #[serde(default)]
    pub manifest_kind: String,
    #[serde(default)]
    pub artifact: Option<Siglip2ArtifactMetadata>,
    #[serde(default)]
    pub storage_dtypes: Vec<String>,
    #[serde(default)]
    pub tensor_count: usize,
    #[serde(default)]
    pub weight_payload_sha256: String,
    #[serde(default)]
    pub source_file_sha256: String,
    #[serde(default)]
    pub parts: Vec<Siglip2BpkPartEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Siglip2BpkPartEntry {
    pub path: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub tensors: usize,
}

#[derive(Debug, Clone)]
pub struct BpkPartsReport {
    pub manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
    /// Requested grouping target after the portable text-embedding transform.
    pub requested_max_part_bytes: u64,
    /// Largest serialized part, including its BPK and safetensors headers.
    pub actual_max_part_bytes: u64,
    pub parts_exceeding_requested_limit: usize,
    pub oversized_tensors: Vec<Siglip2OversizedTensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Siglip2OversizedTensorEntry {
    pub name: String,
    pub tensor_bytes: u64,
    pub part_path: String,
}

#[derive(Debug, Clone)]
struct TensorRecord {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct ShardingMetadata<'a> {
    strategy: &'static str,
    requested_max_part_bytes: u64,
    actual_max_part_bytes: u64,
    limit_enforced: bool,
    parts_exceeding_requested_limit: Vec<&'a Siglip2BpkPartEntry>,
    oversized_tensors: &'a [Siglip2OversizedTensorEntry],
}

pub fn bpk_parts_manifest_path(bpk_path: &Path) -> PathBuf {
    let file_name = bpk_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    bpk_path.with_file_name(format!("{file_name}.parts.json"))
}

pub fn read_bpk_parts_manifest(path: &Path) -> Result<Siglip2BpkPartsManifest, String> {
    let bytes = fs::read(path).map_err(|err| {
        format!(
            "failed to read BPK parts manifest '{}': {err}",
            path.display()
        )
    })?;
    let manifest = serde_json::from_slice(&bytes).map_err(|err| {
        format!(
            "failed to parse BPK parts manifest '{}': {err}",
            path.display()
        )
    })?;
    validate_bpk_parts_layout(path, &manifest)?;
    Ok(manifest)
}

pub fn resolve_part_entry_path(manifest_path: &Path, entry_path: &str) -> Result<PathBuf, String> {
    validate_safe_bundle_file_name(entry_path, "part path")?;
    manifest_path
        .parent()
        .map(|parent| parent.join(entry_path))
        .ok_or_else(|| format!("invalid parts manifest path '{}'", manifest_path.display()))
}

pub fn manifest_is_complete(manifest_path: &Path, source_bpk_path: Option<&Path>) -> bool {
    let Ok(manifest) = read_bpk_parts_manifest(manifest_path) else {
        return false;
    };
    if manifest.parts.is_empty() {
        return false;
    }
    let verify_part_checksums = source_bpk_path.is_some();
    if let Some(source_bpk_path) = source_bpk_path {
        if !source_bpk_path.exists() {
            return false;
        }
        if manifest.source_file
            != source_bpk_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
        {
            return false;
        }
        let Ok(source_metadata) = fs::metadata(source_bpk_path) else {
            return false;
        };
        if manifest.total_bytes > 0 && source_metadata.len() != manifest.total_bytes {
            return false;
        }
        if manifest.source_modified_unix_ms > 0
            && file_modified_unix_ms(source_bpk_path)
                .map(|mtime| mtime != manifest.source_modified_unix_ms)
                .unwrap_or(true)
        {
            return false;
        }
    }
    manifest.parts.iter().all(|entry| {
        let Some(path) = resolve_part_entry_path(manifest_path, &entry.path).ok() else {
            return false;
        };
        let Ok(metadata) = fs::metadata(&path) else {
            return false;
        };
        if !metadata.is_file() || (entry.bytes > 0 && metadata.len() != entry.bytes) {
            return false;
        }
        if verify_part_checksums
            && !entry.sha256.trim().is_empty()
            && sha256_file(&path)
                .map(|actual| !actual.eq_ignore_ascii_case(entry.sha256.trim()))
                .unwrap_or(true)
        {
            return false;
        }
        true
    })
}

pub fn validate_bpk_parts_layout(
    manifest_path: &Path,
    manifest: &Siglip2BpkPartsManifest,
) -> Result<(), String> {
    let manifest_name = manifest_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            format!(
                "invalid BPK parts manifest path '{}'",
                manifest_path.display()
            )
        })?;
    if !manifest_name.ends_with(BPK_PARTS_MANIFEST_SUFFIX) {
        return Err(format!(
            "BPK parts manifest '{}' must end with '{}'",
            manifest_path.display(),
            BPK_PARTS_MANIFEST_SUFFIX
        ));
    }
    validate_safe_bundle_file_name(&manifest.source_file, "source_file").map_err(|err| {
        format!(
            "invalid source_file in BPK parts manifest '{}': {err}",
            manifest_path.display()
        )
    })?;
    if !manifest.source_file.ends_with(BPK_PART_FILE_SUFFIX) {
        return Err(format!(
            "BPK parts manifest '{}' must reference a '.bpk' source file, got '{}'",
            manifest_path.display(),
            manifest.source_file
        ));
    }
    let mut unique_paths = BTreeSet::new();
    for entry in &manifest.parts {
        validate_bpk_part_entry(manifest_path, entry)?;
        if !unique_paths.insert(entry.path.as_str()) {
            return Err(format!(
                "duplicate BPK part entry '{}' in '{}'",
                entry.path,
                manifest_path.display()
            ));
        }
    }
    Ok(())
}

/// Validate the bounded, checksummed F16 manifest contract used by production loaders.
///
/// [`read_bpk_parts_manifest`] and [`validate_bpk_parts_layout`] intentionally remain compatible
/// with legacy/test manifests. Native and browser model loaders must call this stronger validator
/// before allocating backend model tensors or reading any shard payload.
pub fn validate_production_bpk_parts_manifest(
    manifest_path: &Path,
    manifest: &Siglip2BpkPartsManifest,
) -> Result<(), String> {
    validate_bpk_parts_layout(manifest_path, manifest)?;
    if manifest.version != PARTS_MANIFEST_VERSION {
        return Err(format!(
            "unsupported parts manifest version {} (expected {})",
            manifest.version, PARTS_MANIFEST_VERSION
        ));
    }
    if manifest.manifest_kind != "siglip2_bpk_parts" {
        return Err(format!(
            "unsupported parts manifest_kind '{}' (expected 'siglip2_bpk_parts')",
            manifest.manifest_kind
        ));
    }
    if !manifest.model_family.eq_ignore_ascii_case("siglip2") {
        return Err(format!(
            "unsupported parts manifest model_family '{}'",
            manifest.model_family
        ));
    }
    manifest.config.validate_production_profile()?;
    if manifest.parts.is_empty() {
        return Err("parts manifest has no parts".to_string());
    }
    if manifest.parts.len() > MAX_PRODUCTION_MANIFEST_PARTS {
        return Err(format!(
            "parts manifest has {} entries, exceeds production limit {}",
            manifest.parts.len(),
            MAX_PRODUCTION_MANIFEST_PARTS
        ));
    }
    if manifest.total_bytes == 0 || manifest.total_bytes > MAX_PRODUCTION_AGGREGATE_PART_BYTES {
        return Err(format!(
            "parts manifest total_bytes {} must be in 1..={}",
            manifest.total_bytes, MAX_PRODUCTION_AGGREGATE_PART_BYTES
        ));
    }
    if manifest.max_part_bytes == 0 {
        return Err("parts manifest max_part_bytes must be non-zero".to_string());
    }

    let artifact = manifest
        .artifact
        .as_ref()
        .ok_or_else(|| "production parts manifest is missing artifact provenance".to_string())?;
    artifact.validate_for_config(&manifest.config)?;
    if artifact.storage_dtype != "f16" {
        return Err(format!(
            "production parts manifest artifact must use f16 storage, got '{}'",
            artifact.storage_dtype
        ));
    }
    if manifest.storage_dtypes.as_slice() != ["f16"] {
        return Err(format!(
            "production parts manifest storage_dtypes must be exactly ['f16'], got {:?}",
            manifest.storage_dtypes
        ));
    }
    validate_required_sha256(&manifest.weight_payload_sha256, "weight_payload_sha256")?;
    validate_required_sha256(&manifest.source_file_sha256, "source_file_sha256")?;
    if manifest.tensor_count == 0 {
        return Err("production parts manifest tensor_count must be non-zero".to_string());
    }

    let mut aggregate_bytes = 0u64;
    let mut aggregate_tensors = 0usize;
    for part in &manifest.parts {
        if part.bytes == 0 || part.bytes > MAX_PRODUCTION_PART_BYTES {
            return Err(format!(
                "part '{}' bytes {} must be in 1..={}",
                part.path, part.bytes, MAX_PRODUCTION_PART_BYTES
            ));
        }
        validate_required_sha256(&part.sha256, &format!("part '{}' sha256", part.path))?;
        if part.tensors == 0 {
            return Err(format!(
                "part '{}' must declare at least one tensor",
                part.path
            ));
        }
        aggregate_bytes = aggregate_bytes.checked_add(part.bytes).ok_or_else(|| {
            "parts manifest aggregate shard bytes overflow a 64-bit counter".to_string()
        })?;
        aggregate_tensors = aggregate_tensors
            .checked_add(part.tensors)
            .ok_or_else(|| "parts manifest aggregate tensor count overflow".to_string())?;
    }
    if aggregate_bytes > MAX_PRODUCTION_AGGREGATE_PART_BYTES {
        return Err(format!(
            "parts manifest aggregate shard bytes {} exceeds production limit {}",
            aggregate_bytes, MAX_PRODUCTION_AGGREGATE_PART_BYTES
        ));
    }
    if aggregate_tensors != manifest.tensor_count {
        return Err(format!(
            "parts manifest tensor_count {} does not match part-entry total {}",
            manifest.tensor_count, aggregate_tensors
        ));
    }
    Ok(())
}

pub fn validate_bpk_part_entry(
    manifest_path: &Path,
    entry: &Siglip2BpkPartEntry,
) -> Result<(), String> {
    validate_safe_bundle_file_name(&entry.path, "part path").map_err(|err| {
        format!(
            "invalid BPK part entry '{}' in '{}': {err}",
            entry.path,
            manifest_path.display()
        )
    })?;
    let entry_name = entry.path.as_str();
    if !entry_name.ends_with(BPK_PART_FILE_SUFFIX) {
        return Err(format!(
            "BPK part entry '{}' in '{}' must end with '{}'",
            entry.path,
            manifest_path.display(),
            BPK_PART_FILE_SUFFIX
        ));
    }
    let checksum = entry.sha256.trim();
    if !checksum.is_empty()
        && (checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err(format!(
            "BPK part entry '{}' in '{}' has invalid sha256 '{}'",
            entry.path,
            manifest_path.display(),
            entry.sha256
        ));
    }
    Ok(())
}

fn validate_safe_bundle_file_name(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if value.contains('/') || value.contains('\\') {
        return Err(format!("{field} must be a single file name, got '{value}'"));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(format!(
            "{field} contains unsupported characters (allowed: A-Z, a-z, 0-9, '.', '_', '-')"
        ));
    }
    let path = Path::new(value);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(format!(
            "{field} must be a relative file name, got '{value}'"
        ));
    }
    Ok(())
}

fn validate_required_sha256(value: &str, field: &str) -> Result<(), String> {
    let checksum = value.trim();
    if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} must be a non-empty 64-character SHA-256"));
    }
    Ok(())
}

pub fn write_bpk_parts(
    bpk_path: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> Result<Option<BpkPartsReport>, String> {
    if !bpk_path.exists() {
        return Err(format!(
            "BPK does not exist for parting: {}",
            bpk_path.display()
        ));
    }

    let max_part_bytes = max_part_size_mib
        .max(1)
        .checked_mul(ONE_MIB)
        .ok_or_else(|| "max part size overflow".to_string())?;
    let manifest_path = bpk_parts_manifest_path(bpk_path);
    if manifest_path.exists() && !overwrite && manifest_is_complete(&manifest_path, Some(bpk_path))
    {
        let manifest = read_bpk_parts_manifest(&manifest_path)?;
        if manifest.max_part_bytes != max_part_bytes {
            return Err(format!(
                "existing parts manifest '{}' uses requested max {} bytes, but this import requested {}; pass overwrite to regenerate",
                manifest_path.display(),
                manifest.max_part_bytes,
                max_part_bytes
            ));
        }
        return Ok(Some(build_parts_report(manifest_path, &manifest)?));
    }

    if overwrite {
        cleanup_existing_parts(bpk_path, &manifest_path)?;
    }
    ensure_parent_dir(&manifest_path)?;

    let package = read_siglip2_bpk(bpk_path)?;
    let tensor_records = collect_tensor_records(package.weight_blob.as_slice())?;
    let package_header = package.header;
    drop(package.weight_blob);
    if tensor_records.is_empty() {
        return Err(format!("BPK '{}' contains no tensors", bpk_path.display()));
    }
    let (tensor_records, token_embedding_chunked) =
        split_text_token_embedding_for_portable_shards(tensor_records, &package_header.config)?;
    validate_artifact_storage_dtype(&tensor_records, package_header.artifact.as_ref())?;
    let storage_dtypes = collect_storage_dtypes(&tensor_records);
    let tensor_count = tensor_records.len();
    let groups = split_tensor_records(tensor_records, max_part_bytes);
    let source_file_name = bpk_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid BPK file name '{}'", bpk_path.display()))?;
    validate_safe_bundle_file_name(source_file_name, "source BPK file name")?;

    let mut part_entries = Vec::with_capacity(groups.len());
    let mut part_paths = Vec::with_capacity(groups.len());
    let mut oversized_tensors = Vec::new();
    for (index, group) in groups.iter().enumerate() {
        let part_name = format!("{source_file_name}.part-{index:05}.bpk");
        let part_path = bpk_path.with_file_name(&part_name);
        let part_blob = serialize_group(group)?;
        let header = build_bpk_header_with_metadata(
            package_header.config.clone(),
            &part_blob,
            package_header.artifact.clone(),
        );
        write_siglip2_bpk(&part_path, &header, &part_blob)?;
        let bytes = fs::metadata(&part_path)
            .map_err(|err| format!("failed to stat BPK part '{}': {err}", part_path.display()))?
            .len();
        let sha256 = sha256_file(&part_path)?;
        part_entries.push(Siglip2BpkPartEntry {
            path: part_name.clone(),
            bytes,
            sha256,
            tensors: group.len(),
        });
        oversized_tensors.extend(
            group
                .iter()
                .filter(|record| record.data.len() as u64 > max_part_bytes)
                .map(|record| Siglip2OversizedTensorEntry {
                    name: record.name.clone(),
                    tensor_bytes: record.data.len() as u64,
                    part_path: part_name.clone(),
                }),
        );
        part_paths.push(part_path);
    }

    let source_file_sha256 = sha256_file(bpk_path)?;
    let manifest = Siglip2BpkPartsManifest {
        version: default_manifest_version(),
        source_file: source_file_name.to_string(),
        source_modified_unix_ms: file_modified_unix_ms(bpk_path).unwrap_or(0),
        total_bytes: fs::metadata(bpk_path)
            .map_err(|err| format!("failed to stat BPK '{}': {err}", bpk_path.display()))?
            .len(),
        max_part_bytes,
        model_family: default_model_family(),
        config: package_header.config.clone(),
        manifest_kind: "siglip2_bpk_parts".to_string(),
        artifact: package_header.artifact.clone(),
        storage_dtypes,
        tensor_count,
        weight_payload_sha256: package_header.weight_sha256.clone(),
        source_file_sha256,
        parts: part_entries,
    };
    validate_bpk_parts_layout(&manifest_path, &manifest)?;
    let actual_max_part_bytes = manifest
        .parts
        .iter()
        .map(|entry| entry.bytes)
        .max()
        .unwrap_or(0);
    let parts_exceeding_requested_limit = manifest
        .parts
        .iter()
        .filter(|entry| entry.bytes > max_part_bytes)
        .collect::<Vec<_>>();
    let sharding = ShardingMetadata {
        strategy: if token_embedding_chunked {
            "portable_text_embedding_chunks_v1"
        } else {
            "whole_tensor_best_effort"
        },
        requested_max_part_bytes: max_part_bytes,
        actual_max_part_bytes,
        limit_enforced: parts_exceeding_requested_limit.is_empty(),
        parts_exceeding_requested_limit,
        oversized_tensors: &oversized_tensors,
    };
    let manifest_bytes = serialize_manifest_document(&manifest, &sharding)?;
    fs::write(&manifest_path, manifest_bytes).map_err(|err| {
        format!(
            "failed to write BPK parts manifest '{}': {err}",
            manifest_path.display()
        )
    })?;

    Ok(Some(BpkPartsReport {
        manifest_path,
        part_paths,
        total_bytes: manifest.total_bytes,
        requested_max_part_bytes: max_part_bytes,
        actual_max_part_bytes,
        parts_exceeding_requested_limit: sharding.parts_exceeding_requested_limit.len(),
        oversized_tensors,
    }))
}

fn collect_tensor_records(weight_blob: &[u8]) -> Result<Vec<TensorRecord>, String> {
    let tensors = SafeTensors::deserialize(weight_blob)
        .map_err(|err| format!("failed to parse BPK safetensors payload: {err}"))?;
    let mut records = Vec::new();
    for (name, view) in tensors.iter() {
        let record = TensorRecord {
            name: name.to_string(),
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        };
        validate_tensor_record(&record)?;
        records.push(record);
    }
    records.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(records)
}

fn split_text_token_embedding_for_portable_shards(
    records: Vec<TensorRecord>,
    config: &Siglip2Config,
) -> Result<(Vec<TensorRecord>, bool), String> {
    const FULL_KEY: &str = "text.token_embed.weight";

    let mut transformed = Vec::with_capacity(records.len() + 16);
    let mut chunked = false;
    for record in records {
        if record.name.starts_with(TEXT_TOKEN_EMBED_CHUNK_PREFIX) {
            return Err(format!(
                "source BPK unexpectedly contains derived token embedding key '{}'",
                record.name
            ));
        }
        if record.name != FULL_KEY {
            transformed.push(record);
            continue;
        }
        if chunked {
            return Err(format!("source BPK contains duplicate tensor '{FULL_KEY}'"));
        }
        let expected_shape = [config.text_vocab_size, config.hidden_dim];
        if record.shape.as_slice() != expected_shape {
            return Err(format!(
                "tensor '{FULL_KEY}' shape mismatch before chunking: expected {expected_shape:?}, got {:?}",
                record.shape
            ));
        }
        let element_bytes = safetensors_dtype_bytes(record.dtype).ok_or_else(|| {
            format!(
                "tensor '{FULL_KEY}' uses unsupported storage dtype {:?}",
                record.dtype
            )
        })?;
        let row_bytes = config
            .hidden_dim
            .checked_mul(element_bytes)
            .ok_or_else(|| format!("tensor '{FULL_KEY}' row byte count overflow"))?;
        let chunk_bytes = TEXT_TOKEN_EMBED_CHUNK_ROWS
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("tensor '{FULL_KEY}' chunk byte count overflow"))?;
        for (index, data) in record.data.chunks(chunk_bytes).enumerate() {
            if data.len() % row_bytes != 0 {
                return Err(format!(
                    "tensor '{FULL_KEY}' chunk {index} does not contain complete rows"
                ));
            }
            let chunk = TensorRecord {
                name: format!("{TEXT_TOKEN_EMBED_CHUNK_PREFIX}{index:05}"),
                dtype: record.dtype,
                shape: vec![data.len() / row_bytes, config.hidden_dim],
                data: data.to_vec(),
            };
            validate_tensor_record(&chunk)?;
            transformed.push(chunk);
        }
        chunked = true;
    }
    transformed.sort_by(|left, right| left.name.cmp(&right.name));
    Ok((transformed, chunked))
}

fn validate_tensor_record(record: &TensorRecord) -> Result<(), String> {
    if record.name.trim().is_empty() {
        return Err("BPK contains an empty tensor name".to_string());
    }
    if record.shape.contains(&0) {
        return Err(format!(
            "tensor '{}' has a zero-sized dimension in {:?}",
            record.name, record.shape
        ));
    }
    let element_bytes = safetensors_dtype_bytes(record.dtype).ok_or_else(|| {
        format!(
            "tensor '{}' uses unsupported storage dtype {:?}",
            record.name, record.dtype
        )
    })?;
    let elements = record.shape.iter().try_fold(1usize, |count, dim| {
        count.checked_mul(*dim).ok_or_else(|| {
            format!(
                "tensor '{}' shape element count overflows: {:?}",
                record.name, record.shape
            )
        })
    })?;
    let expected_bytes = elements.checked_mul(element_bytes).ok_or_else(|| {
        format!(
            "tensor '{}' byte length overflows for shape {:?}",
            record.name, record.shape
        )
    })?;
    if record.data.len() != expected_bytes {
        return Err(format!(
            "tensor '{}' byte mismatch: shape {:?} with {:?} requires {}, got {}",
            record.name,
            record.shape,
            record.dtype,
            expected_bytes,
            record.data.len()
        ));
    }
    Ok(())
}

fn safetensors_dtype_bytes(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => Some(1),
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => Some(2),
        Dtype::F32 | Dtype::I32 | Dtype::U32 => Some(4),
        Dtype::F64 | Dtype::I64 | Dtype::U64 => Some(8),
        _ => None,
    }
}

fn validate_artifact_storage_dtype(
    records: &[TensorRecord],
    artifact: Option<&Siglip2ArtifactMetadata>,
) -> Result<(), String> {
    let Some(artifact) = artifact else {
        return Ok(());
    };
    let expected = match artifact.storage_dtype.as_str() {
        "f16" => Dtype::F16,
        "f32" => Dtype::F32,
        other => return Err(format!("unsupported artifact storage dtype '{other}'")),
    };
    let mismatched = records
        .iter()
        .filter(|record| record.dtype != expected)
        .take(8)
        .map(|record| format!("{} ({:?})", record.name, record.dtype))
        .collect::<Vec<_>>();
    if !mismatched.is_empty() {
        return Err(format!(
            "artifact declares storage_dtype '{}' but tensors use other dtypes (showing up to 8): {}",
            artifact.storage_dtype,
            mismatched.join(", ")
        ));
    }
    Ok(())
}

fn collect_storage_dtypes(records: &[TensorRecord]) -> Vec<String> {
    records
        .iter()
        .map(|record| format!("{:?}", record.dtype).to_ascii_lowercase())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn split_tensor_records(records: Vec<TensorRecord>, max_part_bytes: u64) -> Vec<Vec<TensorRecord>> {
    // Leave room for the safetensors index and BPK JSON header so the serialized file, rather
    // than only its raw tensor payloads, honors the requested CDN limit.
    let payload_budget = max_part_bytes.saturating_sub(ONE_MIB).max(1);
    let mut groups = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0u64;

    for record in records {
        let record_bytes = record.data.len() as u64;
        if !current.is_empty() && current_bytes.saturating_add(record_bytes) > payload_budget {
            groups.push(current);
            current = Vec::new();
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(record_bytes);
        current.push(record);
    }

    if !current.is_empty() {
        groups.push(current);
    }

    groups
}

fn serialize_group(group: &[TensorRecord]) -> Result<Vec<u8>, String> {
    let mut views = BTreeMap::new();
    for record in group {
        let view = TensorView::new(record.dtype, record.shape.clone(), &record.data)
            .map_err(|err| format!("failed to build TensorView for '{}': {err}", record.name))?;
        views.insert(record.name.clone(), view);
    }
    serialize(&views, None).map_err(|err| format!("failed to serialize safetensors group: {err}"))
}

fn serialize_manifest_document(
    manifest: &Siglip2BpkPartsManifest,
    sharding: &ShardingMetadata<'_>,
) -> Result<Vec<u8>, String> {
    let mut document = serde_json::to_value(manifest)
        .map_err(|err| format!("failed to serialize BPK parts manifest: {err}"))?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| "BPK parts manifest did not serialize to an object".to_string())?;
    object.insert(
        "sharding".to_string(),
        serde_json::to_value(sharding)
            .map_err(|err| format!("failed to serialize sharding metadata: {err}"))?,
    );
    serde_json::to_vec_pretty(&document)
        .map_err(|err| format!("failed to serialize BPK parts manifest: {err}"))
}

fn build_parts_report(
    manifest_path: PathBuf,
    manifest: &Siglip2BpkPartsManifest,
) -> Result<BpkPartsReport, String> {
    let part_paths = manifest
        .parts
        .iter()
        .map(|entry| resolve_part_entry_path(&manifest_path, &entry.path))
        .collect::<Result<Vec<_>, _>>()?;
    let actual_max_part_bytes = manifest
        .parts
        .iter()
        .map(|entry| entry.bytes)
        .max()
        .unwrap_or(0);
    let parts_exceeding_requested_limit = manifest
        .parts
        .iter()
        .filter(|entry| entry.bytes > manifest.max_part_bytes)
        .count();
    let oversized_tensors = read_oversized_tensor_entries(&manifest_path).unwrap_or_default();
    Ok(BpkPartsReport {
        manifest_path,
        part_paths,
        total_bytes: manifest.total_bytes,
        requested_max_part_bytes: manifest.max_part_bytes,
        actual_max_part_bytes,
        parts_exceeding_requested_limit,
        oversized_tensors,
    })
}

fn read_oversized_tensor_entries(
    manifest_path: &Path,
) -> Result<Vec<Siglip2OversizedTensorEntry>, String> {
    let bytes = fs::read(manifest_path).map_err(|err| {
        format!(
            "failed to read BPK parts manifest '{}': {err}",
            manifest_path.display()
        )
    })?;
    let document = serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|err| {
        format!(
            "failed to parse BPK parts manifest '{}': {err}",
            manifest_path.display()
        )
    })?;
    let Some(entries) = document
        .get("sharding")
        .and_then(|value| value.get("oversized_tensors"))
    else {
        return Ok(Vec::new());
    };
    serde_json::from_value(entries.clone()).map_err(|err| {
        format!(
            "failed to parse oversized tensor metadata in '{}': {err}",
            manifest_path.display()
        )
    })
}

fn cleanup_existing_parts(bpk_path: &Path, manifest_path: &Path) -> Result<(), String> {
    if manifest_path.exists() {
        if let Ok(manifest) = read_bpk_parts_manifest(manifest_path) {
            for entry in manifest.parts {
                let part_path = resolve_part_entry_path(manifest_path, &entry.path)?;
                if part_path.exists() {
                    fs::remove_file(&part_path).map_err(|err| {
                        format!(
                            "failed to remove stale part '{}': {err}",
                            part_path.display()
                        )
                    })?;
                }
            }
        }
        fs::remove_file(manifest_path).map_err(|err| {
            format!(
                "failed to remove stale parts manifest '{}': {err}",
                manifest_path.display()
            )
        })?;
    }
    let parent = bpk_path.parent().unwrap_or_else(|| Path::new("."));
    let source_name = bpk_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("invalid BPK path '{}'", bpk_path.display()))?;
    let part_prefix = format!("{source_name}.part-");
    for entry in fs::read_dir(parent).map_err(|err| {
        format!(
            "failed to scan part directory '{}': {err}",
            parent.display()
        )
    })? {
        let entry = entry.map_err(|err| {
            format!(
                "failed to inspect part directory '{}': {err}",
                parent.display()
            )
        })?;
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if file_name.starts_with(&part_prefix) && file_name.ends_with(BPK_PART_FILE_SUFFIX) {
            let path = entry.path();
            if entry
                .file_type()
                .map_err(|err| format!("failed to stat stale part '{}': {err}", path.display()))?
                .is_file()
            {
                fs::remove_file(&path).map_err(|err| {
                    format!("failed to remove stale part '{}': {err}", path.display())
                })?;
            }
        }
    }
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create parent directory '{}': {err}",
                parent.display()
            )
        })?;
    }
    Ok(())
}

fn file_modified_unix_ms(path: &Path) -> Result<u64, String> {
    let modified = fs::metadata(path)
        .map_err(|err| format!("failed to stat '{}': {err}", path.display()))?
        .modified()
        .map_err(|err| format!("failed to read mtime '{}': {err}", path.display()))?;
    let duration = modified
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("mtime before unix epoch '{}': {err}", path.display()))?;
    Ok(duration.as_millis() as u64)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path)
        .map_err(|err| format!("failed to read '{}' for checksum: {err}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|err| format!("failed to read '{}' for checksum: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn default_manifest_version() -> u32 {
    PARTS_MANIFEST_VERSION
}

fn default_model_family() -> String {
    "siglip2".to_string()
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
    };

    use safetensors::tensor::{Dtype, TensorView};
    use tempfile::tempdir;

    use super::{
        PARTS_MANIFEST_VERSION, Siglip2BpkPartEntry, Siglip2BpkPartsManifest, TensorRecord,
        bpk_parts_manifest_path, read_bpk_parts_manifest, resolve_part_entry_path,
        split_text_token_embedding_for_portable_shards, validate_bpk_parts_layout,
        validate_production_bpk_parts_manifest,
    };
    use crate::{
        bpk::{
            Siglip2ArtifactMetadata, build_bpk_header, build_bpk_header_with_metadata,
            write_siglip2_bpk,
        },
        config::Siglip2Config,
    };

    #[test]
    fn resolves_relative_part_path() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let manifest_path = dir.path().join("model.bpk.parts.json");
        let resolved = resolve_part_entry_path(&manifest_path, "model.bpk.part-00000.bpk")?;
        assert_eq!(resolved, dir.path().join("model.bpk.part-00000.bpk"));
        Ok(())
    }

    #[test]
    fn rejects_manifest_path_traversal_and_absolute_paths() {
        let manifest_path = Path::new("/tmp/model.bpk.parts.json");
        for malicious in [
            "../outside.bpk",
            "nested/outside.bpk",
            r"..\outside.bpk",
            "/tmp/outside.bpk",
        ] {
            let err = resolve_part_entry_path(manifest_path, malicious)
                .expect_err("unsafe path must be rejected");
            assert!(
                err.contains("file name") || err.contains("relative"),
                "unexpected error for {malicious:?}: {err}"
            );
        }
    }

    #[test]
    fn manifest_path_is_stable() {
        let path = PathBuf::from("/tmp/model.bpk");
        assert_eq!(
            bpk_parts_manifest_path(&path),
            PathBuf::from("/tmp/model.bpk.parts.json")
        );
    }

    #[test]
    fn portable_sharding_splits_and_preserves_the_full_token_embedding() -> Result<(), String> {
        let mut config = Siglip2Config::tiny_for_tests();
        config.text_vocab_size = 32_769;
        config.hidden_dim = 2;
        let original = (0..config.text_vocab_size * config.hidden_dim * 2)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let record = TensorRecord {
            name: "text.token_embed.weight".to_string(),
            dtype: Dtype::F16,
            shape: vec![config.text_vocab_size, config.hidden_dim],
            data: original.clone(),
        };
        let (chunks, was_chunked) =
            split_text_token_embedding_for_portable_shards(vec![record], &config)?;
        assert!(was_chunked);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].name, "text.token_embed.weight.chunk.00000");
        assert_eq!(chunks[0].shape, [16_384, 2]);
        assert_eq!(chunks[1].shape, [16_384, 2]);
        assert_eq!(chunks[2].shape, [1, 2]);
        let reconstructed = chunks
            .into_iter()
            .flat_map(|chunk| chunk.data)
            .collect::<Vec<_>>();
        assert_eq!(reconstructed, original);
        Ok(())
    }

    #[test]
    fn reads_written_manifest() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let bpk_path = dir.path().join("model.bpk");
        let payload = safetensors::tensor::serialize(
            &BTreeMap::from([(
                "vision.patch_embed.bias".to_string(),
                TensorView::new(Dtype::F32, vec![2], &[0u8; 8])?,
            )]),
            None,
        )?;
        let header = build_bpk_header(Siglip2Config::tiny_for_tests(), &payload);
        write_siglip2_bpk(&bpk_path, &header, &payload)?;
        let report = super::write_bpk_parts(&bpk_path, 1, true)?.expect("parts report");
        let manifest = read_bpk_parts_manifest(&report.manifest_path)?;
        assert_eq!(manifest.version, PARTS_MANIFEST_VERSION);
        assert_eq!(manifest.source_file, "model.bpk");
        assert!(!manifest.parts.is_empty());
        let document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report.manifest_path)?)?;
        assert_eq!(document["manifest_kind"], "siglip2_bpk_parts");
        assert_eq!(document["storage_dtypes"][0], "f32");
        Ok(())
    }

    #[test]
    fn production_manifest_preserves_and_validates_typed_provenance()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let bpk_path = dir.path().join("siglip2-base-patch16-224.bpk");
        let payload = safetensors::tensor::serialize(
            &BTreeMap::from([(
                "logit_scale".to_string(),
                TensorView::new(Dtype::F16, vec![1], &[0u8; 2])?,
            )]),
            None,
        )?;
        let artifact = Siglip2ArtifactMetadata {
            model_variant: "base-patch16-224".to_string(),
            upstream_model_id: "google/siglip2-base-patch16-224".to_string(),
            upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            storage_dtype: "f16".to_string(),
        };
        let header = build_bpk_header_with_metadata(
            Siglip2Config::default(),
            &payload,
            Some(artifact.clone()),
        );
        write_siglip2_bpk(&bpk_path, &header, &payload)?;
        let report = super::write_bpk_parts(&bpk_path, 1, true)?.expect("parts report");
        let mut manifest = read_bpk_parts_manifest(&report.manifest_path)?;
        assert_eq!(manifest.artifact, Some(artifact));
        assert_eq!(manifest.storage_dtypes, ["f16"]);
        validate_production_bpk_parts_manifest(&report.manifest_path, &manifest)?;

        manifest
            .artifact
            .as_mut()
            .expect("artifact")
            .upstream_revision = "fedcba9876543210fedcba9876543210fedcba98".to_string();
        // A different well-formed revision is allowed as outer provenance; individual
        // shard headers must then match it exactly in the production loader.
        validate_production_bpk_parts_manifest(&report.manifest_path, &manifest)?;
        Ok(())
    }

    #[test]
    fn reports_whole_tensor_part_larger_than_requested_limit()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let bpk_path = dir.path().join("oversized.bpk");
        let tensor_bytes = 1024 * 1024 + 2;
        let data = vec![0u8; tensor_bytes];
        let payload = safetensors::tensor::serialize(
            &BTreeMap::from([(
                "some.oversized.weight".to_string(),
                TensorView::new(Dtype::F16, vec![tensor_bytes / 2], &data)?,
            )]),
            None,
        )?;
        let header = build_bpk_header(Siglip2Config::tiny_for_tests(), &payload);
        write_siglip2_bpk(&bpk_path, &header, &payload)?;

        let report = super::write_bpk_parts(&bpk_path, 1, true)?.expect("parts report");
        assert!(report.actual_max_part_bytes > report.requested_max_part_bytes);
        assert_eq!(report.parts_exceeding_requested_limit, 1);
        assert_eq!(report.oversized_tensors.len(), 1);
        assert_eq!(
            report.oversized_tensors[0].tensor_bytes,
            tensor_bytes as u64
        );

        let document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report.manifest_path)?)?;
        assert_eq!(document["sharding"]["strategy"], "whole_tensor_best_effort");
        assert_eq!(document["sharding"]["limit_enforced"], false);
        assert_eq!(
            document["sharding"]["oversized_tensors"][0]["name"],
            "some.oversized.weight"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn cdn_bundler_rejects_non_production_profiles() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let source = temp.path().join("source");
        let destination = temp.path().join("cdn");
        std::fs::create_dir_all(&source)?;
        let variants = [
            ("base-patch16-224", "google/siglip2-base-patch16-224"),
            ("large-patch16-256", "google/siglip2-large-patch16-256"),
            ("so400m-patch14-224", "google/siglip2-so400m-patch14-224"),
        ];
        for (variant, upstream_model_id) in variants {
            let stem = format!("siglip2-{variant}");
            let bpk_path = source.join(format!("{stem}.bpk"));
            let payload = safetensors::tensor::serialize(
                &BTreeMap::from([(
                    "logit_scale".to_string(),
                    TensorView::new(Dtype::F16, vec![1], &[0u8; 2])?,
                )]),
                None,
            )?;
            let header = build_bpk_header_with_metadata(
                Siglip2Config::tiny_for_tests(),
                &payload,
                Some(Siglip2ArtifactMetadata {
                    model_variant: variant.to_string(),
                    upstream_model_id: upstream_model_id.to_string(),
                    upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
                    storage_dtype: "f16".to_string(),
                }),
            );
            write_siglip2_bpk(&bpk_path, &header, &payload)?;
            super::write_bpk_parts(&bpk_path, 1, true)?.expect("parts report");
            std::fs::write(source.join(format!("{stem}.tokenizer.json")), "{}")?;
            std::fs::write(source.join(format!("{stem}.tokenizer_config.json")), "{}")?;
            std::fs::write(
                source.join(format!("{stem}.preprocessor_config.json")),
                "{}",
            )?;
        }

        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/bundle_siglip2_assets.sh");
        let output = std::process::Command::new("bash")
            .arg(script)
            .arg(&source)
            .arg(&destination)
            .env("BURN_SIGLIP2_CDN_BUNDLE_STRICT", "1")
            .output()?;
        assert!(
            !output.status.success(),
            "tiny test profile must be rejected"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("config field"),
            "unexpected bundler error:\nstdout:\n{}\nstderr:\n{stderr}",
            String::from_utf8_lossy(&output.stdout),
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn cdn_bundler_rejects_overlapping_destination_before_deleting_source()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let source = temp.path().join("source");
        std::fs::create_dir_all(&source)?;
        let marker = source.join("must-survive.txt");
        std::fs::write(&marker, "source data")?;

        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/bundle_siglip2_assets.sh");
        let output = std::process::Command::new("bash")
            .arg(script)
            .arg(&source)
            .arg(&source)
            .output()?;
        assert!(!output.status.success(), "overlapping paths must fail");
        assert_eq!(std::fs::read_to_string(&marker)?, "source data");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("overlapping source/destination"),
            "{stderr}"
        );
        Ok(())
    }

    #[test]
    fn rejects_non_bpk_manifest_name() {
        let manifest = Siglip2BpkPartsManifest {
            version: PARTS_MANIFEST_VERSION,
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
                path: "model.bpk.part-00000.bpk".to_string(),
                bytes: 16,
                sha256: String::new(),
                tensors: 1,
            }],
        };
        let err = validate_bpk_parts_layout(Path::new("/tmp/model.parts.json"), &manifest)
            .expect_err("manifest path should enforce .bpk.parts.json");
        assert!(err.contains(".bpk.parts.json"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_non_bpk_part_entry() {
        let manifest = Siglip2BpkPartsManifest {
            version: PARTS_MANIFEST_VERSION,
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
                path: "model.part-00000.safetensors".to_string(),
                bytes: 16,
                sha256: String::new(),
                tensors: 1,
            }],
        };
        let err = validate_bpk_parts_layout(Path::new("/tmp/model.bpk.parts.json"), &manifest)
            .expect_err("part entries should enforce .bpk suffix");
        assert!(err.contains(".bpk"), "unexpected error: {err}");
    }
}
