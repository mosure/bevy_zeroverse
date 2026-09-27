use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{Siglip2Config, Siglip2ModelVariant};

pub const SIGLIP2_BPK_MAGIC: [u8; 4] = *b"BPK1";
pub const SIGLIP2_BPK_VERSION: u32 = 1;

/// Immutable provenance attached to production model artifacts.
///
/// Older BPK files do not contain this block and remain readable. CDN-facing
/// imports should always populate it so an artifact can be traced to an exact
/// upstream checkpoint without relying on a mutable branch or tag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Siglip2ArtifactMetadata {
    pub model_variant: String,
    pub upstream_model_id: String,
    pub upstream_revision: String,
    pub storage_dtype: String,
}

impl Siglip2ArtifactMetadata {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_variant.trim().is_empty() {
            return Err("artifact model_variant must not be empty".to_string());
        }
        if self.upstream_model_id.trim().is_empty() {
            return Err("artifact upstream_model_id must not be empty".to_string());
        }
        let revision = self.upstream_revision.trim();
        if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "artifact upstream_revision must be a full 40-character git commit, got '{}'",
                self.upstream_revision
            ));
        }
        if !matches!(self.storage_dtype.as_str(), "f16" | "f32") {
            return Err(format!(
                "artifact storage_dtype must be 'f16' or 'f32', got '{}'",
                self.storage_dtype
            ));
        }
        Ok(())
    }

    /// Validate that provenance identifies the exact supported architecture in `config`.
    ///
    /// This deliberately validates identity consistency, not authenticity: callers still need
    /// a trusted manifest/checksum transport to establish that the metadata itself is genuine.
    pub fn validate_for_config(&self, config: &Siglip2Config) -> Result<(), String> {
        self.validate()?;
        let variant = Siglip2ModelVariant::ALL
            .into_iter()
            .find(|variant| self.upstream_model_id == variant.hf_model_id())
            .ok_or_else(|| {
                format!(
                    "artifact upstream_model_id '{}' is not one of the supported SigLIP2 checkpoints",
                    self.upstream_model_id
                )
            })?;
        let expected_variant = variant
            .hf_model_id()
            .strip_prefix("google/siglip2-")
            .expect("supported SigLIP2 model ids have a stable prefix");
        if self.model_variant != expected_variant {
            return Err(format!(
                "artifact model_variant '{}' does not match upstream_model_id '{}' (expected '{}')",
                self.model_variant, self.upstream_model_id, expected_variant
            ));
        }
        let expected_config = Siglip2Config::for_variant(variant);
        if config != &expected_config {
            return Err(format!(
                "artifact model_variant '{}' does not match its exact production config",
                self.model_variant
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Siglip2BpkHeader {
    #[serde(default = "default_bpk_version")]
    pub version: u32,
    #[serde(default = "default_model_family")]
    pub model_family: String,
    pub config: Siglip2Config,
    #[serde(default = "default_weight_encoding")]
    pub weight_encoding: String,
    pub weight_bytes: u64,
    #[serde(default)]
    pub weight_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Siglip2ArtifactMetadata>,
}

#[derive(Debug, Clone)]
pub struct Siglip2Bpk {
    pub header: Siglip2BpkHeader,
    pub weight_blob: Vec<u8>,
}

/// Validated zero-copy view over a BPK byte buffer.
///
/// This is the preferred parser for fetched/streamed shards because the
/// safetensors payload borrows the caller's buffer instead of allocating a
/// second model-sized `Vec`.
#[derive(Debug)]
pub struct Siglip2BpkView<'a> {
    pub header: Siglip2BpkHeader,
    pub weight_blob: &'a [u8],
}

pub fn read_siglip2_bpk(path: &Path) -> Result<Siglip2Bpk, String> {
    let mut bytes =
        fs::read(path).map_err(|err| format!("failed to read BPK '{}': {err}", path.display()))?;
    let view = parse_siglip2_bpk_view(bytes.as_slice(), Some(path))?;
    let payload_start = bytes.len() - view.weight_blob.len();
    let payload_len = view.weight_blob.len();
    let header = view.header;
    bytes.copy_within(payload_start.., 0);
    bytes.truncate(payload_len);
    Ok(Siglip2Bpk {
        header,
        weight_blob: bytes,
    })
}

pub fn parse_siglip2_bpk_bytes(
    bytes: &[u8],
    source_path: Option<&Path>,
) -> Result<Siglip2Bpk, String> {
    let view = parse_siglip2_bpk_view(bytes, source_path)?;
    Ok(Siglip2Bpk {
        header: view.header,
        weight_blob: view.weight_blob.to_vec(),
    })
}

pub fn parse_siglip2_bpk_view<'a>(
    bytes: &'a [u8],
    source_path: Option<&Path>,
) -> Result<Siglip2BpkView<'a>, String> {
    if bytes.len() < 12 {
        return Err(format!(
            "invalid BPK (too small, expected >= 12 bytes, got {}){}",
            bytes.len(),
            source_suffix(source_path)
        ));
    }
    if bytes[0..4] != SIGLIP2_BPK_MAGIC {
        return Err(format!("invalid BPK magic{}", source_suffix(source_path)));
    }

    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&bytes[4..12]);
    let header_len = u64::from_le_bytes(len_bytes);
    let header_len = usize::try_from(header_len).map_err(|_| {
        format!(
            "invalid BPK header length {}{}",
            u64::from_le_bytes(len_bytes),
            source_suffix(source_path)
        )
    })?;

    let header_start = 12usize;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| format!("BPK header length overflow{}", source_suffix(source_path)))?;
    if header_end > bytes.len() {
        return Err(format!(
            "BPK header exceeds file bounds (header_end={}, file_len={}){}",
            header_end,
            bytes.len(),
            source_suffix(source_path)
        ));
    }

    let header = serde_json::from_slice::<Siglip2BpkHeader>(&bytes[header_start..header_end])
        .map_err(|err| {
            format!(
                "failed to parse BPK header{}: {err}",
                source_suffix(source_path)
            )
        })?;
    validate_header(&header, source_path)?;

    let payload = &bytes[header_end..];
    if payload.len() as u64 != header.weight_bytes {
        return Err(format!(
            "BPK payload byte mismatch: header={}, actual={}{}",
            header.weight_bytes,
            payload.len(),
            source_suffix(source_path)
        ));
    }
    if !header.weight_sha256.trim().is_empty() {
        let actual = sha256_hex(payload);
        if !actual.eq_ignore_ascii_case(header.weight_sha256.trim()) {
            return Err(format!(
                "BPK payload checksum mismatch: header={}, actual={}{}",
                header.weight_sha256,
                actual,
                source_suffix(source_path)
            ));
        }
    }

    Ok(Siglip2BpkView {
        header,
        weight_blob: payload,
    })
}

pub fn write_siglip2_bpk(
    path: &Path,
    header: &Siglip2BpkHeader,
    weight_blob: &[u8],
) -> Result<(), String> {
    validate_header(header, Some(path))?;
    if weight_blob.len() as u64 != header.weight_bytes {
        return Err(format!(
            "cannot write BPK '{}': header weight_bytes={}, payload bytes={}",
            path.display(),
            header.weight_bytes,
            weight_blob.len()
        ));
    }
    if !header.weight_sha256.trim().is_empty() {
        let actual = sha256_hex(weight_blob);
        if !actual.eq_ignore_ascii_case(header.weight_sha256.trim()) {
            return Err(format!(
                "cannot write BPK '{}': header checksum={}, payload checksum={}",
                path.display(),
                header.weight_sha256,
                actual
            ));
        }
    }

    let header_bytes = serde_json::to_vec(header).map_err(|err| {
        format!(
            "failed to serialize BPK header for '{}': {err}",
            path.display()
        )
    })?;
    let file = fs::File::create(path)
        .map_err(|err| format!("failed to create BPK '{}': {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(&SIGLIP2_BPK_MAGIC)
        .and_then(|_| writer.write_all(&(header_bytes.len() as u64).to_le_bytes()))
        .and_then(|_| writer.write_all(header_bytes.as_slice()))
        .and_then(|_| writer.write_all(weight_blob))
        .and_then(|_| writer.flush())
        .map_err(|err| format!("failed to write BPK '{}': {err}", path.display()))
}

pub fn build_bpk_header(config: Siglip2Config, weight_blob: &[u8]) -> Siglip2BpkHeader {
    build_bpk_header_with_metadata(config, weight_blob, None)
}

pub fn build_bpk_header_with_metadata(
    config: Siglip2Config,
    weight_blob: &[u8],
    artifact: Option<Siglip2ArtifactMetadata>,
) -> Siglip2BpkHeader {
    Siglip2BpkHeader {
        version: SIGLIP2_BPK_VERSION,
        model_family: default_model_family(),
        config,
        weight_encoding: default_weight_encoding(),
        weight_bytes: weight_blob.len() as u64,
        weight_sha256: sha256_hex(weight_blob),
        artifact,
    }
}

fn validate_header(header: &Siglip2BpkHeader, source_path: Option<&Path>) -> Result<(), String> {
    if header.version != SIGLIP2_BPK_VERSION {
        return Err(format!(
            "unsupported BPK version {}{}",
            header.version,
            source_suffix(source_path)
        ));
    }
    if !header.model_family.eq_ignore_ascii_case("siglip2") {
        return Err(format!(
            "unsupported BPK model_family '{}'{}",
            header.model_family,
            source_suffix(source_path)
        ));
    }
    if !header.weight_encoding.eq_ignore_ascii_case("safetensors") {
        return Err(format!(
            "unsupported BPK weight_encoding '{}'{}",
            header.weight_encoding,
            source_suffix(source_path)
        ));
    }
    if let Some(artifact) = &header.artifact {
        artifact.validate()?;
    }
    header.config.validate()
}

fn source_suffix(source_path: Option<&Path>) -> String {
    source_path
        .map(|path| format!(" in '{}'", path.display()))
        .unwrap_or_default()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex::encode(digest.finalize())
}

fn default_bpk_version() -> u32 {
    SIGLIP2_BPK_VERSION
}

fn default_weight_encoding() -> String {
    "safetensors".to_string()
}

fn default_model_family() -> String {
    "siglip2".to_string()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{
        Siglip2ArtifactMetadata, build_bpk_header, build_bpk_header_with_metadata,
        parse_siglip2_bpk_bytes, parse_siglip2_bpk_view, read_siglip2_bpk, write_siglip2_bpk,
    };
    use crate::Siglip2Config;

    #[test]
    fn bpk_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let path = dir.path().join("model.bpk");
        let payload = vec![1u8, 2, 3, 4, 5, 6];
        let header = build_bpk_header(Siglip2Config::tiny_for_tests(), payload.as_slice());
        write_siglip2_bpk(&path, &header, payload.as_slice())?;

        let parsed = read_siglip2_bpk(&path)?;
        assert_eq!(parsed.header.version, 1);
        assert_eq!(parsed.weight_blob, payload);

        let bytes = std::fs::read(&path)?;
        let view = parse_siglip2_bpk_view(bytes.as_slice(), Some(&path))?;
        assert_eq!(view.weight_blob, payload.as_slice());
        Ok(())
    }

    #[test]
    fn rejects_invalid_magic() {
        let bytes = b"NOPE".to_vec();
        let err =
            parse_siglip2_bpk_bytes(bytes.as_slice(), None).expect_err("expected parse failure");
        assert!(err.contains("invalid BPK"), "actual error: {err}");
    }

    #[test]
    fn borrowed_view_rejects_payload_checksum_mismatch() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let path = dir.path().join("model.bpk");
        let payload = vec![1u8, 2, 3, 4];
        let header = build_bpk_header(Siglip2Config::tiny_for_tests(), payload.as_slice());
        write_siglip2_bpk(&path, &header, payload.as_slice())?;
        let mut bytes = std::fs::read(&path)?;
        *bytes.last_mut().expect("BPK payload") ^= 0xff;
        let err = parse_siglip2_bpk_view(bytes.as_slice(), Some(&path))
            .expect_err("corrupted payload must fail checksum validation");
        assert!(err.contains("checksum mismatch"), "unexpected error: {err}");
        Ok(())
    }

    #[test]
    fn production_metadata_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let payload = vec![1u8, 2, 3, 4];
        let metadata = Siglip2ArtifactMetadata {
            model_variant: "base-patch16-224".to_string(),
            upstream_model_id: "google/siglip2-base-patch16-224".to_string(),
            upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            storage_dtype: "f16".to_string(),
        };
        let header = build_bpk_header_with_metadata(
            Siglip2Config::tiny_for_tests(),
            payload.as_slice(),
            Some(metadata.clone()),
        );
        let dir = tempdir()?;
        let path = dir.path().join("model.bpk");
        write_siglip2_bpk(&path, &header, payload.as_slice())?;
        let parsed = read_siglip2_bpk(&path)?;
        assert_eq!(parsed.header.artifact, Some(metadata));
        Ok(())
    }

    #[test]
    fn production_metadata_must_match_exact_variant_config() {
        let config = Siglip2Config::default();
        let metadata = Siglip2ArtifactMetadata {
            model_variant: "base-patch16-224".to_string(),
            upstream_model_id: "google/siglip2-base-patch16-224".to_string(),
            upstream_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            storage_dtype: "f16".to_string(),
        };
        assert!(metadata.validate_for_config(&config).is_ok());

        let mut wrong_variant = metadata.clone();
        wrong_variant.model_variant = "large-patch16-256".to_string();
        assert!(wrong_variant.validate_for_config(&config).is_err());

        let mut wrong_model = metadata.clone();
        wrong_model.upstream_model_id = "google/siglip2-large-patch16-256".to_string();
        assert!(wrong_model.validate_for_config(&config).is_err());

        let mut wrong_config = config;
        wrong_config.num_layers += 1;
        assert!(metadata.validate_for_config(&wrong_config).is_err());
    }
}
