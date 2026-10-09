use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use burn::tensor::{BoolStore, Bytes, DType, TensorData};
use burn_store::{KeyRemapper, burn_pack::Tensor as TensorSnapshot};
use half::{bf16, f16};
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView, serialize},
};
use serde::Deserialize;

use crate::{
    bpk::{Siglip2ArtifactMetadata, build_bpk_header_with_metadata, write_siglip2_bpk},
    config::{Siglip2Config, Siglip2ModelVariant},
    model::expected_weight_specs,
    parts::{BpkPartsReport, write_bpk_parts},
};

pub const DEFAULT_PART_SIZE_MIB: u64 = 64;
pub const SUPPORTED_MODEL_VARIANTS: [&str; 3] = [
    "base-patch16-224",
    "large-patch16-256",
    "so400m-patch14-224",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Siglip2OutputPrecision {
    F16,
    F32,
}

impl Siglip2OutputPrecision {
    pub const fn storage_dtype(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
}

impl std::fmt::Display for Siglip2OutputPrecision {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.storage_dtype())
    }
}

impl std::str::FromStr for Siglip2OutputPrecision {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            _ => Err(format!(
                "unsupported precision '{value}' (expected f16 or f32)"
            )),
        }
    }
}

pub fn expected_upstream_model_id(model_variant: &str) -> Option<&'static str> {
    parse_model_variant(model_variant).map(Siglip2ModelVariant::hf_model_id)
}

fn parse_model_variant(model_variant: &str) -> Option<Siglip2ModelVariant> {
    match model_variant {
        "base-patch16-224" => Some(Siglip2ModelVariant::BasePatch16_224),
        "large-patch16-256" => Some(Siglip2ModelVariant::LargePatch16_256),
        "so400m-patch14-224" => Some(Siglip2ModelVariant::So400mPatch14_224),
        _ => None,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("missing required file '{path}'")]
    MissingFile { path: PathBuf },
    #[error("failed to read '{path}': {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse JSON '{path}': {source}")]
    ParseJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("failed to parse safetensors '{path}': {message}")]
    ParseSafetensors { path: PathBuf, message: String },
    #[error("invalid HF SigLIP configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid key remap rule `{0}` -> `{1}`: {2}")]
    InvalidRemap(String, String, String),
    #[error("missing required remapped tensors: {0}")]
    MissingTensors(String),
    #[error("unexpected remapped tensors: {0}")]
    UnexpectedTensors(String),
    #[error("duplicate remapped tensor '{0}'")]
    DuplicateTensor(String),
    #[error("unsupported safetensors dtype for '{0}': {1:?}")]
    UnsupportedDtype(String, Dtype),
    #[error("invalid tensor bytes for '{0}': {1}")]
    InvalidTensor(String, String),
    #[error("parts generation failed: {0}")]
    Parts(String),
    #[error("failed to write '{path}': {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[derive(Debug, Clone)]
pub struct Siglip2ImportOptions {
    pub write_parts: bool,
    pub parts_max_mib: u64,
    pub parts_overwrite: bool,
    pub copy_tokenizer_assets: bool,
    /// Floating-point dtype stored in the BPK and every shard.
    pub output_precision: Siglip2OutputPrecision,
    /// Required for immutable CDN bundles; optional for local/legacy imports.
    pub artifact_metadata: Option<Siglip2ArtifactMetadata>,
    /// Reject checkpoint drift instead of silently dropping unknown parameters.
    pub reject_unused_source_tensors: bool,
}

impl Default for Siglip2ImportOptions {
    fn default() -> Self {
        Self {
            write_parts: true,
            parts_max_mib: DEFAULT_PART_SIZE_MIB,
            parts_overwrite: false,
            copy_tokenizer_assets: true,
            output_precision: Siglip2OutputPrecision::F16,
            artifact_metadata: None,
            reject_unused_source_tensors: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Siglip2ImportReport {
    pub config: Siglip2Config,
    pub bpk_path: PathBuf,
    pub parts: Option<BpkPartsReport>,
    pub copied_assets: Vec<PathBuf>,
    pub remapped: Vec<(String, String)>,
    pub unused_source_tensors: Vec<String>,
    pub output_precision: Siglip2OutputPrecision,
    pub artifact_metadata: Option<Siglip2ArtifactMetadata>,
}

#[derive(Debug, Clone)]
struct OwnedTensorRecord {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

type RemappedWeights = (Vec<u8>, Vec<(String, String)>, Vec<String>);

#[derive(Debug, Deserialize)]
struct HfSiglipConfig {
    vision_config: HfVisionConfig,
    text_config: HfTextConfig,
}

#[derive(Debug, Deserialize)]
struct HfVisionConfig {
    image_size: Option<usize>,
    patch_size: Option<usize>,
    num_channels: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    layer_norm_eps: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct HfTextConfig {
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    pad_token_id: Option<usize>,
    projection_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    layer_norm_eps: Option<f32>,
}

pub fn import_hf_dir(
    hf_dir: &Path,
    output_base: &Path,
    options: &Siglip2ImportOptions,
) -> Result<Siglip2ImportReport, ImportError> {
    let config_path = hf_dir.join("config.json");
    let weights_path = hf_dir.join("model.safetensors");
    let tokenizer_paths = collect_small_asset_paths(hf_dir);
    import_hf_files(
        &config_path,
        &weights_path,
        output_base,
        &tokenizer_paths,
        options,
    )
}

pub fn import_hf_files(
    config_path: &Path,
    weights_path: &Path,
    output_base: &Path,
    small_asset_paths: &[PathBuf],
    options: &Siglip2ImportOptions,
) -> Result<Siglip2ImportReport, ImportError> {
    ensure_exists(config_path)?;
    ensure_exists(weights_path)?;

    let weights_bytes = fs::read(weights_path).map_err(|source| ImportError::Read {
        path: weights_path.to_path_buf(),
        source,
    })?;
    let source_tensors =
        SafeTensors::deserialize(&weights_bytes).map_err(|err| ImportError::ParseSafetensors {
            path: weights_path.to_path_buf(),
            message: err.to_string(),
        })?;
    let source_records = collect_source_records(&source_tensors)?;

    let hf_config_bytes = fs::read(config_path).map_err(|source| ImportError::Read {
        path: config_path.to_path_buf(),
        source,
    })?;
    let hf_config: HfSiglipConfig =
        serde_json::from_slice(&hf_config_bytes).map_err(|source| ImportError::ParseJson {
            path: config_path.to_path_buf(),
            source,
        })?;
    let config = siglip2_config_from_hf(&hf_config, &source_records)?;
    config
        .validate_production_profile()
        .map_err(ImportError::InvalidConfig)?;
    if let Some(artifact) = options.artifact_metadata.as_ref() {
        validate_artifact_metadata(artifact, &config, options.output_precision)?;
    }

    let (weight_blob, remapped, unused_source_tensors) =
        remap_hf_weights(&config, &source_records, options.output_precision)?;
    if options.reject_unused_source_tensors && !unused_source_tensors.is_empty() {
        return Err(ImportError::UnexpectedTensors(format!(
            "unmapped source checkpoint tensors: {}",
            unused_source_tensors.join(", ")
        )));
    }

    let bpk_path = canonical_bpk_path(output_base);
    if let Some(parent) = bpk_path.parent() {
        fs::create_dir_all(parent).map_err(|source| ImportError::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let header = build_bpk_header_with_metadata(
        config.clone(),
        &weight_blob,
        options.artifact_metadata.clone(),
    );
    write_siglip2_bpk(&bpk_path, &header, &weight_blob)
        .map_err(|message| ImportError::InvalidTensor(bpk_path.display().to_string(), message))?;
    // The sharder reads the just-written BPK. Release the monolithic payload
    // first so production imports do not retain two full model copies.
    drop(weight_blob);

    let parts = if options.write_parts {
        write_bpk_parts(&bpk_path, options.parts_max_mib, options.parts_overwrite)
            .map_err(ImportError::Parts)?
    } else {
        None
    };
    let copied_assets = if options.copy_tokenizer_assets {
        copy_small_assets(output_base, small_asset_paths)?
    } else {
        Vec::new()
    };

    Ok(Siglip2ImportReport {
        config,
        bpk_path,
        parts,
        copied_assets,
        remapped,
        unused_source_tensors,
        output_precision: options.output_precision,
        artifact_metadata: options.artifact_metadata.clone(),
    })
}

fn remap_hf_weights(
    config: &Siglip2Config,
    source_records: &BTreeMap<String, OwnedTensorRecord>,
    output_precision: Siglip2OutputPrecision,
) -> Result<RemappedWeights, ImportError> {
    let mut used_source_keys = BTreeSet::new();

    let remapper = build_key_remapper()?;
    let simple_snapshots = build_simple_snapshots(source_records)?;
    let (_snapshots, remapped) = remapper.remap(simple_snapshots);

    let mut target_records = BTreeMap::new();
    for (new_key, old_key) in &remapped {
        let record = source_records
            .get(old_key)
            .ok_or_else(|| ImportError::MissingTensors(old_key.clone()))?;
        if target_records
            .insert(new_key.clone(), record.clone())
            .is_some()
        {
            return Err(ImportError::DuplicateTensor(new_key.clone()));
        }
        used_source_keys.insert(old_key.clone());
    }

    for special in [
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.head.attention.in_proj_weight",
        "vision_model.head.attention.in_proj_bias",
    ] {
        if source_records.contains_key(special) {
            used_source_keys.insert(special.to_string());
        }
    }
    insert_patch_embedding_weight(source_records, &mut target_records)?;
    insert_split_attention_head_weights(source_records, &mut target_records)?;

    validate_target_record_keys(config, &target_records)?;
    convert_target_records_precision(&mut target_records, output_precision)?;

    let weight_blob = serialize_target_records(&target_records)?;
    let unused_source_tensors = source_records
        .keys()
        .filter(|key| !used_source_keys.contains(*key))
        .cloned()
        .collect::<Vec<_>>();
    Ok((weight_blob, remapped, unused_source_tensors))
}

fn collect_source_records(
    tensors: &SafeTensors<'_>,
) -> Result<BTreeMap<String, OwnedTensorRecord>, ImportError> {
    let mut out = BTreeMap::new();
    for (name, view) in tensors.iter() {
        let record = OwnedTensorRecord {
            dtype: view.dtype(),
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        };
        validate_owned_tensor_record(name, &record)?;
        if out.insert(name.to_string(), record).is_some() {
            return Err(ImportError::DuplicateTensor(name.to_string()));
        }
    }
    Ok(out)
}

fn build_simple_snapshots(
    records: &BTreeMap<String, OwnedTensorRecord>,
) -> Result<Vec<TensorSnapshot>, ImportError> {
    let mut snapshots = Vec::new();
    for (name, record) in records {
        if matches!(
            name.as_str(),
            "vision_model.embeddings.patch_embedding.weight"
                | "vision_model.head.attention.in_proj_weight"
                | "vision_model.head.attention.in_proj_bias"
        ) {
            continue;
        }
        let dtype = burn_dtype_from_safetensors(record.dtype)
            .ok_or_else(|| ImportError::UnsupportedDtype(name.clone(), record.dtype))?;
        let data = TensorData::from_bytes(
            Bytes::from_bytes_vec(record.data.clone()),
            record.shape.clone(),
            dtype,
        );
        snapshots.push(burn_store::bridge::from_data(data, name.to_string(), None));
    }
    Ok(snapshots)
}

fn build_key_remapper() -> Result<KeyRemapper, ImportError> {
    let patterns = vec![
        (
            r"^vision_model\.embeddings\.patch_embedding\.weight$",
            "vision.patch_embed.weight",
        ),
        (
            r"^vision_model\.embeddings\.patch_embedding\.bias$",
            "vision.patch_embed.bias",
        ),
        (
            r"^vision_model\.embeddings\.position_embedding\.weight$",
            "vision.pos_embed",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.weight$",
            "vision.blocks.$1.norm1.gamma",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.bias$",
            "vision.blocks.$1.norm1.beta",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.weight$",
            "vision.blocks.$1.norm2.gamma",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.bias$",
            "vision.blocks.$1.norm2.beta",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$",
            "vision.blocks.$1.attn.q_proj.weight",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias$",
            "vision.blocks.$1.attn.q_proj.bias",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$",
            "vision.blocks.$1.attn.k_proj.weight",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.bias$",
            "vision.blocks.$1.attn.k_proj.bias",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$",
            "vision.blocks.$1.attn.v_proj.weight",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.bias$",
            "vision.blocks.$1.attn.v_proj.bias",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$",
            "vision.blocks.$1.attn.out_proj.weight",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.bias$",
            "vision.blocks.$1.attn.out_proj.bias",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.weight$",
            "vision.blocks.$1.mlp.fc1.weight",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.bias$",
            "vision.blocks.$1.mlp.fc1.bias",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.weight$",
            "vision.blocks.$1.mlp.fc2.weight",
        ),
        (
            r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.bias$",
            "vision.blocks.$1.mlp.fc2.bias",
        ),
        (
            r"^vision_model\.post_layernorm\.weight$",
            "vision.post_norm.gamma",
        ),
        (
            r"^vision_model\.post_layernorm\.bias$",
            "vision.post_norm.beta",
        ),
        (
            r"^vision_model\.head\.attention\.out_proj\.weight$",
            "vision.head.attn.out_proj.weight",
        ),
        (
            r"^vision_model\.head\.attention\.out_proj\.bias$",
            "vision.head.attn.out_proj.bias",
        ),
        (
            r"^vision_model\.head\.layernorm\.weight$",
            "vision.head.layernorm.gamma",
        ),
        (
            r"^vision_model\.head\.layernorm\.bias$",
            "vision.head.layernorm.beta",
        ),
        (
            r"^vision_model\.head\.mlp\.fc1\.weight$",
            "vision.head.mlp.fc1.weight",
        ),
        (
            r"^vision_model\.head\.mlp\.fc1\.bias$",
            "vision.head.mlp.fc1.bias",
        ),
        (
            r"^vision_model\.head\.mlp\.fc2\.weight$",
            "vision.head.mlp.fc2.weight",
        ),
        (
            r"^vision_model\.head\.mlp\.fc2\.bias$",
            "vision.head.mlp.fc2.bias",
        ),
        (r"^vision_model\.head\.probe$", "vision.head.probe"),
        (
            r"^text_model\.embeddings\.token_embedding\.weight$",
            "text.token_embed.weight",
        ),
        (
            r"^text_model\.embeddings\.position_embedding\.weight$",
            "text.pos_embed",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.layer_norm1\.weight$",
            "text.blocks.$1.norm1.gamma",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.layer_norm1\.bias$",
            "text.blocks.$1.norm1.beta",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.layer_norm2\.weight$",
            "text.blocks.$1.norm2.gamma",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.layer_norm2\.bias$",
            "text.blocks.$1.norm2.beta",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$",
            "text.blocks.$1.attn.q_proj.weight",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias$",
            "text.blocks.$1.attn.q_proj.bias",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$",
            "text.blocks.$1.attn.k_proj.weight",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.bias$",
            "text.blocks.$1.attn.k_proj.bias",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$",
            "text.blocks.$1.attn.v_proj.weight",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.bias$",
            "text.blocks.$1.attn.v_proj.bias",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$",
            "text.blocks.$1.attn.out_proj.weight",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.bias$",
            "text.blocks.$1.attn.out_proj.bias",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.weight$",
            "text.blocks.$1.mlp.fc1.weight",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.bias$",
            "text.blocks.$1.mlp.fc1.bias",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.weight$",
            "text.blocks.$1.mlp.fc2.weight",
        ),
        (
            r"^text_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.bias$",
            "text.blocks.$1.mlp.fc2.bias",
        ),
        (
            r"^text_model\.final_layer_norm\.weight$",
            "text.final_norm.gamma",
        ),
        (
            r"^text_model\.final_layer_norm\.bias$",
            "text.final_norm.beta",
        ),
        (r"^text_model\.head\.weight$", "text.projection.weight"),
        (r"^text_model\.head\.bias$", "text.projection.bias"),
        (r"^logit_scale$", "logit_scale"),
        (r"^logit_bias$", "logit_bias"),
    ];
    KeyRemapper::from_patterns(patterns).map_err(|err| {
        ImportError::InvalidRemap(
            "pattern".to_string(),
            "replacement".to_string(),
            err.to_string(),
        )
    })
}

fn insert_split_attention_head_weights(
    source_records: &BTreeMap<String, OwnedTensorRecord>,
    target_records: &mut BTreeMap<String, OwnedTensorRecord>,
) -> Result<(), ImportError> {
    let weight = source_records
        .get("vision_model.head.attention.in_proj_weight")
        .ok_or_else(|| {
            ImportError::MissingTensors("vision_model.head.attention.in_proj_weight".to_string())
        })?;
    let bias = source_records
        .get("vision_model.head.attention.in_proj_bias")
        .ok_or_else(|| {
            ImportError::MissingTensors("vision_model.head.attention.in_proj_bias".to_string())
        })?;
    if weight.shape.len() != 2 || weight.shape[0] % 3 != 0 {
        return Err(ImportError::InvalidTensor(
            "vision_model.head.attention.in_proj_weight".to_string(),
            format!("expected [3*hidden, hidden], got {:?}", weight.shape),
        ));
    }
    if bias.shape != vec![weight.shape[0]] {
        return Err(ImportError::InvalidTensor(
            "vision_model.head.attention.in_proj_bias".to_string(),
            format!("expected [{}], got {:?}", weight.shape[0], bias.shape),
        ));
    }
    let hidden_dim = weight.shape[0] / 3;
    let row_bytes = bytes_per_element(weight.dtype).ok_or_else(|| {
        ImportError::UnsupportedDtype(
            "vision_model.head.attention.in_proj_weight".to_string(),
            weight.dtype,
        )
    })? * weight.shape[1];
    let bias_chunk_bytes = bytes_per_element(bias.dtype).ok_or_else(|| {
        ImportError::UnsupportedDtype(
            "vision_model.head.attention.in_proj_bias".to_string(),
            bias.dtype,
        )
    })? * hidden_dim;

    for (index, name) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
        let weight_start = index * hidden_dim * row_bytes;
        let weight_end = weight_start + hidden_dim * row_bytes;
        let bias_start = index * bias_chunk_bytes;
        let bias_end = bias_start + bias_chunk_bytes;
        insert_target_record(
            target_records,
            format!("vision.head.attn.{name}.weight"),
            OwnedTensorRecord {
                dtype: weight.dtype,
                shape: vec![hidden_dim, weight.shape[1]],
                data: weight.data[weight_start..weight_end].to_vec(),
            },
        )?;
        insert_target_record(
            target_records,
            format!("vision.head.attn.{name}.bias"),
            OwnedTensorRecord {
                dtype: bias.dtype,
                shape: vec![hidden_dim],
                data: bias.data[bias_start..bias_end].to_vec(),
            },
        )?;
    }
    Ok(())
}

fn insert_patch_embedding_weight(
    source_records: &BTreeMap<String, OwnedTensorRecord>,
    target_records: &mut BTreeMap<String, OwnedTensorRecord>,
) -> Result<(), ImportError> {
    let record = source_records
        .get("vision_model.embeddings.patch_embedding.weight")
        .ok_or_else(|| {
            ImportError::MissingTensors(
                "vision_model.embeddings.patch_embedding.weight".to_string(),
            )
        })?;
    if record.shape.len() != 4 {
        return Err(ImportError::InvalidTensor(
            "vision_model.embeddings.patch_embedding.weight".to_string(),
            format!(
                "expected [hidden, channels, patch, patch], got {:?}",
                record.shape
            ),
        ));
    }
    insert_target_record(
        target_records,
        "vision.patch_embed.weight".to_string(),
        OwnedTensorRecord {
            dtype: record.dtype,
            shape: vec![
                record.shape[0],
                record.shape[1] * record.shape[2] * record.shape[3],
            ],
            data: record.data.clone(),
        },
    )
}

fn insert_target_record(
    target_records: &mut BTreeMap<String, OwnedTensorRecord>,
    key: String,
    record: OwnedTensorRecord,
) -> Result<(), ImportError> {
    if target_records.insert(key.clone(), record).is_some() {
        return Err(ImportError::DuplicateTensor(key));
    }
    Ok(())
}

fn validate_target_record_keys(
    config: &Siglip2Config,
    target_records: &BTreeMap<String, OwnedTensorRecord>,
) -> Result<(), ImportError> {
    let expected_specs = expected_weight_specs(config);
    let expected = expected_specs
        .iter()
        .map(|spec| spec.key.clone())
        .collect::<BTreeSet<_>>();
    let actual = target_records.keys().cloned().collect::<BTreeSet<_>>();
    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(ImportError::MissingTensors(missing.join(", ")));
    }
    let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(ImportError::UnexpectedTensors(unexpected.join(", ")));
    }
    for spec in expected_specs {
        let actual = target_records
            .get(&spec.key)
            .ok_or_else(|| ImportError::MissingTensors(spec.key.clone()))?;
        if actual.shape != spec.shape {
            return Err(ImportError::InvalidTensor(
                spec.key,
                format!("expected shape {:?}, got {:?}", spec.shape, actual.shape),
            ));
        }
        validate_owned_tensor_record(&spec.key, actual)?;
        if !matches!(
            actual.dtype,
            Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64
        ) {
            return Err(ImportError::InvalidTensor(
                spec.key,
                format!(
                    "model weight must use a floating dtype, got {:?}",
                    actual.dtype
                ),
            ));
        }
    }
    Ok(())
}

fn validate_owned_tensor_record(name: &str, record: &OwnedTensorRecord) -> Result<(), ImportError> {
    if name.trim().is_empty() {
        return Err(ImportError::InvalidTensor(
            name.to_string(),
            "tensor name must not be empty".to_string(),
        ));
    }
    if record.shape.contains(&0) {
        return Err(ImportError::InvalidTensor(
            name.to_string(),
            format!("shape contains a zero-sized dimension: {:?}", record.shape),
        ));
    }
    let element_bytes = bytes_per_element(record.dtype)
        .ok_or_else(|| ImportError::UnsupportedDtype(name.to_string(), record.dtype))?;
    let elements = record.shape.iter().try_fold(1usize, |count, dim| {
        count.checked_mul(*dim).ok_or_else(|| {
            ImportError::InvalidTensor(
                name.to_string(),
                format!("shape element count overflows: {:?}", record.shape),
            )
        })
    })?;
    let expected_bytes = elements.checked_mul(element_bytes).ok_or_else(|| {
        ImportError::InvalidTensor(
            name.to_string(),
            format!("byte count overflows for shape {:?}", record.shape),
        )
    })?;
    if record.data.len() != expected_bytes {
        return Err(ImportError::InvalidTensor(
            name.to_string(),
            format!(
                "shape {:?} with {:?} requires {} bytes, got {}",
                record.shape,
                record.dtype,
                expected_bytes,
                record.data.len()
            ),
        ));
    }
    Ok(())
}

fn convert_target_records_precision(
    target_records: &mut BTreeMap<String, OwnedTensorRecord>,
    precision: Siglip2OutputPrecision,
) -> Result<(), ImportError> {
    for (name, record) in target_records {
        let target_dtype = match precision {
            Siglip2OutputPrecision::F16 => Dtype::F16,
            Siglip2OutputPrecision::F32 => Dtype::F32,
        };
        if record.dtype == target_dtype {
            visit_floating_record_values(name, record, |_, _| Ok(()))?;
            continue;
        }
        let source_element_bytes = bytes_per_element(record.dtype)
            .ok_or_else(|| ImportError::UnsupportedDtype(name.clone(), record.dtype))?;
        let element_count = record.data.len() / source_element_bytes;
        let target_element_bytes = bytes_per_element(target_dtype).expect("f16/f32 byte width");
        let target_capacity = element_count
            .checked_mul(target_element_bytes)
            .ok_or_else(|| {
                ImportError::InvalidTensor(
                    name.clone(),
                    "converted tensor byte count overflows".to_string(),
                )
            })?;
        let mut converted = Vec::with_capacity(target_capacity);
        visit_floating_record_values(name, record, |index, value| {
            match precision {
                Siglip2OutputPrecision::F16 => {
                    let quantized = f16::from_f32(value);
                    if !quantized.is_finite() {
                        return Err(ImportError::InvalidTensor(
                            name.clone(),
                            format!("value at index {index} ({value}) is outside finite f16 range"),
                        ));
                    }
                    converted.extend_from_slice(&quantized.to_bits().to_le_bytes());
                }
                Siglip2OutputPrecision::F32 => {
                    converted.extend_from_slice(&value.to_le_bytes());
                }
            }
            Ok(())
        })?;
        record.dtype = target_dtype;
        record.data = converted;
        validate_owned_tensor_record(name, record)?;
    }
    Ok(())
}

fn visit_floating_record_values<F>(
    name: &str,
    record: &OwnedTensorRecord,
    mut visit: F,
) -> Result<(), ImportError>
where
    F: FnMut(usize, f32) -> Result<(), ImportError>,
{
    validate_owned_tensor_record(name, record)?;
    let emit = |index: usize, value: f32, visit: &mut F| {
        if !value.is_finite() {
            return Err(ImportError::InvalidTensor(
                name.to_string(),
                format!("non-finite value at index {index}: {value}"),
            ));
        }
        visit(index, value)
    };
    match record.dtype {
        Dtype::F16 => {
            for (index, chunk) in record.data.as_chunks::<2>().0.iter().enumerate() {
                let value = f16::from_bits(u16::from_le_bytes(*chunk)).to_f32();
                emit(index, value, &mut visit)?;
            }
        }
        Dtype::BF16 => {
            for (index, chunk) in record.data.as_chunks::<2>().0.iter().enumerate() {
                let value = bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32();
                emit(index, value, &mut visit)?;
            }
        }
        Dtype::F32 => {
            for (index, chunk) in record.data.as_chunks::<4>().0.iter().enumerate() {
                let value = f32::from_le_bytes(*chunk);
                emit(index, value, &mut visit)?;
            }
        }
        Dtype::F64 => {
            for (index, chunk) in record.data.as_chunks::<8>().0.iter().enumerate() {
                let value = f64::from_le_bytes(*chunk) as f32;
                emit(index, value, &mut visit)?;
            }
        }
        dtype => return Err(ImportError::UnsupportedDtype(name.to_string(), dtype)),
    }
    Ok(())
}

fn serialize_target_records(
    target_records: &BTreeMap<String, OwnedTensorRecord>,
) -> Result<Vec<u8>, ImportError> {
    let mut views = BTreeMap::new();
    for (name, record) in target_records {
        validate_owned_tensor_record(name, record)?;
        let view = TensorView::new(record.dtype, record.shape.clone(), &record.data)
            .map_err(|err| ImportError::InvalidTensor(name.clone(), err.to_string()))?;
        views.insert(name.clone(), view);
    }
    serialize(&views, None).map_err(|err| {
        ImportError::InvalidTensor("target safetensors".to_string(), err.to_string())
    })
}

fn siglip2_config_from_hf(
    config: &HfSiglipConfig,
    source_records: &BTreeMap<String, OwnedTensorRecord>,
) -> Result<Siglip2Config, ImportError> {
    let patch_record = source_records.get("vision_model.embeddings.patch_embedding.weight");
    if let Some(patch_record) = patch_record
        && patch_record.shape.len() != 4
    {
        return Err(ImportError::InvalidTensor(
            "vision_model.embeddings.patch_embedding.weight".to_string(),
            format!(
                "expected 4D patch embedding weight, got {:?}",
                patch_record.shape
            ),
        ));
    }

    let hidden_dim = config
        .vision_config
        .hidden_size
        .or(config.text_config.hidden_size)
        .or_else(|| {
            source_records
                .get("text_model.embeddings.token_embedding.weight")
                .map(|record| record.shape[1])
        })
        .or_else(|| patch_record.map(|record| record.shape[0]))
        .ok_or_else(|| ImportError::InvalidConfig("unable to infer hidden_dim".to_string()))?;
    let intermediate_dim = config
        .vision_config
        .intermediate_size
        .or(config.text_config.intermediate_size)
        .or_else(|| {
            source_records
                .get("vision_model.encoder.layers.0.mlp.fc1.weight")
                .map(|record| record.shape[0])
        })
        .or_else(|| {
            source_records
                .get("text_model.encoder.layers.0.mlp.fc1.weight")
                .map(|record| record.shape[0])
        })
        .ok_or_else(|| {
            ImportError::InvalidConfig("unable to infer intermediate_dim".to_string())
        })?;
    let num_layers = config
        .vision_config
        .num_hidden_layers
        .or(config.text_config.num_hidden_layers)
        .or_else(|| infer_num_layers(source_records, "vision_model.encoder.layers."))
        .or_else(|| infer_num_layers(source_records, "text_model.encoder.layers."))
        .ok_or_else(|| ImportError::InvalidConfig("unable to infer num_layers".to_string()))?;
    let num_heads = config
        .vision_config
        .num_attention_heads
        .or(config.text_config.num_attention_heads)
        .or_else(|| infer_num_heads(hidden_dim))
        .ok_or_else(|| ImportError::InvalidConfig("unable to infer num_heads".to_string()))?;
    let layer_norm_eps = config
        .vision_config
        .layer_norm_eps
        .or(config.text_config.layer_norm_eps)
        .unwrap_or(1e-6);
    let patch_size = config
        .vision_config
        .patch_size
        .or_else(|| patch_record.map(|record| record.shape[2]))
        .ok_or_else(|| ImportError::InvalidConfig("unable to infer patch_size".to_string()))?;
    let channels = config
        .vision_config
        .num_channels
        .or_else(|| patch_record.map(|record| record.shape[1]))
        .ok_or_else(|| ImportError::InvalidConfig("unable to infer channels".to_string()))?;
    let image_token_count = source_records
        .get("vision_model.embeddings.position_embedding.weight")
        .map(|record| record.shape[0])
        .unwrap_or(0);
    let image_size = config.vision_config.image_size.unwrap_or_else(|| {
        let grid = (image_token_count as f64).sqrt() as usize;
        grid * patch_size
    });
    let text_vocab_size = config
        .text_config
        .vocab_size
        .or_else(|| {
            source_records
                .get("text_model.embeddings.token_embedding.weight")
                .map(|record| record.shape[0])
        })
        .ok_or_else(|| ImportError::InvalidConfig("unable to infer text_vocab_size".to_string()))?;
    let text_max_positions = config
        .text_config
        .max_position_embeddings
        .or_else(|| {
            source_records
                .get("text_model.embeddings.position_embedding.weight")
                .map(|record| record.shape[0])
        })
        .ok_or_else(|| {
            ImportError::InvalidConfig("unable to infer text_max_positions".to_string())
        })?;
    let projection_dim = config
        .text_config
        .projection_size
        .or_else(|| {
            source_records
                .get("text_model.head.weight")
                .map(|record| record.shape[0])
        })
        .ok_or_else(|| ImportError::InvalidConfig("unable to infer projection_dim".to_string()))?;

    let vision_hidden = config.vision_config.hidden_size.unwrap_or(hidden_dim);
    let text_hidden = config.text_config.hidden_size.unwrap_or(hidden_dim);
    if vision_hidden != text_hidden {
        return Err(ImportError::InvalidConfig(format!(
            "shared hidden_dim mismatch: vision={}, text={}",
            vision_hidden, text_hidden
        )));
    }
    let vision_intermediate = config
        .vision_config
        .intermediate_size
        .unwrap_or(intermediate_dim);
    let text_intermediate = config
        .text_config
        .intermediate_size
        .unwrap_or(intermediate_dim);
    if vision_intermediate != text_intermediate {
        return Err(ImportError::InvalidConfig(format!(
            "shared intermediate_dim mismatch: vision={}, text={}",
            vision_intermediate, text_intermediate
        )));
    }
    let vision_heads = config
        .vision_config
        .num_attention_heads
        .unwrap_or(num_heads);
    let text_heads = config.text_config.num_attention_heads.unwrap_or(num_heads);
    if vision_heads != text_heads {
        return Err(ImportError::InvalidConfig(format!(
            "shared num_heads mismatch: vision={}, text={}",
            vision_heads, text_heads
        )));
    }
    let vision_layers = config.vision_config.num_hidden_layers.unwrap_or(num_layers);
    let text_layers = config.text_config.num_hidden_layers.unwrap_or(num_layers);
    if vision_layers != text_layers {
        return Err(ImportError::InvalidConfig(format!(
            "shared num_layers mismatch: vision={}, text={}",
            vision_layers, text_layers
        )));
    }
    let text_eps = config.text_config.layer_norm_eps.unwrap_or(layer_norm_eps);
    if (layer_norm_eps - text_eps).abs() > f32::EPSILON {
        return Err(ImportError::InvalidConfig(format!(
            "shared layer_norm_eps mismatch: vision={}, text={}",
            layer_norm_eps, text_eps
        )));
    }
    if projection_dim != hidden_dim {
        return Err(ImportError::InvalidConfig(format!(
            "unsupported text projection size {} (expected hidden size {})",
            projection_dim, hidden_dim
        )));
    }

    Ok(Siglip2Config {
        image_size,
        patch_size,
        channels,
        hidden_dim,
        intermediate_dim,
        projection_dim,
        num_heads,
        num_layers,
        layer_norm_eps,
        text_max_positions,
        text_vocab_size,
        // Published SigLIP2 tokenizer artifacts pad with id 0. The lightweight HF model configs
        // omit this field, so do not inherit the legacy SigLIP default (id 1 / EOS).
        text_pad_token_id: config.text_config.pad_token_id.unwrap_or(0),
    })
}

fn validate_artifact_metadata(
    artifact: &Siglip2ArtifactMetadata,
    config: &Siglip2Config,
    output_precision: Siglip2OutputPrecision,
) -> Result<(), ImportError> {
    artifact.validate().map_err(ImportError::InvalidConfig)?;
    if artifact.storage_dtype != output_precision.storage_dtype() {
        return Err(ImportError::InvalidConfig(format!(
            "artifact storage_dtype '{}' does not match requested output precision '{}'",
            artifact.storage_dtype, output_precision
        )));
    }
    let model_variant = parse_model_variant(&artifact.model_variant).ok_or_else(|| {
        ImportError::InvalidConfig(format!(
            "unsupported model variant '{}' (expected one of: {})",
            artifact.model_variant,
            SUPPORTED_MODEL_VARIANTS.join(", ")
        ))
    })?;
    let expected_model_id = model_variant.hf_model_id();
    if artifact.upstream_model_id != expected_model_id {
        return Err(ImportError::InvalidConfig(format!(
            "model variant '{}' requires upstream id '{}', got '{}'",
            artifact.model_variant, expected_model_id, artifact.upstream_model_id
        )));
    }
    let expected_config = Siglip2Config::for_variant(model_variant);
    if config != &expected_config {
        return Err(ImportError::InvalidConfig(format!(
            "model variant '{}' architecture does not match its production preset: expected {:?}, got {:?}",
            artifact.model_variant, expected_config, config
        )));
    }
    Ok(())
}

fn infer_num_layers(
    source_records: &BTreeMap<String, OwnedTensorRecord>,
    prefix: &str,
) -> Option<usize> {
    let mut max_index = None;
    for key in source_records.keys() {
        let Some(rest) = key.strip_prefix(prefix) else {
            continue;
        };
        let idx = rest.split('.').next()?.parse::<usize>().ok()?;
        max_index = Some(max_index.map_or(idx, |current: usize| current.max(idx)));
    }
    max_index.map(|value| value + 1)
}

fn infer_num_heads(hidden_dim: usize) -> Option<usize> {
    let candidate = hidden_dim / 64;
    if candidate > 0 && hidden_dim.is_multiple_of(candidate) {
        Some(candidate)
    } else {
        None
    }
}

fn ensure_exists(path: &Path) -> Result<(), ImportError> {
    if path.exists() {
        Ok(())
    } else {
        Err(ImportError::MissingFile {
            path: path.to_path_buf(),
        })
    }
}

fn canonical_bpk_path(output_base: &Path) -> PathBuf {
    if output_base.extension().and_then(|value| value.to_str()) == Some("bpk") {
        output_base.to_path_buf()
    } else {
        output_base.with_extension("bpk")
    }
}

fn output_stem(output_base: &Path) -> String {
    if output_base.extension().and_then(|value| value.to_str()) == Some("bpk") {
        output_base
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("siglip2")
            .to_string()
    } else {
        output_base
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("siglip2")
            .to_string()
    }
}

fn copy_small_assets(
    output_base: &Path,
    source_paths: &[PathBuf],
) -> Result<Vec<PathBuf>, ImportError> {
    let mut copied = Vec::new();
    let stem = output_stem(output_base);
    let parent = canonical_bpk_path(output_base)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&parent).map_err(|source| ImportError::Write {
        path: parent.clone(),
        source,
    })?;

    for source_path in source_paths {
        if !source_path.exists() {
            continue;
        }
        let suffix = source_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("asset");
        let dest = parent.join(format!("{stem}.{suffix}"));
        fs::copy(source_path, &dest).map_err(|source| ImportError::Write {
            path: dest.clone(),
            source,
        })?;
        copied.push(dest);
    }

    Ok(copied)
}

fn collect_small_asset_paths(hf_dir: &Path) -> Vec<PathBuf> {
    [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "preprocessor_config.json",
    ]
    .into_iter()
    .map(|name| hf_dir.join(name))
    .collect()
}

fn burn_dtype_from_safetensors(dtype: Dtype) -> Option<DType> {
    match dtype {
        Dtype::BOOL => Some(DType::Bool(BoolStore::Native)),
        Dtype::U8 => Some(DType::U8),
        Dtype::I8 => Some(DType::I8),
        Dtype::I16 => Some(DType::I16),
        Dtype::U16 => Some(DType::U16),
        Dtype::I32 => Some(DType::I32),
        Dtype::U32 => Some(DType::U32),
        Dtype::I64 => Some(DType::I64),
        Dtype::U64 => Some(DType::U64),
        Dtype::F16 => Some(DType::F16),
        Dtype::BF16 => Some(DType::BF16),
        Dtype::F32 => Some(DType::F32),
        Dtype::F64 => Some(DType::F64),
        _ => None,
    }
}

fn bytes_per_element(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => Some(1),
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => Some(2),
        Dtype::F32 | Dtype::I32 | Dtype::U32 => Some(4),
        Dtype::F64 | Dtype::I64 | Dtype::U64 => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use half::f16;
    use safetensors::tensor::Dtype;

    use crate::Siglip2Config;

    use super::{
        ImportError, OwnedTensorRecord, Siglip2OutputPrecision, convert_target_records_precision,
        expected_weight_specs, siglip2_config_from_hf, validate_owned_tensor_record,
        validate_target_record_keys,
    };

    #[test]
    fn target_schema_validation_does_not_require_a_backend() -> Result<(), ImportError> {
        let config = Siglip2Config::tiny_for_tests();
        let records = expected_weight_specs(&config)
            .into_iter()
            .map(|spec| {
                let elements = spec.shape.iter().product::<usize>();
                (
                    spec.key,
                    OwnedTensorRecord {
                        dtype: Dtype::F32,
                        shape: spec.shape,
                        data: vec![0; elements * std::mem::size_of::<f32>()],
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        validate_target_record_keys(&config, &records)
    }

    #[test]
    fn converts_hf_config() -> Result<(), Box<dyn std::error::Error>> {
        let parsed: super::HfSiglipConfig = serde_json::from_str(
            r#"{
                "vision_config": {
                    "image_size": 256,
                    "patch_size": 16,
                    "num_channels": 3,
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "layer_norm_eps": 1e-6
                },
                "text_config": {
                    "vocab_size": 256000,
                    "max_position_embeddings": 64,
                    "pad_token_id": 1,
                    "projection_size": 768,
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "layer_norm_eps": 1e-6
                }
            }"#,
        )?;
        let cfg = siglip2_config_from_hf(&parsed, &BTreeMap::new())?;
        assert_eq!(cfg.image_size, 256);
        assert_eq!(cfg.text_vocab_size, 256000);
        assert_eq!(cfg.projection_dim, 768);
        Ok(())
    }

    #[test]
    fn f16_export_quantizes_floating_weights_with_expected_rounding()
    -> Result<(), Box<dyn std::error::Error>> {
        let input = [0.0f32, 1.0, -2.0, 1.0 / 3.0];
        let mut records = BTreeMap::from([(
            "weight".to_string(),
            OwnedTensorRecord {
                dtype: Dtype::F32,
                shape: vec![input.len()],
                data: input
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            },
        )]);
        convert_target_records_precision(&mut records, Siglip2OutputPrecision::F16)?;
        let record = &records["weight"];
        assert_eq!(record.dtype, Dtype::F16);
        assert_eq!(record.data.len(), input.len() * 2);
        for (index, expected) in input.into_iter().enumerate() {
            let offset = index * 2;
            let actual = f16::from_bits(u16::from_le_bytes([
                record.data[offset],
                record.data[offset + 1],
            ]))
            .to_f32();
            assert_eq!(actual, f16::from_f32(expected).to_f32());
        }
        Ok(())
    }

    #[test]
    fn strict_tensor_validation_rejects_byte_length_mismatch() {
        let record = OwnedTensorRecord {
            dtype: Dtype::F32,
            shape: vec![2],
            data: vec![0; 4],
        };
        let err = validate_owned_tensor_record("weight", &record)
            .expect_err("truncated tensor must fail");
        assert!(err.to_string().contains("requires 8 bytes"));
    }

    #[test]
    fn f16_export_rejects_overflow_instead_of_writing_infinity() {
        let value = 1.0e20f32;
        let mut records = BTreeMap::from([(
            "weight".to_string(),
            OwnedTensorRecord {
                dtype: Dtype::F32,
                shape: vec![1],
                data: value.to_le_bytes().to_vec(),
            },
        )]);
        let err = convert_target_records_precision(&mut records, Siglip2OutputPrecision::F16)
            .expect_err("f16 overflow must fail");
        assert!(err.to_string().contains("finite f16 range"));
    }
}
