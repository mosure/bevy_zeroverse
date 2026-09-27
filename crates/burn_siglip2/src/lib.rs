#![recursion_limit = "256"]

pub mod api;
#[cfg(feature = "bootstrap")]
pub mod bootstrap;
pub mod bpk;
pub mod config;
pub mod hooks;
#[cfg(feature = "import")]
pub mod import;
pub mod loader;
pub mod model;
pub mod parity;
pub mod parts;
#[cfg(feature = "preprocess")]
pub mod preprocess;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm;
#[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
mod wasm_schema;

#[cfg(feature = "flex")]
pub use api::{DefaultFlexBackend, FlexSiglip2Backend, load_backend_flex, run_inference_flex};
#[cfg(feature = "ndarray")]
pub use api::{DefaultNdArrayBackend, Siglip2Backend, load_backend, run_inference};
#[cfg(feature = "wgpu")]
pub use api::{DefaultWgpuBackend, WgpuSiglip2Backend, load_backend_wgpu, run_inference_wgpu};
pub use api::{
    LoadRequest, Siglip2ExecutionEvidence, Siglip2InferenceRequest, Siglip2InferenceResponse,
    Siglip2MultimodalTensorResponse, Siglip2Runtime, Siglip2TensorEmbeddingResponse, WeightSource,
    load_backend_on_device,
};
#[cfg(feature = "bootstrap")]
pub use bootstrap::{
    BootstrapProgressCallback, ModelBootstrapError, Siglip2Artifacts, Siglip2BootstrapConfig,
    default_cache_root, resolve_or_bootstrap_siglip2_weights,
    resolve_or_bootstrap_siglip2_weights_with_config,
    resolve_or_bootstrap_siglip2_weights_with_config_and_progress,
};
pub use bpk::{
    SIGLIP2_BPK_MAGIC, SIGLIP2_BPK_VERSION, Siglip2ArtifactMetadata, Siglip2Bpk, Siglip2BpkHeader,
    Siglip2BpkView, build_bpk_header, build_bpk_header_with_metadata, parse_siglip2_bpk_bytes,
    parse_siglip2_bpk_view, read_siglip2_bpk, write_siglip2_bpk,
};
#[cfg(feature = "flex")]
pub use burn::backend::flex::FlexDevice;
#[cfg(feature = "ndarray")]
pub use burn::backend::ndarray::NdArrayDevice;
#[cfg(feature = "wgpu")]
pub use burn_wgpu::WgpuDevice;
pub use config::{SIGLIP2_DEFAULT_CDN_ROOT_URL, Siglip2Config, Siglip2ModelVariant};
pub use hooks::HookTensor;
#[cfg(feature = "import")]
pub use import::{
    DEFAULT_PART_SIZE_MIB, ImportError, SUPPORTED_MODEL_VARIANTS, Siglip2ImportOptions,
    Siglip2ImportReport, Siglip2OutputPrecision, expected_upstream_model_id, import_hf_dir,
    import_hf_files,
};
pub use loader::{
    PartLoadStats, Siglip2PartsLoader, WebLoadLimits, load_model_from_bpk_path,
    load_model_from_parts_bytes, load_model_from_parts_manifest_path,
    load_model_from_parts_manifest_path_with_stream_reader, load_model_from_safetensors_path,
    read_parts_manifest, validate_manifest_for_web,
};
pub use model::{Siglip2Model, WeightSpec, expected_weight_specs};
pub use parts::{
    BpkPartsReport, PARTS_MANIFEST_VERSION, Siglip2BpkPartEntry, Siglip2BpkPartsManifest,
    Siglip2OversizedTensorEntry, bpk_parts_manifest_path, manifest_is_complete,
    read_bpk_parts_manifest, resolve_part_entry_path, validate_bpk_part_entry,
    validate_bpk_parts_layout, write_bpk_parts,
};
#[cfg(feature = "preprocess")]
pub use preprocess::{
    SIGLIP2_IMAGE_MEAN, SIGLIP2_IMAGE_RESCALE_FACTOR, SIGLIP2_IMAGE_STD,
    SIGLIP2_MAX_DECODER_ALLOCATION_BYTES, SIGLIP2_MAX_ENCODED_IMAGE_BYTES,
    SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION, SIGLIP2_MAX_SOURCE_IMAGE_PIXELS, Siglip2ImageProcessor,
    preprocess_dynamic_image, preprocess_dynamic_images, preprocess_image_bytes,
};
#[cfg(feature = "tokenizer")]
pub use tokenizer::{
    SIGLIP2_MAX_TEXT_BATCH_BYTES, SIGLIP2_MAX_TEXT_BATCH_SIZE, SIGLIP2_MAX_TEXT_BYTES,
    SIGLIP2_MAX_TOKENIZER_CONFIG_BYTES, SIGLIP2_MAX_TOKENIZER_JSON_BYTES, SIGLIP2_TEXT_MAX_LENGTH,
    Siglip2TokenizedBatch, Siglip2Tokenizer,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm::WasmSiglip2;
