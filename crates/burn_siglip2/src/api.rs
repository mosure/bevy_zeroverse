use std::path::PathBuf;

use burn::tensor::Device;
#[cfg(any(feature = "ndarray", feature = "flex", feature = "wgpu"))]
use burn::tensor::TensorData;
use burn::tensor::{Int, Tensor, activation};
use serde::{Deserialize, Serialize};

#[cfg(feature = "preprocess")]
use crate::preprocess::Siglip2ImageProcessor;
#[cfg(feature = "tokenizer")]
use crate::tokenizer::{Siglip2TokenizedBatch, Siglip2Tokenizer};
use crate::{
    config::Siglip2Config,
    hooks::{HookRecorder, HookTensor},
    loader::{
        PartLoadStats, load_model_from_bpk_path, load_model_from_parts_manifest_path,
        load_model_from_safetensors_path,
    },
    model::Siglip2Model,
};

#[cfg(any(feature = "ndarray", feature = "flex"))]
pub type Siglip2Backend = Siglip2Runtime;

/// A SigLIP2 runtime executing on Burn Flex CPU kernels.
#[cfg(feature = "flex")]
pub type FlexSiglip2Backend = Siglip2Runtime;

#[cfg(feature = "wgpu")]
pub type WgpuSiglip2Backend = Siglip2Runtime;

#[derive(Debug, Clone)]
pub enum WeightSource {
    BpkFile(PathBuf),
    SafetensorsFile(PathBuf),
    PartsManifest(PathBuf),
}

#[derive(Debug, Clone)]
pub struct LoadRequest {
    pub source: WeightSource,
    pub config: Option<Siglip2Config>,
    pub verify_checksums: bool,
}

impl LoadRequest {
    pub fn from_bpk(path: impl Into<PathBuf>) -> Self {
        Self {
            source: WeightSource::BpkFile(path.into()),
            config: None,
            verify_checksums: true,
        }
    }

    pub fn from_safetensors(config: Siglip2Config, path: impl Into<PathBuf>) -> Self {
        Self {
            source: WeightSource::SafetensorsFile(path.into()),
            config: Some(config),
            verify_checksums: true,
        }
    }

    pub fn from_parts_manifest(path: impl Into<PathBuf>, verify_checksums: bool) -> Self {
        Self {
            source: WeightSource::PartsManifest(path.into()),
            config: None,
            verify_checksums,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Siglip2ExecutionEvidence {
    pub backend_label: String,
    pub device_label: String,
    pub wgpu_executed: bool,
    pub host_readbacks: usize,
    pub stages_recorded: usize,
}

#[derive(Debug, Clone)]
pub struct Siglip2TensorEmbeddingResponse {
    pub embedding: Tensor<2>,
    pub hooks: Vec<(String, HookTensor)>,
    pub evidence: Siglip2ExecutionEvidence,
}

#[derive(Debug, Clone)]
pub struct Siglip2MultimodalTensorResponse {
    /// Raw vision tower pooler output.
    pub image_embedding: Tensor<2>,
    /// Raw text tower projected pooler output.
    pub text_embedding: Tensor<2>,
    /// L2-normalized image embedding used for similarity scoring.
    pub normalized_image_embedding: Tensor<2>,
    /// L2-normalized text embedding used for similarity scoring.
    pub normalized_text_embedding: Tensor<2>,
    /// Learned-scale and learned-bias similarity matrix, shaped `[images, texts]`.
    pub logits_per_image: Tensor<2>,
    /// Transposed learned similarity matrix, shaped `[texts, images]`.
    pub logits_per_text: Tensor<2>,
    /// Sigmoid-calibrated image-to-text probabilities, shaped `[images, texts]`.
    pub probabilities_per_image: Tensor<2>,
    /// Sigmoid-calibrated text-to-image probabilities, shaped `[texts, images]`.
    pub probabilities_per_text: Tensor<2>,
    pub image_hooks: Vec<(String, HookTensor)>,
    pub text_hooks: Vec<(String, HookTensor)>,
    pub evidence: Siglip2ExecutionEvidence,
}

#[derive(Debug, Clone)]
pub struct Siglip2InferenceRequest {
    pub input: Vec<f32>,
    pub shape: [usize; 4],
    pub capture_hooks: bool,
}

#[derive(Debug, Clone)]
pub struct Siglip2InferenceResponse {
    pub embedding: Vec<f32>,
    pub shape: [usize; 2],
    pub hooks: Vec<(String, HookTensor)>,
    pub evidence: Siglip2ExecutionEvidence,
}

#[derive(Debug, Clone)]
pub struct Siglip2Runtime {
    pub model: Siglip2Model,
    pub device: Device,
    pub load_stats: PartLoadStats,
}

impl Siglip2Runtime {
    pub fn encode_image(
        &self,
        image: Tensor<4>,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        let (embedding, hook, evidence) = self.run_image_tensor(image, capture_hooks)?;
        let hooks = hook
            .map(|hook| hook.into_tensors().into_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        Ok(Siglip2TensorEmbeddingResponse {
            embedding,
            hooks,
            evidence,
        })
    }

    pub fn encode_image_normalized(
        &self,
        image: Tensor<4>,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        normalize_embedding_response(self.encode_image(image, capture_hooks)?)
    }

    #[cfg(feature = "preprocess")]
    pub fn encode_image_bytes(
        &self,
        encoded_image: &[u8],
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        let image = Siglip2ImageProcessor::new(&self.model.config)?
            .preprocess_bytes(encoded_image, &self.device)?;
        self.encode_image(image, capture_hooks)
    }

    #[cfg(feature = "preprocess")]
    pub fn encode_image_bytes_normalized(
        &self,
        encoded_image: &[u8],
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        normalize_embedding_response(self.encode_image_bytes(encoded_image, capture_hooks)?)
    }

    pub fn encode_text(
        &self,
        text: Tensor<3>,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        let (embedding, hook, evidence) = self.run_text_embedding_tensor(text, capture_hooks)?;
        let hooks = hook
            .map(|hook| hook.into_tensors().into_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        Ok(Siglip2TensorEmbeddingResponse {
            embedding,
            hooks,
            evidence,
        })
    }

    pub fn encode_text_normalized(
        &self,
        text: Tensor<3>,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        normalize_embedding_response(self.encode_text(text, capture_hooks)?)
    }

    pub fn encode_text_tokens(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2>>,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        let (embedding, hook, evidence) =
            self.run_text_token_tensor(input_ids, attention_mask, capture_hooks)?;
        let hooks = hook
            .map(|hook| hook.into_tensors().into_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        Ok(Siglip2TensorEmbeddingResponse {
            embedding,
            hooks,
            evidence,
        })
    }

    pub fn encode_text_tokens_normalized(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2>>,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        normalize_embedding_response(self.encode_text_tokens(
            input_ids,
            attention_mask,
            capture_hooks,
        )?)
    }

    #[cfg(feature = "tokenizer")]
    pub fn encode_tokenized_text_batch(
        &self,
        batch: &Siglip2TokenizedBatch,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        // Fixed-resolution SigLIP2 checkpoints were evaluated without forwarding the tokenizer
        // attention mask. Keep that behavior as the ergonomic default; callers that intentionally
        // need masking can opt in through the explicit method below or `encode_text_tokens`.
        self.encode_text_tokens(batch.input_ids_tensor(&self.device)?, None, capture_hooks)
    }

    #[cfg(feature = "tokenizer")]
    pub fn encode_tokenized_text_batch_with_attention_mask(
        &self,
        batch: &Siglip2TokenizedBatch,
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        self.encode_text_tokens(
            batch.input_ids_tensor(&self.device)?,
            Some(batch.attention_mask_tensor(&self.device)?),
            capture_hooks,
        )
    }

    #[cfg(feature = "tokenizer")]
    pub fn encode_text_strings<T: AsRef<str>>(
        &self,
        tokenizer: &Siglip2Tokenizer,
        texts: &[T],
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        let batch = tokenizer.encode_batch(texts)?;
        self.encode_tokenized_text_batch(&batch, capture_hooks)
    }

    #[cfg(feature = "tokenizer")]
    pub fn encode_text_strings_normalized<T: AsRef<str>>(
        &self,
        tokenizer: &Siglip2Tokenizer,
        texts: &[T],
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        normalize_embedding_response(self.encode_text_strings(tokenizer, texts, capture_hooks)?)
    }

    #[cfg(feature = "tokenizer")]
    pub fn encode_text_strings_with_attention_mask<T: AsRef<str>>(
        &self,
        tokenizer: &Siglip2Tokenizer,
        texts: &[T],
        capture_hooks: bool,
    ) -> Result<Siglip2TensorEmbeddingResponse, String> {
        let batch = tokenizer.encode_batch(texts)?;
        self.encode_tokenized_text_batch_with_attention_mask(&batch, capture_hooks)
    }

    pub fn encode_text_and_image(
        &self,
        image: Tensor<4>,
        text: Tensor<3>,
        capture_hooks: bool,
    ) -> Result<Siglip2MultimodalTensorResponse, String> {
        let (image_embedding, image_hook, image_evidence) =
            self.run_image_tensor(image, capture_hooks)?;
        let (text_embedding, text_hook, text_evidence) =
            self.run_text_embedding_tensor(text, capture_hooks)?;

        self.build_multimodal_response(
            image_embedding,
            text_embedding,
            image_hook,
            text_hook,
            image_evidence,
            text_evidence,
        )
    }

    pub fn encode_image_and_text_tokens(
        &self,
        image: Tensor<4>,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2>>,
        capture_hooks: bool,
    ) -> Result<Siglip2MultimodalTensorResponse, String> {
        let (image_embedding, image_hook, image_evidence) =
            self.run_image_tensor(image, capture_hooks)?;
        let (text_embedding, text_hook, text_evidence) =
            self.run_text_token_tensor(input_ids, attention_mask, capture_hooks)?;

        self.build_multimodal_response(
            image_embedding,
            text_embedding,
            image_hook,
            text_hook,
            image_evidence,
            text_evidence,
        )
    }

    #[cfg(all(feature = "preprocess", feature = "tokenizer"))]
    pub fn encode_image_bytes_and_text_strings<T: AsRef<str>>(
        &self,
        encoded_image: &[u8],
        tokenizer: &Siglip2Tokenizer,
        texts: &[T],
        capture_hooks: bool,
    ) -> Result<Siglip2MultimodalTensorResponse, String> {
        let image = Siglip2ImageProcessor::new(&self.model.config)?
            .preprocess_bytes(encoded_image, &self.device)?;
        let batch = tokenizer.encode_batch(texts)?;
        self.encode_image_and_text_tokens(
            image,
            batch.input_ids_tensor(&self.device)?,
            None,
            capture_hooks,
        )
    }

    #[cfg(all(feature = "preprocess", feature = "tokenizer"))]
    pub fn encode_image_bytes_and_text_strings_with_attention_mask<T: AsRef<str>>(
        &self,
        encoded_image: &[u8],
        tokenizer: &Siglip2Tokenizer,
        texts: &[T],
        capture_hooks: bool,
    ) -> Result<Siglip2MultimodalTensorResponse, String> {
        let image = Siglip2ImageProcessor::new(&self.model.config)?
            .preprocess_bytes(encoded_image, &self.device)?;
        let batch = tokenizer.encode_batch(texts)?;
        self.encode_image_and_text_tokens(
            image,
            batch.input_ids_tensor(&self.device)?,
            Some(batch.attention_mask_tensor(&self.device)?),
            capture_hooks,
        )
    }

    fn build_multimodal_response(
        &self,
        image_embedding: Tensor<2>,
        text_embedding: Tensor<2>,
        image_hook: Option<HookRecorder>,
        text_hook: Option<HookRecorder>,
        image_evidence: Siglip2ExecutionEvidence,
        text_evidence: Siglip2ExecutionEvidence,
    ) -> Result<Siglip2MultimodalTensorResponse, String> {
        let normalized_image_embedding = l2_normalize_embeddings(image_embedding.clone())?;
        let normalized_text_embedding = l2_normalize_embeddings(text_embedding.clone())?;
        let logits_per_image = self
            .model
            .similarity_logits(image_embedding.clone(), text_embedding.clone())?;
        let logits_per_text = logits_per_image.clone().swap_dims(0, 1);
        let probabilities_per_image = activation::sigmoid(logits_per_image.clone());
        let probabilities_per_text = probabilities_per_image.clone().swap_dims(0, 1);

        let image_hooks = image_hook
            .map(|hook| hook.into_tensors().into_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        let text_hooks = text_hook
            .map(|hook| hook.into_tensors().into_iter().collect::<Vec<_>>())
            .unwrap_or_default();

        let evidence = Siglip2ExecutionEvidence {
            backend_label: image_evidence.backend_label.clone(),
            device_label: image_evidence.device_label.clone(),
            wgpu_executed: image_evidence.wgpu_executed || text_evidence.wgpu_executed,
            host_readbacks: image_evidence
                .host_readbacks
                .saturating_add(text_evidence.host_readbacks),
            stages_recorded: image_evidence
                .stages_recorded
                .saturating_add(text_evidence.stages_recorded),
        };

        Ok(Siglip2MultimodalTensorResponse {
            image_embedding,
            text_embedding,
            normalized_image_embedding,
            normalized_text_embedding,
            logits_per_image,
            logits_per_text,
            probabilities_per_image,
            probabilities_per_text,
            image_hooks,
            text_hooks,
            evidence,
        })
    }

    fn run_image_tensor(
        &self,
        image: Tensor<4>,
        capture_hooks: bool,
    ) -> Result<(Tensor<2>, Option<HookRecorder>, Siglip2ExecutionEvidence), String> {
        let mut hook = if capture_hooks {
            Some(HookRecorder::new())
        } else {
            None
        };
        let output = self.model.forward_image(image, hook.as_mut())?;
        let evidence = execution_evidence(&self.device, &hook);
        Ok((output, hook, evidence))
    }

    fn run_text_embedding_tensor(
        &self,
        text: Tensor<3>,
        capture_hooks: bool,
    ) -> Result<(Tensor<2>, Option<HookRecorder>, Siglip2ExecutionEvidence), String> {
        let mut hook = if capture_hooks {
            Some(HookRecorder::new())
        } else {
            None
        };
        let output = self.model.forward_text(text, hook.as_mut())?;
        let evidence = execution_evidence(&self.device, &hook);
        Ok((output, hook, evidence))
    }

    fn run_text_token_tensor(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2>>,
        capture_hooks: bool,
    ) -> Result<(Tensor<2>, Option<HookRecorder>, Siglip2ExecutionEvidence), String> {
        let mut hook = if capture_hooks {
            Some(HookRecorder::new())
        } else {
            None
        };
        let output = self
            .model
            .forward_text_tokens(input_ids, attention_mask, hook.as_mut())?;
        let evidence = execution_evidence(&self.device, &hook);
        Ok((output, hook, evidence))
    }
}

fn l2_normalize_embeddings(embedding: Tensor<2>) -> Result<Tensor<2>, String> {
    let [batch, dim] = embedding.shape().dims();
    if batch == 0 || dim == 0 {
        return Err(format!(
            "cannot normalize an empty embedding tensor with shape [{batch}, {dim}]"
        ));
    }
    Ok(Siglip2Model::normalize_embeddings(embedding))
}

fn normalize_embedding_response(
    mut response: Siglip2TensorEmbeddingResponse,
) -> Result<Siglip2TensorEmbeddingResponse, String> {
    response.embedding = l2_normalize_embeddings(response.embedding)?;
    Ok(response)
}

#[cfg(any(feature = "ndarray", feature = "flex"))]
pub fn load_backend(request: LoadRequest) -> Result<Siglip2Backend, String> {
    let device = burn::tensor::Device::flex();
    load_backend_on_device(request, device)
}

/// Load a production BPK or parts manifest into Burn's portable Flex CPU backend.
#[cfg(feature = "flex")]
pub fn load_backend_flex(request: LoadRequest) -> Result<FlexSiglip2Backend, String> {
    let device = burn::tensor::Device::flex();
    load_backend_on_device(request, device)
}

#[cfg(feature = "wgpu")]
pub fn load_backend_wgpu(request: LoadRequest) -> Result<WgpuSiglip2Backend, String> {
    let device = burn::tensor::Device::wgpu(Default::default());
    load_backend_on_device(request, device)
}

/// Load a SigLIP2 runtime on an explicitly selected Burn backend device.
///
/// This is the backend-generic entry point for callers that need to choose a
/// non-default CPU device, GPU adapter, or another compatible Burn backend.
pub fn load_backend_on_device(
    request: LoadRequest,
    device: Device,
) -> Result<Siglip2Runtime, String> {
    load_runtime(&request, device)
}

#[cfg(any(feature = "ndarray", feature = "flex"))]
pub fn run_inference(
    backend: &Siglip2Backend,
    request: Siglip2InferenceRequest,
) -> Result<Siglip2InferenceResponse, String> {
    run_inference_with_runtime(backend, request)
}

/// Run the flat image-inference compatibility API on a Flex CPU runtime.
#[cfg(feature = "flex")]
pub fn run_inference_flex(
    backend: &FlexSiglip2Backend,
    request: Siglip2InferenceRequest,
) -> Result<Siglip2InferenceResponse, String> {
    run_inference_with_runtime(backend, request)
}

#[cfg(feature = "wgpu")]
pub fn run_inference_wgpu(
    backend: &WgpuSiglip2Backend,
    request: Siglip2InferenceRequest,
) -> Result<Siglip2InferenceResponse, String> {
    run_inference_with_runtime(backend, request)
}

fn load_runtime(request: &LoadRequest, device: Device) -> Result<Siglip2Runtime, String> {
    let (model, load_stats) = match &request.source {
        WeightSource::BpkFile(path) => load_model_from_bpk_path(&device, path)?,
        WeightSource::SafetensorsFile(path) => {
            let config = required_config(request, "SafetensorsFile")?;
            load_model_from_safetensors_path(config, &device, path)?
        }
        WeightSource::PartsManifest(path) => {
            load_model_from_parts_manifest_path(&device, path, request.verify_checksums)?
        }
    };

    if let Some(expected) = &request.config
        && expected != &model.config
    {
        return Err(format!(
            "load request config mismatch: expected {:?}, package/model config {:?}",
            expected, model.config
        ));
    }

    Ok(Siglip2Runtime {
        model,
        device,
        load_stats,
    })
}

fn required_config<'a>(
    request: &'a LoadRequest,
    source_label: &str,
) -> Result<&'a Siglip2Config, String> {
    let config = request
        .config
        .as_ref()
        .ok_or_else(|| format!("LoadRequest.config is required for {source_label}"))?;
    config.validate_production_profile()?;
    Ok(config)
}

#[cfg(any(feature = "ndarray", feature = "flex", feature = "wgpu"))]
fn run_inference_with_runtime(
    runtime: &Siglip2Runtime,
    request: Siglip2InferenceRequest,
) -> Result<Siglip2InferenceResponse, String> {
    if request.shape.contains(&0) {
        return Err(format!(
            "inference input shape dimensions must be non-zero, got {:?}",
            request.shape
        ));
    }
    let expected_values = request
        .shape
        .iter()
        .try_fold(1usize, |elements, dimension| {
            elements.checked_mul(*dimension)
        })
        .ok_or_else(|| {
            format!(
                "inference input element count overflows usize for shape {:?}",
                request.shape
            )
        })?;
    if request.input.len() != expected_values {
        return Err(format!(
            "inference input value count mismatch for shape {:?}: expected {expected_values}, got {}",
            request.shape,
            request.input.len()
        ));
    }
    let input = Tensor::<4>::from_data(
        TensorData::new(request.input, request.shape),
        &runtime.device,
    );
    let Siglip2TensorEmbeddingResponse {
        embedding,
        hooks,
        mut evidence,
    } = runtime.encode_image(input, request.capture_hooks)?;
    let [batch, dim] = embedding.shape().dims();
    let embedding_data = embedding.into_data().convert::<f32>();
    let embedding_vec = embedding_data
        .try_to_vec::<f32>()
        .map_err(|err| format!("failed to read embedding tensor: {err:?}"))?;
    evidence.host_readbacks = evidence.host_readbacks.saturating_add(1);

    Ok(Siglip2InferenceResponse {
        shape: [batch, dim],
        embedding: embedding_vec,
        hooks,
        evidence,
    })
}

fn execution_evidence(device: &Device, hook: &Option<HookRecorder>) -> Siglip2ExecutionEvidence {
    let backend_label = format!("{device:?}");
    let device_label = format!("{device:?}");
    let mut evidence = Siglip2ExecutionEvidence {
        backend_label: backend_label.clone(),
        device_label: device_label.clone(),
        wgpu_executed: infer_wgpu_execution(backend_label.as_str(), device_label.as_str()),
        ..Siglip2ExecutionEvidence::default()
    };
    if let Some(hook) = hook {
        evidence.host_readbacks = hook.host_readbacks();
        evidence.stages_recorded = hook.len();
    }
    evidence
}

fn infer_wgpu_execution(backend_label: &str, device_label: &str) -> bool {
    let backend = backend_label.to_ascii_lowercase();
    let device = device_label.to_ascii_lowercase();
    backend.contains("burn_wgpu") || backend.contains("wgpu") || device.contains("wgpu")
}

#[cfg(all(test, any(feature = "ndarray", feature = "flex")))]
mod tests {
    use burn::tensor::{Int, Tensor, TensorData};

    use super::{
        Siglip2InferenceRequest, Siglip2Runtime, execution_evidence, infer_wgpu_execution,
        l2_normalize_embeddings, run_inference_with_runtime,
    };
    use crate::{PartLoadStats, Siglip2Config, Siglip2Model};

    #[test]
    fn detects_wgpu_from_backend_or_device_label() {
        assert!(infer_wgpu_execution(
            "burn_wgpu::CubeBackend<...>",
            "burn_wgpu::WgpuDevice"
        ));
        assert!(infer_wgpu_execution("some::backend", "my::WgpuDevice"));
        assert!(!infer_wgpu_execution(
            "burn_ndarray::NdArray",
            "burn::backend::ndarray::NdArrayDevice"
        ));
    }

    #[test]
    fn execution_evidence_uses_the_backend_reported_name() {
        let device = burn::tensor::Device::flex();
        let evidence = execution_evidence(&device, &None);
        assert_eq!(evidence.backend_label, format!("{device:?}"));
        assert!(evidence.backend_label.contains("Flex"));
        assert!(evidence.device_label.contains("Flex"));
    }

    #[test]
    fn l2_normalization_is_row_wise_and_zero_safe() -> Result<(), String> {
        let device = burn::tensor::Device::flex();
        let embeddings =
            Tensor::<2>::from_data(TensorData::new(vec![3.0, 4.0, 0.0, 0.0], [2, 2]), &device);
        let values = l2_normalize_embeddings(embeddings)?
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|err| format!("failed to read normalized embeddings: {err:?}"))?;

        assert!((values[0] - 0.6).abs() < 1.0e-6);
        assert!((values[1] - 0.8).abs() < 1.0e-6);
        assert!(values[2].abs() < 1.0e-6);
        assert!(values[3].abs() < 1.0e-6);
        Ok(())
    }

    #[test]
    fn l2_normalization_rejects_empty_batches() {
        let device = burn::tensor::Device::flex();
        let embeddings = Tensor::<2>::zeros([0, 4], &device);
        let err = l2_normalize_embeddings(embeddings).expect_err("empty batches must fail");
        assert!(err.contains("empty embedding tensor"));
    }

    #[test]
    fn multimodal_response_has_calibrated_matrix_shapes() -> Result<(), String> {
        let device = burn::tensor::Device::flex();
        let config = Siglip2Config::tiny_for_tests();
        let runtime = Siglip2Runtime {
            model: Siglip2Model::zeros(config.clone(), &device)?,
            device: device.clone(),
            load_stats: PartLoadStats::default(),
        };
        let image = Tensor::<4>::zeros(
            [2, config.channels, config.image_size, config.image_size],
            &device,
        );
        let input_ids = Tensor::<2, Int>::zeros([3, config.text_max_positions], &device);
        let response = runtime.encode_image_and_text_tokens(image, input_ids, None, false)?;

        assert_eq!(response.image_embedding.shape().dims::<2>(), [2, 8]);
        assert_eq!(response.text_embedding.shape().dims::<2>(), [3, 8]);
        assert_eq!(response.logits_per_image.shape().dims::<2>(), [2, 3]);
        assert_eq!(response.logits_per_text.shape().dims::<2>(), [3, 2]);
        assert_eq!(response.probabilities_per_image.shape().dims::<2>(), [2, 3]);
        assert_eq!(response.probabilities_per_text.shape().dims::<2>(), [3, 2]);
        let probabilities = response
            .probabilities_per_image
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|err| format!("failed to read similarity probabilities: {err:?}"))?;
        assert!(
            probabilities
                .iter()
                .all(|value| (*value - 0.5).abs() < 1.0e-6)
        );
        Ok(())
    }

    #[test]
    fn flat_inference_rejects_malformed_value_count_without_panicking() -> Result<(), String> {
        let device = burn::tensor::Device::flex();
        let config = Siglip2Config::tiny_for_tests();
        let runtime = Siglip2Runtime {
            model: Siglip2Model::zeros(config.clone(), &device)?,
            device: device.clone(),
            load_stats: PartLoadStats::default(),
        };
        let shape = [1, config.channels, config.image_size, config.image_size];
        let expected_values = shape.into_iter().product::<usize>();
        let error = run_inference_with_runtime(
            &runtime,
            Siglip2InferenceRequest {
                input: vec![0.0; expected_values - 1],
                shape,
                capture_hooks: false,
            },
        )
        .expect_err("a short input buffer must return an error");

        assert!(error.contains("inference input value count mismatch"));
        assert!(error.contains(&format!("expected {expected_values}")));
        Ok(())
    }

    #[test]
    fn flat_inference_rejects_shape_product_overflow_without_panicking() -> Result<(), String> {
        let device = burn::tensor::Device::flex();
        let runtime = Siglip2Runtime {
            model: Siglip2Model::zeros(Siglip2Config::tiny_for_tests(), &device)?,
            device: device.clone(),
            load_stats: PartLoadStats::default(),
        };
        let error = run_inference_with_runtime(
            &runtime,
            Siglip2InferenceRequest {
                input: Vec::new(),
                shape: [usize::MAX, 2, 1, 1],
                capture_hooks: false,
            },
        )
        .expect_err("an overflowing shape must return an error");

        assert!(error.contains("element count overflows usize"));
        Ok(())
    }

    #[test]
    fn flat_inference_rejects_zero_dimensions_without_panicking() -> Result<(), String> {
        let device = burn::tensor::Device::flex();
        let runtime = Siglip2Runtime {
            model: Siglip2Model::zeros(Siglip2Config::tiny_for_tests(), &device)?,
            device: device.clone(),
            load_stats: PartLoadStats::default(),
        };
        let error = run_inference_with_runtime(
            &runtime,
            Siglip2InferenceRequest {
                input: Vec::new(),
                shape: [0, 3, 8, 8],
                capture_hooks: false,
            },
        )
        .expect_err("a zero-sized tensor must return an error");

        assert!(error.contains("shape dimensions must be non-zero"));
        Ok(())
    }
}
