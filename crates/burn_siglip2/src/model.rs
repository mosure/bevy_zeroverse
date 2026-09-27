use std::collections::{BTreeMap, BTreeSet};

use burn::tensor::{Int, Tensor, TensorData, backend::Backend, module::embedding};

use crate::{config::Siglip2Config, hooks::HookRecorder};

pub(crate) const TEXT_TOKEN_EMBED_CHUNK_PREFIX: &str = "text.token_embed.weight.chunk.";
pub(crate) const TEXT_TOKEN_EMBED_CHUNK_ROWS: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSpec {
    pub key: String,
    pub shape: Vec<usize>,
}

/// Return the canonical model-weight schema without requiring a concrete backend.
pub fn expected_weight_specs(config: &Siglip2Config) -> Vec<WeightSpec> {
    let mut specs = Vec::new();

    specs.push(WeightSpec {
        key: "vision.patch_embed.weight".to_string(),
        shape: vec![config.hidden_dim, config.patch_dim()],
    });
    specs.push(WeightSpec {
        key: "vision.patch_embed.bias".to_string(),
        shape: vec![config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "vision.pos_embed".to_string(),
        shape: vec![config.image_token_count(), config.hidden_dim],
    });
    append_block_specs(&mut specs, "vision.blocks", config);
    specs.push(WeightSpec {
        key: "vision.post_norm.gamma".to_string(),
        shape: vec![config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "vision.post_norm.beta".to_string(),
        shape: vec![config.hidden_dim],
    });
    append_pool_head_specs(&mut specs, config);

    specs.push(WeightSpec {
        key: "text.token_embed.weight".to_string(),
        shape: vec![config.text_vocab_size, config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "text.pos_embed".to_string(),
        shape: vec![config.text_max_positions, config.hidden_dim],
    });
    append_block_specs(&mut specs, "text.blocks", config);
    specs.push(WeightSpec {
        key: "text.final_norm.gamma".to_string(),
        shape: vec![config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "text.final_norm.beta".to_string(),
        shape: vec![config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "text.projection.weight".to_string(),
        shape: vec![config.projection_dim, config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "text.projection.bias".to_string(),
        shape: vec![config.projection_dim],
    });

    specs.push(WeightSpec {
        key: "logit_scale".to_string(),
        shape: vec![1],
    });
    specs.push(WeightSpec {
        key: "logit_bias".to_string(),
        shape: vec![1],
    });

    specs
}

pub(crate) fn expected_weight_key_set(config: &Siglip2Config) -> BTreeSet<String> {
    expected_weight_specs(config)
        .into_iter()
        .map(|spec| spec.key)
        .collect()
}

pub(crate) fn expected_text_token_embedding_chunk_specs(config: &Siglip2Config) -> Vec<WeightSpec> {
    (0..config.text_vocab_size)
        .step_by(TEXT_TOKEN_EMBED_CHUNK_ROWS)
        .enumerate()
        .map(|(index, start)| WeightSpec {
            key: format!("{TEXT_TOKEN_EMBED_CHUNK_PREFIX}{index:05}"),
            shape: vec![
                (config.text_vocab_size - start).min(TEXT_TOKEN_EMBED_CHUNK_ROWS),
                config.hidden_dim,
            ],
        })
        .collect()
}

pub(crate) fn validate_loaded_weight_keys(
    config: &Siglip2Config,
    loaded: &BTreeSet<String>,
) -> Result<(), String> {
    const FULL_KEY: &str = "text.token_embed.weight";

    let chunk_keys = expected_text_token_embedding_chunk_specs(config)
        .into_iter()
        .map(|spec| spec.key)
        .collect::<BTreeSet<_>>();
    let loaded_chunk_keys = loaded
        .iter()
        .filter(|key| key.starts_with(TEXT_TOKEN_EMBED_CHUNK_PREFIX))
        .cloned()
        .collect::<BTreeSet<_>>();
    if loaded.contains(FULL_KEY) && !loaded_chunk_keys.is_empty() {
        return Err(
            "text token embedding must use either one full tensor or canonical chunks, never both"
                .to_string(),
        );
    }
    let unexpected_chunks = loaded_chunk_keys
        .difference(&chunk_keys)
        .take(16)
        .cloned()
        .collect::<Vec<_>>();
    if !unexpected_chunks.is_empty() {
        return Err(format!(
            "unexpected text token embedding chunks (showing up to 16): {}",
            unexpected_chunks.join(", ")
        ));
    }

    let mut expected = expected_weight_key_set(config);
    if !loaded.contains(FULL_KEY) {
        expected.remove(FULL_KEY);
        expected.extend(chunk_keys);
    }
    let missing = expected
        .difference(loaded)
        .take(16)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "missing required weight tensors (showing up to 16): {}",
            missing.join(", ")
        ));
    }
    let unexpected = loaded
        .difference(&expected)
        .take(16)
        .cloned()
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(format!(
            "unexpected weight tensors (showing up to 16): {}",
            unexpected.join(", ")
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Linear<B: Backend> {
    weight: Tensor<B, 2>, // [out, in]
    bias: Tensor<B, 1>,   // [out]
}

impl<B: Backend> Linear<B> {
    fn zeros(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            weight: Tensor::<B, 2>::zeros([output_dim, input_dim], device),
            bias: Tensor::<B, 1>::zeros([output_dim], device),
        }
    }

    fn set_weight(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        expected: [usize; 2],
        device: &B::Device,
    ) -> Result<(), String> {
        expect_shape(key, shape, &expected)?;
        expect_values_shape(key, &values, &expected)?;
        self.weight = Tensor::<B, 2>::from_data(TensorData::new(values, expected), device);
        Ok(())
    }

    fn set_bias(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        expected: [usize; 1],
        device: &B::Device,
    ) -> Result<(), String> {
        expect_shape(key, shape, &expected)?;
        expect_values_shape(key, &values, &expected)?;
        self.bias = Tensor::<B, 1>::from_data(TensorData::new(values, expected), device);
        Ok(())
    }

    fn forward_3d(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, input_dim] = input.shape().dims();
        let [output_dim, expected_input] = self.weight.shape().dims();
        assert_eq!(
            input_dim, expected_input,
            "linear input dim mismatch: got {input_dim}, expected {expected_input}"
        );
        let output = input
            .reshape([batch * tokens, input_dim])
            .matmul(self.weight.clone().swap_dims(0, 1));
        let bias = self
            .bias
            .clone()
            .reshape([1, output_dim])
            .expand([(batch * tokens) as i64, -1]);
        output.add(bias).reshape([batch, tokens, output_dim])
    }

    fn forward_2d(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, _] = input.shape().dims();
        let [output_dim, _] = self.weight.shape().dims();
        let output = input.matmul(self.weight.clone().swap_dims(0, 1));
        let bias = self
            .bias
            .clone()
            .reshape([1, output_dim])
            .expand([batch as i64, -1]);
        output.add(bias)
    }
}

#[derive(Debug, Clone)]
struct EmbeddingTable<B: Backend> {
    // Keep individual backend buffers below WebGPU's guaranteed binding limits. The published
    // 256k-token table is 0.75-1.15 GiB as f32 and cannot be represented by one portable WebGPU
    // storage buffer.
    weights: Vec<Tensor<B, 2>>, // each [chunk_vocab, hidden]
    vocab_size: usize,
    hidden_dim: usize,
    chunk_rows: usize,
}

impl<B: Backend> EmbeddingTable<B> {
    fn zeros(vocab_size: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self::zeros_with_chunk_rows(vocab_size, hidden_dim, TEXT_TOKEN_EMBED_CHUNK_ROWS, device)
    }

    fn zeros_with_chunk_rows(
        vocab_size: usize,
        hidden_dim: usize,
        chunk_rows: usize,
        device: &B::Device,
    ) -> Self {
        let chunk_rows = chunk_rows.max(1);
        let weights = (0..vocab_size)
            .step_by(chunk_rows)
            .map(|start| {
                Tensor::<B, 2>::zeros([(vocab_size - start).min(chunk_rows), hidden_dim], device)
            })
            .collect();
        Self {
            weights,
            vocab_size,
            hidden_dim,
            chunk_rows,
        }
    }

    fn set_weight(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        expected: [usize; 2],
        device: &B::Device,
    ) -> Result<(), String> {
        *self = Self::from_weight_values(key, shape, values, expected, self.chunk_rows, device)?;
        Ok(())
    }

    fn from_weight_values(
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        expected: [usize; 2],
        chunk_rows: usize,
        device: &B::Device,
    ) -> Result<Self, String> {
        expect_shape(key, shape, &expected)?;
        expect_values_shape(key, &values, &expected)?;
        let chunk_rows = chunk_rows.max(1);
        let values_per_chunk = chunk_rows
            .checked_mul(expected[1])
            .ok_or_else(|| format!("embedding chunk size overflows for '{key}'"))?;
        let weights = values
            .chunks(values_per_chunk)
            .map(|values| {
                let rows = values.len() / expected[1];
                Tensor::<B, 2>::from_data(
                    TensorData::new(values.to_vec(), [rows, expected[1]]),
                    device,
                )
            })
            .collect();
        Ok(Self {
            weights,
            vocab_size: expected[0],
            hidden_dim: expected[1],
            chunk_rows,
        })
    }

    fn set_weight_chunk(
        &mut self,
        key: &str,
        chunk_index: usize,
        shape: &[usize],
        values: Vec<f32>,
        device: &B::Device,
    ) -> Result<(), String> {
        let start = chunk_index
            .checked_mul(self.chunk_rows)
            .ok_or_else(|| format!("embedding chunk index overflows for '{key}'"))?;
        if start >= self.vocab_size {
            return Err(format!(
                "embedding chunk index {chunk_index} is out of range for '{key}'"
            ));
        }
        let rows = (self.vocab_size - start).min(self.chunk_rows);
        expect_shape(key, shape, &[rows, self.hidden_dim])?;
        expect_values_shape(key, &values, &[rows, self.hidden_dim])?;
        let slot = self.weights.get_mut(chunk_index).ok_or_else(|| {
            format!("embedding chunk index {chunk_index} is out of range for '{key}'")
        })?;
        *slot = Tensor::<B, 2>::from_data(TensorData::new(values, [rows, self.hidden_dim]), device);
        Ok(())
    }

    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, sequence] = input_ids.shape().dims();
        let mut output: Option<Tensor<B, 3>> = None;
        for (chunk_index, weight) in self.weights.iter().enumerate() {
            let start = chunk_index * self.chunk_rows;
            let rows = weight.shape().dims::<2>()[0];
            let end = start + rows;
            let in_chunk = input_ids
                .clone()
                .greater_equal_elem(start as i64)
                .bool_and(input_ids.clone().lower_elem(end as i64))
                .float()
                .reshape([batch, sequence, 1]);
            let local_ids = input_ids
                .clone()
                .sub_scalar(start as i64)
                .clamp(0i64, rows.saturating_sub(1) as i64);
            let selected = embedding(weight.clone(), local_ids).mul(in_chunk);
            output = Some(match output {
                Some(accumulated) => accumulated.add(selected),
                None => selected,
            });
        }
        output.unwrap_or_else(|| {
            Tensor::<B, 3>::zeros([batch, sequence, self.hidden_dim], &input_ids.device())
        })
    }
}

#[derive(Debug, Clone)]
struct LayerNorm<B: Backend> {
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    eps: f32,
}

impl<B: Backend> LayerNorm<B> {
    fn identity(hidden_dim: usize, eps: f32, device: &B::Device) -> Self {
        Self {
            gamma: Tensor::<B, 1>::ones([hidden_dim], device),
            beta: Tensor::<B, 1>::zeros([hidden_dim], device),
            eps,
        }
    }

    fn set_gamma(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        expected: [usize; 1],
        device: &B::Device,
    ) -> Result<(), String> {
        expect_shape(key, shape, &expected)?;
        expect_values_shape(key, &values, &expected)?;
        self.gamma = Tensor::<B, 1>::from_data(TensorData::new(values, expected), device);
        Ok(())
    }

    fn set_beta(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        expected: [usize; 1],
        device: &B::Device,
    ) -> Result<(), String> {
        expect_shape(key, shape, &expected)?;
        expect_values_shape(key, &values, &expected)?;
        self.beta = Tensor::<B, 1>::from_data(TensorData::new(values, expected), device);
        Ok(())
    }

    fn forward_3d(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, hidden_dim] = input.shape().dims();
        let mean = input.clone().mean_dim(2);
        let centered = input.clone().sub(mean);
        // WGSL `pow(x, 2.0)` is undefined for negative `x` on some WebGPU adapters. Explicit
        // multiplication is both exact for this integer power and portable across native/Wasm.
        let var = centered.clone().mul(centered.clone()).mean_dim(2);
        let normalized = centered.div(var.add_scalar(self.eps).sqrt());
        let gamma = self.gamma.clone().reshape([1, 1, hidden_dim]).expand([
            batch as i64,
            tokens as i64,
            -1,
        ]);
        let beta =
            self.beta
                .clone()
                .reshape([1, 1, hidden_dim])
                .expand([batch as i64, tokens as i64, -1]);
        normalized.mul(gamma).add(beta)
    }
}

#[derive(Debug, Clone)]
struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    fn zeros(config: &Siglip2Config, device: &B::Device) -> Self {
        Self {
            q_proj: Linear::zeros(config.hidden_dim, config.hidden_dim, device),
            k_proj: Linear::zeros(config.hidden_dim, config.hidden_dim, device),
            v_proj: Linear::zeros(config.hidden_dim, config.hidden_dim, device),
            out_proj: Linear::zeros(config.hidden_dim, config.hidden_dim, device),
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
        }
    }

    fn forward(
        &self,
        query_states: Tensor<B, 3>,
        key_value_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 4>>,
        prefix: &str,
        mut hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 3>, String> {
        let [batch, query_len, hidden_dim] = query_states.shape().dims();
        let [kv_batch, key_len, kv_hidden_dim] = key_value_states.shape().dims();
        if batch != kv_batch {
            return Err(format!(
                "attention batch mismatch: query batch {batch}, key/value batch {kv_batch}"
            ));
        }
        if hidden_dim != kv_hidden_dim {
            return Err(format!(
                "attention hidden mismatch: query hidden {hidden_dim}, key/value hidden {kv_hidden_dim}"
            ));
        }

        let q = self.q_proj.forward_3d(query_states);
        let k = self.k_proj.forward_3d(key_value_states.clone());
        let v = self.v_proj.forward_3d(key_value_states);
        record_tensor(&mut hook, &format!("{prefix}.q"), &q)?;
        record_tensor(&mut hook, &format!("{prefix}.k"), &k)?;
        record_tensor(&mut hook, &format!("{prefix}.v"), &v)?;

        let q = q
            .reshape([batch, query_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .reshape([batch, key_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([batch, key_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        let mut logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((self.head_dim as f32).powf(-0.5));
        if let Some(mask) = attention_mask {
            let mask = broadcast_attention_mask(mask, batch, self.num_heads, query_len, key_len)?;
            logits = logits.add(mask);
        }
        let attn = softmax_last_dim_4d(logits);
        record_tensor(&mut hook, &format!("{prefix}.probs"), &attn)?;

        let hidden = attn
            .matmul(v)
            .permute([0, 2, 1, 3])
            .reshape([batch, query_len, hidden_dim]);
        let output = self.out_proj.forward_3d(hidden);
        record_tensor(&mut hook, &format!("{prefix}.out"), &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct Siglip2Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    norm2: LayerNorm<B>,
    mlp_fc1: Linear<B>,
    mlp_fc2: Linear<B>,
}

impl<B: Backend> Siglip2Block<B> {
    fn zeros(config: &Siglip2Config, device: &B::Device) -> Self {
        Self {
            norm1: LayerNorm::identity(config.hidden_dim, config.layer_norm_eps, device),
            attn: Attention::zeros(config, device),
            norm2: LayerNorm::identity(config.hidden_dim, config.layer_norm_eps, device),
            mlp_fc1: Linear::zeros(config.hidden_dim, config.intermediate_dim, device),
            mlp_fc2: Linear::zeros(config.intermediate_dim, config.hidden_dim, device),
        }
    }

    fn forward(
        &self,
        block_prefix: &str,
        block_idx: usize,
        input: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 4>>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 3>, String> {
        let prefix = format!("{block_prefix}.{block_idx}");
        let attn_input = self.norm1.forward_3d(input.clone());
        let attn_out = self.attn.forward(
            attn_input.clone(),
            attn_input,
            attention_mask,
            &format!("{prefix}.attn"),
            hook.as_deref_mut(),
        )?;
        let hidden = input.add(attn_out);

        let mlp_input = self.norm2.forward_3d(hidden.clone());
        let mlp_hidden = self.mlp_fc1.forward_3d(mlp_input);
        let mlp_hidden = gelu(mlp_hidden);
        let mlp_out = self.mlp_fc2.forward_3d(mlp_hidden);
        record_tensor(&mut hook, &format!("{prefix}.mlp.out"), &mlp_out)?;

        Ok(hidden.add(mlp_out))
    }
}

#[derive(Debug, Clone)]
struct AttentionPoolHead<B: Backend> {
    probe: Tensor<B, 3>,
    attn: Attention<B>,
    layernorm: LayerNorm<B>,
    mlp_fc1: Linear<B>,
    mlp_fc2: Linear<B>,
}

impl<B: Backend> AttentionPoolHead<B> {
    fn zeros(config: &Siglip2Config, device: &B::Device) -> Self {
        Self {
            probe: Tensor::<B, 3>::zeros([1, 1, config.hidden_dim], device),
            attn: Attention::zeros(config, device),
            layernorm: LayerNorm::identity(config.hidden_dim, config.layer_norm_eps, device),
            mlp_fc1: Linear::zeros(config.hidden_dim, config.intermediate_dim, device),
            mlp_fc2: Linear::zeros(config.intermediate_dim, config.hidden_dim, device),
        }
    }

    fn set_probe(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        expected: [usize; 3],
        device: &B::Device,
    ) -> Result<(), String> {
        expect_shape(key, shape, &expected)?;
        expect_values_shape(key, &values, &expected)?;
        self.probe = Tensor::<B, 3>::from_data(TensorData::new(values, expected), device);
        Ok(())
    }

    fn forward(
        &self,
        hidden_state: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 2>, String> {
        let [batch, _, hidden_dim] = hidden_state.shape().dims();
        let probe = self.probe.clone().expand([batch as i64, -1, -1]);
        let hidden_state = self.attn.forward(
            probe,
            hidden_state,
            None,
            "vision.head.attn",
            hook.as_deref_mut(),
        )?;
        let residual = hidden_state.clone();
        let hidden_state = self.layernorm.forward_3d(hidden_state);
        record_tensor(&mut hook, "vision.head.layernorm", &hidden_state)?;
        let mlp_hidden = self.mlp_fc1.forward_3d(hidden_state);
        let mlp_hidden = gelu(mlp_hidden);
        let mlp_out = self.mlp_fc2.forward_3d(mlp_hidden);
        let pooled = residual.add(mlp_out);
        record_tensor(&mut hook, "vision.head.output", &pooled)?;
        Ok(pooled
            .slice([0..batch, 0..1, 0..hidden_dim])
            .reshape([batch, hidden_dim]))
    }
}

#[derive(Debug, Clone)]
pub struct Siglip2Model<B: Backend> {
    pub config: Siglip2Config,
    vision_patch_embed: Linear<B>,
    vision_pos_embed: Tensor<B, 2>,
    vision_blocks: Vec<Siglip2Block<B>>,
    vision_post_norm: LayerNorm<B>,
    vision_head: AttentionPoolHead<B>,
    text_token_embed: EmbeddingTable<B>,
    text_pos_embed: Tensor<B, 2>,
    text_blocks: Vec<Siglip2Block<B>>,
    text_final_norm: LayerNorm<B>,
    text_projection: Linear<B>,
    logit_scale: Tensor<B, 1>,
    logit_bias: Tensor<B, 1>,
}

/// Incrementally constructs a model from backend tensors without first allocating a complete
/// zero-filled model.
///
/// Production checkpoints are much larger in memory than their F16 transport representation
/// because Burn materializes the weights using the backend float element type. Keeping the
/// builder separate from [`Siglip2Model::zeros`] ensures sharded loaders only allocate each real
/// tensor once. The final model is assembled by moving the populated tensors into their runtime
/// structures after the complete key set has been validated.
pub(crate) struct Siglip2ModelBuilder<B: Backend> {
    config: Siglip2Config,
    expected_shapes: BTreeMap<String, Vec<usize>>,
    expected_text_chunk_shapes: BTreeMap<String, Vec<usize>>,
    loaded: BTreeSet<String>,
    tensors: BTreeMap<String, PendingTensor<B>>,
    text_token_embed: PendingEmbeddingTable<B>,
}

enum PendingTensor<B: Backend> {
    D1(Tensor<B, 1>),
    D2(Tensor<B, 2>),
    D3(Tensor<B, 3>),
}

enum PendingEmbeddingTable<B: Backend> {
    Empty,
    Full(EmbeddingTable<B>),
    Chunks(Vec<Option<Tensor<B, 2>>>),
}

impl<B: Backend> Siglip2ModelBuilder<B> {
    pub(crate) fn new(config: Siglip2Config) -> Result<Self, String> {
        config.validate()?;
        if config.projection_dim != config.hidden_dim {
            return Err(format!(
                "projection_dim ({}) must equal hidden_dim ({}) for runtime compatibility",
                config.projection_dim, config.hidden_dim
            ));
        }
        let expected_shapes = Siglip2Model::<B>::expected_weight_specs(&config)
            .into_iter()
            .map(|spec| (spec.key, spec.shape))
            .collect();
        let expected_text_chunk_shapes =
            Siglip2Model::<B>::expected_text_token_embedding_chunk_specs(&config)
                .into_iter()
                .map(|spec| (spec.key, spec.shape))
                .collect();
        Ok(Self {
            config,
            expected_shapes,
            expected_text_chunk_shapes,
            loaded: BTreeSet::new(),
            tensors: BTreeMap::new(),
            text_token_embed: PendingEmbeddingTable::Empty,
        })
    }

    pub(crate) fn config(&self) -> &Siglip2Config {
        &self.config
    }

    pub(crate) fn apply_weight(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        device: &B::Device,
    ) -> Result<(), String> {
        if self.loaded.contains(key) {
            return Err(format!("duplicate weight key loaded: '{key}'"));
        }

        const FULL_TEXT_EMBEDDING: &str = "text.token_embed.weight";
        if key == FULL_TEXT_EMBEDDING {
            if matches!(self.text_token_embed, PendingEmbeddingTable::Chunks(_)) {
                return Err(
                    "cannot apply a full text token embedding after chunk tensors were loaded"
                        .to_string(),
                );
            }
            let expected = [self.config.text_vocab_size, self.config.hidden_dim];
            let table = EmbeddingTable::from_weight_values(
                key,
                shape,
                values,
                expected,
                TEXT_TOKEN_EMBED_CHUNK_ROWS,
                device,
            )?;
            self.text_token_embed = PendingEmbeddingTable::Full(table);
            self.loaded.insert(key.to_string());
            return Ok(());
        }

        if let Some(index) = key.strip_prefix(TEXT_TOKEN_EMBED_CHUNK_PREFIX) {
            if matches!(self.text_token_embed, PendingEmbeddingTable::Full(_)) {
                return Err(
                    "cannot apply text token embedding chunks after the full tensor was loaded"
                        .to_string(),
                );
            }
            let chunk_index = index
                .parse::<usize>()
                .map_err(|_| format!("invalid text token embedding chunk key '{key}'"))?;
            let canonical = format!("{TEXT_TOKEN_EMBED_CHUNK_PREFIX}{chunk_index:05}");
            if key != canonical {
                return Err(format!(
                    "non-canonical text token embedding chunk key '{key}', expected '{canonical}'"
                ));
            }
            let expected = self
                .expected_text_chunk_shapes
                .get(key)
                .ok_or_else(|| format!("unknown weight key '{key}'"))?;
            expect_shape(key, shape, expected)?;
            expect_values_shape(key, &values, expected)?;
            let expected = [expected[0], expected[1]];
            let tensor = Tensor::<B, 2>::from_data(TensorData::new(values, expected), device);
            if matches!(self.text_token_embed, PendingEmbeddingTable::Empty) {
                self.text_token_embed = PendingEmbeddingTable::Chunks(
                    (0..self.expected_text_chunk_shapes.len())
                        .map(|_| None)
                        .collect(),
                );
            }
            let PendingEmbeddingTable::Chunks(chunks) = &mut self.text_token_embed else {
                return Err("internal text embedding builder state mismatch".to_string());
            };
            let slot = chunks.get_mut(chunk_index).ok_or_else(|| {
                format!("embedding chunk index {chunk_index} is out of range for '{key}'")
            })?;
            if slot.is_some() {
                return Err(format!("duplicate weight key loaded: '{key}'"));
            }
            *slot = Some(tensor);
            self.loaded.insert(key.to_string());
            return Ok(());
        }

        let expected = self
            .expected_shapes
            .get(key)
            .ok_or_else(|| format!("unknown weight key '{key}'"))?;
        expect_shape(key, shape, expected)?;
        expect_values_shape(key, &values, expected)?;
        let tensor = match expected.as_slice() {
            [length] => PendingTensor::D1(Tensor::<B, 1>::from_data(
                TensorData::new(values, [*length]),
                device,
            )),
            [rows, columns] => PendingTensor::D2(Tensor::<B, 2>::from_data(
                TensorData::new(values, [*rows, *columns]),
                device,
            )),
            [first, second, third] => PendingTensor::D3(Tensor::<B, 3>::from_data(
                TensorData::new(values, [*first, *second, *third]),
                device,
            )),
            _ => {
                return Err(format!(
                    "unsupported expected tensor rank {} for '{key}'",
                    expected.len()
                ));
            }
        };
        self.tensors.insert(key.to_string(), tensor);
        self.loaded.insert(key.to_string());
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<Siglip2Model<B>, String> {
        validate_loaded_weight_keys(&self.config, &self.loaded)?;

        let text_token_embed = match self.text_token_embed {
            PendingEmbeddingTable::Full(table) => table,
            PendingEmbeddingTable::Chunks(chunks) => {
                let weights = chunks
                    .into_iter()
                    .enumerate()
                    .map(|(index, tensor)| {
                        tensor.ok_or_else(|| {
                            format!(
                                "missing required text token embedding chunk {index:05} during model assembly"
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                EmbeddingTable {
                    weights,
                    vocab_size: self.config.text_vocab_size,
                    hidden_dim: self.config.hidden_dim,
                    chunk_rows: TEXT_TOKEN_EMBED_CHUNK_ROWS,
                }
            }
            PendingEmbeddingTable::Empty => {
                return Err(
                    "missing required text token embedding during model assembly".to_string(),
                );
            }
        };

        let vision_patch_embed = take_linear(&mut self.tensors, "vision.patch_embed")?;
        let vision_pos_embed = take_tensor_2(&mut self.tensors, "vision.pos_embed")?;
        let mut vision_blocks = Vec::with_capacity(self.config.num_layers);
        for index in 0..self.config.num_layers {
            vision_blocks.push(take_block(
                &mut self.tensors,
                &format!("vision.blocks.{index}"),
                &self.config,
            )?);
        }
        let vision_post_norm = take_layer_norm(
            &mut self.tensors,
            "vision.post_norm",
            self.config.layer_norm_eps,
        )?;
        let vision_head = take_pool_head(&mut self.tensors, &self.config)?;
        let text_pos_embed = take_tensor_2(&mut self.tensors, "text.pos_embed")?;
        let mut text_blocks = Vec::with_capacity(self.config.num_layers);
        for index in 0..self.config.num_layers {
            text_blocks.push(take_block(
                &mut self.tensors,
                &format!("text.blocks.{index}"),
                &self.config,
            )?);
        }
        let text_final_norm = take_layer_norm(
            &mut self.tensors,
            "text.final_norm",
            self.config.layer_norm_eps,
        )?;
        let text_projection = take_linear(&mut self.tensors, "text.projection")?;
        let logit_scale = take_tensor_1(&mut self.tensors, "logit_scale")?;
        let logit_bias = take_tensor_1(&mut self.tensors, "logit_bias")?;

        if let Some(key) = self.tensors.keys().next() {
            return Err(format!(
                "internal model builder left tensor '{key}' unconsumed during assembly"
            ));
        }

        Ok(Siglip2Model {
            config: self.config,
            vision_patch_embed,
            vision_pos_embed,
            vision_blocks,
            vision_post_norm,
            vision_head,
            text_token_embed,
            text_pos_embed,
            text_blocks,
            text_final_norm,
            text_projection,
            logit_scale,
            logit_bias,
        })
    }
}

fn take_tensor_1<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    key: &str,
) -> Result<Tensor<B, 1>, String> {
    match tensors.remove(key) {
        Some(PendingTensor::D1(tensor)) => Ok(tensor),
        Some(_) => Err(format!("internal tensor rank mismatch for '{key}'")),
        None => Err(format!("missing tensor '{key}' during model assembly")),
    }
}

fn take_tensor_2<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    key: &str,
) -> Result<Tensor<B, 2>, String> {
    match tensors.remove(key) {
        Some(PendingTensor::D2(tensor)) => Ok(tensor),
        Some(_) => Err(format!("internal tensor rank mismatch for '{key}'")),
        None => Err(format!("missing tensor '{key}' during model assembly")),
    }
}

fn take_tensor_3<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    key: &str,
) -> Result<Tensor<B, 3>, String> {
    match tensors.remove(key) {
        Some(PendingTensor::D3(tensor)) => Ok(tensor),
        Some(_) => Err(format!("internal tensor rank mismatch for '{key}'")),
        None => Err(format!("missing tensor '{key}' during model assembly")),
    }
}

fn take_linear<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    prefix: &str,
) -> Result<Linear<B>, String> {
    Ok(Linear {
        weight: take_tensor_2(tensors, &format!("{prefix}.weight"))?,
        bias: take_tensor_1(tensors, &format!("{prefix}.bias"))?,
    })
}

fn take_layer_norm<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm<B>, String> {
    Ok(LayerNorm {
        gamma: take_tensor_1(tensors, &format!("{prefix}.gamma"))?,
        beta: take_tensor_1(tensors, &format!("{prefix}.beta"))?,
        eps,
    })
}

fn take_attention<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    prefix: &str,
    config: &Siglip2Config,
) -> Result<Attention<B>, String> {
    Ok(Attention {
        q_proj: take_linear(tensors, &format!("{prefix}.q_proj"))?,
        k_proj: take_linear(tensors, &format!("{prefix}.k_proj"))?,
        v_proj: take_linear(tensors, &format!("{prefix}.v_proj"))?,
        out_proj: take_linear(tensors, &format!("{prefix}.out_proj"))?,
        num_heads: config.num_heads,
        head_dim: config.head_dim(),
    })
}

fn take_block<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    prefix: &str,
    config: &Siglip2Config,
) -> Result<Siglip2Block<B>, String> {
    Ok(Siglip2Block {
        norm1: take_layer_norm(tensors, &format!("{prefix}.norm1"), config.layer_norm_eps)?,
        attn: take_attention(tensors, &format!("{prefix}.attn"), config)?,
        norm2: take_layer_norm(tensors, &format!("{prefix}.norm2"), config.layer_norm_eps)?,
        mlp_fc1: take_linear(tensors, &format!("{prefix}.mlp.fc1"))?,
        mlp_fc2: take_linear(tensors, &format!("{prefix}.mlp.fc2"))?,
    })
}

fn take_pool_head<B: Backend>(
    tensors: &mut BTreeMap<String, PendingTensor<B>>,
    config: &Siglip2Config,
) -> Result<AttentionPoolHead<B>, String> {
    Ok(AttentionPoolHead {
        probe: take_tensor_3(tensors, "vision.head.probe")?,
        attn: take_attention(tensors, "vision.head.attn", config)?,
        layernorm: take_layer_norm(tensors, "vision.head.layernorm", config.layer_norm_eps)?,
        mlp_fc1: take_linear(tensors, "vision.head.mlp.fc1")?,
        mlp_fc2: take_linear(tensors, "vision.head.mlp.fc2")?,
    })
}

impl<B: Backend> Siglip2Model<B> {
    pub fn zeros(config: Siglip2Config, device: &B::Device) -> Result<Self, String> {
        config.validate()?;
        if config.projection_dim != config.hidden_dim {
            return Err(format!(
                "projection_dim ({}) must equal hidden_dim ({}) for runtime compatibility",
                config.projection_dim, config.hidden_dim
            ));
        }
        let vision_patch_embed = Linear::zeros(config.patch_dim(), config.hidden_dim, device);
        let vision_pos_embed =
            Tensor::<B, 2>::zeros([config.image_token_count(), config.hidden_dim], device);
        let vision_blocks = (0..config.num_layers)
            .map(|_| Siglip2Block::zeros(&config, device))
            .collect::<Vec<_>>();
        let vision_post_norm =
            LayerNorm::identity(config.hidden_dim, config.layer_norm_eps, device);
        let vision_head = AttentionPoolHead::zeros(&config, device);
        let text_token_embed =
            EmbeddingTable::zeros(config.text_vocab_size, config.hidden_dim, device);
        let text_pos_embed =
            Tensor::<B, 2>::zeros([config.text_max_positions, config.hidden_dim], device);
        let text_blocks = (0..config.num_layers)
            .map(|_| Siglip2Block::zeros(&config, device))
            .collect::<Vec<_>>();
        let text_final_norm = LayerNorm::identity(config.hidden_dim, config.layer_norm_eps, device);
        let text_projection = Linear::zeros(config.hidden_dim, config.projection_dim, device);
        let logit_scale = Tensor::<B, 1>::zeros([1], device);
        let logit_bias = Tensor::<B, 1>::zeros([1], device);
        Ok(Self {
            config,
            vision_patch_embed,
            vision_pos_embed,
            vision_blocks,
            vision_post_norm,
            vision_head,
            text_token_embed,
            text_pos_embed,
            text_blocks,
            text_final_norm,
            text_projection,
            logit_scale,
            logit_bias,
        })
    }

    pub fn backend_label() -> String {
        std::any::type_name::<B>().to_string()
    }

    pub fn expected_weight_specs(config: &Siglip2Config) -> Vec<WeightSpec> {
        crate::model::expected_weight_specs(config)
    }

    pub fn expected_weight_key_set(config: &Siglip2Config) -> BTreeSet<String> {
        crate::model::expected_weight_key_set(config)
    }

    pub(crate) fn expected_text_token_embedding_chunk_specs(
        config: &Siglip2Config,
    ) -> Vec<WeightSpec> {
        crate::model::expected_text_token_embedding_chunk_specs(config)
    }

    pub fn apply_weight(
        &mut self,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        device: &B::Device,
    ) -> Result<(), String> {
        if key == "vision.patch_embed.weight" {
            return self.vision_patch_embed.set_weight(
                key,
                shape,
                values,
                [self.config.hidden_dim, self.config.patch_dim()],
                device,
            );
        }
        if key == "vision.patch_embed.bias" {
            return self.vision_patch_embed.set_bias(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            );
        }
        if key == "vision.pos_embed" {
            let expected = [self.config.image_token_count(), self.config.hidden_dim];
            expect_shape(key, shape, &expected)?;
            expect_values_shape(key, &values, &expected)?;
            self.vision_pos_embed =
                Tensor::<B, 2>::from_data(TensorData::new(values, expected), device);
            return Ok(());
        }
        if key == "vision.post_norm.gamma" {
            return self.vision_post_norm.set_gamma(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            );
        }
        if key == "vision.post_norm.beta" {
            return self.vision_post_norm.set_beta(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            );
        }
        if let Some(rest) = key.strip_prefix("vision.head.") {
            return self.apply_pool_head_weight(rest, key, shape, values, device);
        }

        if key == "text.token_embed.weight" {
            return self.text_token_embed.set_weight(
                key,
                shape,
                values,
                [self.config.text_vocab_size, self.config.hidden_dim],
                device,
            );
        }
        if let Some(index) = key.strip_prefix(TEXT_TOKEN_EMBED_CHUNK_PREFIX) {
            let chunk_index = index
                .parse::<usize>()
                .map_err(|_| format!("invalid text token embedding chunk key '{key}'"))?;
            let canonical = format!("{TEXT_TOKEN_EMBED_CHUNK_PREFIX}{chunk_index:05}");
            if key != canonical {
                return Err(format!(
                    "non-canonical text token embedding chunk key '{key}', expected '{canonical}'"
                ));
            }
            return self
                .text_token_embed
                .set_weight_chunk(key, chunk_index, shape, values, device);
        }
        if key == "text.pos_embed" {
            let expected = [self.config.text_max_positions, self.config.hidden_dim];
            expect_shape(key, shape, &expected)?;
            expect_values_shape(key, &values, &expected)?;
            self.text_pos_embed =
                Tensor::<B, 2>::from_data(TensorData::new(values, expected), device);
            return Ok(());
        }
        if key == "text.final_norm.gamma" {
            return self.text_final_norm.set_gamma(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            );
        }
        if key == "text.final_norm.beta" {
            return self.text_final_norm.set_beta(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            );
        }
        if key == "text.projection.weight" {
            return self.text_projection.set_weight(
                key,
                shape,
                values,
                [self.config.projection_dim, self.config.hidden_dim],
                device,
            );
        }
        if key == "text.projection.bias" {
            return self.text_projection.set_bias(
                key,
                shape,
                values,
                [self.config.projection_dim],
                device,
            );
        }
        if key == "logit_scale" {
            expect_shape(key, shape, &[1])?;
            expect_values_len(key, &values, 1)?;
            self.logit_scale = Tensor::<B, 1>::from_data(TensorData::new(values, [1]), device);
            return Ok(());
        }
        if key == "logit_bias" {
            expect_shape(key, shape, &[1])?;
            expect_values_len(key, &values, 1)?;
            self.logit_bias = Tensor::<B, 1>::from_data(TensorData::new(values, [1]), device);
            return Ok(());
        }

        if let Some(rest) = key.strip_prefix("vision.blocks.") {
            return apply_block_weight(
                &mut self.vision_blocks,
                &self.config,
                rest,
                key,
                shape,
                values,
                device,
            );
        }
        if let Some(rest) = key.strip_prefix("text.blocks.") {
            return apply_block_weight(
                &mut self.text_blocks,
                &self.config,
                rest,
                key,
                shape,
                values,
                device,
            );
        }

        Err(format!("unknown weight key '{key}'"))
    }

    pub fn forward_image(
        &self,
        image: Tensor<B, 4>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 2>, String> {
        let [batch, channels, height, width] = image.shape().dims();
        expect_nonzero_batch("image input", batch)?;
        if channels != self.config.channels {
            return Err(format!(
                "input channel mismatch: got {channels}, expected {}",
                self.config.channels
            ));
        }
        if height != self.config.image_size || width != self.config.image_size {
            return Err(format!(
                "input spatial mismatch: got [{height},{width}], expected [{},{}]",
                self.config.image_size, self.config.image_size
            ));
        }

        record_tensor(&mut hook, "vision.input", &image)?;
        let patches = patchify(image, self.config.patch_size);
        record_tensor(&mut hook, "vision.patches", &patches)?;

        let mut tokens = self.vision_patch_embed.forward_3d(patches);
        tokens = add_position_embedding(
            tokens,
            self.vision_pos_embed.clone(),
            self.config.image_token_count(),
            self.config.hidden_dim,
        )?;
        record_tensor(&mut hook, "vision.tokens.plus_pos", &tokens)?;

        for (idx, block) in self.vision_blocks.iter().enumerate() {
            tokens = block.forward("vision.blocks", idx, tokens, None, hook.as_deref_mut())?;
        }
        let tokens = self.vision_post_norm.forward_3d(tokens);
        record_tensor(&mut hook, "vision.post_norm", &tokens)?;
        let pooled = self.vision_head.forward(tokens, hook.as_deref_mut())?;
        record_tensor(&mut hook, "vision.embedding", &pooled)?;
        Ok(pooled)
    }

    pub fn forward_text(
        &self,
        text_embeddings: Tensor<B, 3>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 2>, String> {
        let [batch, seq_len, hidden] = text_embeddings.shape().dims();
        expect_nonzero_batch("text embedding input", batch)?;
        if hidden != self.config.hidden_dim {
            return Err(format!(
                "text hidden dim mismatch: got {hidden}, expected {}",
                self.config.hidden_dim
            ));
        }
        if seq_len == 0 {
            return Err("text sequence length must be > 0".to_string());
        }
        if seq_len > self.config.text_max_positions {
            return Err(format!(
                "text sequence length {seq_len} exceeds text_max_positions {}",
                self.config.text_max_positions
            ));
        }
        record_tensor(&mut hook, "text.input_embeddings", &text_embeddings)?;
        self.encode_text_embeddings(text_embeddings, None, hook)
    }

    pub fn forward_text_tokens(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2>>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 2>, String> {
        let [batch, seq_len] = input_ids.shape().dims();
        expect_nonzero_batch("text token input", batch)?;
        if seq_len == 0 {
            return Err("text sequence length must be > 0".to_string());
        }
        if seq_len > self.config.text_max_positions {
            return Err(format!(
                "text sequence length {seq_len} exceeds text_max_positions {}",
                self.config.text_max_positions
            ));
        }
        if let Some(mask) = &attention_mask {
            let [mask_batch, mask_len] = mask.shape().dims();
            if mask_batch != batch || mask_len != seq_len {
                return Err(format!(
                    "attention mask shape mismatch: expected [{batch}, {seq_len}], got [{mask_batch}, {mask_len}]"
                ));
            }
        }

        let embeddings = self.text_token_embed.forward(input_ids);
        record_tensor(&mut hook, "text.token_embeddings", &embeddings)?;
        let attention_mask = attention_mask.map(|mask| prepare_text_attention_mask(mask));
        self.encode_text_embeddings(embeddings, attention_mask, hook)
    }

    pub fn forward(
        &self,
        image: Tensor<B, 4>,
        hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 2>, String> {
        self.forward_image(image, hook)
    }

    /// L2-normalize a batch of image or text embeddings along its feature dimension.
    pub fn normalize_embeddings(embeddings: Tensor<B, 2>) -> Tensor<B, 2> {
        let squared = embeddings.clone().mul(embeddings.clone());
        let norm = squared.sum_dim(1).add_scalar(1.0e-12).sqrt();
        embeddings.div(norm)
    }

    /// Compute calibrated SigLIP image/text logits.
    ///
    /// Both towers are normalized before the cosine matrix is multiplied by the learned
    /// exponential logit scale and shifted by the learned logit bias. The output layout is
    /// `[image_batch, text_batch]`, matching Hugging Face's `logits_per_image`.
    pub fn similarity_logits(
        &self,
        image_embeddings: Tensor<B, 2>,
        text_embeddings: Tensor<B, 2>,
    ) -> Result<Tensor<B, 2>, String> {
        let [image_batch, image_dim] = image_embeddings.shape().dims();
        let [text_batch, text_dim] = text_embeddings.shape().dims();
        expect_nonzero_batch("image embedding input", image_batch)?;
        expect_nonzero_batch("text embedding input", text_batch)?;
        if image_dim != self.config.projection_dim || text_dim != self.config.projection_dim {
            return Err(format!(
                "similarity embedding dimension mismatch: image={image_dim}, text={text_dim}, expected {}",
                self.config.projection_dim
            ));
        }

        let image_embeddings = Self::normalize_embeddings(image_embeddings);
        let text_embeddings = Self::normalize_embeddings(text_embeddings);
        let cosine = image_embeddings.matmul(text_embeddings.swap_dims(0, 1));
        let scale = self
            .logit_scale
            .clone()
            .exp()
            .reshape([1, 1])
            .expand([image_batch as i64, text_batch as i64]);
        let bias = self
            .logit_bias
            .clone()
            .reshape([1, 1])
            .expand([image_batch as i64, text_batch as i64]);
        Ok(cosine.mul(scale).add(bias))
    }

    fn encode_text_embeddings(
        &self,
        mut embeddings: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 4>>,
        mut hook: Option<&mut HookRecorder>,
    ) -> Result<Tensor<B, 2>, String> {
        let [batch, seq_len, hidden] = embeddings.shape().dims();
        embeddings = add_position_embedding(
            embeddings,
            self.text_pos_embed.clone(),
            self.config.text_max_positions,
            hidden,
        )?;
        record_tensor(&mut hook, "text.tokens.plus_pos", &embeddings)?;

        let block_mask = attention_mask.clone();
        let mut tokens = embeddings;
        for (idx, block) in self.text_blocks.iter().enumerate() {
            tokens = block.forward(
                "text.blocks",
                idx,
                tokens,
                block_mask.clone(),
                hook.as_deref_mut(),
            )?;
        }

        let tokens = self.text_final_norm.forward_3d(tokens);
        record_tensor(&mut hook, "text.final_norm", &tokens)?;
        let pooled = tokens
            .slice([0..batch, (seq_len - 1)..seq_len, 0..hidden])
            .reshape([batch, hidden]);
        let embedding = self.text_projection.forward_2d(pooled);
        record_tensor(&mut hook, "text.embedding", &embedding)?;
        Ok(embedding)
    }

    fn apply_pool_head_weight(
        &mut self,
        rest: &str,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
        device: &B::Device,
    ) -> Result<(), String> {
        match rest {
            "probe" => self.vision_head.set_probe(
                key,
                shape,
                values,
                [1, 1, self.config.hidden_dim],
                device,
            ),
            "attn.q_proj.weight" => self.vision_head.attn.q_proj.set_weight(
                key,
                shape,
                values,
                [self.config.hidden_dim, self.config.hidden_dim],
                device,
            ),
            "attn.q_proj.bias" => self.vision_head.attn.q_proj.set_bias(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            ),
            "attn.k_proj.weight" => self.vision_head.attn.k_proj.set_weight(
                key,
                shape,
                values,
                [self.config.hidden_dim, self.config.hidden_dim],
                device,
            ),
            "attn.k_proj.bias" => self.vision_head.attn.k_proj.set_bias(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            ),
            "attn.v_proj.weight" => self.vision_head.attn.v_proj.set_weight(
                key,
                shape,
                values,
                [self.config.hidden_dim, self.config.hidden_dim],
                device,
            ),
            "attn.v_proj.bias" => self.vision_head.attn.v_proj.set_bias(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            ),
            "attn.out_proj.weight" => self.vision_head.attn.out_proj.set_weight(
                key,
                shape,
                values,
                [self.config.hidden_dim, self.config.hidden_dim],
                device,
            ),
            "attn.out_proj.bias" => self.vision_head.attn.out_proj.set_bias(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            ),
            "layernorm.gamma" => self.vision_head.layernorm.set_gamma(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            ),
            "layernorm.beta" => self.vision_head.layernorm.set_beta(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            ),
            "mlp.fc1.weight" => self.vision_head.mlp_fc1.set_weight(
                key,
                shape,
                values,
                [self.config.intermediate_dim, self.config.hidden_dim],
                device,
            ),
            "mlp.fc1.bias" => self.vision_head.mlp_fc1.set_bias(
                key,
                shape,
                values,
                [self.config.intermediate_dim],
                device,
            ),
            "mlp.fc2.weight" => self.vision_head.mlp_fc2.set_weight(
                key,
                shape,
                values,
                [self.config.hidden_dim, self.config.intermediate_dim],
                device,
            ),
            "mlp.fc2.bias" => self.vision_head.mlp_fc2.set_bias(
                key,
                shape,
                values,
                [self.config.hidden_dim],
                device,
            ),
            _ => Err(format!("unknown weight key '{key}'")),
        }
    }
}

fn append_block_specs(specs: &mut Vec<WeightSpec>, prefix: &str, config: &Siglip2Config) {
    for idx in 0..config.num_layers {
        let block_prefix = format!("{prefix}.{idx}");
        specs.push(WeightSpec {
            key: format!("{block_prefix}.norm1.gamma"),
            shape: vec![config.hidden_dim],
        });
        specs.push(WeightSpec {
            key: format!("{block_prefix}.norm1.beta"),
            shape: vec![config.hidden_dim],
        });
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            specs.push(WeightSpec {
                key: format!("{block_prefix}.attn.{proj}.weight"),
                shape: vec![config.hidden_dim, config.hidden_dim],
            });
            specs.push(WeightSpec {
                key: format!("{block_prefix}.attn.{proj}.bias"),
                shape: vec![config.hidden_dim],
            });
        }
        specs.push(WeightSpec {
            key: format!("{block_prefix}.norm2.gamma"),
            shape: vec![config.hidden_dim],
        });
        specs.push(WeightSpec {
            key: format!("{block_prefix}.norm2.beta"),
            shape: vec![config.hidden_dim],
        });
        specs.push(WeightSpec {
            key: format!("{block_prefix}.mlp.fc1.weight"),
            shape: vec![config.intermediate_dim, config.hidden_dim],
        });
        specs.push(WeightSpec {
            key: format!("{block_prefix}.mlp.fc1.bias"),
            shape: vec![config.intermediate_dim],
        });
        specs.push(WeightSpec {
            key: format!("{block_prefix}.mlp.fc2.weight"),
            shape: vec![config.hidden_dim, config.intermediate_dim],
        });
        specs.push(WeightSpec {
            key: format!("{block_prefix}.mlp.fc2.bias"),
            shape: vec![config.hidden_dim],
        });
    }
}

fn append_pool_head_specs(specs: &mut Vec<WeightSpec>, config: &Siglip2Config) {
    specs.push(WeightSpec {
        key: "vision.head.probe".to_string(),
        shape: vec![1, 1, config.hidden_dim],
    });
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        specs.push(WeightSpec {
            key: format!("vision.head.attn.{proj}.weight"),
            shape: vec![config.hidden_dim, config.hidden_dim],
        });
        specs.push(WeightSpec {
            key: format!("vision.head.attn.{proj}.bias"),
            shape: vec![config.hidden_dim],
        });
    }
    specs.push(WeightSpec {
        key: "vision.head.layernorm.gamma".to_string(),
        shape: vec![config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "vision.head.layernorm.beta".to_string(),
        shape: vec![config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "vision.head.mlp.fc1.weight".to_string(),
        shape: vec![config.intermediate_dim, config.hidden_dim],
    });
    specs.push(WeightSpec {
        key: "vision.head.mlp.fc1.bias".to_string(),
        shape: vec![config.intermediate_dim],
    });
    specs.push(WeightSpec {
        key: "vision.head.mlp.fc2.weight".to_string(),
        shape: vec![config.hidden_dim, config.intermediate_dim],
    });
    specs.push(WeightSpec {
        key: "vision.head.mlp.fc2.bias".to_string(),
        shape: vec![config.hidden_dim],
    });
}

fn apply_block_weight<B: Backend>(
    blocks: &mut [Siglip2Block<B>],
    config: &Siglip2Config,
    rest: &str,
    key: &str,
    shape: &[usize],
    values: Vec<f32>,
    device: &B::Device,
) -> Result<(), String> {
    let mut parts = rest.split('.');
    let block_idx = parts
        .next()
        .ok_or_else(|| format!("missing block index in key '{key}'"))?
        .parse::<usize>()
        .map_err(|err| format!("invalid block index in key '{key}': {err}"))?;
    let block = blocks
        .get_mut(block_idx)
        .ok_or_else(|| format!("block index out of range in key '{key}'"))?;

    let tail = parts.collect::<Vec<_>>();
    match tail.as_slice() {
        ["norm1", "gamma"] => {
            block
                .norm1
                .set_gamma(key, shape, values, [config.hidden_dim], device)
        }
        ["norm1", "beta"] => block
            .norm1
            .set_beta(key, shape, values, [config.hidden_dim], device),
        ["attn", "q_proj", "weight"] => block.attn.q_proj.set_weight(
            key,
            shape,
            values,
            [config.hidden_dim, config.hidden_dim],
            device,
        ),
        ["attn", "q_proj", "bias"] => {
            block
                .attn
                .q_proj
                .set_bias(key, shape, values, [config.hidden_dim], device)
        }
        ["attn", "k_proj", "weight"] => block.attn.k_proj.set_weight(
            key,
            shape,
            values,
            [config.hidden_dim, config.hidden_dim],
            device,
        ),
        ["attn", "k_proj", "bias"] => {
            block
                .attn
                .k_proj
                .set_bias(key, shape, values, [config.hidden_dim], device)
        }
        ["attn", "v_proj", "weight"] => block.attn.v_proj.set_weight(
            key,
            shape,
            values,
            [config.hidden_dim, config.hidden_dim],
            device,
        ),
        ["attn", "v_proj", "bias"] => {
            block
                .attn
                .v_proj
                .set_bias(key, shape, values, [config.hidden_dim], device)
        }
        ["attn", "out_proj", "weight"] => block.attn.out_proj.set_weight(
            key,
            shape,
            values,
            [config.hidden_dim, config.hidden_dim],
            device,
        ),
        ["attn", "out_proj", "bias"] => {
            block
                .attn
                .out_proj
                .set_bias(key, shape, values, [config.hidden_dim], device)
        }
        ["norm2", "gamma"] => {
            block
                .norm2
                .set_gamma(key, shape, values, [config.hidden_dim], device)
        }
        ["norm2", "beta"] => block
            .norm2
            .set_beta(key, shape, values, [config.hidden_dim], device),
        ["mlp", "fc1", "weight"] => block.mlp_fc1.set_weight(
            key,
            shape,
            values,
            [config.intermediate_dim, config.hidden_dim],
            device,
        ),
        ["mlp", "fc1", "bias"] => {
            block
                .mlp_fc1
                .set_bias(key, shape, values, [config.intermediate_dim], device)
        }
        ["mlp", "fc2", "weight"] => block.mlp_fc2.set_weight(
            key,
            shape,
            values,
            [config.hidden_dim, config.intermediate_dim],
            device,
        ),
        ["mlp", "fc2", "bias"] => {
            block
                .mlp_fc2
                .set_bias(key, shape, values, [config.hidden_dim], device)
        }
        _ => Err(format!("unknown weight key '{key}'")),
    }
}

fn record_tensor<B: Backend, const D: usize>(
    hook: &mut Option<&mut HookRecorder>,
    name: &str,
    tensor: &Tensor<B, D>,
) -> Result<(), String> {
    if let Some(hook) = hook.as_deref_mut() {
        hook.record_tensor(name, tensor)?;
    }
    Ok(())
}

fn add_position_embedding<B: Backend>(
    tokens: Tensor<B, 3>,
    pos_embed: Tensor<B, 2>,
    max_tokens: usize,
    hidden_dim: usize,
) -> Result<Tensor<B, 3>, String> {
    let [batch, token_count, hidden] = tokens.shape().dims();
    if hidden != hidden_dim {
        return Err(format!(
            "token hidden dim mismatch: got {hidden}, expected {hidden_dim}"
        ));
    }
    if token_count > max_tokens {
        return Err(format!(
            "token count exceeds position table: got {token_count}, max {max_tokens}"
        ));
    }
    let pos = pos_embed
        .slice([0..token_count, 0..hidden])
        .reshape([1, token_count, hidden])
        .expand([batch as i64, -1, -1]);
    Ok(tokens.add(pos))
}

fn prepare_text_attention_mask<B: Backend>(mask: Tensor<B, 2>) -> Tensor<B, 4> {
    let [batch, seq_len] = mask.shape().dims();
    mask.mul_scalar(-1.0)
        .add_scalar(1.0)
        .mul_scalar(-1.0e9)
        .reshape([batch, 1, 1, seq_len])
}

fn broadcast_attention_mask<B: Backend>(
    mask: Tensor<B, 4>,
    batch: usize,
    num_heads: usize,
    query_len: usize,
    key_len: usize,
) -> Result<Tensor<B, 4>, String> {
    let [mask_batch, mask_heads, mask_query, mask_key] = mask.shape().dims();
    if mask_batch != batch {
        return Err(format!(
            "attention mask batch mismatch: expected {batch}, got {mask_batch}"
        ));
    }
    if mask_key != key_len {
        return Err(format!(
            "attention mask key length mismatch: expected {key_len}, got {mask_key}"
        ));
    }
    if mask_heads != 1 && mask_heads != num_heads {
        return Err(format!(
            "attention mask head dimension mismatch: expected 1 or {num_heads}, got {mask_heads}"
        ));
    }
    if mask_query != 1 && mask_query != query_len {
        return Err(format!(
            "attention mask query dimension mismatch: expected 1 or {query_len}, got {mask_query}"
        ));
    }
    Ok(mask.expand([
        batch as i64,
        num_heads as i64,
        query_len as i64,
        key_len as i64,
    ]))
}

fn softmax_last_dim_4d<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let max = tensor.clone().max_dim(3);
    let exp = tensor.sub(max).exp();
    let denom = exp.clone().sum_dim(3);
    exp.div(denom)
}

fn gelu<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    // Hugging Face `gelu_pytorch_tanh`, which is the activation recorded in every
    // supported fixed-resolution SigLIP 2 checkpoint.
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    // Avoid WGSL `pow` for negative inputs; integer powers via multiplication are portable.
    let cubic = tensor
        .clone()
        .mul(tensor.clone())
        .mul(tensor.clone())
        .mul_scalar(0.044_715);
    let inner = tensor
        .clone()
        .add(cubic)
        .mul_scalar(SQRT_2_OVER_PI)
        // `tanh` is already exactly saturated in f32 at this range. The explicit clamp keeps
        // WebGPU implementations from evaluating their exp-based approximation at huge values;
        // SwiftShader and some Metal drivers can otherwise produce NaNs for valid SigLIP2 MLP
        // activations. This is bitwise-equivalent to unclamped f32 tanh-GELU for |x| >= 10.
        .clamp(-10.0, 10.0)
        .tanh()
        .add_scalar(1.0);
    tensor.mul(inner).mul_scalar(0.5)
}

fn patchify<B: Backend>(image: Tensor<B, 4>, patch_size: usize) -> Tensor<B, 3> {
    let [batch, channels, height, width] = image.shape().dims();
    assert_eq!(
        height % patch_size,
        0,
        "patchify height must divide patch size"
    );
    assert_eq!(
        width % patch_size,
        0,
        "patchify width must divide patch size"
    );
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    image
        .reshape([batch, channels, grid_h, patch_size, grid_w, patch_size])
        .permute([0, 2, 4, 1, 3, 5])
        .reshape([batch, grid_h * grid_w, channels * patch_size * patch_size])
}

fn expect_nonzero_batch(context: &str, batch: usize) -> Result<(), String> {
    if batch > 0 {
        return Ok(());
    }
    Err(format!("{context} batch size must be > 0"))
}

fn expect_shape(key: &str, actual: &[usize], expected: &[usize]) -> Result<(), String> {
    if actual == expected {
        return Ok(());
    }
    Err(format!(
        "shape mismatch for '{key}': expected {expected:?}, got {actual:?}"
    ))
}

fn expect_values_len(key: &str, values: &[f32], expected: usize) -> Result<(), String> {
    if values.len() == expected {
        return Ok(());
    }
    Err(format!(
        "value count mismatch for '{key}': expected {expected}, got {}",
        values.len()
    ))
}

fn expect_values_shape(key: &str, values: &[f32], shape: &[usize]) -> Result<(), String> {
    let expected = shape.iter().try_fold(1usize, |elements, dimension| {
        elements.checked_mul(*dimension)
    });
    let expected = expected
        .ok_or_else(|| format!("element count overflows usize for '{key}' with shape {shape:?}"))?;
    expect_values_len(key, values, expected)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::{
        backend::NdArray,
        tensor::{Int, Tensor, TensorData},
    };

    use crate::{config::Siglip2Config, hooks::HookRecorder};

    use super::{
        EmbeddingTable, Siglip2Model, Siglip2ModelBuilder, TEXT_TOKEN_EMBED_CHUNK_ROWS, gelu,
    };

    fn populate_builder(
        builder: &mut Siglip2ModelBuilder<NdArray>,
        config: &Siglip2Config,
        embedding_values: &[f32],
        use_embedding_chunks: bool,
        device: &burn::backend::ndarray::NdArrayDevice,
    ) -> Result<(), String> {
        for spec in Siglip2Model::<NdArray>::expected_weight_specs(config) {
            if spec.key == "text.token_embed.weight" {
                continue;
            }
            let values = vec![0.0; spec.shape.iter().product()];
            builder.apply_weight(&spec.key, &spec.shape, values, device)?;
        }

        if use_embedding_chunks {
            for (index, spec) in
                Siglip2Model::<NdArray>::expected_text_token_embedding_chunk_specs(config)
                    .into_iter()
                    .enumerate()
            {
                let start_row = index * TEXT_TOKEN_EMBED_CHUNK_ROWS;
                let end_row = start_row + spec.shape[0];
                let start = start_row * config.hidden_dim;
                let end = end_row * config.hidden_dim;
                builder.apply_weight(
                    &spec.key,
                    &spec.shape,
                    embedding_values[start..end].to_vec(),
                    device,
                )?;
            }
        } else {
            builder.apply_weight(
                "text.token_embed.weight",
                &[config.text_vocab_size, config.hidden_dim],
                embedding_values.to_vec(),
                device,
            )?;
        }
        Ok(())
    }

    #[test]
    fn model_builder_rejects_duplicate_unexpected_and_incomplete_weights() -> Result<(), String> {
        let config = Siglip2Config::tiny_for_tests();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut builder = Siglip2ModelBuilder::<NdArray>::new(config)?;

        builder.apply_weight("logit_scale", &[1], vec![1.0], &device)?;
        let duplicate = builder
            .apply_weight("logit_scale", &[1], vec![1.0], &device)
            .expect_err("duplicate tensor must fail");
        assert!(duplicate.contains("duplicate weight key"));

        let unexpected = builder
            .apply_weight("unknown.weight", &[1], vec![0.0], &device)
            .expect_err("unexpected tensor must fail");
        assert!(unexpected.contains("unknown weight key"));

        let incomplete = builder.finish().expect_err("incomplete model must fail");
        assert!(incomplete.contains("missing required weight tensors"));
        Ok(())
    }

    #[test]
    fn model_builder_full_and_chunked_embeddings_are_equivalent() -> Result<(), String> {
        let mut config = Siglip2Config::tiny_for_tests();
        config.text_vocab_size = TEXT_TOKEN_EMBED_CHUNK_ROWS + 2;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let embedding_values = (0..config.text_vocab_size * config.hidden_dim)
            .map(|index| (index % 997) as f32 / 997.0)
            .collect::<Vec<_>>();

        let mut full_builder = Siglip2ModelBuilder::<NdArray>::new(config.clone())?;
        populate_builder(
            &mut full_builder,
            &config,
            &embedding_values,
            false,
            &device,
        )?;
        let full = full_builder.finish()?;

        let mut chunked_builder = Siglip2ModelBuilder::<NdArray>::new(config.clone())?;
        populate_builder(
            &mut chunked_builder,
            &config,
            &embedding_values,
            true,
            &device,
        )?;
        let chunked = chunked_builder.finish()?;

        let ids = Tensor::<NdArray, 2, Int>::from_data(
            TensorData::new(
                vec![
                    0i64,
                    (TEXT_TOKEN_EMBED_CHUNK_ROWS - 1) as i64,
                    TEXT_TOKEN_EMBED_CHUNK_ROWS as i64,
                    (TEXT_TOKEN_EMBED_CHUNK_ROWS + 1) as i64,
                ],
                [1, 4],
            ),
            &device,
        );
        let full_values = full
            .text_token_embed
            .forward(ids.clone())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| format!("full embedding readback failed: {error:?}"))?;
        let chunked_values = chunked
            .text_token_embed
            .forward(ids)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| format!("chunked embedding readback failed: {error:?}"))?;
        assert_eq!(full_values, chunked_values);
        Ok(())
    }

    #[test]
    fn chunked_embedding_matches_rows_across_chunk_boundaries() -> Result<(), String> {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut table = EmbeddingTable::<NdArray>::zeros_with_chunk_rows(5, 2, 2, &device);
        let values = vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
        table.set_weight("embedding", &[5, 2], values.clone(), [5, 2], &device)?;
        let ids = Tensor::<NdArray, 2, Int>::from_data(
            TensorData::new(vec![0i64, 1, 2, 3, 4], [1, 5]),
            &device,
        );
        let actual = table
            .forward(ids)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("embedding readback failed: {err:?}"))?;
        assert_eq!(actual, values);
        Ok(())
    }

    #[test]
    fn chunked_embedding_can_be_loaded_without_a_full_table_allocation() -> Result<(), String> {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut table = EmbeddingTable::<NdArray>::zeros_with_chunk_rows(5, 2, 2, &device);
        table.set_weight_chunk("chunk.0", 0, &[2, 2], vec![0.0, 1.0, 10.0, 11.0], &device)?;
        table.set_weight_chunk("chunk.1", 1, &[2, 2], vec![20.0, 21.0, 30.0, 31.0], &device)?;
        table.set_weight_chunk("chunk.2", 2, &[1, 2], vec![40.0, 41.0], &device)?;
        let ids = Tensor::<NdArray, 2, Int>::from_data(
            TensorData::new(vec![0i64, 1, 2, 3, 4], [1, 5]),
            &device,
        );
        let actual = table
            .forward(ids)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("embedding readback failed: {err:?}"))?;
        assert_eq!(
            actual,
            vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]
        );
        Ok(())
    }

    #[test]
    fn chunked_embedding_rejects_malformed_value_lengths_without_panicking() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut table = EmbeddingTable::<NdArray>::zeros_with_chunk_rows(5, 2, 2, &device);
        let error = table
            .set_weight_chunk("chunk.0", 0, &[2, 2], vec![1.0, 2.0, 3.0], &device)
            .expect_err("short value buffer must fail");
        assert!(error.contains("value count mismatch"));
    }

    #[test]
    fn apply_weight_rejects_all_malformed_tensor_buffers_without_panicking() -> Result<(), String> {
        let cfg = Siglip2Config::tiny_for_tests();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let mut model = Siglip2Model::<NdArray>::zeros(cfg.clone(), &device)?;
        let cases = [
            (
                "vision.patch_embed.weight",
                vec![cfg.hidden_dim, cfg.patch_dim()],
            ),
            ("vision.patch_embed.bias", vec![cfg.hidden_dim]),
            (
                "vision.pos_embed",
                vec![cfg.image_token_count(), cfg.hidden_dim],
            ),
            ("vision.post_norm.gamma", vec![cfg.hidden_dim]),
            ("vision.post_norm.beta", vec![cfg.hidden_dim]),
            ("vision.head.probe", vec![1, 1, cfg.hidden_dim]),
            (
                "text.token_embed.weight",
                vec![cfg.text_vocab_size, cfg.hidden_dim],
            ),
            (
                "text.token_embed.weight.chunk.00000",
                vec![cfg.text_vocab_size, cfg.hidden_dim],
            ),
            (
                "text.pos_embed",
                vec![cfg.text_max_positions, cfg.hidden_dim],
            ),
            ("text.final_norm.gamma", vec![cfg.hidden_dim]),
            ("text.final_norm.beta", vec![cfg.hidden_dim]),
            (
                "text.projection.weight",
                vec![cfg.projection_dim, cfg.hidden_dim],
            ),
            ("text.projection.bias", vec![cfg.projection_dim]),
            ("logit_scale", vec![1]),
            ("logit_bias", vec![1]),
        ];

        for (key, shape) in cases {
            let expected_values = shape.iter().product::<usize>();
            let error = model
                .apply_weight(key, &shape, vec![0.0; expected_values - 1], &device)
                .expect_err("a short weight buffer must return an error");
            assert!(
                error.contains("value count mismatch"),
                "unexpected error for {key}: {error}"
            );
        }

        let error = model
            .apply_weight("logit_bias", &[1], vec![0.0, 1.0], &device)
            .expect_err("a long weight buffer must return an error");
        assert!(error.contains("value count mismatch"));
        Ok(())
    }

    #[test]
    fn gelu_matches_pytorch_tanh_reference() -> Result<(), String> {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let input = Tensor::<NdArray, 1>::from_data(
            TensorData::new(vec![-1.0f32, 0.0, 1.0, 2.0], [4]),
            &device,
        );
        let actual = gelu(input)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("gelu readback failed: {err:?}"))?;
        let expected = [-0.158_808, 0.0, 0.841_192, 1.954_598];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn gelu_extreme_inputs_remain_finite_and_saturate() -> Result<(), String> {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let input_values = [-1.0e6f32, -100.0, -20.0, -10.0, 10.0, 20.0, 100.0, 1.0e6];
        let input = Tensor::<NdArray, 1>::from_data(
            TensorData::new(input_values.to_vec(), [input_values.len()]),
            &device,
        );
        let actual = gelu(input)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("gelu readback failed: {err:?}"))?;

        assert!(actual.iter().all(|value| value.is_finite()));
        for value in &actual[..4] {
            assert!(
                value.abs() <= 1.0e-6,
                "negative GELU tail did not saturate: {value}"
            );
        }
        for (actual, expected) in actual[4..].iter().zip(&input_values[4..]) {
            assert_eq!(actual, expected, "positive GELU tail did not saturate");
        }
        Ok(())
    }

    #[test]
    fn similarity_logits_normalizes_both_towers() -> Result<(), String> {
        let cfg = Siglip2Config::tiny_for_tests();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = Siglip2Model::<NdArray>::zeros(cfg.clone(), &device)?;
        let mut image = vec![0.0f32; cfg.projection_dim * 2];
        image[0] = 3.0;
        image[1] = 4.0;
        image[cfg.projection_dim + 1] = 2.0;
        let mut text = vec![0.0f32; cfg.projection_dim * 2];
        text[0] = 1.0;
        text[cfg.projection_dim + 1] = 1.0;
        let logits = model.similarity_logits(
            Tensor::<NdArray, 2>::from_data(
                TensorData::new(image, [2, cfg.projection_dim]),
                &device,
            ),
            Tensor::<NdArray, 2>::from_data(
                TensorData::new(text, [2, cfg.projection_dim]),
                &device,
            ),
        )?;
        let actual = logits
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("logit readback failed: {err:?}"))?;
        let expected = [0.6f32, 0.8, 0.0, 1.0];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn expected_weight_specs_cover_dual_tower_keys() {
        let cfg = Siglip2Config::tiny_for_tests();
        let specs = Siglip2Model::<NdArray>::expected_weight_specs(&cfg);
        assert!(!specs.is_empty());
        assert!(
            specs
                .iter()
                .any(|spec| spec.key == "vision.patch_embed.weight")
        );
        assert!(
            specs
                .iter()
                .any(|spec| spec.key == "text.token_embed.weight")
        );
        assert!(specs.iter().any(|spec| spec.key == "vision.head.probe"));
    }

    #[test]
    fn zeros_model_forward_image_has_expected_shape() -> Result<(), String> {
        let cfg = Siglip2Config::tiny_for_tests();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = Siglip2Model::<NdArray>::zeros(cfg.clone(), &device)?;
        let input =
            Tensor::<NdArray, 4>::zeros([2, cfg.channels, cfg.image_size, cfg.image_size], &device);
        let output = model.forward(input, None)?;
        assert_eq!(output.shape().dims(), [2, cfg.hidden_dim]);
        Ok(())
    }

    #[test]
    fn zeros_model_forward_text_tokens_has_expected_shape() -> Result<(), String> {
        let cfg = Siglip2Config::tiny_for_tests();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = Siglip2Model::<NdArray>::zeros(cfg.clone(), &device)?;
        let input_ids = Tensor::<NdArray, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3, 0, 4, 5, 0, 0], [2, 4]),
            &device,
        );
        let attention_mask = Tensor::<NdArray, 2>::from_data(
            TensorData::new(vec![1.0f32, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], [2, 4]),
            &device,
        );
        let output = model.forward_text_tokens(input_ids, Some(attention_mask), None)?;
        assert_eq!(output.shape().dims(), [2, cfg.projection_dim]);
        Ok(())
    }

    #[test]
    fn public_tensor_paths_reject_zero_batches_before_backend_operations() -> Result<(), String> {
        let cfg = Siglip2Config::tiny_for_tests();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = Siglip2Model::<NdArray>::zeros(cfg.clone(), &device)?;

        let image =
            Tensor::<NdArray, 4>::zeros([0, cfg.channels, cfg.image_size, cfg.image_size], &device);
        let image_error = model
            .forward_image(image, None)
            .expect_err("a zero image batch must fail before patchification");
        assert_eq!(image_error, "image input batch size must be > 0");

        let text_embeddings = Tensor::<NdArray, 3>::zeros([0, 2, cfg.hidden_dim], &device);
        let text_error = model
            .forward_text(text_embeddings, None)
            .expect_err("a zero text embedding batch must fail before encoding");
        assert_eq!(text_error, "text embedding input batch size must be > 0");

        let input_ids = Tensor::<NdArray, 2, Int>::zeros([0, 2], &device);
        let token_error = model
            .forward_text_tokens(input_ids, None, None)
            .expect_err("a zero token batch must fail before embedding lookup");
        assert_eq!(token_error, "text token input batch size must be > 0");

        let empty_images = Tensor::<NdArray, 2>::zeros([0, cfg.projection_dim], &device);
        let one_text = Tensor::<NdArray, 2>::zeros([1, cfg.projection_dim], &device);
        let image_similarity_error = model
            .similarity_logits(empty_images, one_text)
            .expect_err("a zero image embedding batch must fail before normalization");
        assert_eq!(
            image_similarity_error,
            "image embedding input batch size must be > 0"
        );

        let one_image = Tensor::<NdArray, 2>::zeros([1, cfg.projection_dim], &device);
        let empty_texts = Tensor::<NdArray, 2>::zeros([0, cfg.projection_dim], &device);
        let text_similarity_error = model
            .similarity_logits(one_image, empty_texts)
            .expect_err("a zero text embedding batch must fail before normalization");
        assert_eq!(
            text_similarity_error,
            "text embedding input batch size must be > 0"
        );
        Ok(())
    }

    #[test]
    fn forward_records_hooks_when_enabled() -> Result<(), String> {
        let cfg = Siglip2Config::tiny_for_tests();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let model = Siglip2Model::<NdArray>::zeros(cfg.clone(), &device)?;
        let input =
            Tensor::<NdArray, 4>::zeros([1, cfg.channels, cfg.image_size, cfg.image_size], &device);
        let mut hook = HookRecorder::new();
        let _ = model.forward(input, Some(&mut hook))?;
        assert!(hook.len() > 3);
        Ok(())
    }
}
