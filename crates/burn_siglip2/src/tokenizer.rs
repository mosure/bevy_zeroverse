use std::path::{Path, PathBuf};

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use crate::config::Siglip2Config;

/// SigLIP 2 text towers are trained with a fixed context window of 64 tokens.
pub const SIGLIP2_TEXT_MAX_LENGTH: usize = 64;
/// Maximum number of raw text inputs accepted by one tokenizer call.
pub const SIGLIP2_MAX_TEXT_BATCH_SIZE: usize = 256;
/// Maximum UTF-8 byte length of one raw text input.
pub const SIGLIP2_MAX_TEXT_BYTES: usize = 64 * 1024;
/// Maximum combined UTF-8 byte length accepted by one tokenizer call.
pub const SIGLIP2_MAX_TEXT_BATCH_BYTES: usize = 1024 * 1024;
/// Maximum accepted serialized `tokenizer.json` size.
pub const SIGLIP2_MAX_TOKENIZER_JSON_BYTES: usize = 64 * 1024 * 1024;
/// Maximum accepted serialized `tokenizer_config.json` size.
pub const SIGLIP2_MAX_TOKENIZER_CONFIG_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Siglip2Tokenizer {
    inner: Tokenizer,
    max_length: usize,
    pad_id: u32,
    pad_token: String,
    eos_id: u32,
    eos_token: String,
    do_lower_case: bool,
    expected_vocab_size: usize,
    artifact_sha256: String,
}

#[derive(Debug, Clone)]
pub struct Siglip2TokenizedBatch {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<f32>,
    pub shape: [usize; 2],
}

impl Siglip2Tokenizer {
    pub fn from_file(path: &Path, config: &Siglip2Config) -> Result<Self, String> {
        validate_text_config(config)?;
        let tokenizer_json =
            read_file_bounded(path, SIGLIP2_MAX_TOKENIZER_JSON_BYTES, "tokenizer JSON")?;
        let tokenizer_config_path = adjacent_tokenizer_config_path(path);
        let mut tokenizer_config =
            read_optional_file_bounded(&tokenizer_config_path, SIGLIP2_MAX_TOKENIZER_CONFIG_BYTES)?;
        if tokenizer_config.is_none() {
            let fallback = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("tokenizer_config.json");
            if fallback != tokenizer_config_path {
                tokenizer_config =
                    read_optional_file_bounded(&fallback, SIGLIP2_MAX_TOKENIZER_CONFIG_BYTES)?;
            }
        }
        Self::from_bytes(&tokenizer_json, tokenizer_config.as_deref(), config)
            .map_err(|err| format!("failed to load tokenizer '{}': {err}", path.display()))
    }

    pub fn from_hf_dir(dir: &Path, config: &Siglip2Config) -> Result<Self, String> {
        validate_text_config(config)?;
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer_json = read_file_bounded(
            &tokenizer_path,
            SIGLIP2_MAX_TOKENIZER_JSON_BYTES,
            "tokenizer JSON",
        )?;
        let tokenizer_config_path = dir.join("tokenizer_config.json");
        let tokenizer_config =
            read_optional_file_bounded(&tokenizer_config_path, SIGLIP2_MAX_TOKENIZER_CONFIG_BYTES)?;
        Self::from_bytes(&tokenizer_json, tokenizer_config.as_deref(), config).map_err(|err| {
            format!(
                "failed to load tokenizer '{}': {err}",
                tokenizer_path.display()
            )
        })
    }

    /// Loads tokenizer artifacts from memory, which is the preferred entry point for WASM.
    pub fn from_bytes(
        tokenizer_json: &[u8],
        tokenizer_config_json: Option<&[u8]>,
        config: &Siglip2Config,
    ) -> Result<Self, String> {
        validate_text_config(config)?;
        validate_artifact_len(
            tokenizer_json.len(),
            SIGLIP2_MAX_TOKENIZER_JSON_BYTES,
            "tokenizer JSON",
        )?;
        if let Some(bytes) = tokenizer_config_json {
            validate_artifact_len(
                bytes.len(),
                SIGLIP2_MAX_TOKENIZER_CONFIG_BYTES,
                "tokenizer config JSON",
            )?;
        }
        let artifact_sha256 = tokenizer_artifact_sha256(tokenizer_json, tokenizer_config_json);
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|err| format!("failed to load tokenizer JSON: {err}"))?;
        let tokenizer_config = tokenizer_config_json
            .map(|bytes| {
                serde_json::from_slice(bytes)
                    .map_err(|err| format!("failed to parse tokenizer config JSON: {err}"))
            })
            .transpose()?;
        Self::from_inner(
            tokenizer,
            tokenizer_config.as_ref(),
            artifact_sha256,
            config,
        )
    }

    fn from_inner(
        mut tokenizer: Tokenizer,
        tokenizer_config: Option<&HfTokenizerConfig>,
        artifact_sha256: String,
        config: &Siglip2Config,
    ) -> Result<Self, String> {
        let effective_vocab_size = tokenizer.get_vocab_size(true);
        if effective_vocab_size != config.text_vocab_size {
            return Err(format!(
                "tokenizer effective vocabulary size {effective_vocab_size} does not match model config text_vocab_size {}",
                config.text_vocab_size
            ));
        }

        if let Some(side) = tokenizer_config.and_then(|config| config.padding_side.as_deref())
            && !side.eq_ignore_ascii_case("right")
        {
            return Err(format!(
                "SigLIP 2 requires right padding, but tokenizer_config.json declares '{side}'"
            ));
        }
        if tokenizer_config.and_then(|config| config.add_eos_token) == Some(false) {
            return Err(
                "SigLIP 2 requires tokenizer_config.json add_eos_token to be true".to_string(),
            );
        }

        // tokenizer.json is authoritative for the token string, while the artifact and model
        // config must agree on its numeric id before inference is allowed.
        let (pad_id, pad_token) = if let Some(padding) = tokenizer.get_padding() {
            (padding.pad_id, padding.pad_token.clone())
        } else if let Some(pad_token) = tokenizer_config
            .and_then(|config| config.pad_token.as_ref())
            .map(HfToken::content)
        {
            let pad_id = tokenizer.token_to_id(pad_token).ok_or_else(|| {
                format!(
                    "tokenizer_config.json pad token '{pad_token}' is missing from tokenizer vocabulary"
                )
            })?;
            (pad_id, pad_token.to_string())
        } else {
            return Err(
                "tokenizer has no padding metadata; expected tokenizer.json padding or tokenizer_config.json pad_token"
                    .to_string(),
            );
        };
        let vocabulary_pad_id = tokenizer.token_to_id(&pad_token).ok_or_else(|| {
            format!("tokenizer pad token '{pad_token}' is missing from tokenizer vocabulary")
        })?;
        if vocabulary_pad_id != pad_id {
            return Err(format!(
                "tokenizer padding is inconsistent: token '{pad_token}' has vocabulary id {vocabulary_pad_id}, but padding declares id {pad_id}"
            ));
        }
        let configured_pad_id = u32::try_from(config.text_pad_token_id).map_err(|_| {
            format!(
                "model config text_pad_token_id {} cannot be represented by the tokenizer's u32 token ids",
                config.text_pad_token_id
            )
        })?;
        if pad_id != configured_pad_id {
            return Err(format!(
                "tokenizer pad id {pad_id} does not match model config text_pad_token_id {configured_pad_id}"
            ));
        }

        let configured_eos = tokenizer_config
            .and_then(|config| config.eos_token.as_ref())
            .map(HfToken::content);
        let (eos_id, eos_token) = derive_appended_eos(&tokenizer, configured_eos)?;
        if eos_id as usize >= effective_vocab_size {
            return Err(format!(
                "tokenizer EOS id {eos_id} is outside model vocabulary [0, {effective_vocab_size})"
            ));
        }
        let do_lower_case = match tokenizer_config.and_then(|config| config.do_lower_case) {
            Some(false) => {
                return Err(
                    "SigLIP 2 requires tokenizer_config.json do_lower_case to be true".to_string(),
                );
            }
            Some(true) | None => true,
        };

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(SIGLIP2_TEXT_MAX_LENGTH),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token: pad_token.clone(),
        }));

        Ok(Self {
            inner: tokenizer,
            max_length: SIGLIP2_TEXT_MAX_LENGTH,
            pad_id,
            pad_token,
            eos_id,
            eos_token,
            do_lower_case,
            expected_vocab_size: effective_vocab_size,
            artifact_sha256,
        })
    }

    pub const fn max_length(&self) -> usize {
        self.max_length
    }

    pub const fn pad_token_id(&self) -> u32 {
        self.pad_id
    }

    pub fn pad_token(&self) -> &str {
        &self.pad_token
    }

    pub const fn eos_token_id(&self) -> u32 {
        self.eos_id
    }

    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    pub const fn do_lower_case(&self) -> bool {
        self.do_lower_case
    }

    /// SHA-256 over the exact tokenizer/config bytes parsed by this instance.
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub const fn vocab_size(&self) -> usize {
        self.expected_vocab_size
    }

    pub fn encode_batch<T: AsRef<str>>(
        &self,
        texts: &[T],
    ) -> Result<Siglip2TokenizedBatch, String> {
        validate_text_batch(texts)?;
        let mut tokenizer = self.inner.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|err| format!("failed to configure tokenizer truncation: {err}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(self.max_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: self.pad_id,
            pad_type_id: 0,
            pad_token: self.pad_token.clone(),
        }));

        let inputs = texts
            .iter()
            .map(|value| {
                let value = value.as_ref();
                if self.do_lower_case {
                    value.to_lowercase()
                } else {
                    value.to_string()
                }
            })
            .collect::<Vec<_>>();
        let encodings = tokenizer
            .encode_batch(inputs, true)
            .map_err(|err| format!("failed to tokenize batch: {err}"))?;

        let mut input_ids = Vec::with_capacity(texts.len() * self.max_length);
        let mut attention_mask = Vec::with_capacity(texts.len() * self.max_length);
        for (batch_index, encoding) in encodings.into_iter().enumerate() {
            if encoding.len() != self.max_length {
                return Err(format!(
                    "tokenizer produced {} tokens, expected fixed length {}",
                    encoding.len(),
                    self.max_length
                ));
            }
            if let Some((token_index, token_id)) = encoding
                .get_ids()
                .iter()
                .copied()
                .enumerate()
                .find(|(_, token_id)| *token_id as usize >= self.expected_vocab_size)
            {
                return Err(format!(
                    "tokenizer emitted id {token_id} outside model vocabulary [0, {}) at batch index {batch_index}, token index {token_index}",
                    self.expected_vocab_size
                ));
            }
            let last_active = encoding
                .get_attention_mask()
                .iter()
                .rposition(|value| *value != 0)
                .ok_or_else(|| "tokenizer produced an empty attention mask".to_string())?;
            if encoding.get_ids()[last_active] != self.eos_id {
                return Err(format!(
                    "tokenizer post-processor did not place EOS id {} at the end of the active sequence",
                    self.eos_id
                ));
            }
            if encoding.get_attention_mask()[last_active + 1..]
                .iter()
                .any(|value| *value != 0)
                || encoding.get_ids()[last_active + 1..]
                    .iter()
                    .any(|value| *value != self.pad_id)
            {
                return Err("tokenizer did not apply right padding consistently".to_string());
            }
            input_ids.extend(encoding.get_ids().iter().map(|value| *value as i64));
            attention_mask.extend(
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|value| *value as f32),
            );
        }
        Ok(Siglip2TokenizedBatch {
            input_ids,
            attention_mask,
            shape: [texts.len(), self.max_length],
        })
    }
}

fn validate_text_config(config: &Siglip2Config) -> Result<(), String> {
    if config.text_max_positions != SIGLIP2_TEXT_MAX_LENGTH {
        return Err(format!(
            "SigLIP 2 tokenizer requires text_max_positions {SIGLIP2_TEXT_MAX_LENGTH}, but model config declares {}",
            config.text_max_positions
        ));
    }
    if config.text_vocab_size == 0 {
        return Err("model config text_vocab_size must be greater than zero".to_string());
    }
    if config.text_pad_token_id >= config.text_vocab_size {
        return Err(format!(
            "model config text_pad_token_id {} must be less than text_vocab_size {}",
            config.text_pad_token_id, config.text_vocab_size
        ));
    }
    Ok(())
}

fn validate_text_batch<T: AsRef<str>>(texts: &[T]) -> Result<(), String> {
    if texts.is_empty() {
        return Err("tokenizer encode_batch requires at least one input".to_string());
    }
    if texts.len() > SIGLIP2_MAX_TEXT_BATCH_SIZE {
        return Err(format!(
            "tokenizer batch contains {} texts, exceeding limit {SIGLIP2_MAX_TEXT_BATCH_SIZE}",
            texts.len()
        ));
    }

    let mut total_bytes = 0usize;
    for (index, text) in texts.iter().enumerate() {
        let text_bytes = text.as_ref().len();
        if text_bytes > SIGLIP2_MAX_TEXT_BYTES {
            return Err(format!(
                "text at batch index {index} is {text_bytes} UTF-8 bytes, exceeding per-text limit {SIGLIP2_MAX_TEXT_BYTES}"
            ));
        }
        total_bytes = total_bytes.checked_add(text_bytes).ok_or_else(|| {
            "combined tokenizer input UTF-8 byte length overflowed usize".to_string()
        })?;
        if total_bytes > SIGLIP2_MAX_TEXT_BATCH_BYTES {
            return Err(format!(
                "tokenizer batch is {total_bytes} UTF-8 bytes, exceeding total limit {SIGLIP2_MAX_TEXT_BATCH_BYTES}"
            ));
        }
    }
    Ok(())
}

fn adjacent_tokenizer_config_path(tokenizer_path: &Path) -> PathBuf {
    let parent = tokenizer_path.parent().unwrap_or_else(|| Path::new("."));
    let Some(file_name) = tokenizer_path.file_name().and_then(|name| name.to_str()) else {
        return parent.join("tokenizer_config.json");
    };
    let Some(prefix) = file_name.strip_suffix(".tokenizer.json") else {
        return parent.join("tokenizer_config.json");
    };
    parent.join(format!("{prefix}.tokenizer_config.json"))
}

fn read_optional_file_bounded(path: &Path, max_len: usize) -> Result<Option<Vec<u8>>, String> {
    if !path.exists() {
        return Ok(None);
    }
    read_file_bounded(path, max_len, "tokenizer config JSON").map(Some)
}

fn read_file_bounded(path: &Path, max_len: usize, label: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|err| format!("failed to open {label} '{}': {err}", path.display()))?;
    let declared_len = file
        .metadata()
        .map_err(|err| format!("failed to inspect {label} '{}': {err}", path.display()))?
        .len();
    if declared_len > max_len as u64 {
        return Err(format!(
            "{label} '{}' is {declared_len} bytes; maximum is {max_len}",
            path.display()
        ));
    }

    let mut bytes = Vec::with_capacity(declared_len as usize);
    file.take(max_len as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read {label} '{}': {err}", path.display()))?;
    validate_artifact_len(bytes.len(), max_len, label)?;
    Ok(bytes)
}

fn validate_artifact_len(len: usize, max_len: usize, label: &str) -> Result<(), String> {
    if len > max_len {
        return Err(format!(
            "{label} is {len} bytes, exceeding the {max_len}-byte limit"
        ));
    }
    Ok(())
}

fn tokenizer_artifact_sha256(
    tokenizer_json: &[u8],
    tokenizer_config_json: Option<&[u8]>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"burn_siglip2.tokenizer-artifacts.v1\0");
    hasher.update((tokenizer_json.len() as u64).to_le_bytes());
    hasher.update(tokenizer_json);
    match tokenizer_config_json {
        Some(bytes) => {
            hasher.update([1]);
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
        None => hasher.update([0]),
    }
    format!("{:x}", hasher.finalize())
}

fn derive_appended_eos(
    tokenizer: &Tokenizer,
    configured_eos: Option<&str>,
) -> Result<(u32, String), String> {
    let mut probe = tokenizer.clone();
    probe.with_padding(None);
    probe
        .with_truncation(None)
        .map_err(|err| format!("failed to disable tokenizer truncation for validation: {err}"))?;

    let plain = probe
        .encode("siglip2 tokenizer contract probe", false)
        .map_err(|err| format!("failed to validate tokenizer without special tokens: {err}"))?;
    let processed = probe
        .encode("siglip2 tokenizer contract probe", true)
        .map_err(|err| format!("failed to validate tokenizer post-processor: {err}"))?;
    if processed.len() != plain.len() + 1
        || processed.get_ids().get(..plain.len()) != Some(plain.get_ids())
        || processed
            .get_special_tokens_mask()
            .get(..plain.len())
            .is_none_or(|mask| mask.iter().any(|value| *value != 0))
    {
        return Err(
            "tokenizer post-processor must preserve the plain token sequence and append exactly one EOS token"
                .to_string(),
        );
    }

    let appended_id = *processed
        .get_ids()
        .last()
        .ok_or_else(|| "tokenizer post-processor must append an EOS token".to_string())?;
    if processed.get_special_tokens_mask().last().copied() != Some(1) {
        return Err("tokenizer post-processor's final token is not marked special".to_string());
    }

    if let Some(eos_token) = configured_eos {
        let eos_id = tokenizer.token_to_id(eos_token).ok_or_else(|| {
            format!(
                "tokenizer_config.json EOS token '{eos_token}' is missing from tokenizer vocabulary"
            )
        })?;
        if appended_id != eos_id {
            return Err(format!(
                "tokenizer post-processor appends id {appended_id}, but tokenizer_config.json declares EOS token '{eos_token}' with id {eos_id}"
            ));
        }
        return Ok((eos_id, eos_token.to_string()));
    }

    let eos_token = tokenizer
        .id_to_token(appended_id)
        .ok_or_else(|| format!("tokenizer post-processor appends unknown EOS id {appended_id}"))?;
    Ok((appended_id, eos_token))
}

impl Siglip2TokenizedBatch {
    /// Validates the public flat buffers before constructing Burn tensors.
    pub fn validate(&self) -> Result<(), String> {
        if self.shape.contains(&0) {
            return Err(format!(
                "tokenized batch shape dimensions must be non-zero, got {:?}",
                self.shape
            ));
        }
        let expected = self.shape[0].checked_mul(self.shape[1]).ok_or_else(|| {
            format!(
                "tokenized batch element count overflows usize for shape {:?}",
                self.shape
            )
        })?;
        if self.input_ids.len() != expected {
            return Err(format!(
                "tokenized input_ids value count mismatch for shape {:?}: expected {expected}, got {}",
                self.shape,
                self.input_ids.len()
            ));
        }
        if self.attention_mask.len() != expected {
            return Err(format!(
                "tokenized attention_mask value count mismatch for shape {:?}: expected {expected}, got {}",
                self.shape,
                self.attention_mask.len()
            ));
        }
        Ok(())
    }

    pub fn input_ids_tensor<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<Tensor<B, 2, Int>, String> {
        self.validate()?;
        Ok(Tensor::<B, 2, Int>::from_data(
            TensorData::new(self.input_ids.clone(), self.shape),
            device,
        ))
    }

    pub fn attention_mask_tensor<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, String> {
        self.validate()?;
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(self.attention_mask.clone(), self.shape),
            device,
        ))
    }
}

#[derive(Debug, Deserialize)]
struct HfTokenizerConfig {
    #[serde(default)]
    pad_token: Option<HfToken>,
    #[serde(default)]
    eos_token: Option<HfToken>,
    #[serde(default)]
    padding_side: Option<String>,
    #[serde(default)]
    add_eos_token: Option<bool>,
    #[serde(default)]
    do_lower_case: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HfToken {
    Content(String),
    Detailed { content: String },
}

impl HfToken {
    fn content(&self) -> &str {
        match self {
            Self::Content(content) => content,
            Self::Detailed { content } => content,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tokenizers::{
        AddedToken, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer,
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing,
    };

    use super::{
        HfToken, HfTokenizerConfig, SIGLIP2_MAX_TEXT_BATCH_BYTES, SIGLIP2_MAX_TEXT_BATCH_SIZE,
        SIGLIP2_MAX_TEXT_BYTES, SIGLIP2_MAX_TOKENIZER_JSON_BYTES, SIGLIP2_TEXT_MAX_LENGTH,
        Siglip2TokenizedBatch, Siglip2Tokenizer,
    };
    use crate::config::Siglip2Config;

    #[test]
    fn parses_hf_pad_token_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let as_string: HfTokenizerConfig = serde_json::from_str(r#"{"pad_token":"<pad>"}"#)?;
        assert!(matches!(
            as_string.pad_token,
            Some(HfToken::Content(ref value)) if value == "<pad>"
        ));

        let as_object: HfTokenizerConfig =
            serde_json::from_str(r#"{"pad_token":{"content":"<pad>"}}"#)?;
        assert!(matches!(
            as_object.pad_token,
            Some(HfToken::Detailed { ref content }) if content == "<pad>"
        ));
        Ok(())
    }

    #[test]
    fn appends_eos_then_right_pads_with_json_id_and_preserves_mask()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        write_word_level_fixture(dir.path())?;
        let tokenizer_path = dir.path().join("siglip2-test.tokenizer.json");
        fs::rename(dir.path().join("tokenizer.json"), &tokenizer_path)?;
        fs::rename(
            dir.path().join("tokenizer_config.json"),
            dir.path().join("siglip2-test.tokenizer_config.json"),
        )?;

        let model_config = tokenizer_fixture_config();
        let tokenizer = Siglip2Tokenizer::from_file(&tokenizer_path, &model_config)?;

        assert_eq!(tokenizer.max_length(), SIGLIP2_TEXT_MAX_LENGTH);
        assert_eq!(tokenizer.vocab_size(), 5);
        assert_eq!(tokenizer.pad_token_id(), 0);
        assert_eq!(tokenizer.pad_token(), "<pad>");
        assert_eq!(tokenizer.eos_token_id(), 1);
        assert_eq!(tokenizer.eos_token(), "<eos>");
        assert!(tokenizer.do_lower_case());

        let batch = tokenizer.encode_batch(&["HELLO world", "hello"])?;
        assert_eq!(batch.shape, [2, SIGLIP2_TEXT_MAX_LENGTH]);
        assert_eq!(&batch.input_ids[..4], &[3, 4, 1, 0]);
        assert_eq!(&batch.attention_mask[..4], &[1.0, 1.0, 1.0, 0.0]);
        assert!(
            batch.input_ids[3..SIGLIP2_TEXT_MAX_LENGTH]
                .iter()
                .all(|id| *id == 0)
        );
        assert!(
            batch.attention_mask[3..SIGLIP2_TEXT_MAX_LENGTH]
                .iter()
                .all(|value| *value == 0.0)
        );

        let second = SIGLIP2_TEXT_MAX_LENGTH;
        assert_eq!(&batch.input_ids[second..second + 3], &[3, 1, 0]);
        assert_eq!(&batch.attention_mask[second..second + 3], &[1.0, 1.0, 0.0]);
        assert!(batch.input_ids[second + 2..].iter().all(|id| *id == 0));
        assert!(
            batch.attention_mask[second + 2..]
                .iter()
                .all(|value| *value == 0.0)
        );
        Ok(())
    }

    #[test]
    fn rejects_model_and_tokenizer_contract_mismatches()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        write_word_level_fixture(dir.path())?;

        let mut wrong_positions = tokenizer_fixture_config();
        wrong_positions.text_max_positions = SIGLIP2_TEXT_MAX_LENGTH - 1;
        let error = Siglip2Tokenizer::from_hf_dir(dir.path(), &wrong_positions).unwrap_err();
        assert!(error.contains("text_max_positions 64"), "{error}");

        let mut wrong_vocab = tokenizer_fixture_config();
        wrong_vocab.text_vocab_size += 1;
        let error = Siglip2Tokenizer::from_hf_dir(dir.path(), &wrong_vocab).unwrap_err();
        assert!(error.contains("effective vocabulary size 5"), "{error}");
        assert!(error.contains("text_vocab_size 6"), "{error}");

        let mut wrong_pad = tokenizer_fixture_config();
        wrong_pad.text_pad_token_id = 1;
        let error = Siglip2Tokenizer::from_hf_dir(dir.path(), &wrong_pad).unwrap_err();
        assert!(error.contains("tokenizer pad id 0"), "{error}");
        assert!(error.contains("text_pad_token_id 1"), "{error}");
        Ok(())
    }

    #[test]
    fn rejects_post_processor_that_adds_bos_before_eos()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        write_word_level_fixture(dir.path())?;
        let tokenizer_path = dir.path().join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)?;
        tokenizer.add_special_tokens(&[AddedToken::from("<bos>", true)]);
        let bos_id = tokenizer
            .token_to_id("<bos>")
            .ok_or("missing newly added BOS token")?;
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<bos> $A <eos>")?
                .try_pair("<bos> $A <eos> $B:1 <eos>:1")?
                .special_tokens(vec![("<bos>", bos_id), ("<eos>", 1)])
                .build()?,
        ));
        tokenizer.save(&tokenizer_path, true)?;

        let mut config = tokenizer_fixture_config();
        config.text_vocab_size += 1;
        let error = Siglip2Tokenizer::from_hf_dir(dir.path(), &config)
            .expect_err("BOS+EOS post-processing must not be accepted");
        assert!(error.contains("append exactly one EOS"), "{error}");
        Ok(())
    }

    #[test]
    fn rejects_oversized_raw_text_before_tokenization()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        write_word_level_fixture(dir.path())?;
        let tokenizer = Siglip2Tokenizer::from_hf_dir(dir.path(), &tokenizer_fixture_config())?;

        let too_many = vec!["hello"; SIGLIP2_MAX_TEXT_BATCH_SIZE + 1];
        let error = tokenizer.encode_batch(&too_many).unwrap_err();
        assert!(error.contains("exceeding limit 256"), "{error}");

        // The limit is deliberately measured in encoded UTF-8 bytes, not Unicode scalar count.
        let too_long = "é".repeat(SIGLIP2_MAX_TEXT_BYTES / 2 + 1);
        let error = tokenizer.encode_batch(&[too_long]).unwrap_err();
        assert!(error.contains("UTF-8 bytes"), "{error}");
        assert!(error.contains("per-text limit 65536"), "{error}");

        let total_too_large = vec![
            "a".repeat(SIGLIP2_MAX_TEXT_BYTES);
            SIGLIP2_MAX_TEXT_BATCH_BYTES / SIGLIP2_MAX_TEXT_BYTES + 1
        ];
        let error = tokenizer.encode_batch(&total_too_large).unwrap_err();
        assert!(error.contains("exceeding total limit 1048576"), "{error}");
        Ok(())
    }

    #[test]
    fn rejects_token_ids_outside_the_model_vocabulary()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        write_word_level_fixture(dir.path())?;
        let mut tokenizer = Siglip2Tokenizer::from_hf_dir(dir.path(), &tokenizer_fixture_config())?;

        // Simulate a compromised tokenizer emitting an id beyond the model embedding table.
        tokenizer.expected_vocab_size = 4;
        let error = tokenizer.encode_batch(&["world"]).unwrap_err();
        assert!(error.contains("emitted id 4"), "{error}");
        assert!(error.contains("outside model vocabulary [0, 4)"), "{error}");
        Ok(())
    }

    #[test]
    fn public_tokenized_batch_rejects_malformed_flat_buffers_without_panicking() {
        let malformed_ids = Siglip2TokenizedBatch {
            input_ids: vec![1],
            attention_mask: vec![1.0, 0.0],
            shape: [1, 2],
        };
        let error = malformed_ids
            .validate()
            .expect_err("short input_ids must fail");
        assert!(error.contains("input_ids value count mismatch"), "{error}");

        let malformed_mask = Siglip2TokenizedBatch {
            input_ids: vec![1, 0],
            attention_mask: vec![1.0],
            shape: [1, 2],
        };
        let error = malformed_mask
            .validate()
            .expect_err("short attention mask must fail");
        assert!(
            error.contains("attention_mask value count mismatch"),
            "{error}"
        );

        let overflow = Siglip2TokenizedBatch {
            input_ids: Vec::new(),
            attention_mask: Vec::new(),
            shape: [usize::MAX, 2],
        };
        let error = overflow
            .validate()
            .expect_err("overflowing tokenized shape must fail");
        assert!(error.contains("overflows usize"), "{error}");
    }

    #[test]
    fn tokenizer_fingerprint_covers_exact_tokenizer_and_config_bytes()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        write_word_level_fixture(dir.path())?;
        let config = tokenizer_fixture_config();
        let first = Siglip2Tokenizer::from_hf_dir(dir.path(), &config)?;
        assert_eq!(first.artifact_sha256().len(), 64);

        let config_path = dir.path().join("tokenizer_config.json");
        let mut config_json = fs::read(&config_path)?;
        config_json.push(b'\n');
        fs::write(&config_path, config_json)?;
        let second = Siglip2Tokenizer::from_hf_dir(dir.path(), &config)?;
        assert_ne!(first.artifact_sha256(), second.artifact_sha256());

        let tokenizer_path = dir.path().join("tokenizer.json");
        let mut tokenizer_json = fs::read(&tokenizer_path)?;
        tokenizer_json.push(b'\n');
        fs::write(&tokenizer_path, tokenizer_json)?;
        let third = Siglip2Tokenizer::from_hf_dir(dir.path(), &config)?;
        assert_ne!(second.artifact_sha256(), third.artifact_sha256());
        Ok(())
    }

    #[test]
    fn tokenizer_enforces_canonical_lowercasing_with_or_without_sidecar()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        write_word_level_fixture(dir.path())?;
        let config = tokenizer_fixture_config();

        fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{
                "add_eos_token": true,
                "do_lower_case": false,
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "padding_side": "right"
            }"#,
        )?;
        let error = Siglip2Tokenizer::from_hf_dir(dir.path(), &config)
            .expect_err("explicitly disabled lowercasing must fail");
        assert!(error.contains("do_lower_case"), "{error}");

        fs::remove_file(dir.path().join("tokenizer_config.json"))?;
        let tokenizer = Siglip2Tokenizer::from_hf_dir(dir.path(), &config)?;
        assert!(tokenizer.do_lower_case());
        let batch = tokenizer.encode_batch(&["HELLO"])?;
        assert_eq!(batch.input_ids[0], 3);
        Ok(())
    }

    #[test]
    fn tokenizer_file_size_is_rejected_before_parsing_or_allocation()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tokenizer.json");
        fs::File::create(&path)?.set_len(SIGLIP2_MAX_TOKENIZER_JSON_BYTES as u64 + 1)?;
        let error = Siglip2Tokenizer::from_file(&path, &tokenizer_fixture_config())
            .expect_err("oversized tokenizer must fail");
        assert!(error.contains("maximum"), "{error}");
        Ok(())
    }

    fn tokenizer_fixture_config() -> Siglip2Config {
        let mut config = Siglip2Config::tiny_for_tests();
        config.text_max_positions = SIGLIP2_TEXT_MAX_LENGTH;
        config.text_vocab_size = 5;
        config.text_pad_token_id = 0;
        config
    }

    fn write_word_level_fixture(
        dir: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("<pad>".to_string(), 0),
                    ("<eos>".to_string(), 1),
                    ("<unk>".to_string(), 2),
                    ("hello".to_string(), 3),
                    ("world".to_string(), 4),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<unk>".to_string())
            .build()?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<pad>", true),
            AddedToken::from("<eos>", true),
        ]);
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("$A <eos>")?
                .try_pair("$A <eos> $B:1 <eos>:1")?
                .special_tokens(vec![("<eos>", 1)])
                .build()?,
        ));
        // Deliberately serialize a different fixed length. Loading must retain the JSON token/id
        // metadata while enforcing SigLIP 2's production length of 64.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "<pad>".to_string(),
        }));
        tokenizer.save(dir.join("tokenizer.json"), true)?;
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{
                "add_eos_token": true,
                "do_lower_case": true,
                "eos_token": {"content": "<eos>", "special": true},
                "pad_token": "<pad>",
                "padding_side": "right"
            }"#,
        )?;
        Ok(())
    }
}
