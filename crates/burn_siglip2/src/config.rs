use serde::{Deserialize, Serialize};

/// Public CDN root containing the versioned SigLIP2 bundle index and model directories.
pub const SIGLIP2_DEFAULT_CDN_ROOT_URL: &str = "https://aberration.technology/model/siglip2";

/// The three smallest fixed-resolution SigLIP 2 architecture scales released by Google.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Siglip2ModelVariant {
    BasePatch16_224,
    LargePatch16_256,
    So400mPatch14_224,
}

impl Siglip2ModelVariant {
    pub const ALL: [Self; 3] = [
        Self::BasePatch16_224,
        Self::LargePatch16_256,
        Self::So400mPatch14_224,
    ];

    pub const fn hf_model_id(self) -> &'static str {
        match self {
            Self::BasePatch16_224 => "google/siglip2-base-patch16-224",
            Self::LargePatch16_256 => "google/siglip2-large-patch16-256",
            Self::So400mPatch14_224 => "google/siglip2-so400m-patch14-224",
        }
    }

    /// Canonical directory name used by the public CDN and command-line model selection.
    pub const fn model_size(self) -> &'static str {
        match self {
            Self::BasePatch16_224 => "base-patch16-224",
            Self::LargePatch16_256 => "large-patch16-256",
            Self::So400mPatch14_224 => "so400m-patch14-224",
        }
    }

    /// Resolve a supported short or canonical model-size name.
    pub fn from_model_size(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "base" | "base-patch16-224" => Some(Self::BasePatch16_224),
            "large" | "large-patch16-256" => Some(Self::LargePatch16_256),
            "so400m" | "so400m-patch14-224" => Some(Self::So400mPatch14_224),
            _ => None,
        }
    }

    /// Complete hash-bearing browser bundle manifest URL for this model size.
    pub fn cdn_bundle_manifest_url(self) -> String {
        format!(
            "{}/{}/bundle.manifest.json",
            SIGLIP2_DEFAULT_CDN_ROOT_URL,
            self.model_size()
        )
    }

    pub const fn model_stem(self) -> &'static str {
        match self {
            Self::BasePatch16_224 => "siglip2-base-patch16-224",
            Self::LargePatch16_256 => "siglip2-large-patch16-256",
            Self::So400mPatch14_224 => "siglip2-so400m-patch14-224",
        }
    }
}

pub const SIGLIP2_MIN_IMAGE_SIZE: usize = 224;
pub const SIGLIP2_MIN_PATCH_SIZE: usize = 14;
pub const SIGLIP2_MIN_HIDDEN_DIM: usize = 512;
pub const SIGLIP2_MIN_INTERMEDIATE_DIM: usize = 2048;
pub const SIGLIP2_MIN_PROJECTION_DIM: usize = 512;
pub const SIGLIP2_MIN_NUM_HEADS: usize = 8;
pub const SIGLIP2_MIN_NUM_LAYERS: usize = 12;
pub const SIGLIP2_MIN_TEXT_MAX_POSITIONS: usize = 64;
pub const SIGLIP2_MIN_TEXT_VOCAB_SIZE: usize = 32_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Siglip2Config {
    pub image_size: usize,
    pub patch_size: usize,
    pub channels: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub projection_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_norm_eps: f32,
    #[serde(default = "default_text_max_positions")]
    pub text_max_positions: usize,
    #[serde(default = "default_text_vocab_size")]
    pub text_vocab_size: usize,
    #[serde(default = "default_text_pad_token_id")]
    pub text_pad_token_id: usize,
}

impl Default for Siglip2Config {
    fn default() -> Self {
        Self::for_variant(Siglip2ModelVariant::BasePatch16_224)
    }
}

impl Siglip2Config {
    pub const fn for_variant(variant: Siglip2ModelVariant) -> Self {
        match variant {
            Siglip2ModelVariant::BasePatch16_224 => Self {
                image_size: 224,
                patch_size: 16,
                channels: 3,
                hidden_dim: 768,
                intermediate_dim: 3072,
                projection_dim: 768,
                num_heads: 12,
                num_layers: 12,
                layer_norm_eps: 1e-6,
                text_max_positions: 64,
                text_vocab_size: 256_000,
                text_pad_token_id: 0,
            },
            Siglip2ModelVariant::LargePatch16_256 => Self {
                image_size: 256,
                patch_size: 16,
                channels: 3,
                hidden_dim: 1024,
                intermediate_dim: 4096,
                projection_dim: 1024,
                num_heads: 16,
                num_layers: 24,
                layer_norm_eps: 1e-6,
                text_max_positions: 64,
                text_vocab_size: 256_000,
                text_pad_token_id: 0,
            },
            Siglip2ModelVariant::So400mPatch14_224 => Self {
                image_size: 224,
                patch_size: 14,
                channels: 3,
                hidden_dim: 1152,
                intermediate_dim: 4304,
                projection_dim: 1152,
                num_heads: 16,
                num_layers: 27,
                layer_norm_eps: 1e-6,
                text_max_positions: 64,
                text_vocab_size: 256_000,
                text_pad_token_id: 0,
            },
        }
    }

    pub fn supported_variant(&self) -> Option<Siglip2ModelVariant> {
        Siglip2ModelVariant::ALL
            .into_iter()
            .find(|variant| self == &Self::for_variant(*variant))
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            image_size: 8,
            patch_size: 4,
            channels: 3,
            hidden_dim: 8,
            intermediate_dim: 16,
            projection_dim: 8,
            num_heads: 2,
            num_layers: 2,
            layer_norm_eps: 1e-5,
            text_max_positions: 4,
            text_vocab_size: 32,
            text_pad_token_id: 0,
        }
    }

    pub fn patch_dim(&self) -> usize {
        self.channels
            .saturating_mul(self.patch_size)
            .saturating_mul(self.patch_size)
    }

    pub fn image_token_count(&self) -> usize {
        let patch_grid = self.image_size / self.patch_size.max(1);
        patch_grid.saturating_mul(patch_grid)
    }

    pub fn token_count(&self) -> usize {
        self.image_token_count()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads.max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.image_size == 0 {
            return Err("image_size must be > 0".to_string());
        }
        if self.patch_size == 0 {
            return Err("patch_size must be > 0".to_string());
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(format!(
                "image_size ({}) must be divisible by patch_size ({})",
                self.image_size, self.patch_size
            ));
        }
        if self.channels == 0 {
            return Err("channels must be > 0".to_string());
        }
        if self.hidden_dim == 0
            || self.intermediate_dim == 0
            || self.projection_dim == 0
            || self.num_heads == 0
            || self.num_layers == 0
        {
            return Err(
                "hidden_dim/intermediate_dim/projection_dim/num_heads/num_layers must all be > 0"
                    .to_string(),
            );
        }
        if self.text_max_positions == 0 {
            return Err("text_max_positions must be > 0".to_string());
        }
        if self.text_vocab_size == 0 {
            return Err("text_vocab_size must be > 0".to_string());
        }
        if self.text_pad_token_id >= self.text_vocab_size {
            return Err(format!(
                "text_pad_token_id ({}) must be < text_vocab_size ({})",
                self.text_pad_token_id, self.text_vocab_size
            ));
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(format!(
                "hidden_dim ({}) must be divisible by num_heads ({})",
                self.hidden_dim, self.num_heads
            ));
        }
        if !(self.layer_norm_eps.is_finite() && self.layer_norm_eps > 0.0) {
            return Err(format!(
                "layer_norm_eps must be finite and > 0, got {}",
                self.layer_norm_eps
            ));
        }
        Ok(())
    }

    pub fn validate_production_profile(&self) -> Result<(), String> {
        self.validate()?;

        if self.supported_variant().is_none() {
            return Err(
                "config does not match an exact supported production SigLIP2 profile; supported profiles are base-patch16-224, large-patch16-256, and so400m-patch14-224"
                    .to_string(),
            );
        }
        Ok(())
    }
}

const fn default_text_max_positions() -> usize {
    64
}

const fn default_text_vocab_size() -> usize {
    256_000
}

const fn default_text_pad_token_id() -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::{SIGLIP2_DEFAULT_CDN_ROOT_URL, Siglip2Config, Siglip2ModelVariant};

    #[test]
    fn tiny_config_is_valid() {
        let cfg = Siglip2Config::tiny_for_tests();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.patch_dim(), 48);
        assert_eq!(cfg.image_token_count(), 4);
    }

    #[test]
    fn default_config_meets_production_profile() {
        let cfg = Siglip2Config::default();
        assert!(cfg.validate_production_profile().is_ok());
        assert_eq!(
            cfg.supported_variant(),
            Some(Siglip2ModelVariant::BasePatch16_224)
        );
    }

    #[test]
    fn all_supported_variants_are_valid_and_round_trip() {
        for variant in Siglip2ModelVariant::ALL {
            let cfg = Siglip2Config::for_variant(variant);
            assert!(cfg.validate_production_profile().is_ok());
            assert_eq!(cfg.supported_variant(), Some(variant));
            assert!(variant.hf_model_id().starts_with("google/siglip2-"));
            assert_eq!(
                Siglip2ModelVariant::from_model_size(variant.model_size()),
                Some(variant)
            );
            assert_eq!(
                variant.cdn_bundle_manifest_url(),
                format!(
                    "{}/{}/bundle.manifest.json",
                    SIGLIP2_DEFAULT_CDN_ROOT_URL,
                    variant.model_size()
                )
            );
        }
    }

    #[test]
    fn model_size_aliases_are_bounded_to_supported_variants() {
        assert_eq!(
            Siglip2ModelVariant::from_model_size("base"),
            Some(Siglip2ModelVariant::BasePatch16_224)
        );
        assert_eq!(
            Siglip2ModelVariant::from_model_size("LARGE"),
            Some(Siglip2ModelVariant::LargePatch16_256)
        );
        assert_eq!(
            Siglip2ModelVariant::from_model_size("so400m"),
            Some(Siglip2ModelVariant::So400mPatch14_224)
        );
        assert_eq!(Siglip2ModelVariant::from_model_size("g-opt"), None);
    }

    #[test]
    fn tiny_config_is_rejected_by_production_profile() {
        let cfg = Siglip2Config::tiny_for_tests();
        assert!(cfg.validate_production_profile().is_err());
    }

    #[test]
    fn altered_near_presets_are_rejected_by_production_profile() {
        let preset = Siglip2Config::for_variant(Siglip2ModelVariant::BasePatch16_224);
        let mut altered = Vec::new();

        let mut config = preset.clone();
        config.image_size += config.patch_size;
        altered.push(config);
        let mut config = preset.clone();
        config.patch_size = 14;
        altered.push(config);
        let mut config = preset.clone();
        config.channels += 1;
        altered.push(config);
        let mut config = preset.clone();
        config.hidden_dim += config.num_heads;
        altered.push(config);
        let mut config = preset.clone();
        config.intermediate_dim += 1;
        altered.push(config);
        let mut config = preset.clone();
        config.projection_dim += 1;
        altered.push(config);
        let mut config = preset.clone();
        config.num_heads = 8;
        altered.push(config);
        let mut config = preset.clone();
        config.num_layers += 1;
        altered.push(config);
        let mut config = preset.clone();
        config.layer_norm_eps *= 2.0;
        altered.push(config);
        let mut config = preset.clone();
        config.text_max_positions += 1;
        altered.push(config);
        let mut config = preset.clone();
        config.text_vocab_size += 1;
        altered.push(config);
        let mut config = preset;
        config.text_pad_token_id = 1;
        altered.push(config);

        for config in altered {
            assert!(
                config.validate().is_ok(),
                "near-preset must be structurally valid"
            );
            let error = config
                .validate_production_profile()
                .expect_err("altered preset must not be accepted for production loading");
            assert!(
                error.contains("exact supported production SigLIP2 profile"),
                "unexpected error: {error}"
            );
        }
    }
}
