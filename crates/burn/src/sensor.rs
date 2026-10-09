//! Optional, reproducible RGB corruptions on the CPU export worker. These are
//! post-tonemapping augmentations, not a physical sensor simulation.
use anyhow::{Context as ContextExt, Result, ensure};
use bevy_zeroverse::{
    render::color::{ColorEncoding, linear_to_srgb},
    sample::Sample,
    scene::procedural_indoor::layout::stream,
};
use image::{ImageBuffer, Rgb};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SensorSettings {
    pub seed: u64,
    pub white_balance_gain: [[f32; 2]; 3],
    pub blur_sigma_pixels: [f32; 2],
    pub noise_std_srgb: [f32; 2],
    pub jpeg_quality: Option<[u8; 2]>,
}
impl Default for SensorSettings {
    fn default() -> Self {
        Self {
            seed: 0,
            white_balance_gain: [[1.; 2]; 3],
            blur_sigma_pixels: [0.; 2],
            noise_std_srgb: [0.; 2],
            jpeg_quality: None,
        }
    }
}
impl SensorSettings {
    pub fn parse(json: &str) -> Result<Self> {
        let settings: Self = serde_json::from_str(json).context("rgb_sensor JSON")?;
        settings.validate()?;
        Ok(settings)
    }
    pub fn validate(&self) -> Result<()> {
        for (range, min, max) in self
            .white_balance_gain
            .iter()
            .map(|r| (r, 0.25, 4.))
            .chain([
                (&self.blur_sigma_pixels, 0., 4.),
                (&self.noise_std_srgb, 0., 0.15),
            ])
        {
            ensure!(
                range.iter().all(|v| v.is_finite())
                    && range[0] >= min
                    && range[1] <= max
                    && range[0] <= range[1],
                "invalid rgb_sensor range {range:?}"
            );
        }
        ensure!(
            self.jpeg_quality
                .is_none_or(|[lo, hi]| lo > 0 && lo <= hi && hi <= 100),
            "rgb_sensor jpeg_quality must be ordered in 1..=100"
        );
        Ok(())
    }
    pub fn apply(&self, sample: &mut Sample, size: [u32; 2]) -> Result<()> {
        self.validate()?;
        ensure!(
            size[0] > 0 && size[1] > 0,
            "sensor requires positive image dimensions"
        );
        let scene_seed = sample
            .indoor
            .as_ref()
            .context("rgb_sensor requires seeded indoor scenes")?
            .seed;
        ensure!(
            sample.color_encoding != ColorEncoding::Legacy,
            "sensor requires explicit RGB encoding"
        );
        ensure!(
            sample
                .indoor_render_metadata
                .as_ref()
                .and_then(|m| m.get("rgb_sensor"))
                .is_none(),
            "sensor transforms already applied"
        );
        let root_seed = stream(scene_seed, self.seed ^ 0x53454e534f52).random::<u64>();
        let sample_range =
            |range: [f32; 2], index| stream(root_seed, index).random_range(range[0]..=range[1]);
        let gains =
            std::array::from_fn::<_, 3, _>(|i| sample_range(self.white_balance_gain[i], i as u64));
        let sigma = sample_range(self.blur_sigma_pixels, 3);
        let noise = sample_range(self.noise_std_srgb, 4);
        let quality = self
            .jpeg_quality
            .map(|[lo, hi]| stream(root_seed, 5).random_range(lo..=hi));
        for (index, view) in sample.views.iter_mut().enumerate() {
            let rgba = crate::chunk::decode_rgba_bytes(&view.color, size[0], size[1])?;
            ensure!(rgba.iter().all(|v| v.is_finite()), "non-finite RGB input");
            let mut rgb: ImageBuffer<Rgb<f32>, Vec<f32>> =
                ImageBuffer::from_fn(size[0], size[1], |x, y| {
                    let pixel = (y as usize * size[0] as usize + x as usize) * 4;
                    Rgb(std::array::from_fn(|c| {
                        let v = rgba[pixel + c].clamp(0., 1.);
                        let linear = if sample.color_encoding == ColorEncoding::Srgb {
                            srgb_to_linear(v)
                        } else {
                            v
                        };
                        linear_to_srgb(linear * gains[c]).clamp(0., 1.)
                    }))
                });
            if sigma > 0. {
                rgb = image::imageops::blur(&rgb, sigma);
            }
            let mut rng = stream(root_seed, 100 + index as u64);
            if noise > 0. {
                for value in rgb.as_mut() {
                    let u = rng.random::<f32>().max(f32::MIN_POSITIVE);
                    let v = rng.random::<f32>();
                    let gaussian = (-2. * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos();
                    *value = (*value + noise * gaussian).clamp(0., 1.);
                }
            }
            if let Some(quality) = quality {
                let bytes: Vec<_> = rgb
                    .as_raw()
                    .iter()
                    .map(|v| (v.clamp(0., 1.) * 255.).round() as u8)
                    .collect();
                let mut compressed = Vec::new();
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut compressed, quality)
                    .encode(&bytes, size[0], size[1], image::ExtendedColorType::Rgb8)?;
                let decoded = image::load_from_memory(&compressed)?.to_rgb32f();
                rgb = decoded;
            }
            let output: Vec<_> = rgb
                .as_raw()
                .as_chunks::<3>()
                .0
                .iter()
                .enumerate()
                .flat_map(|(p, v)| [v[0], v[1], v[2], rgba[p * 4 + 3]])
                .collect();
            view.color = bytemuck::cast_slice(&output).to_vec();
        }
        sample.color_encoding = ColorEncoding::Srgb;
        let metadata = sample
            .indoor_render_metadata
            .get_or_insert_with(|| serde_json::json!({}));
        metadata.as_object_mut().context("indoor render metadata must be an object")?.insert("rgb_sensor".into(), serde_json::json!({
            "schema_version":1, "requested":self, "seed":root_seed,
            "white_balance_gain":gains, "blur_sigma_pixels":sigma, "noise_std_srgb":noise, "jpeg_quality":quality,
            "order":["white_balance_display_linear", "srgb_transfer", "gaussian_blur_srgb", "independent_gaussian_noise_srgb", "optional_jpeg_encode_decode"],
            "scope":"parameters shared across scene; independent noise stream 100 + view index; geometry annotations and alpha untouched"
        }));
        Ok(())
    }
}
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_zeroverse::{
        sample::View,
        scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest},
    };
    fn sample() -> Sample {
        let rgba: Vec<f32> = (0..64)
            .flat_map(|p| [p as f32 / 64., 0.4, 0.6, 1.])
            .collect();
        Sample {
            indoor: Some(IndoorManifest::generate(0, IndoorLayout::Mixed, 0.65, 1).unwrap()),
            color_encoding: ColorEncoding::TonemappedLinear,
            views: vec![View {
                color: bytemuck::cast_slice(&rgba).to_vec(),
                depth: vec![1, 2, 3],
                normal: vec![4, 5, 6],
                semantic: vec![7],
                ..Default::default()
            }],
            ..Default::default()
        }
    }
    #[test]
    fn seeded_factors_only_modify_rgb() {
        let original = sample();
        let settings = SensorSettings {
            noise_std_srgb: [0.05; 2],
            blur_sigma_pixels: [0.5; 2],
            jpeg_quality: Some([75; 2]),
            ..Default::default()
        };
        let mut a = original.clone();
        let mut b = original.clone();
        settings.apply(&mut a, [8, 8]).unwrap();
        settings.apply(&mut b, [8, 8]).unwrap();
        assert_eq!(a.views[0].color, b.views[0].color);
        assert_eq!(a.indoor_render_metadata, b.indoor_render_metadata);
        assert_ne!(a.views[0].color, original.views[0].color);
        for s in [&mut a, &mut b] {
            s.views[0].color = original.views[0].color.clone();
        }
        assert_eq!(
            serde_json::to_value(&a.views).unwrap(),
            serde_json::to_value(&original.views).unwrap()
        );
        let mut c = original.clone();
        SensorSettings {
            seed: 1,
            ..settings.clone()
        }
        .apply(&mut c, [8, 8])
        .unwrap();
        let mut d = original.clone();
        settings.apply(&mut d, [8, 8]).unwrap();
        assert_ne!(c.views[0].color, d.views[0].color);
        assert!(settings.apply(&mut c, [8, 8]).is_err());
        assert_eq!(
            c.indoor.as_ref().unwrap().objects,
            original.indoor.as_ref().unwrap().objects
        );
    }
    #[test]
    fn neutral_settings_preserve_display_values() {
        let mut s = sample();
        s.color_encoding = ColorEncoding::Srgb;
        let old = s.views[0].color.clone();
        SensorSettings::default().apply(&mut s, [8, 8]).unwrap();
        let a = crate::chunk::decode_rgba_bytes(&old, 8, 8).unwrap();
        let b = crate::chunk::decode_rgba_bytes(&s.views[0].color, 8, 8).unwrap();
        assert!(a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
    }
    #[test]
    fn each_sensor_factor_has_an_isolated_effect_and_keeps_annotations() {
        let original = sample();
        let mut reference = original.clone();
        SensorSettings::default()
            .apply(&mut reference, [8, 8])
            .unwrap();
        for settings in [
            SensorSettings {
                white_balance_gain: [[0.8; 2], [1.; 2], [1.; 2]],
                ..Default::default()
            },
            SensorSettings {
                blur_sigma_pixels: [0.8; 2],
                ..Default::default()
            },
            SensorSettings {
                noise_std_srgb: [0.01; 2],
                ..Default::default()
            },
            SensorSettings {
                jpeg_quality: Some([50; 2]),
                ..Default::default()
            },
        ] {
            let mut changed = original.clone();
            settings.apply(&mut changed, [8, 8]).unwrap();
            assert_ne!(changed.views[0].color, reference.views[0].color);
            changed.views[0].color = original.views[0].color.clone();
            assert_eq!(changed.views, original.views);
            assert_eq!(changed.indoor, original.indoor);
            assert_eq!(
                changed.indoor_render_metadata.as_ref().unwrap()["rgb_sensor"]["seed"],
                reference.indoor_render_metadata.as_ref().unwrap()["rgb_sensor"]["seed"]
            );
        }
    }
}
