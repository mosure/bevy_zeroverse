//! Seeded, display-linear RGB image formation for logical indoor capture cameras.
//! Optical parameters are fixed on the camera. Only read/shot noise changes with time.
use bevy::prelude::Component;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Probability selects cameras, independently of the rig and the scene contents.
/// The remaining controls bound the sampled severity of each effect in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SensorArtifactSettings {
    /// Change the optical realization of a fixed room without moving any geometry or cameras.
    pub seed_offset: u32,
    pub probability: f32,
    pub blur: f32,
    pub smudge: f32,
    pub flare: f32,
    pub noise: f32,
}
impl Default for SensorArtifactSettings {
    fn default() -> Self {
        Self {
            seed_offset: 0,
            probability: 0.,
            blur: 0.85,
            smudge: 0.75,
            flare: 0.65,
            noise: 0.8,
        }
    }
}
impl SensorArtifactSettings {
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.probability,
            self.blur,
            self.smudge,
            self.flare,
            self.noise,
        ]
        .into_iter()
        .all(|x| (0.0..=1.).contains(&x))
        {
            Ok(())
        } else {
            Err(
                "sensor probability, blur, smudge, flare and noise must be finite and in [0,1]"
                    .into(),
            )
        }
    }
    pub fn profile(&self, room_seed: u64, camera: usize) -> Option<CameraSensorProfile> {
        if self.probability == 0. || [self.blur, self.smudge, self.flare, self.noise] == [0.; 4] {
            return None;
        }
        let mut rng = crate::scene::procedural_indoor::layout::stream(
            room_seed ^ u64::from(self.seed_offset).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            0x5345_4e53_0000_0000 | camera as u64,
        );
        if rng.random::<f32>() >= self.probability {
            return None;
        }
        // Always draw the same variates: interventions on one control must preserve the rest.
        let axis = rng.random_range(0.0..std::f32::consts::PI);
        let blur = (0.18 + 2.7 * rng.random::<f32>().powi(2)) * self.blur;
        let aspect = rng.random_range(0.15..1.);
        let smear_angle = rng.random_range(0.0..std::f32::consts::PI);
        let (sin, cos) = smear_angle.sin_cos();
        let width = rng.random_range(0.09..0.34);
        Some(CameraSensorProfile {
            blur_axes_px512: [blur, blur * aspect],
            blur_direction: [axis.cos(), axis.sin()],
            smudge_center_uv: [rng.random_range(0.04..0.96), rng.random_range(0.04..0.96)],
            smudge_direction: [cos, sin],
            smudge_radii: [width, width * rng.random_range(0.28..0.82)],
            smudge_mix: self.smudge * (0.10 + 0.58 * rng.random::<f32>().powi(2)),
            scatter_px512: rng.random_range(3.5..13.),
            flare_gain: self.flare * (0.012 + 0.15 * rng.random::<f32>().powi(2)),
            ghost_scale: rng.random_range(0.48..1.18),
            highlight_knee: rng.random_range(0.53..0.82),
            ghost_radius_px512: rng.random_range(2.8..9.),
            ghost_tint: [
                rng.random_range(0.68..1.),
                rng.random_range(0.74..1.),
                rng.random_range(0.63..1.),
            ],
            read_std: self.noise * rng.random_range(0.0007..0.006),
            shot_std: self.noise * (0.007 + 0.041 * rng.random::<f32>().powi(2)),
            noise_key: rng.random(),
        })
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraSensorProfile {
    /// Blur and scatter lengths are output pixels at a 512-pixel short edge.
    pub blur_axes_px512: [f32; 2],
    pub blur_direction: [f32; 2],
    pub smudge_center_uv: [f32; 2],
    pub smudge_direction: [f32; 2],
    /// Semiaxes in units of the short edge, invariant to sensor aspect.
    pub smudge_radii: [f32; 2],
    pub smudge_mix: f32,
    pub scatter_px512: f32,
    pub flare_gain: f32,
    pub ghost_scale: f32,
    pub highlight_knee: f32,
    pub ghost_radius_px512: f32,
    pub ghost_tint: [f32; 3],
    /// Display-linear per-channel variance = read_std² + shot_std² × signal.
    pub read_std: f32,
    pub shot_std: f32,
    pub noise_key: u32,
}
impl CameraSensorProfile {
    pub fn noise_seed(&self, normalized_time: f32) -> u32 {
        self.noise_key
            ^ if normalized_time == 0. {
                0
            } else {
                normalized_time.to_bits()
            }
    }

    pub(crate) fn uniform(&self, factor: u32, center: [f32; 2], time: f32) -> ResolveParams {
        ResolveParams {
            header: [factor, self.noise_seed(time), 0, 0],
            blur: [
                self.blur_direction[0],
                self.blur_direction[1],
                self.blur_axes_px512[0],
                self.blur_axes_px512[1],
            ],
            patch: [
                self.smudge_center_uv[0],
                self.smudge_center_uv[1],
                self.smudge_radii[0],
                self.smudge_radii[1],
            ],
            scatter: [
                self.smudge_direction[0],
                self.smudge_direction[1],
                self.smudge_mix,
                self.scatter_px512,
            ],
            ghost: [center[0], center[1], self.ghost_scale, self.highlight_knee],
            tint: [
                self.ghost_tint[0],
                self.ghost_tint[1],
                self.ghost_tint[2],
                self.flare_gain,
            ],
            sensor: [self.read_std, self.shot_std, self.ghost_radius_px512, 0.],
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ResolveParams {
    pub header: [u32; 4],
    blur: [f32; 4],
    patch: [f32; 4],
    scatter: [f32; 4],
    ghost: [f32; 4],
    tint: [f32; 4],
    sensor: [f32; 4],
}

pub fn captures_rgb(config: &crate::app::BevyZeroverseConfig) -> bool {
    if config.render_modes.is_empty() {
        config.render_mode == super::RenderMode::Color
    } else {
        config.render_modes.contains(&super::RenderMode::Color)
    }
}

pub(crate) fn metadata(
    scene: &crate::scene::procedural_indoor::layout::IndoorManifest,
    views: &[crate::sample::View],
    camera_count: usize,
    unbounded: bool,
) -> Option<serde_json::Value> {
    let policy = scene.appearance.as_ref()?.sensor;
    if policy.probability == 0. || camera_count == 0 {
        return None;
    }
    let cameras: Vec<_> = (0..camera_count)
        .map(|i| policy.profile(scene.seed, i))
        .collect();
    let noise_seeds: Vec<_> = views
        .chunks(camera_count)
        .map(|step| {
            step.iter()
                .zip(&cameras)
                .map(|(v, c)| c.map(|c| c.noise_seed(v.time)))
                .collect::<Vec<_>>()
        })
        .collect();
    Some(
        serde_json::json!({"schema_version": 1, "stage":if unbounded { "unbounded_linear_rgb" } else { "tonemapped_linear_rgb_before_srgb" },
        "policy": policy, "cameras": cameras, "noise_seeds_by_step": noise_seeds}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_replayable_cameras_and_factors() {
        let mut s = SensorArtifactSettings {
            probability: 0.65,
            ..Default::default()
        };
        let mut pairs = [0usize; 4];
        for seed in 0..1024 {
            let a = s.profile(seed, 0);
            let b = s.profile(seed, 1);
            pairs[2 * usize::from(a.is_some()) + usize::from(b.is_some())] += 1;
            if let (Some(a), Some(b)) = (a, b) {
                assert_ne!(a.noise_key, b.noise_key);
                assert_eq!(Some(a), s.profile(seed, 0));
                let resampled = SensorArtifactSettings {
                    seed_offset: 17,
                    probability: 1.,
                    ..s
                }
                .profile(seed, 0)
                .unwrap();
                assert_ne!(resampled.noise_key, a.noise_key);
                assert_ne!(a.noise_seed(0.), a.noise_seed(0.5));
                assert_eq!(a.noise_seed(0.), a.noise_seed(-0.));
                let isolated = SensorArtifactSettings { blur: 0., ..s }
                    .profile(seed, 0)
                    .unwrap();
                assert_eq!(
                    CameraSensorProfile {
                        blur_axes_px512: [0.; 2],
                        ..a
                    },
                    isolated
                );
            }
        }
        assert!(pairs.into_iter().all(|n| n > 70));
        assert!(SensorArtifactSettings::default().profile(7, 0).is_none());
        s.probability = 1.;
        assert!(SensorArtifactSettings {
            blur: 0.,
            smudge: 0.,
            flare: 0.,
            noise: 0.,
            ..s
        }
        .profile(7, 0)
        .is_none());
        assert!(SensorArtifactSettings {
            noise: f32::NAN,
            ..s
        }
        .validate()
        .is_err());
    }
}
