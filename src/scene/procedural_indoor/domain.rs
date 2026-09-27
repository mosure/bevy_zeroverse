//! Continuous scene-level priors. These describe the sampled domain, rather than
//! choosing a complete room from a small preset bank. Stored in each manifest.
use super::{
    layout::{stream, IndoorManifest, LightingMood},
    materials::kelvin_rgb,
};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

fn log_sample(rng: &mut impl Rng, lo: f32, hi: f32) -> f32 {
    rng.random_range(lo.ln()..hi.ln()).exp()
}

pub fn room_size(seed: u64) -> Vec3 {
    let mut rng = stream(seed, 211);
    let area = log_sample(&mut rng, 38.0, 270.0);
    let aspect = log_sample(&mut rng, 0.42, 2.4);
    let width = (area * aspect).sqrt().clamp(5.6, 21.0);
    Vec3::new(
        width,
        rng.random_range(2.65..4.8),
        (area / width).clamp(5.6, 21.0),
    )
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Photometry {
    /// Direct normal solar illuminance; independent from electric work-plane target.
    pub sun_lux: f32,
    /// Isotropic upper-hemisphere radiance, shared with native GI and Cycles.
    pub sky_radiance: Vec3,
    pub environment_intensity: f32,
    pub ev100: f32,
    pub active_fraction: f32,
    pub circuit_contrast: f32,
    pub sun_kelvin: f32,
    #[serde(default)]
    pub fixture_gradient: Vec2,
    #[serde(default)]
    pub temperature_gradient: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneDomain {
    pub photometry: Photometry,
    pub target_lux: f32,
    pub fixture_kelvin: f32,
    pub facade_pier_fraction: f32,
    pub blind_coverage: f32,
    pub blind_tilt: f32,
    pub ceiling_relief: f32,
    pub ceiling_coverage: f32,
    pub clutter: f32,
    pub service_density: f32,
    pub disorder: f32,
}

impl SceneDomain {
    pub fn sample(seed: u64) -> Self {
        // Domain separation prevents adding props from changing exposure or sun.
        let mut light = stream(seed, 212);
        // Mix continuous priors: common occupied/daylit conditions retain most
        // mass, while a broad component covers every logarithmic lighting decade.
        let sun_lux = if light.random_bool(0.65) {
            log_sample(&mut light, 3000.0, 100_000.0)
        } else {
            log_sample(&mut light, 0.1, 100_000.0)
        };
        let target_lux = if light.random_bool(0.75) {
            log_sample(&mut light, 150.0, 900.0)
        } else {
            log_sample(&mut light, 3.0, 1000.0)
        };
        let sky = (sun_lux * log_sample(&mut light, 0.008, 0.10)).clamp(0.03, 2400.0);
        let sky_radiance = kelvin_rgb(light.random_range(5000.0..10000.0)) * sky;
        // Partial adaptation preserves real brightness differences. Independent
        // exposure offsets cover under/overexposure without normalizing each image.
        let ev100 = (6.4
            + 0.45 * ((target_lux + sun_lux * 0.025) / 450.0).log2()
            + light.random_range(-1.1..1.3))
        .clamp(2.0, 10.5);
        let photometry = Photometry {
            sun_lux,
            sky_radiance,
            ev100,
            environment_intensity: (0.07 * target_lux + 0.09 * sky).clamp(0.25, 160.0),
            active_fraction: light.random_range(0.35..1.0),
            circuit_contrast: light.random_range(0.0..0.85),
            sun_kelvin: light.random_range(2600.0..6900.0),
            fixture_gradient: Vec2::new(
                light.random_range(-0.6..0.6),
                light.random_range(-0.6..0.6),
            ),
            temperature_gradient: light.random_range(-1100.0..1100.0),
        };
        let fixture_kelvin = light.random_range(2200.0..7500.0);
        let mut rng = stream(seed, 213);
        Self {
            photometry,
            target_lux,
            fixture_kelvin,
            facade_pier_fraction: rng.random_range(0.07..0.58),
            blind_coverage: rng.random_range(0.08..0.96),
            blind_tilt: rng.random_range(-1.2..1.2),
            ceiling_relief: rng.random_range(0.0_f32..1.0).powi(2) * 0.32,
            ceiling_coverage: rng.random_range(0.28..0.86),
            clutter: rng.random_range(0.0_f32..1.0).powf(0.7),
            service_density: rng.random_range(0.45..1.65),
            disorder: rng.random_range(0.0..1.0),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        let p = &self.photometry;
        for (value, lo, hi) in [
            (self.target_lux, 0.1, 2000.0),
            (self.fixture_kelvin, 1800.0, 10000.0),
            (p.sun_lux, 0.0, 150000.0),
            (p.ev100, -2.0, 16.0),
            (p.environment_intensity, 0.0, 500.0),
            (p.active_fraction, 0.0, 1.0),
            (p.circuit_contrast, 0.0, 1.0),
            (p.sun_kelvin, 1800.0, 10000.0),
            (p.temperature_gradient, -1500.0, 1500.0),
            (self.facade_pier_fraction, 0.02, 0.70),
            (self.blind_coverage, 0.0, 1.0),
            (self.blind_tilt, -1.5, 1.5),
            (self.ceiling_relief, 0.0, 0.40),
            (self.ceiling_coverage, 0.1, 0.95),
            (self.clutter, 0.0, 1.0),
            (self.service_density, 0.0, 2.0),
            (self.disorder, 0.0, 1.0),
        ] {
            if !value.is_finite() || !(lo..=hi).contains(&value) {
                return Err("invalid continuous scene domain parameter".into());
            }
        }
        if !p.sky_radiance.is_finite()
            || p.sky_radiance.min_element() < 0.0
            || !p.fixture_gradient.is_finite()
            || p.fixture_gradient.abs().max_element() > 0.8
        {
            return Err("invalid sky radiance".into());
        }
        Ok(())
    }
}

impl IndoorManifest {
    pub(crate) fn minimum_main_objects(&self) -> usize {
        self.domain()
            .map_or(6, |d| 3 + (d.clutter * self.density * 5.0) as usize)
    }
    pub fn domain(&self) -> Option<&SceneDomain> {
        self.program.as_ref()?.domain.as_ref()
    }
    pub fn sky_radiance(&self) -> Vec3 {
        self.domain().map_or_else(
            || match self.lighting {
                LightingMood::Daylight => Vec3::new(360.0, 440.0, 560.0),
                LightingMood::Overcast => Vec3::new(420.0, 460.0, 510.0),
                LightingMood::Evening => Vec3::new(16.0, 21.0, 32.0),
            },
            |d| d.photometry.sky_radiance,
        )
    }
    pub fn ev100(&self) -> f32 {
        self.domain().map_or_else(
            || match self.lighting {
                LightingMood::Daylight => 6.7,
                LightingMood::Overcast => 6.0,
                LightingMood::Evening => 5.6,
            },
            |d| d.photometry.ev100,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_domain_covers_decades_jointly_and_replays_full_width_seeds() {
        let mut joint = [0u32; 18];
        let mut smallest = f32::INFINITY;
        let mut largest = 0.0_f32;
        for seed in 0..4096 {
            let domain = SceneDomain::sample(seed);
            domain.validate().unwrap();
            let sun = (domain.photometry.sun_lux.log10() + 1.0).floor() as usize;
            let electric = domain.target_lux.log10().floor() as usize;
            joint[sun.min(5) * 3 + electric.min(2)] += 1;
            let size = room_size(seed);
            smallest = smallest.min(size.x * size.z);
            largest = largest.max(size.x * size.z);
        }
        assert!(
            joint.iter().all(|&count| count > 3),
            "missing lighting combinations: {joint:?}"
        );
        assert!(smallest < 42.0 && largest > 250.0);
        for seed in [0, u64::MAX, 1 << 40] {
            assert_eq!(SceneDomain::sample(seed), SceneDomain::sample(seed));
            assert_ne!(
                SceneDomain::sample(seed),
                SceneDomain::sample(seed.wrapping_add(1 << 32))
            );
        }
    }
    #[test]
    fn electric_dimming_reaches_renderer_without_old_brightness_floor() {
        let mut scene = IndoorManifest::generate_with_humans(
            0,
            super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.0,
        )
        .unwrap();
        scene.target_lux = 5.0;
        let dim = super::super::architecture::fixture_photometry(&scene, 0).1;
        scene.target_lux = 50.0;
        let bright = super::super::architecture::fixture_photometry(&scene, 0).1;
        assert!(dim > 0.0 && dim < 900.0);
        assert!((bright / dim - 10.0).abs() < 0.001);
        scene.daylight_lux = 0.2;
        assert_eq!(super::super::architecture::sun_illuminance(&scene), 0.2);
        assert!(scene.sky_radiance().is_finite());
    }
}
