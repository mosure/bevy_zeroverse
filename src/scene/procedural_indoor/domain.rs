//! Continuous scene-level priors. These describe the sampled domain, rather than
//! choosing a complete room from a small preset bank. Stored in each manifest.
use super::{
    layout::{stream, IndoorManifest, LightingMood},
    materials::kelvin_rgb,
};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

mod exposure;
pub use super::sky::SkyDistribution;
pub use exposure::IndoorExposureMeter;

fn log_sample(rng: &mut impl Rng, lo: f32, hi: f32) -> f32 {
    rng.random_range(lo.ln()..hi.ln()).exp()
}

pub fn room_size(seed: u64) -> Vec3 {
    let mut rng = stream(seed, 211);
    let area = log_sample(&mut rng, 38.0, 270.0);
    let mut aspect = log_sample(&mut rng, 0.42, 2.4);
    // At fixed area A both sqrt(A*r) and sqrt(A/r) must fit the span.
    // Condition the log-aspect prior instead of piling samples on a boundary.
    let lo = (5.6 * 5.6 / area).max(area / (21. * 21.)).max(0.42);
    let hi = (21. * 21. / area).min(area / (5.6 * 5.6)).min(2.4);
    if !(lo..=hi).contains(&aspect) {
        aspect = log_sample(&mut stream(seed, 0x5350_414e_4649_5400), lo, hi);
    }
    let width = (area * aspect).sqrt();
    Vec3::new(width, rng.random_range(2.65..4.8), area / width)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Photometry {
    /// Direct normal solar illuminance; independent from electric work-plane target.
    pub sun_lux: f32,
    /// Cosine-weighted mean upper-hemisphere radiance in linear RGB.
    pub sky_radiance: Vec3,
    /// Archived rooms without this field have a uniform upper hemisphere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sky_distribution: Option<SkyDistribution>,
    pub environment_intensity: f32,
    pub ev100: f32,
    /// Recorded before any explicit appearance intervention. Absent in older manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meter: Option<IndoorExposureMeter>,
    pub active_fraction: f32,
    pub circuit_contrast: f32,
    pub sun_kelvin: f32,
    #[serde(default)]
    pub fixture_gradient: Vec2,
    #[serde(default)]
    pub temperature_gradient: f32,
}

impl Photometry {
    /// Diffuse horizontal illuminance is pi times the cosine-weighted mean luminance.
    pub fn diffuse_sky_lux(&self) -> f32 {
        std::f32::consts::PI * self.sky_radiance.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    pub(crate) fn mood(&self) -> LightingMood {
        let diffuse = self.diffuse_sky_lux();
        if self.sun_lux > 8000. && self.sun_lux > diffuse {
            LightingMood::Daylight
        } else if self.sun_lux.max(diffuse) > 100. {
            LightingMood::Overcast
        } else {
            LightingMood::Evening
        }
    }
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
        let clear_sun = if light.random_bool(0.65) {
            log_sample(&mut light, 3000.0, 100_000.0)
        } else {
            log_sample(&mut light, 0.1, 100_000.0)
        };
        let target_lux = if light.random_bool(0.75) {
            log_sample(&mut light, 150.0, 900.0)
        } else {
            log_sample(&mut light, 3.0, 1000.0)
        };
        let sky_ratio = log_sample(&mut light, 0.008, 0.10);
        let sky_kelvin = light.random_range(5000.0..10000.0);
        // Clear through heavy cloud. A bright overcast sky can illuminate
        // a room even when the direct beam is almost completely attenuated.
        let mut weather = stream(seed, 0x434c_4f55_4453);
        let transmission = (-9. * weather.random::<f32>().powi(4)).exp();
        let sun_lux = clear_sun * transmission;
        let diffuse_lux = clear_sun
            * (std::f32::consts::PI * sky_ratio
                + (1. - transmission) * weather.random_range(0.16..0.55));
        // A smooth shoulder preserves variation at the bright end. Store
        // linear RGB normalized to the sampled photopic sky luminance.
        let diffuse_lux = diffuse_lux / (1. + diffuse_lux / 20_000.);
        let sky = diffuse_lux / std::f32::consts::PI;
        let kelvin = 6500. + (sky_kelvin - 6500.) * (0.35 + 0.65 * transmission);
        let rgb = kelvin_rgb(kelvin);
        let linear = Color::srgb(rgb.x, rgb.y, rgb.z).to_linear();
        let chroma = Vec3::new(linear.red, linear.green, linear.blue);
        let sky_radiance = chroma * (sky / chroma.dot(Vec3::new(0.2126, 0.7152, 0.0722)));
        // Partial adaptation preserves real brightness differences. Independent
        // exposure offsets cover under/overexposure without normalizing each image.
        let exposure_variation = light.random_range(-1.1..1.3);
        let meter = IndoorExposureMeter {
            electric_lux: target_lux,
            direct_daylight_lux: 0.025 * sun_lux,
            diffuse_daylight_lux: 0.025 * diffuse_lux,
            bias_ev: 0.70 * exposure_variation,
        };
        // Full or near-full circuits are common while intermittent use still
        // spans mostly unlit through fully occupied installations.
        let circuit_draw = light.random_range(0.0_f32..1.);
        let photometry = Photometry {
            sun_lux,
            sky_radiance,
            sky_distribution: Some(SkyDistribution::sample(seed, transmission)),
            ev100: meter.ev100(),
            meter: Some(meter),
            environment_intensity: (0.07 * target_lux + 0.09 * sky).clamp(0.25, 160.0),
            active_fraction: 1. - 0.65 * (1. - circuit_draw).powi(2),
            circuit_contrast: light.random_range(0.0..0.85),
            sun_kelvin: light.random_range(2600.0..6900.0),
            fixture_gradient: Vec2::new(
                light.random_range(-0.6..0.6),
                light.random_range(-0.6..0.6),
            ),
            temperature_gradient: light.random_range(-1100.0..1100.0),
        };
        // Keep tungsten and cool daylight-matched installations as tails of
        // the room lighting mix. Individual fixtures keep their own tint/gradient.
        let u = light.random::<f32>();
        let fixture_kelvin = if u < 0.10 {
            2200. + 800. * u / 0.10
        } else if u < 0.82 {
            3000. + 2000. * (u - 0.10) / 0.72
        } else {
            5000. + 2500. * (u - 0.82) / 0.18
        };
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
        if p.meter.is_some_and(|m| !m.valid()) {
            return Err("invalid indoor exposure estimate".into());
        }
        if !p.sky_radiance.is_finite()
            || p.sky_radiance.min_element() < 0.0
            || p.sky_distribution.is_some_and(|s| !s.valid())
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
    fn meter_ev(electric_lux: f32, sun_lux: f32, bias_ev: f32) -> f32 {
        IndoorExposureMeter {
            electric_lux,
            direct_daylight_lux: 0.025 * sun_lux,
            diffuse_daylight_lux: 0.,
            bias_ev,
        }
        .ev100()
    }
    #[test]
    fn exposure_preserves_lighting_range_without_washing_out_daylight() {
        let sampled = SceneDomain::sample(6).photometry;
        assert_eq!(sampled.ev100, sampled.meter.unwrap().ev100());
        // A lighting octave is partially compensated, leaving 0.38 stops of
        // brightness variation. This holds before the intentional dark limit.
        for electric in [30., 300., 900.] {
            for sun in [100., 10_000., 50_000.] {
                let a = meter_ev(electric, sun, 0.);
                let b = meter_ev(electric * 2., sun * 2., 0.);
                assert!((b - a - 0.62).abs() < 1e-5);
            }
        }
        assert!(meter_ev(3., 0.1, 0.) < 3.);
        assert!(meter_ev(900., 100_000., 0.) > 8.);
    }
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
    fn room_spans_preserve_requested_floor_area_without_clamped_edges() {
        for seed in 0..4096 {
            let area = log_sample(&mut stream(seed, 211), 38., 270.);
            let size = room_size(seed);
            assert!((size.x * size.z - area).abs() < 0.0001);
            for side in [size.x, size.z] {
                assert!((5.6..=21.).contains(&side));
                assert_ne!(side, 5.6);
                assert_ne!(side, 21.);
            }
        }
    }

    #[test]
    fn diffuse_daylight_can_be_bright_without_direct_sun() {
        let mut overcast = 0;
        let mut sunny = 0;
        let mut dark = 0;
        for seed in 0..4096 {
            let p = SceneDomain::sample(seed).photometry;
            let sky = p.diffuse_sky_lux();
            if sky > 1000. && p.sun_lux < 500. {
                assert_eq!(p.mood(), LightingMood::Overcast);
                overcast += 1;
            }
            sunny += usize::from(p.sun_lux > 10000. && p.sun_lux > 3. * sky);
            dark += usize::from(p.sun_lux < 10. && sky < 10.);
        }
        assert!(
            overcast > 100 && sunny > 100 && dark > 100,
            "{overcast}/{sunny}/{dark}"
        );
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
        let positions = super::super::architecture::fixture_positions(&scene);
        let count = super::super::architecture::primary_fixture_count(&scene, &positions);
        let index = super::super::architecture::energized_fixture(&scene, count);
        let dim = super::super::architecture::fixture_photometry(&scene, index).1;
        scene.target_lux = 50.0;
        let bright = super::super::architecture::fixture_photometry(&scene, index).1;
        assert!(dim > 0.0 && dim < 900.0);
        assert!((bright / dim - 10.0).abs() < 0.001);
        scene.daylight_lux = 0.2;
        assert_eq!(super::super::architecture::sun_illuminance(&scene), 0.2);
        assert!(scene.sky_radiance().is_finite());
    }
}
