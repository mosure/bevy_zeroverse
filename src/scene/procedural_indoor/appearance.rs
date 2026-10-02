//! Appearance factors applied after furnishing, before material and light creation.
//! They never resample architecture, objects, people, cameras or family identity.
use super::{
    domain::SceneDomain,
    layout::{stream, IndoorManifest, LightingMood},
    materials,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AppearanceSettings {
    pub material_seed: Option<u64>,
    pub lighting_seed: Option<u64>,
    /// Scale material relief and pattern contrast; geometry detail is unchanged.
    pub material_detail: f32,
    /// Uniform dimming of electric, sun, sky and ambient illumination.
    pub illumination_scale: f32,
    /// Added to EV100: positive values darken the renderer's exposure.
    pub exposure_ev100_offset: f32,
}
impl Default for AppearanceSettings {
    fn default() -> Self {
        Self {
            material_seed: None,
            lighting_seed: None,
            material_detail: 1.,
            illumination_scale: 1.,
            exposure_ev100_offset: 0.,
        }
    }
}
impl AppearanceSettings {
    pub fn parse(json: &str) -> Result<Self, String> {
        let settings: Self =
            serde_json::from_str(json).map_err(|e| format!("indoor_appearance: {e}"))?;
        settings.validate()?;
        Ok(settings)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.material_detail)
            || !(0.05..=1.0).contains(&self.illumination_scale)
            || !(-2.0..=2.0).contains(&self.exposure_ev100_offset)
        {
            return Err("indoor_appearance requires material_detail in [0,1], illumination_scale in [0.05,1], exposure_ev100_offset in [-2,2]".into());
        }
        Ok(())
    }
}
impl IndoorManifest {
    pub fn material_seed(&self) -> u64 {
        self.appearance
            .as_ref()
            .and_then(|a| a.material_seed)
            .unwrap_or(self.seed)
    }
    pub fn apply_appearance(&mut self, settings: AppearanceSettings) -> Result<(), String> {
        settings.validate()?;
        if self.appearance.is_some() {
            return Err(
                "apply appearance to a fresh manifest; repeated dimming is ambiguous".into(),
            );
        }
        let program = self
            .program
            .as_mut()
            .ok_or("appearance factors require a procedural program")?;
        let domain = program
            .domain
            .as_mut()
            .ok_or("appearance factors require photometry")?;
        if let Some(seed) = settings.material_seed {
            program.materials = materials::program::sample(seed);
        }
        for recipe in &mut program.materials {
            recipe.relief_m *= settings.material_detail;
            recipe.contrast *= settings.material_detail;
        }
        if let Some(seed) = settings.lighting_seed {
            let lighting = SceneDomain::sample(seed);
            domain.photometry = lighting.photometry;
            domain.target_lux = lighting.target_lux;
            domain.fixture_kelvin = lighting.fixture_kelvin;
            let mut rng = stream(seed, 348);
            self.sun_elevation = rng.random_range(0.08..1.3);
            self.sun_azimuth = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
        }
        domain.target_lux *= settings.illumination_scale;
        domain.photometry.sun_lux *= settings.illumination_scale;
        domain.photometry.sky_radiance *= settings.illumination_scale;
        domain.photometry.environment_intensity *= settings.illumination_scale;
        domain.photometry.ev100 += settings.exposure_ev100_offset;
        domain.validate()?;
        self.target_lux = domain.target_lux;
        self.daylight_lux = domain.photometry.sun_lux;
        self.light_kelvin = domain.fixture_kelvin;
        self.lighting = if self.daylight_lux > 8000. {
            LightingMood::Daylight
        } else if self.daylight_lux > 100. {
            LightingMood::Overcast
        } else {
            LightingMood::Evening
        };
        self.appearance = Some(settings);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn factors_preserve_geometry_people_cameras_and_the_scene_family_seed() {
        let original =
            IndoorManifest::generate(7, super::super::layout::IndoorLayout::Mixed, 0.65, 2)
                .unwrap();
        for settings in [
            AppearanceSettings {
                material_seed: Some(55),
                ..Default::default()
            },
            AppearanceSettings {
                lighting_seed: Some(88),
                ..Default::default()
            },
            AppearanceSettings {
                material_detail: 0.2,
                ..Default::default()
            },
            AppearanceSettings {
                illumination_scale: 0.05,
                exposure_ev100_offset: 1.,
                ..Default::default()
            },
        ] {
            let mut scene = original.clone();
            scene.apply_appearance(settings.clone()).unwrap();
            assert_eq!(scene.seed, original.seed);
            assert_eq!(scene.objects, original.objects);
            assert_eq!(scene.humans, original.humans);
            assert_eq!(scene.cameras, original.cameras);
            assert_eq!(scene.envelope, original.envelope);
            assert_eq!(
                scene.program.as_ref().unwrap().partitions,
                original.program.as_ref().unwrap().partitions
            );
            super::super::validation::validate_layout(&scene).unwrap();
            if settings.lighting_seed.is_none() && settings.illumination_scale == 1. {
                assert_eq!(
                    scene.domain().unwrap().photometry,
                    original.domain().unwrap().photometry
                );
            }
            if settings.material_seed.is_none() && settings.material_detail == 1. {
                assert_eq!(
                    scene.program.as_ref().unwrap().materials,
                    original.program.as_ref().unwrap().materials
                );
            }
            assert!(scene.apply_appearance(settings).is_err());
        }
    }
}
