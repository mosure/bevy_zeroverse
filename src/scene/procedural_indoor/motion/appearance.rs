//! Reversible changes to emitted RGB on the calibrated scene timeline.
use super::super::{
    architecture,
    layout::{IndoorManifest, ObjectKind},
    materials::{self, screens, variants},
    photometry,
    preparation::SceneGeometry,
};
use super::*;
mod playback;
pub(super) use playback::install;
pub(crate) use playback::{AppearancePlayback, DimmedLight};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AppearanceTimeline {
    pub screens: Vec<ScreenChange>,
    pub lights: Vec<LightChange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScreenChange {
    pub object_id: usize,
    /// Normalized instant when the next page becomes visible on every camera.
    pub at: f32,
    pub initial_content_seed: u64,
    pub next_page_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightSource {
    Object(usize),
    /// Index in the room's ceiling fixture installation.
    Ceiling(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightChange {
    pub source: LightSource,
    pub curve: DimmerCurve,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DimmerCurve {
    pub interval: [f32; 2],
    /// Multiplier of the source's calibrated rest luminance and luminous flux.
    pub end_multiplier: f32,
}

impl DimmerCurve {
    pub fn multiplier_at(self, progress: f32) -> f32 {
        let t =
            ((progress - self.interval[0]) / (self.interval[1] - self.interval[0])).clamp(0., 1.);
        1. + (self.end_multiplier - 1.) * t * t * (3. - 2. * t)
    }

    fn sample(seed: u64, fraction: f32) -> Option<Self> {
        let mut rng = stream(seed, u64::from_le_bytes(*b"LightDim"));
        if rng.random::<f32>() >= fraction {
            return None;
        }
        let start = rng.random_range(0.08..0.69);
        let duration = rng.random_range(0.08..0.29);
        let end_multiplier = if rng.random_bool(0.68) {
            rng.random_range(0.12..0.78)
        } else {
            rng.random_range(1.10..1.40)
        };
        Some(Self {
            interval: [start, start + duration],
            end_multiplier,
        })
    }
}

impl LightSource {
    pub fn dimmer(self, scene: &IndoorManifest) -> Option<DimmerCurve> {
        let lights = &scene.indoor_motion.as_ref()?.appearance.lights;
        lights
            .binary_search_by_key(&self, |l| l.source)
            .ok()
            .map(|i| lights[i].curve)
    }

    /// Shared probes are static across the exposure sequence. Bake the flux
    /// present throughout the sweep; direct illumination follows the live curve.
    pub(crate) fn baked_multiplier(self, scene: &IndoorManifest) -> f32 {
        self.dimmer(scene).map_or(1., |c| c.end_multiplier.min(1.))
    }
}

pub(crate) fn display(kind: ObjectKind) -> Option<(Surface, screens::DisplayKind)> {
    use screens::DisplayKind;
    Some(match kind {
        ObjectKind::Laptop | ObjectKind::Monitor => (Surface::Screen, DisplayKind::Desktop),
        ObjectKind::Phone => (Surface::PhoneScreen, DisplayKind::Phone),
        ObjectKind::Display => (Surface::Television, DisplayKind::Television),
        _ => return None,
    })
}

impl ScreenChange {
    fn sample(
        scene: &IndoorManifest,
        obj: &super::super::layout::IndoorObject,
        assembly: &Assembly,
        fraction: f32,
    ) -> Option<Self> {
        let (surface, kind) = display(obj.kind)?;
        let mut rng = stream(obj.seed, u64::from_le_bytes(*b"PageTurn"));
        if rng.random::<f32>() >= fraction {
            return None;
        }
        let ((_, label), _) = assembly
            .parts
            .iter()
            .find(|((s, _), g)| *s == surface && !g.indices.is_empty())?;
        let stock = scene.material_seed();
        let slot = materials::finish_slot(surface, label)
            .filter(|&s| scene.program.is_some() && materials::registered_finish(surface, s));
        let initial_content_seed =
            slot.map_or(stock, |s| variants::content_seed(stock, surface, s));
        (!screens::parameters_for(initial_content_seed, kind).dormant).then(|| Self {
            object_id: obj.id,
            at: rng.random_range(0.16..0.85),
            initial_content_seed,
            next_page_seed: rng.random(),
        })
    }
}

impl AppearanceTimeline {
    pub fn is_empty(&self) -> bool {
        self.lights.is_empty() && self.screens.is_empty()
    }

    pub(super) fn sample(
        scene: &IndoorManifest,
        geometry: &SceneGeometry,
        policy: &IndoorMotionConfig,
    ) -> Self {
        let mut timeline = Self::default();
        if scene.generator_version < super::TEMPORAL_PROGRAM_VERSION
            || (policy.screen_change_fraction == 0. && policy.light_change_fraction == 0.)
        {
            return timeline;
        }
        for (obj, assembly) in scene.objects.iter().zip(&geometry.objects) {
            if policy.screen_change_fraction > 0. {
                if let Some(change) =
                    ScreenChange::sample(scene, obj, assembly, policy.screen_change_fraction)
                {
                    timeline.screens.push(change);
                }
            }
            if policy.light_change_fraction > 0.
                && photometry::is_lamp_at(obj.kind, scene.generator_version)
                && photometry::lamp(scene, obj).lumens > 0.
            {
                if let Some(curve) = DimmerCurve::sample(
                    obj.seed ^ photometry::lighting_seed(scene).rotate_left(21),
                    policy.light_change_fraction,
                ) {
                    timeline.lights.push(LightChange {
                        source: LightSource::Object(obj.id),
                        curve,
                    });
                }
            }
        }
        if policy.light_change_fraction > 0. {
            let positions = architecture::fixture_positions(scene);
            let seed = photometry::lighting_seed(scene);
            // Reuse the actual installation wiring, including switches for joined rooms.
            let wiring = scene
                .domain()
                .and_then(|d| d.photometry.ceiling_circuits.as_ref());
            let circuit_count = wiring.map_or(2, |w| w.circuits.len());
            let circuits: Vec<_> = (0..circuit_count)
                .map(|i| {
                    DimmerCurve::sample(
                        seed ^ u64::from_le_bytes(*b"Circuit0") ^ i as u64,
                        policy.light_change_fraction,
                    )
                })
                .collect();
            for (i, p, _, _) in architecture::powered_ceiling_fixtures(scene, &positions) {
                let circuit = wiring.map_or_else(
                    || Some(usize::from(p.z > scene.room_size.z * 0.5)),
                    |w| w.mount_circuits.get(i).copied(),
                );
                if let Some(curve) = circuit.and_then(|id| circuits.get(id)).copied().flatten() {
                    timeline.lights.push(LightChange {
                        source: LightSource::Ceiling(i),
                        curve,
                    });
                }
            }
        }
        timeline
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod capture_test;
