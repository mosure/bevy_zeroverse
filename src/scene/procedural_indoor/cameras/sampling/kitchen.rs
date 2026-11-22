//! Reference framing of the installed work surfaces. The eyes, lenses and paths
//! still come from the camera policy. Only the first few eligible proposals
//! prefer a working view, so an occluded galley never blocks camera admission.
use super::super::super::{
    layout::{stream, IndoorManifest, ObjectKind},
    objects::kitchen,
};
use bevy::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
mod composition;
pub(super) use composition::fixture_label;

struct WorkSurface {
    zone: usize,
    center: Vec3,
    across: Vec3,
    front: Vec3,
    size: Vec2,
    island: bool,
}

pub(super) struct KitchenFraming {
    surfaces: Vec<WorkSurface>,
    rng: ChaCha8Rng,
    remaining: u8,
    paired_tries: u8,
}

impl KitchenFraming {
    pub(super) fn for_slot(scene: &IndoorManifest, index: usize) -> Option<Self> {
        if scene.assembly_diagnostics.kitchens.is_empty()
            || scene.camera_settings.overlap_mixture.is_some()
            || scene.camera_settings.handheld.is_some()
            || scene.camera_settings.ceiling_mounted(scene.seed, index)
        {
            return None;
        }
        let mut rng = stream(scene.seed, 0x4b49_5443_4856_4945 ^ index as u64);
        // Some sessions follow dining, circulation or other ordinary subjects.
        if !rng.random_bool(0.82) {
            return None;
        }
        let mut surfaces = Vec::new();
        for group in &scene.assembly_diagnostics.kitchens {
            for &id in &group.units {
                let Some(o) = scene.objects.get(id) else {
                    continue;
                };
                if o.neighbor
                    || !matches!(
                        o.kind,
                        ObjectKind::KitchenSink | ObjectKind::Range | ObjectKind::KitchenCounter
                    )
                {
                    continue;
                }
                // An open door expands the reserved footprint past the slab.
                let tf = o.transform();
                let (center, size) = if o.kind == ObjectKind::Range {
                    kitchen::appliances::cooking_surface(o, &kitchen::parameters(o))
                } else {
                    let (lo, hi) = kitchen::counter_top_bounds(o);
                    let mid = (lo + hi) * 0.5;
                    let height = if o.kind == ObjectKind::KitchenSink {
                        o.size.y * 0.69
                    } else {
                        o.size.y
                    };
                    (tf.transform_point(Vec3::new(mid.x, height, mid.y)), hi - lo)
                };
                surfaces.push(WorkSurface {
                    zone: group.zone,
                    center,
                    across: tf.rotation * Vec3::X,
                    front: tf.rotation * Vec3::Z,
                    size,
                    island: kitchen::is_island(o),
                });
            }
        }
        (!surfaces.is_empty()).then_some(Self {
            surfaces,
            rng,
            remaining: 12,
            paired_tries: 16,
        })
    }

    /// None tries another eye within the bounded preference budget. When a
    /// person is required, split the angular field between their trunk and the
    /// work surface. A Euclidean midpoint would favor the farther subject.
    pub(super) fn target(
        &mut self,
        zone: usize,
        eye: Vec3,
        ordinary: Vec3,
        require_person: bool,
    ) -> Option<Vec3> {
        if self.remaining == 0 {
            return Some(ordinary);
        }
        let mut in_zone = self.surfaces.iter().filter(|s| s.zone == zone);
        let count = in_zone.clone().count();
        if count == 0 {
            return Some(ordinary);
        }
        self.remaining -= 1;
        let surface = in_zone.nth(self.rng.random_range(0..count))?;
        let work = surface.center
            + surface.across * surface.size.x * self.rng.random_range(-0.30..0.30)
            + surface.front * surface.size.y * self.rng.random_range(-0.16..0.28)
            + Vec3::Y * self.rng.random_range(0.03..0.24);
        let delta = work - eye;
        // See the working face of a run. A freestanding island is finished on
        // both sides. Avoid grazing the appliance from the neighboring cabinet.
        if delta.length_squared() < 1.5_f32.powi(2)
            || (!surface.island && -delta.dot(surface.front) < 0.65)
        {
            return None;
        }
        if !require_person {
            return Some(work);
        }
        let person = ordinary - eye;
        let person_dir = person.try_normalize()?;
        let work_dir = delta.normalize();
        if person_dir.dot(work_dir) < 0.50 {
            return None;
        }
        let weight = self.rng.random_range(0.32..0.54);
        let direction = person_dir.lerp(work_dir, weight).normalize();
        Some(eye + direction * person.length().lerp(delta.length(), weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run() -> KitchenFraming {
        KitchenFraming {
            surfaces: vec![WorkSurface {
                zone: 2,
                center: Vec3::new(0., 0.90, 0.),
                across: Vec3::X,
                front: Vec3::Z,
                size: Vec2::new(1.30, 0.66),
                island: false,
            }],
            rng: stream(15, 20),
            remaining: 12,
            paired_tries: 16,
        }
    }

    #[test]
    fn kitchen_preferences_replay_and_exhaust_without_changing_other_zones() {
        let mut a = run();
        let mut b = run();
        let eye = Vec3::new(0., 1.60, 3.);
        let human = Vec3::new(1., 1.24, 1.);
        assert_eq!(a.target(1, eye, human, true), Some(human));
        for _ in 0..12 {
            let target = a.target(2, eye, human, true).unwrap();
            assert_eq!(Some(target), b.target(2, eye, human, true));
            let dir = (target - eye).normalize();
            assert!(dir.dot((human - eye).normalize()) > 0.90);
            assert!(dir.dot((a.surfaces[0].center - eye).normalize()) > 0.90);
        }
        assert_eq!(a.target(2, eye, human, true), Some(human));
        let mut behind = run();
        for _ in 0..12 {
            assert!(behind.target(2, -eye, human, false).is_none());
        }
        assert_eq!(behind.target(2, -eye, human, false), Some(human));
        // Subjects on opposite sides of the observer cannot share a pinhole frame.
        assert!(run().target(2, eye, eye + Vec3::Z * 2., true).is_none());
    }
}
