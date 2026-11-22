//! When the visitor and equipment are far apart, choose an eye from which
//! both fit in one photograph. Changing only the aim of an eye between them
//! cannot fit two subjects on opposite sides of a pinhole sensor.
use super::*;
use crate::scene::procedural_indoor::humans::IndoorHuman;
use rand::seq::{IndexedRandom, IteratorRandom};

pub(in super::super) fn fixture_label(label: &str) -> bool {
    matches!(
        label,
        "counter" | "sink" | "stove" | "oven" | "refrigerator" | "dishwasher" | "cabinet"
    )
}

pub(in super::super) struct Composition {
    pub(in super::super) eye: Vec2,
    person: Vec3,
    work: Vec3,
    weight: f32,
}

impl Composition {
    pub(in super::super) fn target(&self, eye: Vec3) -> Vec3 {
        let a = self.person - eye;
        let b = self.work - eye;
        let direction = a.normalize().lerp(b.normalize(), self.weight).normalize();
        eye + direction * a.length().lerp(b.length(), self.weight)
    }
}

impl KitchenFraming {
    pub(in super::super) fn composition(
        &mut self,
        scene: &IndoorManifest,
        people: &[&IndoorHuman],
    ) -> Option<Composition> {
        // Most room views already find a useful angle. Only try this after
        // ordinary kitchen proposals have failed, inside the existing search cap.
        if self.remaining > 7 || self.paired_tries == 0 {
            return None;
        }
        self.paired_tries -= 1;
        // This recovery is only used by the person-framing reference slot.
        // An absent, nearby or obstructed pair must not advance the ordinary
        // work-surface target stream, even after multiple failed proposals.
        let mut rng = stream(
            scene.seed,
            0x5041_4952_4652_414d ^ u64::from(self.paired_tries),
        );
        let human = people.choose(&mut rng)?;
        let person = human.transform().transform_point(human.joints[2]);
        let (zone_id, zone) = scene
            .program
            .as_ref()?
            .zones
            .iter()
            .enumerate()
            .find(|(_, z)| person.xz().cmpge(z.min).all() && person.xz().cmple(z.max).all())?;
        let surfaces = self.surfaces.iter().filter(|s| s.zone == zone_id);
        if surfaces
            .clone()
            .any(|s| s.center.xz().distance_squared(person.xz()) < 2.2_f32.powi(2))
        {
            return None;
        }
        let surface = surfaces.choose(&mut rng)?;
        let work = surface.center
            + surface.across * surface.size.x * rng.random_range(-0.32..0.32)
            + surface.front * surface.size.y * rng.random_range(-0.14..0.24)
            + Vec3::Y * rng.random_range(0.035..0.18);
        let eye = surface.pair_eye(person.xz(), work.xz(), zone.min, zone.max, &mut rng)?;
        Some(Composition {
            eye,
            person,
            work,
            weight: rng.random_range(0.33..0.52),
        })
    }
}

impl WorkSurface {
    fn pair_eye(
        &self,
        person: Vec2,
        work: Vec2,
        min: Vec2,
        max: Vec2,
        rng: &mut ChaCha8Rng,
    ) -> Option<Vec2> {
        let lo = min + Vec2::splat(0.51);
        let hi = max - Vec2::splat(0.51);
        if hi.cmple(lo).any() {
            return None;
        }
        let preferred = rng.random_range(16.0_f32..39.).to_radians().cos();
        // Cheap plan-view ranking precedes one actual floor/roof, body, path
        // and calibrated first-surface test. Nothing is clamped onto a wall.
        (0..12)
            .filter_map(|_| {
                let uv = Vec2::new(rng.random(), rng.random());
                let eye = lo + uv * (hi - lo);
                let a = person - eye;
                let b = work - eye;
                let da = a.length();
                let db = b.length();
                if !(1.45..7.0).contains(&da)
                    || !(1.50..9.).contains(&db)
                    || (!self.island && -b.dot(self.front.xz()) < 0.65)
                {
                    return None;
                }
                let cosine = a.dot(b) / (da * db);
                if cosine < 0.52 {
                    return None;
                }
                // A very remote person occupies too few pixels. A slab seen
                // edge-on or at the far end of a hall is a weak second subject.
                let score = (cosine - preferred).abs() + 0.038 * (da - 3.2).abs() + 0.026 * db;
                Some((score, eye))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, eye)| eye)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distant_subjects_have_replayable_front_side_eyes_and_spread() {
        let s = WorkSurface {
            zone: 0,
            center: Vec3::new(0., 0.9, -2.8),
            across: Vec3::X,
            front: Vec3::Z,
            size: Vec2::new(1.1, 0.65),
            island: false,
        };
        // A symmetric aisle has valid viewpoints on either side of both
        // subjects. An off-centre galley can physically admit only one side.
        let h = Vec3::new(0., 1.4, 0.9);
        let mut eyes = Vec::new();
        for seed in 0..96 {
            let draw = || {
                s.pair_eye(
                    h.xz(),
                    s.center.xz(),
                    Vec2::new(-4.6, -3.5),
                    Vec2::new(4.6, 3.5),
                    &mut stream(seed, 45),
                )
            };
            let eye = draw();
            assert_eq!(eye, draw());
            if let Some(eye) = eye {
                let c = Composition {
                    eye,
                    person: h,
                    work: s.center,
                    weight: 0.4,
                };
                let eye = Vec3::new(eye.x, 1.58, eye.y);
                let axis = (c.target(eye) - eye).normalize();
                assert!(
                    (eye - s.center).dot(s.front) > 0.64
                        && axis.dot((h - eye).normalize()) > 0.85
                        && axis.dot((s.center - eye).normalize()) > 0.75
                );
                eyes.push(eye.xz());
            }
        }
        assert!(eyes.len() > 55 && eyes.iter().any(|p| p.x < -2.) && eyes.iter().any(|p| p.x > 2.));
        assert!(s
            .pair_eye(
                h.xz(),
                s.center.xz(),
                Vec2::ZERO,
                Vec2::ONE,
                &mut stream(0, 1)
            )
            .is_none());
    }
}
