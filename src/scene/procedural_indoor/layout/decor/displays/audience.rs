//! Furniture supplies viewing positions independently of the capture rig.
use super::*;
use crate::scene::procedural_indoor::envelope::polygon;

pub(super) struct Audience {
    pub(super) eyes: Vec<(Option<usize>, Vec3)>,
    pub(super) teaching_front: Option<Vec2>,
}

impl Audience {
    pub(super) fn sample(scene: &IndoorManifest) -> Self {
        let region = scene.primary_region();
        let mut rng = stream(scene.seed, u64::from_le_bytes(*b"TVSeats!"));
        let inside =
            |o: &IndoorObject| !o.neighbor && region.contains_projection(o.position.xz(), 0.);
        let mut eyes = Vec::with_capacity(5);
        for seats in [true, false] {
            for (i, o) in scene
                .objects
                .iter()
                .filter(|o| {
                    (if seats {
                        matches!(
                            o.kind,
                            ObjectKind::Chair | ObjectKind::Sofa | ObjectKind::Bench
                        )
                    } else {
                        matches!(o.kind, ObjectKind::Desk | ObjectKind::Table)
                    }) && inside(o)
                })
                .enumerate()
            {
                let eye = (
                    Some(o.id),
                    o.position + Vec3::Y * if seats { 1.20 } else { 1.55 },
                );
                if eyes.len() < 5 {
                    eyes.push(eye);
                } else {
                    let j = rng.random_range(0..=i);
                    if j < eyes.len() {
                        eyes[j] = eye;
                    }
                }
            }
            if !eyes.is_empty() {
                break;
            }
        }
        if eyes.is_empty() {
            let center = (region.min + region.max) * 0.5;
            if region.contains_projection(center, 0.) {
                eyes.push((
                    None,
                    Vec3::new(center.x, scene.floor_height(center) + 1.6, center.y),
                ));
            }
        }
        let teaching_front = scene
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::TeachingBoard && inside(o))
            .map(|o| o.position.xz());
        Self {
            eyes,
            teaching_front,
        }
    }

    /// Check complete eye-to-screen segments. A chair's own high back is
    /// excluded from its eye probe; every other furnishing remains an occluder.
    pub(super) fn score(
        &self,
        scene: &IndoorManifest,
        o: &IndoorObject,
        space: &MountSpace,
        host: Option<Bounds>,
    ) -> Option<f32> {
        let front = o.transform().rotation * Vec3::Z;
        let screen = o.position + Vec3::Y * (o.size.y * 0.5) + front * (o.size.z * 0.5 + 0.01);
        self.score_at(scene, o, space, host, screen)
    }

    pub(super) fn score_at(
        &self,
        scene: &IndoorManifest,
        o: &IndoorObject,
        space: &MountSpace,
        host: Option<Bounds>,
        screen: Vec3,
    ) -> Option<f32> {
        let front = o.transform().rotation * Vec3::Z;
        let projection = o.kind == ObjectKind::ProjectionScreen;
        let mut visible = 0;
        let mut score = 0.;
        for &(seat, eye) in &self.eyes {
            let ray = eye - screen;
            let distance = ray.length();
            if distance < o.size.x.max(1.) {
                continue;
            }
            let facing = ray.dot(front) / distance;
            if facing < 0.36 {
                continue;
            }
            let orientation = self.teaching_front.map_or(1., |board| {
                (screen.xz() - eye.xz())
                    .normalize_or_zero()
                    .dot((board - eye.xz()).normalize_or_zero())
            });
            // A primary projection should be visible while facing the teaching
            // area. TVs can also serve as side repeaters or peripheral signage.
            if orientation < if projection { 0.34 } else { -0.25 } {
                continue;
            }
            if scene
                .envelope
                .as_ref()
                .is_some_and(|e| !polygon::segment_inside(&e.footprint, screen.xz(), eye.xz(), 0.))
                || space
                    .structure
                    .iter()
                    .any(|&(a, b)| Some((a, b)) != host && segment_hits_box(eye, screen, a, b))
                || space.objects.iter().any(|&(id, (a, b))| {
                    id != o.id && Some(id) != seat && segment_hits_box(eye, screen, a, b)
                })
            {
                continue;
            }
            let range_fit = (o.size.x * 3.8 / distance).min(distance / (o.size.x * 3.8));
            visible += 1;
            score += 0.6 + facing + 0.5 * range_fit + 0.4 * orientation;
        }
        let required = if projection {
            self.eyes.len().div_ceil(2).max(1)
        } else {
            1
        };
        (visible >= required).then(|| score / self.eyes.len() as f32)
    }
}
