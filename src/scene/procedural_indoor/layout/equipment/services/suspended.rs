//! Ceiling-backed devices and decorative strands. Disjoint physical reservations
//! preserve the empty space below a strand for people and moving furniture.
use super::*;

pub(super) fn detector_mounted(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    let (lo, hi) = o.bounds();
    let corners = [
        lo.xz(),
        hi.xz(),
        Vec2::new(lo.x, hi.z),
        Vec2::new(hi.x, lo.z),
    ];
    let backed = if scene.generator_version < 189 {
        // Older manifests placed the mounting plate against the nominal roof.
        corners
            .iter()
            .all(|&p| (hi.y - scene.ceiling_height(p) + 0.001).abs() < 0.004)
    } else {
        detector_support_height(scene, o.position.xz(), o.size)
            .is_some_and(|height| (height - hi.y).abs() < 0.0005)
    };
    backed
        && corners.into_iter().all(|p| {
            scene.in_primary_room(Vec3::new(p.x, lo.y, p.y), 0.25)
                && lo.y > scene.floor_height(p) + 2.1
        })
}

fn detector_support_height(scene: &IndoorManifest, center: Vec2, size: Vec3) -> Option<f32> {
    let radius = size.x.min(size.z) * 0.5;
    if let Some(grid) = architecture::ceiling::CeilingGrid::for_scene(scene) {
        return grid.fixture_support(scene, center, radius);
    }
    let (lo, hi) = (center - radius, center + radius);
    if scene.envelope.as_ref().is_some_and(|e| {
        e.soffits.iter().any(|patch| {
            patch.overlaps(lo, hi) && !(patch.min.cmple(lo).all() && patch.max.cmpge(hi).all())
        })
    }) {
        return None;
    }
    // On a pitched slab only the straight mounting sleeve may enter plaster.
    // Seat it to the high side, leaving the complete inlet band below the low side.
    let heights =
        [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)].map(|p| scene.ceiling_height(p));
    let low = heights.into_iter().fold(f32::INFINITY, f32::min);
    let high = heights.into_iter().fold(f32::NEG_INFINITY, f32::max);
    (high - low < signals::fire::detector::sleeve_depth(size) - 0.0003).then_some(high)
}

pub(super) fn string_mounted(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    let p = signals::string_program(o);
    let tf = o.transform();
    if !p.anchors(o.size).all(|q| {
        let q = tf.transform_point(q.with_y(o.size.y));
        [Vec2::ZERO, Vec2::splat(0.008), -Vec2::splat(0.008)]
            .into_iter()
            .all(|d| (q.y - scene.ceiling_height(q.xz() + d) + 0.001).abs() < 0.004)
    }) {
        return false;
    }
    o.obstacle_bounds().all(|(lo, hi)| {
        [
            lo.xz(),
            hi.xz(),
            Vec2::new(lo.x, hi.z),
            Vec2::new(hi.x, lo.z),
        ]
        .into_iter()
        .all(|q| {
            scene.in_primary_room(Vec3::new(q.x, lo.y, q.y), 0.08)
                && hi.y <= scene.ceiling_height(q) + 0.003
                && lo.y > scene.floor_height(q) + 2.12
        }) && lo.y > architecture::ceiling::highest_floor(scene, lo.xz(), hi.xz()) + 2.12
    })
}

impl IndoorManifest {
    pub(in super::super::super::super) fn furnish_detectors(&mut self, structure: &[(Vec3, Vec3)]) {
        let mut r = sampling::Domain::Detectors.rng(self.seed);
        let count = if self.room_size.x * self.room_size.z > 60. && r.random_bool(0.65) {
            2
        } else {
            1
        };
        // Coverage scales with room area. Keep the second fixing free to land
        // at any azimuth while avoiding redundant clusters above one workstation.
        let separation = ((self.room_size.x * self.room_size.z).sqrt()
            * r.random_range(0.23..0.34))
        .clamp(2., 4.5);
        let matched_product = r.random_bool(0.68);
        let mut first: Option<usize> = None;
        let footprint = self.room_size.xz();
        let point = |r: &mut ChaCha8Rng| {
            Vec2::new(r.random_range(-0.38..0.38), r.random_range(-0.38..0.38)) * footprint
        };
        for ordinal in 0..count {
            let mut r = sampling::Domain::Detectors.proposal(self.seed, ordinal);
            let diameter = r.random_range(0.045..0.092) * 2.;
            let size = Vec3::new(diameter, r.random_range(0.025..0.067), diameter);
            let mut at = point(&mut r);
            let mut o = self.candidate(
                ObjectKind::FireDetector,
                Vec3::ZERO,
                size,
                r.random_range(0.0..2. * PI),
                &mut r,
            );
            o.variant = r.random_range(0..4);
            o.solid = false;
            if let Some(index) = first.filter(|_| matched_product) {
                let mate = &self.objects[index];
                o.size = mate.size;
                // As for paired waste bins, the seed identifies a manufactured
                // product. IDs and transforms identify its installed instances.
                o.seed = mate.seed;
                o.variant = mate.variant;
            }
            // Retry the mounting position without re-rolling the purchased device.
            for attempt in 0..24 {
                if attempt != 0 {
                    at = point(&mut r);
                }
                if first.is_some_and(|i| {
                    self.objects[i].position.xz().distance_squared(at) < separation * separation
                }) {
                    continue;
                }
                let Some(height) = detector_support_height(self, at, o.size) else {
                    continue;
                };
                o.position = Vec3::new(at.x, height - o.size.y, at.y);
                if mounted_clear_against(self, &o, structure) {
                    first = Some(o.id);
                    self.objects.push(o);
                    break;
                }
            }
        }
    }
    pub(in super::super::super::super) fn furnish_string_lights(
        &mut self,
        structure: &[(Vec3, Vec3)],
    ) {
        let mut r = sampling::Domain::StringLights.rng(self.seed);
        let social = matches!(self.layout, IndoorLayout::Cafe | IndoorLayout::Gallery);
        if !r.random_bool(if social { 0.78 } else { 0.32 }) {
            return;
        }
        let runs = if social {
            r.random_range(1..=3)
        } else {
            r.random_range(1..=2)
        };
        let mut practical_installed = false;
        for ordinal in 0..runs {
            let mut r = sampling::Domain::StringLights.proposal(self.seed, ordinal);
            let suspended = r.random_bool(0.68);
            let practical = !practical_installed && r.random_bool(0.54);
            for attempt in 0..32 {
                // Wall runs are an explicit alternative, not a floating fallback.
                let volume = suspended && attempt < 28;
                let (size, position, yaw) = if volume {
                    let span = self.room_size.x.min(self.room_size.z);
                    let mut size = Vec3::new(
                        (span * r.random_range(0.42..0.90) * (1. - 0.30 * attempt as f32 / 28.))
                            .clamp(1.8, 7.5),
                        r.random_range(0.34..0.90),
                        (span * r.random_range(0.06..0.46)).clamp(0.24, 3.4),
                    );
                    let mut pos = Vec3::new(
                        r.random_range(-0.36..0.36) * self.room_size.x,
                        0.,
                        r.random_range(-0.36..0.36) * self.room_size.z,
                    );
                    // Scale the requested drop to available headroom. Curves
                    // still receive the exact segment/bulb clearance check.
                    let roof = self.ceiling_height(pos.xz());
                    size.y = size
                        .y
                        .min((roof - self.floor_height(pos.xz()) - 2.18).max(0.20));
                    pos.y = roof - size.y - 0.001;
                    (size, pos, r.random_range(0.0..2. * PI))
                } else {
                    let size = Vec3::new(
                        r.random_range(1.4..4.2),
                        r.random_range(0.24..0.60),
                        r.random_range(0.08..0.16),
                    );
                    let (tf, span) = self.attachment_wall(&mut r);
                    if size.x + 0.25 >= span {
                        continue;
                    }
                    let x =
                        r.random_range((-span + size.x) * 0.5 + 0.1..(span - size.x) * 0.5 - 0.1);
                    let mut pos = tf.transform_point(Vec3::new(x, 0., size.z * 0.5 + 0.001));
                    pos.y = self.ceiling_height(pos.xz()) - size.y - r.random_range(0.25..0.50);
                    (size, pos, tf.rotation.to_euler(EulerRot::YXZ).0)
                };
                if !volume && position.y < 2.12 {
                    continue;
                }
                let mut o = self.candidate(ObjectKind::StringLights, position, size, yaw, &mut r);
                o.variant = u32::from(volume) | (u32::from(practical) << 1);
                o.solid = false;
                if mounted_clear_against(self, &o, structure) {
                    practical_installed |= practical;
                    self.objects.push(o);
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::envelope::FloorPatch;

    #[test]
    fn detector_sleeve_fits_pitched_plaster_but_cannot_bridge_a_soffit_or_enter_the_intakes() {
        let mut scene =
            IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.ceiling_installation = None;
        let y = scene.room_size.y;
        let env = scene.envelope.as_mut().unwrap();
        env.ceiling_drop = Vec2::ZERO;
        env.soffits = vec![FloorPatch {
            min: Vec2::new(0.013, -0.3),
            max: Vec2::new(0.024, 0.3),
            height: y - 0.12,
        }];
        assert!(detector_support_height(&scene, Vec2::ZERO, Vec3::new(0.16, 0.05, 0.16)).is_none());
        assert_eq!(
            detector_support_height(&scene, Vec2::new(0.21, 0.), Vec3::new(0.16, 0.05, 0.16)),
            Some(y)
        );
        scene.envelope.as_mut().unwrap().soffits.clear();
        let size = Vec3::new(0.16, 0.05, 0.16);
        scene.envelope.as_mut().unwrap().ceiling_drop = scene.room_size.xz() * 0.02;
        let mount = detector_support_height(&scene, Vec2::ZERO, size).unwrap();
        let highest = scene
            .ceiling_height(-size.xz() * 0.5)
            .max(scene.ceiling_height(size.xz() * 0.5));
        assert!((mount - highest).abs() < 1e-5);
        let low = scene.minimum_ceiling_height(-size.xz() * 0.5, size.xz() * 0.5);
        assert!(mount > low && mount - low < signals::fire::detector::sleeve_depth(size));
        scene.envelope.as_mut().unwrap().ceiling_drop = scene.room_size.xz() * 0.2;
        assert!(detector_support_height(&scene, Vec2::ZERO, Vec3::new(0.16, 0.05, 0.16)).is_none());
    }
}
